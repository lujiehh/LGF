"""融合推理：全局模块注意力 + 局部模块LLM + 门控融合"""
import os
import sys

# 将开源包根目录加入搜索路径，以便 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
import requests
from datetime import datetime, timedelta

import torch
from torch.utils.data import DataLoader

from global_module import (
    RSTConfig, ModelConfig,
    RSTParser, RSTGraphBuilder, EDUEncoder, RGAT,
    AttentionPooling, TripletClassifier,
)
from train_global import GlobalDataset, collate_fn
from fusion_module import (
    extract_top_k_edu_sentences, compute_p_local, fixed_gate, LearnableGate,
)


# ============================================================
# LLM 调用（从 local_1.py 复用）
# ============================================================

import config
API_ENDPOINT = config.LLM_API_ENDPOINT
MODEL_NAME = config.LLM_MODEL_NAME


def call_local_llm(prompt, system_prompt=None, max_tokens=1024, temperature=0.0, top_p=1.0):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,
        "top_logprobs": 1,
    }
    if system_prompt:
        payload["messages"].append({"role": "system", "content": system_prompt})
    payload["messages"].append({"role": "user", "content": prompt})

    try:
        response = requests.post(f"{API_ENDPOINT}/chat/completions", headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()
            confidence = None
            choice = result["choices"][0]
            if "logprobs" in choice and choice["logprobs"] is not None:
                if "content" in choice["logprobs"] and len(choice["logprobs"]["content"]) > 0:
                    total_logprob = sum(t["logprob"] for t in choice["logprobs"]["content"] if "logprob" in t)
                    avg_logprob = total_logprob / len(choice["logprobs"]["content"])
                    confidence = min(max(0, 10**avg_logprob), 1)
            return generated_text, confidence
        else:
            print(f"  LLM API失败: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"  LLM API异常: {e}")
        return None, None


def parse_llm_answer(response):
    """解析LLM输出为True/False"""
    if response is None:
        return None
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', response, flags=re.DOTALL)
    if "true" in cleaned.lower():
        return True
    elif "false" in cleaned.lower():
        return False
    return None


def get_entity_sentences(premise, head_entity, tail_entity):
    """实体句子选择（fallback策略）"""
    entity_sentences = []
    seen = set()
    for sentence in premise:
        if (head_entity in sentence or tail_entity in sentence) and sentence not in seen:
            entity_sentences.append(sentence)
            seen.add(sentence)
    return entity_sentences if entity_sentences else premise


# ============================================================
# 全局模块推理
# ============================================================

def load_global_model(model_path, device, rel_list):
    model_config = ModelConfig()
    encoder = EDUEncoder(model_config).to(device)
    rgat = RGAT(model_config, rel_list).to(device)
    classifier = TripletClassifier(model_config).to(device)
    attention_pooling = AttentionPooling(model_config.hidden_dim).to(device)

    state = torch.load(model_path, map_location=device)
    encoder.load_state_dict(state["encoder"])
    rgat.load_state_dict(state["rgat"])
    classifier.load_state_dict(state["classifier"])
    attention_pooling.load_state_dict(state["attention_pooling"])

    encoder.eval()
    rgat.eval()
    classifier.eval()
    attention_pooling.eval()
    return encoder, rgat, classifier, attention_pooling


def global_inference_single(rst_result, hypothesis, encoder, rgat, classifier, attention_pooling, graph_builder, device):
    """对单个样本跑全局模块推理"""
    with torch.no_grad():
        hypo_feats = encoder.encode_hypotheses_batch([hypothesis])
        all_node_feats = encoder.encode_all_nodes(rst_result.nodes, rst_result.edges)
        g = graph_builder.build_graph(rst_result, all_node_feats)
        g = g.to(device)

        node_feats_updated = rgat(g, g.ndata["feat"])
        node_types = g.ndata["node_type"]
        query = hypo_feats[0]
        graph_repr, attn_weights = attention_pooling(node_feats_updated, query, node_types)

        p_global = classifier(graph_repr.unsqueeze(0), query.unsqueeze(0)).item()
        num_edus = rst_result.num_edus

    return p_global, attn_weights, num_edus


# ============================================================
# 主流程
# ============================================================

def main():
    t_start = time.time()
    device = torch.device(config.DEVICE)
    model_path = config.GLOBAL_MODEL_PATH
    data_path = config.DATA_PATH
    output_path = config.FUSION_RESULT_PATH
    cuda_device = int(config.DEVICE.split(":")[1]) if config.DEVICE.startswith("cuda") else -1

    # 加载全局模块
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载全局模块...")
    state = torch.load(model_path, map_location=device)
    rel_list = state["rel_list"]
    rst_config = RSTConfig(cuda_device=cuda_device)
    rst_parser = RSTParser(rst_config)
    graph_builder = RSTGraphBuilder(rst_config)
    encoder, rgat, classifier, attention_pooling = load_global_model(model_path, device, rel_list)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 全局模块加载完成")

    # 加载原始数据
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 和训练时一致的划分
    import random
    random.seed(42)
    random.shuffle(all_data)
    n = len(all_data)
    test_data = all_data[int(n * 0.7):]  # train_ratio=0.7
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 测试集: {len(test_data)} 样本")

    # 加载RST缓存
    cache_path = os.path.join(os.path.dirname(data_path), "NLI_Input_total_1_rst_cache.pkl")
    cache = None
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] RST缓存: {len(cache)} 条")

    results = []
    total = len(test_data)

    system_prompt = "你是一个逻辑推理专家，擅长判断前提句是否能直接推断出假设句。"

    for i, item in enumerate(test_data):
        premise = item.get("premise", [])
        hypothesis = item.get("hypothesis", "")
        is_negative = item.get("is_negative", False)
        label = 0.0 if is_negative else 1.0

        # 提取实体（fallback用）
        head_entity = ""
        tail_entity = ""
        if "head_type_pair" in item and isinstance(item["head_type_pair"], list) and len(item["head_type_pair"]) >= 2:
            head_entity = item["head_type_pair"][0]
        if "tail_type_pair" in item and isinstance(item["tail_type_pair"], list) and len(item["tail_type_pair"]) >= 2:
            tail_entity = item["tail_type_pair"][0]

        if isinstance(premise, list):
            doc_text = " ".join(premise)
        else:
            doc_text = str(premise)
            premise = [doc_text]

        # ---- Stage 1: 全局模块推理 ----
        rst_result = None
        p_global = None
        attn_weights = None
        num_edus = 0
        attention_sentences = []
        attention_sentence_indices = []

        cache_key = doc_text
        if cache is not None and cache_key in cache:
            rst_result = cache[cache_key]

        if rst_result is None or rst_result.num_nodes == 0:
            try:
                rst_result = rst_parser.parse(doc_text)
            except Exception as e:
                rst_result = None

        if rst_result is not None and rst_result.num_nodes > 0:
            p_global, attn_weights, num_edus = global_inference_single(
                rst_result, hypothesis, encoder, rgat, classifier, attention_pooling, graph_builder, device
            )

            # ---- Stage 2: Top-K EDU 映射回句子 ----
            selected_sents, sent_indices, top_edu_info = extract_top_k_edu_sentences(
                rst_result, attn_weights, premise, top_k=10
            )
            if selected_sents:
                attention_sentences = selected_sents
                attention_sentence_indices = sent_indices
            else:
                # EDU映射失败，直接用EDU文本
                attention_sentences = [node.text for node in rst_result.nodes if node.node_type == 0][:10]
                attention_sentence_indices = []

        # ---- Stage 3: 局部模块推理 ----
        # 选择前提句子：优先用attention选择的，fallback到实体选择
        if attention_sentences:
            premise_for_llm = ' '.join(attention_sentences)
            selection_strategy = "attention"
        else:
            entity_sents = get_entity_sentences(premise, head_entity, tail_entity)
            premise_for_llm = ' '.join(entity_sents)
            selection_strategy = "entity"

        user_prompt = f"""
请判断是否可以通过前提句直接推断出假设句。
前提：{premise_for_llm}
假设：{hypothesis}

请只输出true或false，不要添加任何其他解释：
"""

        llm_response, llm_confidence = call_local_llm(user_prompt, system_prompt=system_prompt)
        llm_answer = parse_llm_answer(llm_response)

        # ---- Stage 4: 门控融合 ----
        p_local = compute_p_local(llm_answer, llm_confidence)

        if p_global is not None and llm_answer is not None:
            gate = fixed_gate(p_global, num_edus, llm_confidence if llm_confidence is not None else 0.5)
            p_final = gate * p_global + (1 - gate) * p_local
        elif p_global is not None:
            # LLM失败，只信全局
            gate = 1.0
            p_final = p_global
        elif llm_answer is not None:
            # 全局失败，只信局部
            gate = 0.0
            p_final = p_local
        else:
            # 都失败
            gate = 0.5
            p_final = 0.5

        final_pred = 1.0 if p_final > 0.5 else 0.0

        result = {
            "idx": i,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "head_entity": head_entity,
            "tail_entity": tail_entity,
            "selection_strategy": selection_strategy,
            "attention_sentences": attention_sentences,
            "num_edus": num_edus,
            "p_global": p_global,
            "llm_answer": llm_answer,
            "llm_confidence": llm_confidence,
            "p_local": p_local,
            "gate": gate,
            "p_final": p_final,
            "final_pred": final_pred,
        }
        results.append(result)

        # 进度输出
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_start
            avg = elapsed / (i + 1)
            remaining = avg * (total - i - 1)
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i+1}/{total} | "
                  f"已用 {elapsed:.0f}s | 预计剩余 {remaining:.0f}s | "
                  f"预计完成 {eta.strftime('%H:%M:%S')}", flush=True)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 计算指标
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    labels = [r["label"] for r in results]
    preds = [r["final_pred"] for r in results]

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    p = precision_score(labels, preds, average="macro", zero_division=0)
    r = recall_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)

    # 对比：global-only
    global_preds = [1.0 if r["p_global"] is not None and r["p_global"] > 0.5 else 0.0 for r in results]
    g_acc = accuracy_score(labels, global_preds)
    g_f1 = f1_score(labels, global_preds, average="macro", zero_division=0)
    g_p = precision_score(labels, global_preds, average="macro", zero_division=0)
    g_r = recall_score(labels, global_preds, average="macro", zero_division=0)
    g_cm = confusion_matrix(labels, global_preds)

    # 对比：local-only
    local_preds = [1.0 if r["llm_answer"] is True else 0.0 for r in results]
    l_acc = accuracy_score(labels, local_preds)
    l_f1 = f1_score(labels, local_preds, average="macro", zero_division=0)
    l_p = precision_score(labels, local_preds, average="macro", zero_division=0)
    l_r = recall_score(labels, local_preds, average="macro", zero_division=0)
    l_cm = confusion_matrix(labels, local_preds)

    print(f"\n{'='*60}")
    print(f"全局模块结果:")
    print(f"  Acc: {g_acc:.4f} | F1: {g_f1:.4f} | P: {g_p:.4f} | R: {g_r:.4f}")
    print(f"  Confusion Matrix:")
    print(f"              Pred=0  Pred=1")
    print(f"  Actual=0   {g_cm[0][0]:>5d}   {g_cm[0][1]:>5d}")
    print(f"  Actual=1   {g_cm[1][0]:>5d}   {g_cm[1][1]:>5d}")

    print(f"\n局部模块结果:")
    print(f"  Acc: {l_acc:.4f} | F1: {l_f1:.4f} | P: {l_p:.4f} | R: {l_r:.4f}")
    print(f"  Confusion Matrix:")
    print(f"              Pred=0  Pred=1")
    print(f"  Actual=0   {l_cm[0][0]:>5d}   {l_cm[0][1]:>5d}")
    print(f"  Actual=1   {l_cm[1][0]:>5d}   {l_cm[1][1]:>5d}")

    print(f"\n融合结果:")
    print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | P: {p:.4f} | R: {r:.4f}")
    print(f"  Confusion Matrix:")
    print(f"              Pred=0  Pred=1")
    print(f"  Actual=0   {cm[0][0]:>5d}   {cm[0][1]:>5d}")
    print(f"  Actual=1   {cm[1][0]:>5d}   {cm[1][1]:>5d}")

    # 策略统计
    attn_count = sum(1 for r in results if r["selection_strategy"] == "attention")
    entity_count = sum(1 for r in results if r["selection_strategy"] == "entity")
    print(f"\n句子选择策略: attention={attn_count}, entity={entity_count}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 完成，总耗时 {time.time()-t_start:.1f}s")
    print(f"结果保存到: {output_path}")


if __name__ == "__main__":
    main()