"""融合模块：全局模块（RST+RGAT）+ 局部模块（LLM）"""

import torch
import torch.nn as nn


def map_edu_to_sentences(edu_text: str, premise_sentences: list) -> list:
    """将EDU文本映射回原始premise句子

    Args:
        edu_text: EDU文本（RST解析的子句单元）
        premise_sentences: 原始premise句子列表

    Returns:
        匹配到的句子索引列表
    """
    matched = []
    edu_clean = edu_text.strip()
    if not edu_clean:
        return matched

    for idx, sent in enumerate(premise_sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        # EDU是句子的子串
        if edu_clean in sent_clean or sent_clean in edu_clean:
            matched.append(idx)
            continue
        # 去除标点后匹配
        import re
        edu_no_punc = re.sub(r'[^\w]', '', edu_clean).lower()
        sent_no_punc = re.sub(r'[^\w]', '', sent_clean).lower()
        if edu_no_punc and len(edu_no_punc) > 5 and edu_no_punc in sent_no_punc:
            matched.append(idx)

    return matched


def extract_top_k_edu_sentences(rst_result, attn_weights, premise_sentences, top_k=10):
    """提取注意力最高的Top-K EDU对应的原始句子

    Args:
        rst_result: RST解析结果
        attn_weights: 注意力权重tensor（全节点长度，非EDU位置为0）
        premise_sentences: 原始premise句子列表
        top_k: 选取Top-K个EDU

    Returns:
        selected_sentences: 去重后的句子列表
        sentence_indices: 句子在premise中的原始索引
        top_edu_info: Top-K EDU的详细信息 [(weight, text, sent_idx), ...]
    """
    # 收集EDU节点的权重
    edu_pairs = []
    for j, node in enumerate(rst_result.nodes):
        if node.node_type == 0 and j < len(attn_weights):
            edu_pairs.append((attn_weights[j].item(), j, node))

    # 按权重降序排序
    edu_pairs.sort(key=lambda x: x[0], reverse=True)

    # 取Top-K
    actual_k = min(top_k, len(edu_pairs))
    top_edus = edu_pairs[:actual_k]

    # 映射到原始句子
    selected_indices = set()
    top_edu_info = []

    for weight, j, node in top_edus:
        matched = map_edu_to_sentences(node.text, premise_sentences)
        if matched:
            for idx in matched:
                selected_indices.add(idx)
            top_edu_info.append((weight, node.text, matched[0]))
        else:
            # 无法映射，把EDU文本本身当句子用
            pseudo_idx = len(premise_sentences) + len(top_edu_info)
            selected_indices.add(-len(top_edu_info) - 1)  # 负索引标记
            top_edu_info.append((weight, node.text, -1))

    # 按原始顺序组织选中的句子
    selected_sentences = []
    sentence_indices = []
    for idx in sorted(selected_indices):
        if idx >= 0:
            selected_sentences.append(premise_sentences[idx])
            sentence_indices.append(idx)

    # 补上无法映射的EDU文本
    for weight, text, idx in top_edu_info:
        if idx < 0:
            selected_sentences.append(text)
            sentence_indices.append(idx)

    return selected_sentences, sentence_indices, top_edu_info


def compute_p_local(answer, confidence):
    """计算局部模块的概率映射

    P_local = (C_i + 1) / 2,  当 L_i = 1 (answer = True)
    P_local = (- C_i + 1) / 2,  当 L_i = 0 (answer = False)
    """
    if answer is None or confidence is None:
        return 0.5  # fallback

    if answer:
        return (confidence + 1) / 2
    else:
        return (- confidence + 1) / 2


def fixed_gate(p_global, num_edus, confidence, max_edus=50):
    """固定门控（启发式）

    - EDU多（篇章结构丰富）→ 更信任全局模块
    - 局部置信度高 → 更信任局部模块
    - 全局得分极端 → 更信任全局模块
    """
    edus_norm = min(num_edus / max_edus, 1.0)
    extremity = abs(p_global - 0.5)

    gate = 0.5 + 0.2 * edus_norm - 0.2 * confidence + 0.1 * extremity
    return max(0.1, min(0.9, gate))


class LearnableGate(nn.Module):
    """可学习门控

    gate = sigmoid(w1 * P_global + w2 * num_edus_norm + w3 * C_i + b)
    """

    def __init__(self, max_edus=50):
        super().__init__()
        self.max_edus = max_edus
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.w3 = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, p_global, num_edus, confidence):
        num_edus_norm = min(num_edus, self.max_edus) / self.max_edus
        x = self.w1 * p_global + self.w2 * num_edus_norm + self.w3 * confidence + self.b
        return torch.sigmoid(x)
