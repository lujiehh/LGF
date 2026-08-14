"""
全局模块训练脚本
基于 RST + RGAT + Attention Pooling + Classifier 的 NLI 验证模型
"""
import os
import sys

# 将开源包根目录加入搜索路径，以便 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import random
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from global_module import (
    RSTConfig, ModelConfig, TrainConfig,
    RSTParser, RSTGraphBuilder, EDUEncoder, RGAT,
    AttentionPooling, TripletClassifier,
)


# ============================================================
# 数据集
# ============================================================

class GlobalDataset(Dataset):
    """全局验证数据集

    数据格式: NLI_Input_total.json
    每个样本: {
        "premise": ["sentence1", "sentence2", ...],
        "hypothesis": "假设文本",
        "is_negative": false/true
    }
    """

    def __init__(
        self,
        data_path: str,
        rst_parser: RSTParser,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.0,
        seed: int = 42,
        cache_path: str = None,
    ):
        self.data_path = data_path
        self.split = split
        self.rst_parser = rst_parser
        self.samples: List[Dict] = []

        if cache_path is None:
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            cache_dir = os.path.dirname(data_path)
            cache_path = os.path.join(cache_dir, f"{base_name}_rst_cache.pkl")
        self.cache_path = cache_path

        self._load_data(train_ratio, val_ratio, seed)

    def _load_data(self, train_ratio: float, val_ratio: float, seed: int):
        with open(self.data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        random.seed(seed)
        random.shuffle(all_data)

        n = len(all_data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if self.split == "train":
            data = all_data[:train_end]
        elif self.split == "val":
            data = all_data[train_end:val_end]
        else:
            data = all_data[val_end:]

        print(f"[{self.split}] Loading {len(data)} samples...", flush=True)

        cache = self._load_cache()
        cache_updated = False

        for i, item in enumerate(data):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)} samples", flush=True)

            premise = item.get("premise", [])
            hypothesis = item.get("hypothesis", "")
            is_negative = item.get("is_negative", False)

            if isinstance(premise, list):
                doc_text = " ".join(premise)
            else:
                doc_text = str(premise)

            cache_key = doc_text
            if cache is not None and cache_key in cache:
                rst_result = cache[cache_key]
            else:
                try:
                    rst_result = self.rst_parser.parse(doc_text)
                except Exception as e:
                    print(f"  Warning: RST parse failed for sample {i}: {e}", flush=True)
                    continue

                if rst_result.num_nodes == 0:
                    continue

                if cache is None:
                    cache = {}
                cache[cache_key] = rst_result
                cache_updated = True

            self.samples.append({
                "rst_result": rst_result,
                "hypothesis": hypothesis,
                "label": 0.0 if is_negative else 1.0,
            })

        if cache_updated:
            self._save_cache(cache)

        print(f"[{self.split}] Loaded {len(self.samples)} valid samples", flush=True)

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            print(f"[{self.split}] Loading RST cache from {self.cache_path}", flush=True)
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        print(f"[{self.split}] No RST cache found, will parse from scratch", flush=True)
        return None

    def _save_cache(self, cache: dict):
        print(f"[{self.split}] Saving RST cache to {self.cache_path} ({len(cache)} entries)", flush=True)
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(cache, f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    hypotheses = []
    labels = []
    rst_results = []

    for sample in batch:
        hypotheses.append(sample["hypothesis"])
        labels.append(sample["label"])
        rst_results.append(sample["rst_result"])

    return {
        "hypotheses": hypotheses,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "rst_results": rst_results,
    }


# ============================================================
# 训练器
# ============================================================

class GlobalTrainer:
    """全局模块训练器"""

    def __init__(self, train_config=None, model_config=None, rst_config=None):
        self.train_config = train_config or TrainConfig()
        self.model_config = model_config or ModelConfig()
        self.rst_config = rst_config or RSTConfig()
        self.device = torch.device(self.train_config.device)

        self.rst_parser = RSTParser(self.rst_config)
        self.graph_builder = RSTGraphBuilder(self.rst_config)
        self.encoder = EDUEncoder(self.model_config).to(self.device)

        self.rgat = None
        self.classifier = TripletClassifier(self.model_config).to(self.device)
        self.attention_pooling = None

        self.criterion = nn.BCELoss()
        self.optimizer = None
        self.best_f1 = 0.0

    def train(self):
        """训练主循环"""
        train_dataset = GlobalDataset(
            data_path=self.train_config.data_path,
            rst_parser=self.rst_parser,
            split="train",
            train_ratio=self.train_config.train_ratio,
            val_ratio=self.train_config.val_ratio,
            seed=self.train_config.seed,
        )

        # 收集所有关系类型
        all_relations = set()
        relation_counter = Counter()
        for sample in train_dataset:
            for rel in sample["rst_result"].relation_types:
                all_relations.add(rel)
                relation_counter[rel] += 1

        rel_list = sorted(all_relations)
        print(f"\n{'='*60}")
        print(f"Collected {len(rel_list)} relation types from training data:")
        for rel, count in relation_counter.most_common():
            print(f"  {rel}: {count}")
        print(f"{'='*60}")

        # 创建 RGAT 和 Attention Pooling
        self.rgat = RGAT(self.model_config, rel_list).to(self.device)
        self.attention_pooling = AttentionPooling(self.model_config.hidden_dim).to(self.device)

        # 优化器
        params = []
        for module in [self.encoder, self.rgat, self.classifier, self.attention_pooling]:
            params.extend([p for p in module.parameters() if p.requires_grad])
        self.optimizer = Adam(params, lr=self.train_config.learning_rate)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # 加载测试集
        test_dataset = GlobalDataset(
            data_path=self.train_config.data_path,
            rst_parser=self.rst_parser,
            split="test",
            train_ratio=self.train_config.train_ratio,
            val_ratio=self.train_config.val_ratio,
            seed=self.train_config.seed,
        )

        print(f"\nTraining on {len(train_dataset)} samples, {len(train_loader)} batches per epoch")
        print(f"Testing on {len(test_dataset)} samples")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_config.batch_size}")
        print(f"Learning rate: {self.train_config.learning_rate}")

        for epoch in range(self.train_config.num_epochs):
            self.encoder.train()
            self.rgat.train()
            self.classifier.train()
            self.attention_pooling.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                loss, batch_correct, batch_total = self._train_step(batch, epoch, batch_idx)

                total_loss += loss
                correct += batch_correct
                total += batch_total

                if (batch_idx + 1) % 50 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    running_acc = correct / max(total, 1)
                    print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {avg_loss:.4f} | Acc: {running_acc:.4f}", flush=True)

            avg_loss = total_loss / max(len(train_loader), 1)
            acc = correct / max(total, 1)
            print(f"Epoch {epoch+1}/{self.train_config.num_epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}", flush=True)

            # 每 epoch 评估并保存最优模型
            metrics = self.evaluate(test_dataset)
            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.save_model("best_model.pt")
                print(f"  -> New best F1: {self.best_f1:.4f}, model saved!")

        print(f"\n{'='*60}")
        print("Relation statistics during training:")
        for rel, count in self.graph_builder.get_relation_stats().items():
            print(f"  {rel}: {count}")
        print(f"{'='*60}")
        print(f"Training complete. Best F1: {self.best_f1:.4f}")

    def _train_step(self, batch, epoch, batch_idx):
        """单步训练"""
        self.optimizer.zero_grad()

        rst_results = batch["rst_results"]
        hypotheses = batch["hypotheses"]
        labels = batch["labels"].to(self.device)

        batch_graph_repr = []
        hypo_feats = self.encoder.encode_hypotheses_batch(hypotheses)

        for i, rst_result in enumerate(rst_results):
            # 第1个epoch前3个样本打印RST树
            if epoch == 0 and batch_idx == 0 and i < 3:
                tree_str = self.rst_parser.format_tree(rst_result)
                print(f"\n--- RST Tree (sample {i}) ---")
                print(tree_str)
                print(f"  Nodes: {rst_result.num_nodes} (EDU: {rst_result.num_edus}, Branch: {rst_result.num_branches})")
                print(f"  Edges: {len(rst_result.edges)}")
                print(f"  Relations: {rst_result.relation_types}")
                print(f"  Hypothesis: {hypotheses[i]}")
                print(f"  Label: {labels[i].item():.0f}")

            all_node_feats = self.encoder.encode_all_nodes(rst_result.nodes, rst_result.edges)
            g = self.graph_builder.build_graph(rst_result, all_node_feats)
            g = g.to(self.device)

            node_feats_updated = self.rgat(g, g.ndata["feat"])
            node_types = g.ndata["node_type"]

            query = hypo_feats[i]
            graph_repr, attn_weights = self.attention_pooling(node_feats_updated, query, node_types)

            if epoch == 0 and batch_idx == 0 and i < 3:
                self._print_attention(rst_result, attn_weights)

            batch_graph_repr.append(graph_repr)

        graph_feats = torch.stack(batch_graph_repr)
        p_global = self.classifier(graph_feats, hypo_feats)

        loss = self.criterion(p_global, labels)
        loss.backward()
        self.optimizer.step()

        preds = (p_global > 0.5).float()
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        return loss.item(), correct, total

    def evaluate(self, dataset):
        """评估模型"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        self.encoder.eval()
        self.rgat.eval()
        self.classifier.eval()
        self.attention_pooling.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                rst_results = batch["rst_results"]
                hypotheses = batch["hypotheses"]
                labels = batch["labels"].to(self.device)

                batch_graph_repr = []
                hypo_feats = self.encoder.encode_hypotheses_batch(hypotheses)

                for i, rst_result in enumerate(rst_results):
                    all_node_feats = self.encoder.encode_all_nodes(rst_result.nodes, rst_result.edges)
                    g = self.graph_builder.build_graph(rst_result, all_node_feats)
                    g = g.to(self.device)

                    node_feats_updated = self.rgat(g, g.ndata["feat"])
                    node_types = g.ndata["node_type"]
                    query = hypo_feats[i]
                    graph_repr, _ = self.attention_pooling(node_feats_updated, query, node_types)
                    batch_graph_repr.append(graph_repr)

                graph_feats = torch.stack(batch_graph_repr)
                p_global = self.classifier(graph_feats, hypo_feats)
                loss = self.criterion(p_global, labels)
                total_loss += loss.item()

                preds = (p_global > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n[test] Loss: {total_loss / len(dataloader):.4f} | "
              f"Acc: {accuracy:.4f} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")
        print(f"Confusion Matrix:")
        print(f"              Pred=0  Pred=1")
        print(f"  Actual=0   {cm[0][0]:>5d}   {cm[0][1]:>5d}")
        print(f"  Actual=1   {cm[1][0]:>5d}   {cm[1][1]:>5d}")

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
        }

    def save_model(self, filename: str):
        """保存模型权重"""
        import config
        save_dir = config.OUTPUT_DIR
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        state = {
            "encoder": self.encoder.state_dict(),
            "rgat": self.rgat.state_dict(),
            "classifier": self.classifier.state_dict(),
            "attention_pooling": self.attention_pooling.state_dict(),
            "rel_list": self.rgat.rel_names if self.rgat else [],
        }
        torch.save(state, path)

    def _print_attention(self, rst_result, attn_weights):
        """打印注意力分布"""
        print(f"\n  --- Attention Weights ---")
        for j, node in enumerate(rst_result.nodes):
            if j < len(attn_weights):
                weight = attn_weights[j].item()
                if node.node_type == 0:
                    print(f"    EDU[{node.id}] w={weight:.4f}: {node.text}")
                else:
                    print(f"    Branch[{node.id}] w={weight:.4f}: {node.text}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import config

    # 设备：读取配置；RST 解析器与训练共用同一 device 字符串。
    device = config.DEVICE
    cuda_device = int(device.split(":")[1]) if device.startswith("cuda") else -1

    rst_config = RSTConfig(cuda_device=cuda_device)
    model_config = ModelConfig()
    train_config = TrainConfig(
        data_path=config.DATA_PATH,
        batch_size=config.TRAIN_BATCH_SIZE,
        learning_rate=config.TRAIN_LEARNING_RATE,
        num_epochs=config.TRAIN_NUM_EPOCHS,
        device=device,
        seed=config.TRAIN_SEED,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.TRAIN_VAL_RATIO,
    )

    trainer = GlobalTrainer(
        train_config=train_config,
        model_config=model_config,
        rst_config=rst_config,
    )
    trainer.train()
