"""
全局验证模块 - 独立脚本版本
整合 RST 解析、图构建、编码、RGAT、注意力池化、分类器为单一脚本
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, HeteroGraphConv
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 配置
# ============================================================

@dataclass
class RSTConfig:
    rst_tool_path: str = "isanlp_rst"
    cuda_device: int = 4
    rst_relations: list = field(default_factory=list)


@dataclass
class ModelConfig:
    encoder_name: str = "xlm-roberta-base"
    encoder_dim: int = 768
    hidden_dim: int = 256
    num_rgat_heads_layer1: int = 4
    num_rgat_heads_layer2: int = 1
    classifier_hidden_dim: int = 128
    dropout: float = 0.1
    freeze_encoder: bool = True


@dataclass
class TrainConfig:
    data_path: str = "Long-Premise_Textual_Entailment/SummaC/NLI_Input_total.json"
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 20
    device: str = "cpu"
    seed: int = 42
    max_edu_length: int = 512
    train_ratio: float = 0.7
    val_ratio: float = 0.0


# ============================================================
# RST 解析器 (isanlp-rst)
# ============================================================

@dataclass
class RSTNode:
    """RST树节点"""
    id: int
    text: str = ""
    node_type: int = 0  # 0=EDU叶节点, 1=Branch分支节点
    relation: str = ""
    nuclearity: str = ""


@dataclass
class RSTEdge:
    """RST树边"""
    src: int
    dst: int
    relation: str


@dataclass
class RSTParseResult:
    """RST解析结果"""
    nodes: List[RSTNode] = field(default_factory=list)
    edges: List[RSTEdge] = field(default_factory=list)
    text: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edus(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == 0)

    @property
    def num_branches(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == 1)

    @property
    def relation_types(self) -> List[str]:
        return list(set(e.relation for e in self.edges))


class RSTParser:
    """isanlp-rst RST解析器"""

    def __init__(self, config: RSTConfig = None):
        self.config = config or RSTConfig()
        self._parser = None

    def _init_parser(self):
        if self._parser is None:
            from isanlp_rst.parser import Parser
            cuda_device = self.config.cuda_device if self.config.cuda_device >= 0 else -1
            self._parser = Parser(
                hf_model_name="tchewik/isanlp_rst_v3",
                hf_model_version="rstdt",
                cuda_device=cuda_device,
            )

    def parse(self, text: str) -> RSTParseResult:
        self._init_parser()
        result = self._parser(text)
        rst_trees = result.get("rst", [])
        if not rst_trees:
            return RSTParseResult(text=text)

        root = rst_trees[0]
        parse_result = RSTParseResult(text=text)
        self._traverse(root, parse_result)
        return parse_result

    def _traverse(self, unit, result: RSTParseResult, parent_id: Optional[int] = None):
        from isanlp.annotation_rst import DiscourseUnit
        node_id = len(result.nodes)

        if unit.left is None and unit.right is None:
            edu_text = unit.text.strip() if hasattr(unit, "text") and unit.text else ""
            result.nodes.append(RSTNode(id=node_id, text=edu_text, node_type=0, relation="", nuclearity=""))
        else:
            branch_text = unit.text.strip() if hasattr(unit, "text") and unit.text else ""
            result.nodes.append(RSTNode(
                id=node_id, text=branch_text, node_type=1,
                relation=unit.relation.lower() if unit.relation else "span",
                nuclearity=unit.nuclearity if unit.nuclearity else "NN",
            ))
            for child in [unit.left, unit.right]:
                if child is not None:
                    edge_relation = self._make_edge_relation(unit, child)
                    result.edges.append(RSTEdge(src=node_id, dst=len(result.nodes), relation=edge_relation))
                    self._traverse(child, result, parent_id=node_id)

    def _make_edge_relation(self, parent, child) -> str:
        relation = parent.relation.lower() if parent.relation else "span"
        nuclearity = parent.nuclearity if parent.nuclearity else "NN"
        return f"{relation}_{nuclearity}"

    def format_tree(self, result: RSTParseResult, indent: int = 0) -> str:
        lines = []
        children_map = {}
        for edge in result.edges:
            children_map.setdefault(edge.src, []).append(edge)

        def _print(node_id, depth):
            node = result.nodes[node_id]
            prefix = "  " * depth
            if node.node_type == 0:
                lines.append(f"{prefix}EDU[{node.id}]: {node.text}")
            else:
                lines.append(f"{prefix}Branch[{node.id}] ({node.relation}/{node.nuclearity})")
                for edge in children_map.get(node_id, []):
                    _print(edge.dst, depth + 1)

        child_ids = set(e.dst for e in result.edges)
        root_id = None
        for node in result.nodes:
            if node.node_type == 1 and node.id not in child_ids:
                root_id = node.id
                break
        if root_id is not None:
            _print(root_id, 0)
        else:
            for node in result.nodes:
                if node.node_type == 0:
                    lines.append(f"EDU[{node.id}]: {node.text}")
        return "\n".join(lines)


# ============================================================
# RST 解析器 (Stanza DM-RST)
# ============================================================

@dataclass
class EDU:
    text: str
    start_idx: int
    end_idx: int
    index: int = -1


@dataclass
class RSTNodeStanza:
    node_id: int
    is_leaf: bool
    text: str = ""
    edu: Optional[EDU] = None
    relation: str = "Root"
    nuclearity: str = "nucleus"
    children: List['RSTNodeStanza'] = field(default_factory=list)
    parent: Optional['RSTNodeStanza'] = None
    span_start: int = -1
    span_end: int = -1

    def leaf_nodes(self) -> List['RSTNodeStanza']:
        if self.is_leaf:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.leaf_nodes())
        return sorted(result, key=lambda n: n.span_start)

    def branch_nodes(self) -> List['RSTNodeStanza']:
        if self.is_leaf:
            return []
        result = [self]
        for child in self.children:
            result.extend(child.branch_nodes())
        return result


@dataclass
class RSTEdgeStanza:
    source_id: int
    target_id: int
    relation: str
    nuclearity: str


class RSTParserStanza:
    """RST 解析器 - 使用 Stanza 或简化版本"""

    def __init__(self, use_stanza: bool = True):
        self.use_stanza = use_stanza
        self._stanza_pipeline = None

    def _get_stanza_pipeline(self):
        if self._stanza_pipeline is None:
            import stanza
            stanza.download('en', verbose=False)
            self._stanza_pipeline = stanza.Pipeline(
                'en', processors='tokenize,pos,constituency,rst', verbose=False
            )
        return self._stanza_pipeline

    def parse(self, text: str) -> Tuple[List[EDU], RSTNodeStanza, List[RSTEdgeStanza]]:
        if self.use_stanza:
            try:
                return self._parse_with_stanza(text)
            except Exception as e:
                print(f"Stanza RST parsing failed: {e}, falling back to simple parser")
        return self._parse_simple(text)

    def _parse_with_stanza(self, text: str):
        nlp = self._get_stanza_pipeline()
        doc = nlp(text)
        edus = []
        edu_idx = 0
        for sentence in doc.sentences:
            if hasattr(sentence, 'constituency') and sentence.constituency is not None:
                edu_texts = self._extract_edus_from_constituency(sentence.constituency, sentence.text)
                char_offset = text.find(sentence.text)
                for edu_text in edu_texts:
                    start = text.find(edu_text, char_offset)
                    end = start + len(edu_text)
                    edus.append(EDU(text=edu_text, start_idx=start, end_idx=end, index=edu_idx))
                    edu_idx += 1
                    char_offset = end
            else:
                start = text.find(sentence.text)
                end = start + len(sentence.text)
                edus.append(EDU(text=sentence.text, start_idx=start, end_idx=end, index=edu_idx))
                edu_idx += 1
        if not edus:
            return self._parse_simple(text)
        root, edges = self._build_rst_tree(edus)
        return edus, root, edges

    def _extract_edus_from_constituency(self, tree, sentence_text: str) -> List[str]:
        edus = []
        if hasattr(tree, 'children') and tree.children:
            for child in tree.children:
                child_edus = self._extract_edus_from_constituency(child, sentence_text)
                edus.extend(child_edus)
        elif hasattr(tree, 'label') and tree.label:
            edus.append(str(tree))
        else:
            edus.append(sentence_text)
        return edus if edus else [sentence_text]

    def _build_rst_tree(self, edus: List[EDU]):
        node_id_counter = [0]
        edges = []

        def make_leaf_node(edu: EDU) -> RSTNodeStanza:
            node_id = node_id_counter[0]
            node_id_counter[0] += 1
            return RSTNodeStanza(
                node_id=node_id, is_leaf=True, text=edu.text, edu=edu,
                relation="Root", nuclearity="nucleus",
                span_start=edu.index, span_end=edu.index,
            )

        def make_branch_node(children: List[RSTNodeStanza], relation: str = "Elaboration") -> RSTNodeStanza:
            node_id = node_id_counter[0]
            node_id_counter[0] += 1
            span_start = min(c.span_start for c in children)
            span_end = max(c.span_end for c in children)
            node = RSTNodeStanza(
                node_id=node_id, is_leaf=False,
                text=" ".join(c.text for c in children),
                relation=relation, nuclearity="nucleus",
                span_start=span_start, span_end=span_end, children=children,
            )
            for child in children:
                child.parent = node
            return node

        nodes = [make_leaf_node(edu) for edu in edus]
        relation_cycle = ["Elaboration", "Explanation", "Background", "Cause", "Condition"]

        while len(nodes) > 1:
            new_nodes = []
            i = 0
            pair_idx = 0
            while i < len(nodes):
                if i + 1 < len(nodes):
                    relation = relation_cycle[pair_idx % len(relation_cycle)]
                    branch = make_branch_node([nodes[i], nodes[i + 1]], relation)
                    edges.append(RSTEdgeStanza(branch.node_id, nodes[i].node_id, relation, "nucleus"))
                    edges.append(RSTEdgeStanza(branch.node_id, nodes[i + 1].node_id, relation, "satellite"))
                    edges.append(RSTEdgeStanza(nodes[i].node_id, branch.node_id, relation, "nucleus"))
                    edges.append(RSTEdgeStanza(nodes[i + 1].node_id, branch.node_id, relation, "satellite"))
                    new_nodes.append(branch)
                    i += 2
                    pair_idx += 1
                else:
                    new_nodes.append(nodes[i])
                    i += 1
            nodes = new_nodes

        root = nodes[0]
        root.relation = "Root"
        return root, edges

    def _parse_simple(self, text: str):
        sentences = self._split_sentences(text)
        edus = []
        current_idx = 0
        for i, sent in enumerate(sentences):
            edu = EDU(text=sent.strip(), start_idx=current_idx, end_idx=current_idx + len(sent), index=i)
            edus.append(edu)
            current_idx += len(sent) + 1
        if not edus:
            edu = EDU(text=text, start_idx=0, end_idx=len(text), index=0)
            edus = [edu]
        root, edges = self._build_rst_tree(edus)
        return edus, root, edges

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# ============================================================
# RST 图构建器
# ============================================================

class RSTGraphBuilder:
    """将RSTParseResult转为DGL异构图"""

    def __init__(self, rst_config=None):
        self.rst_config = rst_config
        self.relation_counter = Counter()

    def build_graph(self, rst_result: RSTParseResult, node_embeddings: torch.Tensor) -> dgl.DGLGraph:
        if rst_result.num_nodes == 0:
            raise ValueError("Cannot build graph from empty RST result")

        num_nodes = rst_result.num_nodes
        node_types = torch.tensor([n.node_type for n in rst_result.nodes])

        # 收集所有边，按关系类型分组
        edge_dict = {}
        for edge in rst_result.edges:
            etype = edge.relation
            if etype not in edge_dict:
                edge_dict[etype] = ([], [])
            edge_dict[etype][0].append(edge.src)
            edge_dict[etype][1].append(edge.dst)
            self.relation_counter[etype] += 1

        # 添加反向边
        for etype, (srcs, dsts) in list(edge_dict.items()):
            rev_etype = f"rev_{etype}"
            if rev_etype not in edge_dict:
                edge_dict[rev_etype] = (list(dsts), list(srcs))

        # 构建DGL异构图
        graph_data = {}
        for etype, (srcs, dsts) in edge_dict.items():
            src_tensor = torch.tensor(srcs, dtype=torch.int64)
            dst_tensor = torch.tensor(dsts, dtype=torch.int64)
            graph_data[("node", etype, "node")] = (src_tensor, dst_tensor)

        if not graph_data:
            graph_data[("node", "self_loop", "node")] = (
                torch.arange(num_nodes), torch.arange(num_nodes)
            )

        g = dgl.heterograph(graph_data)

        # 设置节点特征（CPU上）
        g.nodes["node"].data["feat"] = node_embeddings.cpu()
        g.nodes["node"].data["node_type"] = node_types

        # 转为同构图
        g = dgl.to_homogeneous(g, ndata=["feat", "node_type"])
        return g

    def get_relation_stats(self) -> dict:
        return dict(self.relation_counter.most_common())

    def reset_relation_counter(self):
        self.relation_counter.clear()


# ============================================================
# EDU 编码器
# ============================================================

class EDUEncoder(nn.Module):
    """逐EDU编码文档节点 + 编码假设"""

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
        self.bert = AutoModel.from_pretrained(config.encoder_name)

        if config.freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.projection = None
        if config.encoder_dim != config.hidden_dim:
            self.projection = nn.Linear(config.encoder_dim, config.hidden_dim)

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    def encode_edus(self, edu_texts: List[str], max_length: int = 512) -> torch.Tensor:
        if not edu_texts:
            return torch.zeros(0, self.config.hidden_dim, device=self.bert.device)
        encoded = self.tokenizer(
            edu_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.bert.device)
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            outputs = self.bert(**encoded)
            cls_repr = outputs.last_hidden_state[:, 0, :]
        if self.projection is not None:
            cls_repr = self.projection(cls_repr)
        return cls_repr

    def encode_all_nodes(self, nodes: list, edges: list = None, max_length: int = 512) -> torch.Tensor:
        edu_nodes = [n for n in nodes if n.node_type == 0]
        branch_nodes = [n for n in nodes if n.node_type == 1]

        id_to_idx = {}
        edu_idx = 0
        for n in edu_nodes:
            id_to_idx[n.id] = edu_idx
            edu_idx += 1
        branch_idx = len(edu_nodes)
        for n in branch_nodes:
            id_to_idx[n.id] = branch_idx
            branch_idx += 1

        num_nodes = len(nodes)
        hidden_dim = self.config.hidden_dim
        device = self.bert.device
        node_embeddings = torch.zeros(num_nodes, hidden_dim, device=device)

        edu_texts = [n.text.strip() for n in edu_nodes]
        for i, t in enumerate(edu_texts):
            if not t:
                edu_texts[i] = "[EMPTY]"

        if edu_texts:
            edu_embeddings = self.encode_edus(edu_texts, max_length)
            node_embeddings[:len(edu_nodes)] = edu_embeddings

        if edges and branch_nodes:
            children_map = {}
            for edge in edges:
                parent_id = edge.src
                child_id = edge.dst
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(child_id)

            for i, branch in enumerate(branch_nodes):
                branch_idx = len(edu_nodes) + i
                child_ids = children_map.get(branch.id, [])
                if child_ids:
                    child_indices = [id_to_idx.get(cid, None) for cid in child_ids]
                    child_indices = [idx for idx in child_indices if idx is not None]
                    if child_indices:
                        child_embeddings = node_embeddings[child_indices]
                        node_embeddings[branch_idx] = child_embeddings.mean(dim=0)
        return node_embeddings

    def encode_hypothesis(self, hypothesis: str, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            hypothesis, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.bert.device)
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            outputs = self.bert(**encoded)
            cls_repr = outputs.last_hidden_state[:, 0, :]
        if self.projection is not None:
            cls_repr = self.projection(cls_repr)
        return cls_repr.squeeze(0)

    def encode_hypotheses_batch(self, hypotheses: List[str], max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            hypotheses, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.bert.device)
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            outputs = self.bert(**encoded)
            cls_repr = outputs.last_hidden_state[:, 0, :]
        if self.projection is not None:
            cls_repr = self.projection(cls_repr)
        return cls_repr


# ============================================================
# RGAT
# ============================================================

class RGAT(nn.Module):
    """2层RGAT：HeteroGraphConv + GATConv"""

    def __init__(self, config: ModelConfig, rel_names: list):
        super().__init__()
        self.config = config
        self.rel_names = rel_names
        in_dim = config.hidden_dim
        hidden_dim = config.hidden_dim
        out_dim = config.hidden_dim
        heads1 = config.num_rgat_heads_layer1
        heads2 = config.num_rgat_heads_layer2

        self.conv1 = HeteroGraphConv(
            {rel: GATConv(in_dim, hidden_dim, num_heads=heads1, residual=True,
                          activation=F.elu, allow_zero_in_degree=True) for rel in rel_names},
            aggregate="mean",
        )
        self.rel_embeddings = nn.Parameter(torch.randn(len(rel_names), hidden_dim))

        self.conv2 = HeteroGraphConv(
            {rel: GATConv(hidden_dim * heads1, out_dim, num_heads=heads2, residual=True,
                          allow_zero_in_degree=True) for rel in rel_names},
            aggregate="mean",
        )
        self.res_fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim * heads1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        # 转为异构图
        hg = self._to_hetero(g, node_feats)

        h = self.conv1(hg, {"node": node_feats})["node"]
        h = h.view(h.size(0), -1)
        h = self._add_rel_embed(hg, h)
        h = self.norm(h)
        h = self.dropout(F.elu(h))
        h = self.conv2(hg, {"node": h})["node"]
        h = h.squeeze(1) if h.dim() == 3 else h
        h = h + self.res_fc(node_feats)
        return h

    def _to_hetero(self, g, node_feats):
        """将同构图转为异构图

        dgl.to_homogeneous 在 edata 中自动加 _TYPE（原始边类型索引）
        _TYPE 值范围是 [0, num_hetero_edge_types)，包含正向和反向边
        """
        if "_TYPE" in g.edata:
            edge_type_ids = g.edata["_TYPE"]
        elif "edge_type" in g.edata:
            edge_type_ids = g.edata["edge_type"]
        else:
            edge_dict = {("node", self.rel_names[0], "node"): g.edges()}
            hg = dgl.heterograph(edge_dict)
            hg = hg.to(node_feats.device)
            hg.nodes["node"].data["feat"] = node_feats
            return hg

        # 在CPU上做mask，避免GPU越界
        edge_type_ids_cpu = edge_type_ids.cpu()
        src_cpu = g.edges()[0].cpu()
        dst_cpu = g.edges()[1].cpu()

        edge_dict = {}
        for i, rel_name in enumerate(self.rel_names):
            mask = edge_type_ids_cpu == i
            if mask.any():
                edge_dict[("node", rel_name, "node")] = (src_cpu[mask], dst_cpu[mask])

        # 处理不在 self.rel_names 中的边类型（如 rev_ 反向边）
        unique_types = edge_type_ids_cpu.unique()
        for t in unique_types:
            t_val = t.item()
            if t_val >= len(self.rel_names):
                # 这是一条反向边或其他不在 rel_names 中的边类型
                # 跳过，因为 HeteroGraphConv 只处理 rel_names 中定义的边类型
                pass

        if not edge_dict:
            edge_dict = {("node", self.rel_names[0], "node"): g.edges()}
        hg = dgl.heterograph(edge_dict)
        hg = hg.to(node_feats.device)
        hg.nodes["node"].data["feat"] = node_feats
        return hg

    def _add_rel_embed(self, hg, h):
        for etype in hg.etypes:
            _, dst = hg.edges(etype=etype)
            rel_idx = self.rel_names.index(etype) if etype in self.rel_names else 0
            heads = self.config.num_rgat_heads_layer1
            rel_emb = self.rel_embeddings[rel_idx].repeat(heads).unsqueeze(0)
            h[dst] = h[dst] + rel_emb.expand(dst.size(0), -1)
        return h


# ============================================================
# Attention Pooling
# ============================================================

class AttentionPooling(nn.Module):
    """Query引导的Attention池化"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feats: torch.Tensor, query: torch.Tensor,
                node_types: torch.Tensor = None) -> tuple:
        Q = self.query_proj(query)
        K = self.key_proj(node_feats)
        V = self.value_proj(node_feats)

        # 只对EDU节点(node_type==0)做pooling
        if node_types is not None:
            edu_mask = (node_types == 0)
            K_edu = K[edu_mask]
            V_edu = V[edu_mask]
            scores = F.cosine_similarity(K_edu, Q.unsqueeze(0), dim=-1)
            attention_weights = F.softmax(scores, dim=0)
            graph_repr = torch.matmul(attention_weights.unsqueeze(0), V_edu).squeeze(0)
            # 返回全长度权重，非EDU位置填0
            full_weights = torch.zeros(node_feats.size(0), device=node_feats.device)
            full_weights[edu_mask] = attention_weights
            return graph_repr, full_weights
        else:
            scores = F.cosine_similarity(K, Q.unsqueeze(0), dim=-1)
            attention_weights = F.softmax(scores, dim=0)
            graph_repr = torch.matmul(attention_weights.unsqueeze(0), V).squeeze(0)
            return graph_repr, attention_weights


# ============================================================
# 分类器
# ============================================================

class TripletClassifier(nn.Module):
    """MLP分类器"""

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, graph_repr: torch.Tensor, hypo_repr: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([graph_repr, hypo_repr], dim=-1)
        return self.mlp(combined).squeeze(-1)


# ============================================================
# 解释性模块
# ============================================================

class InterpretabilityModule(nn.Module):
    """EDU-level 解释性模块"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** 0.5
        self.node_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def compute_node_importance(self, attention_weights: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if attention_weights.numel() == 0:
            return torch.ones(num_nodes, device=attention_weights.device) / max(num_nodes, 1)
        return F.softmax(attention_weights, dim=0)

    def compute_interaction_features(self, node_feats: torch.Tensor, hypo_repr: torch.Tensor,
                                     importance: torch.Tensor) -> torch.Tensor:
        weighted_feats = importance.unsqueeze(-1) * node_feats
        scores = torch.matmul(weighted_feats, hypo_repr) / self.scale
        attn_weights = F.softmax(scores, dim=0)
        return attn_weights.unsqueeze(-1) * weighted_feats

    def forward(self, node_feats: torch.Tensor, hypo_repr: torch.Tensor,
                attention_weights: torch.Tensor, is_leaf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.shape[0]
        importance = self.compute_node_importance(attention_weights, num_nodes)
        interaction = self.compute_interaction_features(node_feats, hypo_repr, importance)
        weighted_feats = importance.unsqueeze(-1) * node_feats
        combined = torch.cat([weighted_feats, interaction], dim=-1)
        logits = self.node_classifier(combined).squeeze(-1)
        probs = torch.sigmoid(logits)
        leaf_probs = probs[is_leaf]
        return leaf_probs, importance

    def extract_explanation(self, node_probs: torch.Tensor, top_k: int = 3) -> List[int]:
        k = min(top_k, node_probs.numel())
        if k == 0:
            return []
        _, indices = torch.topk(node_probs, k)
        return indices.tolist()
