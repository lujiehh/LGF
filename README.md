# LGF：RST 增强的古生物知识图谱验证方法

本仓库是论文《RST 增强的古生物知识图谱验证》的**主实验开源代码**。

LGF（Local-Global Fusion）通过 **修辞结构理论（RST）** 解析文献段落的句间逻辑关系，融合**段落逻辑信息（全局模块）** 与 **局部语义证据（局部模块）**，判断一个待验证三元组（hypothesis）是否能被文献段落（premise）支持。

## 目录结构

```
open_source/
├── config.py                    # ★ 唯一个性化配置文件（路径 / LLM 服务地址）
├── requirements.txt             # 依赖清单
├── README.md
├── data/
│   └── NLI_Input_total_1_sample.json   # 20 条样例子集（完整数据见 dataset/）
├── dataset/
│   └── README.md                # 完整数据集获取方式与字段说明
├── model/
│   ├── global_module.py         # 全局模块：RST 解析 + RGAT 图注意力 + 注意力池化 + 分类器
│   ├── fusion_module.py         # 融合模块：Top-K 证据筛选 + 门控融合
│   ├── train_global.py          # ★ 训练入口（全局模块）
│   ├── fusion_inference.py      # ★ 推理入口（全局 + 局部 + 门控融合）
│   └── local_1.py               # ★ 局部 LLM 推理入口（实体证据筛选 + LLM 判断）
└── output/                      # 模型权重与推理结果输出目录
```

## 方法概述

| 模块 | 作用 |
|---|---|
| **全局模块** | 用 RST 把段落解析为句间修辞关系树，构建异构图，经 RGAT 聚合得到段落逻辑表示，再通过假设引导的注意力池化得到全局预测概率 `p_global` |
| **局部模块** | 以待验证三元组的自然语言假设为查询，筛选与目标语义最相关的证据句子，交给 LLM 判断得到局部预测概率 `p_local` |
| **门控融合** | 用启发式门控 `fixed_gate` 动态调整全局/局部信息贡献，得到最终概率 `p_final` |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 全局模块使用 `isanlp-rst` 做 RST 解析（`tchewik/isanlp_rst_v3`，需联网下载模型），首次运行较慢。如未安装，可参考 requirements.txt 中注释安装 `stanza` 备选解析方案。

### 2. 配置

编辑 [config.py](config.py)，按需修改：

```python
DEVICE          = "cpu"                    # 有 CUDA 时改为 "cuda:0"
DATA_PATH       = "data/NLI_Input_total_1_sample.json"
LLM_API_ENDPOINT = "http://localhost:8001/v1"   # 你的本地 LLM 服务（OpenAI 兼容）
LLM_MODEL_NAME   = "Llama3.1-8B-Instruct"
```

> 局部模块需要一个 **OpenAI 兼容的 Chat Completions 服务**（vLLM、llama.cpp、本地推理服务均可），并开启 `logprobs` 以计算置信度。若无可用服务，仅全局模块（`train_global.py`）可独立运行。

### 3. 训练全局模块

```bash
cd model
python train_global.py
```

训练完成后模型保存在 `output/best_model.pt`。

### 4. 局部模块推理（可选，需 LLM 服务）

```bash
cd model
python local_1.py
```

### 5. 融合推理（完整 LGF）

```bash
cd model
python fusion_inference.py
```

输出结果与评测指标（全局 / 局部 / 融合三类在测试集上的 Acc / F1 / P / R）打印在控制台，并保存到 `output/fusion_results.json`。

## 数据格式

每个样本为一个 JSON 对象：

```json
{
  "premise": ["句子1", "句子2", "..."],
  "hypothesis": "由三元组生成的自然语言假设句",
  "is_negative": false,
  "head_type": "头实体类型描述",
  "tail_type": "尾实体类型描述",
  "head_type_pair": ["头实体", "location"],
  "tail_type_pair": ["尾实体", "section"]
}
```

模型中 `label = 0.0` 表示负样本（`is_negative=true`），`1.0` 表示正样本。

## License

代码遵循 MIT License。数据集的使用请参考 [dataset/README.md](dataset/README.md)。

## 引用

如使用本代码，请引用我们的论文：

```
@article{yourPaper,
  title={RST增强的古生物知识图谱验证},
  ...
}
```

（引用信息在论文正式发表后补充。）
