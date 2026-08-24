# LGF: RST-Enhanced Local-Global Fusion for Paleontology Knowledge Graph Validation

This repository provides the **main experimental code** for our paper on RST-enhanced knowledge graph validation in paleontology.

**LGF (Local-Global Fusion)** parses inter-sentence logical relations in scientific paragraphs using **Rhetorical Structure Theory (RST)**, then fuses **paragraph-level logical information (global module)** with **local semantic evidence (local module)** to determine whether a candidate triple (hypothesis) is supported by the source paragraph (premise).

The **full dataset (2,569 samples) is open-sourced in this repository** — see [data/](data/).

## Repository Structure

```
open_source/
├── config.py                    # ★ Only file you need to personalize (paths / LLM service)
├── requirements.txt             # Dependencies
├── README.md
├── data/
│   └── NLI_Input_total_1.json   # ★ Complete dataset (2,569 samples) — fully open-sourced
├── dataset/
│   └── README.md                # Dataset format, statistics, and documentation
├── model/
│   ├── global_module.py         # Global module: RST parsing + RGAT + attention pooling + classifier
│   ├── fusion_module.py         # Fusion module: Top-K evidence selection + gated fusion
│   ├── train_global.py          # ★ Training entry (global module)
│   ├── fusion_inference.py      # ★ Inference entry (global + local + gated fusion)
│   └── local_1.py               # ★ Local LLM inference entry (entity evidence selection + LLM judgment)
└── output/                      # Directory for model weights and inference results
```

---

## Method Overview

| Module | Role |
|---|---|
| **Global module** | Parses a paragraph into an RST-based discourse relation tree, builds a heterogeneous graph, aggregates via RGAT to obtain a paragraph-level logical representation, then produces the global prediction probability `p_global` through hypothesis-guided attention pooling |
| **Local module** | Uses the natural-language hypothesis of the candidate triple as a query to retrieve the evidence sentences most relevant to the target semantics, and asks an LLM to produce the local prediction probability `p_local` |
| **Gated fusion** | Uses a heuristic gate `fixed_gate` to dynamically balance the global/local contributions and obtain the final probability `p_final` |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> The global module uses `isanlp-rst` for RST parsing (the `tchewik/isanlp_rst_v3` model is downloaded from the network, so the first run is slow). If it is not installed, an optional `stanza`-based parser is noted in `requirements.txt`.

### 2. Configure

Edit [config.py](config.py) as needed:

```python
DEVICE           = "cpu"                    # change to "cuda:0" if CUDA is available
DATA_PATH        = "data/NLI_Input_total_1.json"
LLM_API_ENDPOINT = "http://localhost:8001/v1"   # your local OpenAI-compatible LLM service
LLM_MODEL_NAME   = "Llama3.1-8B-Instruct"
```

> The local module requires an **OpenAI-compatible Chat Completions service** (vLLM, llama.cpp, or any local inference server) with `logprobs` enabled to compute confidence scores. If no such service is available, the global module (`train_global.py`) can still run standalone.

### 3. Train the Global Module

```bash
cd model
python train_global.py
```

After training, the model is saved to `output/best_model.pt`.

### 4. Local Module Inference (optional; requires an LLM service)

```bash
cd model
python local_1.py
```

### 5. Fusion Inference (full LGF)

```bash
cd model
python fusion_inference.py
```

Prediction results and evaluation metrics (Acc / F1 / P / R for the global, local, and fused variants on the test set) are printed to the console and saved to `output/fusion_results.json`.

### 6. Pretrained Models

The trained model checkpoints are **not hosted in this GitHub repository** because each file is ~1.1–1.2 GB (exceeding GitHub's 100 MB single-file limit). The following checkpoints are available:

| Checkpoint | Description | Approx. size |
|---|---|---|
| `best_model.pt` | Full global model using RST discourse structure | ~1.2 GB |
| `best_model_wo_rst.pt` | Global model without RST (ablation) | ~1.1 GB |

If you need the checkpoints, please contact us by email (see [Contact](#contact)). The full dataset is already included in this repository under `data/NLI_Input_total_1.json`.

---

## Data

The **complete dataset is open-sourced** in this repository at [data/NLI_Input_total_1.json](data/NLI_Input_total_1.json), containing **2,569 NLI-based knowledge graph triple validation samples** (no sampling needed). See [dataset/README.md](dataset/README.md) for dataset statistics, the train/test split, and the data-construction procedure.

Each sample is a JSON object:

```json
{
  "premise": ["Sentence 1", "Sentence 2", "..."],
  "hypothesis": "Natural-language hypothesis generated from the triple",
  "is_negative": false,
  "head_type": "head entity type description",
  "tail_type": "tail entity type description",
  "head_type_pair": ["head entity", "location"],
  "tail_type_pair": ["tail entity", "section"]
}
```

In the model, `label = 0.0` corresponds to a negative sample (`is_negative = true`) and `1.0` to a positive sample.

---

## License

The code is released under the MIT License. For dataset usage, please refer to [dataset/README.md](dataset/README.md).

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourPaper,
  title  = {RST-Enhanced Paleontology Knowledge Graph Validation},
  year   = {2026}
}
```

(Citation details will be completed after the paper is formally published.)

---

## Contact

For questions or to request the pretrained model checkpoints, please contact:

**Jie Lu**
Email: [15226155582@163.com](mailto:15226155582@163.com)