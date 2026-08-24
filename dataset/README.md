# Dataset: RST-Enhanced Paleontology Knowledge Graph Validation

This file documents only the dataset. The code, method, and usage instructions are in the repository root [README.md](../README.md).

## Overview

The **complete dataset is fully open-sourced** in this repository at [data/NLI_Input_total_1.json](../data/NLI_Input_total_1.json).

It contains **2,569 NLI-based knowledge graph triple validation samples** constructed from paleontology scientific literature. Each sample asks whether a candidate knowledge graph triple (hypothesis) is supported by a source paragraph (premise) split into sentences.

## File

| File | Description |
|---|---|
| `data/NLI_Input_total_1.json` | Complete dataset (2,569 samples), stored as a JSON list |

## Data Format

Each item is a JSON object:

| Field | Type | Description |
|---|---|---|
| `premise` | `list[str]` | Source paragraph represented as a list of sentences |
| `hypothesis` | `str` | Natural-language hypothesis generated from the candidate triple |
| `is_negative` | `bool` | `true` = negative sample, `false` = positive sample |
| `head_type` | `str` | Natural-language description of the head entity type |
| `tail_type` | `str` | Natural-language description of the tail entity type |
| `head_type_pair` | `list` | `[head entity text, lowercase head entity type]` |
| `tail_type_pair` | `list` | `[tail entity text, lowercase tail entity type]` |

Example:

```json
{
  "premise": ["Sentence 1.", "Sentence 2."],
  "hypothesis": "The candidate triple expressed as a natural-language statement.",
  "is_negative": false,
  "head_type": "head entity type",
  "tail_type": "tail entity type",
  "head_type_pair": ["head entity", "location"],
  "tail_type_pair": ["tail entity", "section"]
}
```

Model label mapping: `label = 1.0` for positive samples (`is_negative = false`), `label = 0.0` for negative samples (`is_negative = true`).

## Statistics

| Statistic | Value |
|---|---:|
| Total samples | 2,569 |
| Positive samples | 1,637 |
| Negative samples | 932 |
| Positive-to-negative ratio | ≈ 1.76 : 1 |

Evidence length: **7.8 sentences per instance on average** (range 2–25).

## Data Split

| Split | Samples |
|---|---:|
| Training | 1,798 |
| Test | 771 |
| Total | 2,569 |

## Construction

The dataset was built from paleontology scientific literature. Steps:

1. Extract entities and relations from the literature to form candidate triples.
2. Convert each triple into a natural-language hypothesis.
3. Collect the relevant paragraph from the source as evidence and split it into sentences (`premise`).
4. Annotate whether the triple is consistent with its supporting evidence (`is_negative`).

It is designed to evaluate triple validation under **long texts, multiple evidence sentences, and dispersed factual information**.

## Data Availability

The complete dataset is released for **academic and research purposes**. Users are responsible for complying with the copyright and usage conditions of the original source materials and should properly cite the original publications where required. The dataset should not be redistributed for commercial purposes without authorization.