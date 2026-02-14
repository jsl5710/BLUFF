# BLUFF Benchmark Results

This document summarizes the comprehensive experimental results from the BLUFF benchmark evaluation. All results use macro-averaged F1 scores.

---

## Experimental Setup

### Encoder Models

| Model | HuggingFace ID | Parameters | Pre-training Languages |
|-------|----------------|------------|----------------------|
| mBERT | `bert-base-multilingual-cased` | 110M | 104 |
| XLM-RoBERTa-base | `xlm-roberta-base` | 270M | 100 |
| XLM-RoBERTa-large | `xlm-roberta-large` | 550M | 100 |
| XLM-E | `microsoft/xlm-e-base` | 270M | 100 |
| XLM-T | `xlm-t-base` | 270M | 100 |
| mDeBERTa-v3-base | `microsoft/mdeberta-v3-base` | 86M | 100 |
| mDeBERTa-v3-large | `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large` | 304M | 100 |
| Glot500 | `cis-lmu/glot500-base` | 395M | 511 |
| S-BERT (LaBSE) | `sentence-transformers/LaBSE` | 471M | 109 |

### Decoder Models

| Model | Provider | Parameters |
|-------|----------|------------|
| GPT-4o | OpenAI | N/A |
| GPT-4o-mini | OpenAI | N/A |
| Claude 3.5 Sonnet | Anthropic | N/A |
| Gemini 1.5 Pro | Google | N/A |
| Gemini 2.0 Flash | Google | N/A |
| Llama 3.3 70B | Meta | 70B |
| Qwen3-8B | Alibaba | 8B |

### Evaluation Settings

- **Cross-lingual:** Train on English only, evaluate on all 78 languages
- **Multilingual:** Train on stratified multilingual sample, evaluate on all 78 languages
- **Three prompt regimens (decoder):** Cross-lingual, Native, English-Translated

---

## Task 1: Binary Veracity Classification (Real vs. Fake)

### Cross-lingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap | Average F1 |
|-------|-------------|--------------|-----|------------|
| mBERT | 78.2 | 68.1 | 10.1 | 71.4 |
| XLM-R-base | 80.5 | 70.8 | 9.7 | 73.9 |
| XLM-R-large | 83.4 | 73.5 | 9.9 | 76.7 |
| mDeBERTa-v3 | 85.1 | 74.2 | 10.9 | 77.7 |
| S-BERT (LaBSE) | 81.9 | 72.0 | 9.9 | 75.2 |

### Multilingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap | Average F1 |
|-------|-------------|--------------|-----|------------|
| mBERT | 82.6 | 78.3 | 4.3 | 79.7 |
| XLM-R-base | 85.1 | 81.6 | 3.5 | 82.7 |
| XLM-R-large | 87.3 | 83.4 | 3.9 | 84.7 |
| mDeBERTa-v3 | **98.3** | 82.1 | 16.2 | 87.4 |
| S-BERT (LaBSE) | 84.7 | 81.0 | 3.7 | **97.2** (avg) |

### Decoder Models (Best Prompt Regimen)

| Model | Big-Head F1 | Long-Tail F1 | Average F1 |
|-------|-------------|--------------|------------|
| GPT-4o | 62.3 | 55.8 | 57.9 |
| Claude 3.5 Sonnet | 60.1 | 53.2 | 55.4 |
| Qwen3-8B | **65.9** | 58.1 | 60.6 |

**Key Finding:** Encoder models substantially outperform decoder models. Multilingual training reduces the big-head vs. long-tail gap from ~9.9 to ~3.7 points.

---

## Task 2: Multi-class Veracity Classification (8 classes)

### Cross-lingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap |
|-------|-------------|--------------|-----|
| XLM-R-large | 66.4 | 55.7 | 10.7 |

### Multilingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap |
|-------|-------------|--------------|-----|
| S-BERT (LaBSE) | 66.8 | 70.3 | -3.5 |

**Key Finding:** Multiclass veracity shows the largest cross-lingual gaps. Decoder models fall below the 12.5% random baseline, indicating fundamental inability to perform fine-grained veracity classification.

---

## Task 3: Binary Synthetic Text Detection (Human vs. Machine)

### Cross-lingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap |
|-------|-------------|--------------|-----|
| Best Encoder | 87.3 | 83.0 | 4.2 |

### Multilingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap |
|-------|-------------|--------------|-----|
| S-BERT (LaBSE) | 88.7 | **93.2** | -4.5 |

**Key Finding:** Reversed performance gap in multilingual setting — long-tail languages outperform big-head, possibly due to more distinctive LLM artifacts in low-resource generation.

---

## Task 4: Multi-class Authorship Attribution (HWT/MGT/MTT/HAT)

### Cross-lingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap |
|-------|-------------|--------------|-----|
| Best Encoder | 80.6 | 62.1 | **18.5** |

### Multilingual Setting

| Model | Big-Head F1 | Long-Tail F1 | Gap |
|-------|-------------|--------------|-----|
| S-BERT (LaBSE) | 79.4 | 82.0 | -2.6 |

**Key Finding:** Cross-lingual authorship attribution shows the largest performance gap (18.5 points), highlighting the challenge of fine-grained classification across diverse linguistic contexts.

---

## Linguistic Transfer Analysis

### Script-Based Transfer

| Transfer Type | Average F1 |
|---------------|------------|
| Same-script | 68.8 |
| Cross-script | 52.4 |
| **Gap** | **16.4** |

Notable patterns: Latin→Cyrillic transfers well (77%), while Latin→Arabic is challenging (47%).

### Syntax-Based Transfer

| Transfer Type | Average F1 |
|---------------|------------|
| SVO→SVO | 75.0 |
| SVO→SOV | 69.0 |
| VSO targets | 28–63 |

### Family-Based Transfer

| Transfer Type | Average F1 |
|---------------|------------|
| Within-family | 66.6 |
| Cross-family | 51.2 |
| **Gap** | **15.4** |

Best: Indo-European (85%); Worst: Creole languages (23%).

---

## External Evaluation

Evaluated on 28 external sources, 36,612 samples across 53 languages.

| Model | Overall F1 | Big-Head F1 | Long-Tail F1 |
|-------|------------|-------------|--------------|
| mDeBERTa-v3 | **67.3** | 65.1 | 59.5 |
| XLM-E (Hindi) | — | — | 67.5 |
| XLM-T (Chinese) | — | — | 62.0 |
| mBERT (Portuguese) | — | — | 61.5 |

---

## Key Takeaways

1. **Long-tail degradation:** 9.0–25.3% in cross-lingual settings, reduced to 0.1–7.9% with multilingual training
2. **Linguistically-informed training** outperforms random batching by 15–16 points
3. **Multiclass tasks** show largest gaps (15.0–25.3% cross-lingual)
4. **Decoder models fail** on all fine-grained classification tasks
5. **S-BERT (LaBSE)** recommended for balanced performance across all tasks
6. **mDeBERTa** recommended for high-resource language performance
7. **Script similarity** is the strongest predictor of cross-lingual transfer success

---

## Reproducing Results

See [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) for step-by-step reproduction instructions, and the `scripts/` directory for ready-to-run experiment scripts.
