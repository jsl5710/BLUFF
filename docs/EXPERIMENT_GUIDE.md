# BLUFF Experiment Reproduction Guide

This guide provides step-by-step instructions to reproduce all experiments reported in the BLUFF paper.

## Prerequisites

1. **Hardware:** GPU with ≥24GB VRAM (A100 recommended for large models)
2. **Software:** Python 3.10+, CUDA 11.8+
3. **API Keys** (for decoder experiments): OpenAI, Anthropic, Google AI

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
bash scripts/download_data.sh
```

## Experiment Overview

| Experiment | Type | Models | Details |
|------------|------|--------|---------|
| Multilingual | Encoder | mBERT, XLM-R, mDeBERTa, Glot500 | Train all langs, eval head vs. tail |
| Cross-lingual | Encoder | Same | Transfer across family/script/syntax |
| External | Encoder | Same | Held-out dataset evaluation |
| Cross-lingual Prompt | Decoder | GPT-4o, Claude, Gemini, Llama, Qwen, Aya | English prompt, original input |
| Native Prompt | Decoder | Same | Target-language prompt and input |
| English-Translated | Decoder | Same | English prompt, translated input |

## 1. Encoder Experiments

### 1.1 Multilingual (Head vs. Tail)

```bash
# All models, all tasks
bash scripts/run_encoder_experiments.sh --task task1_veracity_binary --gpu 0

# Single model
python src/evaluation/encoder/train.py \
    --model xlm-roberta-large \
    --task task1_veracity_binary \
    --experiment multilingual
```

### 1.2 Cross-lingual Evaluation

After training, evaluate transfer performance:

```bash
python src/evaluation/encoder/evaluate.py \
    --model outputs/encoder/task1_veracity_binary/multilingual/xlm-roberta-large/best_model \
    --task task1_veracity_binary \
    --split test \
    --disaggregate resource_category language_family script syntax language
```

### 1.3 All Four Tasks

```bash
for task in task1_veracity_binary task2_veracity_multiclass task3_authorship_binary task4_authorship_multiclass; do
    bash scripts/run_encoder_experiments.sh --task $task --gpu 0
done
```

## 2. Decoder Experiments

### 2.1 Setup API Keys

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### 2.2 Run All Prompt Regimens

```bash
bash scripts/run_decoder_experiments.sh --task task1_veracity_binary
```

### 2.3 Single Configuration

```bash
python src/evaluation/decoder/inference.py \
    --model gpt4o \
    --task task1_veracity_binary \
    --prompt_type native \
    --split test
```

## 3. Generating Results Tables

```bash
python scripts/generate_results_tables.py \
    --results_dir outputs/ \
    --output_format latex \
    --output_file docs/RESULTS.md
```

## Expected Results Directory Structure

```
outputs/
├── encoder/
│   ├── task1_veracity_binary/
│   │   └── multilingual/
│   │       ├── xlm-roberta-large/
│   │       │   ├── best_model/
│   │       │   ├── results.json
│   │       │   └── eval_test_results.json
│   │       └── ...
│   └── ...
└── decoder/
    ├── task1_veracity_binary/
    │   ├── crosslingual/
    │   │   ├── gpt-4o/
    │   │   │   ├── predictions_test.jsonl
    │   │   │   └── results_test.json
    │   │   └── ...
    │   ├── native/
    │   └── english_translated/
    └── ...
```

## Troubleshooting

- **OOM errors:** Reduce `batch_size` in `configs/encoder_models.yaml` or use gradient accumulation
- **API rate limits:** Adjust `rate_limit_delay` in `configs/decoder_models.yaml`
- **Missing languages:** Some languages may not be supported by all models; check model documentation
