#!/bin/bash
# Run all encoder-based experiments for BLUFF benchmark
# Usage: bash scripts/run_encoder_experiments.sh [--task TASK] [--gpu GPU_ID]

set -e

TASK="task1_veracity_binary"
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU

MODELS=(
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
    "xlm-roberta-large"
    "microsoft/mdeberta-v3-base"
    "cis-lmu/glot500-base"
)

echo "============================================"
echo "  BLUFF Encoder Experiments"
echo "  Task: ${TASK}"
echo "  GPU: ${GPU}"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">> Training: ${MODEL}"
    echo "--------------------------------------------"

    # Multilingual experiment
    python src/evaluation/encoder/train.py \
        --model "${MODEL}" \
        --task "${TASK}" \
        --experiment multilingual \
        --config configs/encoder_models.yaml

    # Evaluate with disaggregated results
    MODEL_DIR="outputs/encoder/${TASK}/multilingual/${MODEL//\//_}/best_model"
    python src/evaluation/encoder/evaluate.py \
        --model "${MODEL_DIR}" \
        --task "${TASK}" \
        --split test \
        --disaggregate resource_category language_family script syntax language

    echo ">> Completed: ${MODEL}"
done

echo ""
echo "============================================"
echo "  All encoder experiments complete!"
echo "============================================"
