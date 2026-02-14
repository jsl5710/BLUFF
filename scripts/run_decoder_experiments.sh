#!/bin/bash
# Run all decoder-based experiments for BLUFF benchmark
# Usage: bash scripts/run_decoder_experiments.sh [--task TASK]

set -e

TASK="task1_veracity_binary"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODELS=("gpt4o" "gpt4o_mini" "claude_sonnet" "gemini_pro" "llama3_70b" "qwen2_72b" "aya_expanse")
PROMPT_TYPES=("crosslingual" "native" "english_translated")

echo "============================================"
echo "  BLUFF Decoder Experiments"
echo "  Task: ${TASK}"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    for PROMPT in "${PROMPT_TYPES[@]}"; do
        echo ""
        echo ">> ${MODEL} | ${PROMPT}"
        echo "--------------------------------------------"

        python src/evaluation/decoder/inference.py \
            --model "${MODEL}" \
            --task "${TASK}" \
            --prompt_type "${PROMPT}" \
            --split test \
            --config configs/decoder_models.yaml

        echo ">> Done: ${MODEL} | ${PROMPT}"
    done
done

echo ""
echo "============================================"
echo "  All decoder experiments complete!"
echo "============================================"
