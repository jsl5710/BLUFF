#!/bin/bash
# Download BLUFF dataset from HuggingFace
# Usage: bash scripts/download_data.sh [--task TASK] [--split SPLIT]

set -e

DATASET_NAME="jsl5710/BLUFF"
OUTPUT_DIR="data/splits"
TASK="all"
SPLIT="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/download_data.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TASK      Task to download (task1_veracity_binary, task2_veracity_multiclass,"
            echo "                   task3_authorship_binary, task4_authorship_multiclass, or 'all')"
            echo "  --split SPLIT    Split to download (train, dev, test, or 'all')"
            echo "  --output DIR     Output directory (default: data/splits)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  BLUFF Dataset Download"
echo "============================================"
echo "  Dataset: ${DATASET_NAME}"
echo "  Task:    ${TASK}"
echo "  Split:   ${SPLIT}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

TASKS=("task1_veracity_binary" "task2_veracity_multiclass" "task3_authorship_binary" "task4_authorship_multiclass")

if [ "${TASK}" != "all" ]; then
    TASKS=("${TASK}")
fi

for task in "${TASKS[@]}"; do
    echo ""
    echo ">> Downloading: ${task}"
    mkdir -p "${OUTPUT_DIR}/${task}"

    python -c "
from datasets import load_dataset
import json, os

task = '${task}'
split = '${SPLIT}'
output_dir = '${OUTPUT_DIR}/${task}'

dataset = load_dataset('${DATASET_NAME}', task)

splits = list(dataset.keys()) if split == 'all' else [split]

for s in splits:
    if s in dataset:
        output_file = os.path.join(output_dir, f'{s}.jsonl')
        dataset[s].to_json(output_file)
        print(f'  Saved {s}: {len(dataset[s])} samples -> {output_file}')
    else:
        print(f'  Warning: Split \"{s}\" not found in {task}')
"
    echo "  Done: ${task}"
done

echo ""
echo "============================================"
echo "  Download complete!"
echo "============================================"
