#!/bin/bash
# Download BLUFF dataset from HuggingFace
# Usage: bash scripts/download_data.sh [--subset SUBSET] [--output DIR]

set -e

REPO_ID="jsl5710/BLUFF"
SUBSET="all"
OUTPUT_DIR="data"

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset) SUBSET="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/download_data.sh [OPTIONS]"
            echo ""
            echo "Downloads BLUFF dataset from HuggingFace: https://huggingface.co/datasets/${REPO_ID}"
            echo ""
            echo "Options:"
            echo "  --subset SUBSET  Data subset to download:"
            echo "                     all        - Download everything (~3.9 GB)"
            echo "                     meta_data  - Metadata CSVs only (~100 MB)"
            echo "                     processed  - Cleaned text data by model/language (~1.4 GB)"
            echo "                     raw        - Original source data (~2.4 GB)"
            echo "                     splits     - Train/val split definitions only (~10 MB)"
            echo "  --output DIR     Output directory (default: data)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash scripts/download_data.sh                          # Download everything"
            echo "  bash scripts/download_data.sh --subset splits          # Only split definitions"
            echo "  bash scripts/download_data.sh --subset meta_data       # Only metadata"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  BLUFF Dataset Download"
echo "============================================"
echo "  Repository: ${REPO_ID}"
echo "  Subset:     ${SUBSET}"
echo "  Output:     ${OUTPUT_DIR}"
echo "============================================"

# Determine allow_patterns based on subset
if [ "${SUBSET}" == "all" ]; then
    PATTERN_ARG=""
elif [ "${SUBSET}" == "meta_data" ]; then
    PATTERN_ARG="--include 'data/meta_data/*'"
elif [ "${SUBSET}" == "processed" ]; then
    PATTERN_ARG="--include 'data/processed/**'"
elif [ "${SUBSET}" == "raw" ]; then
    PATTERN_ARG="--include 'data/raw/**'"
elif [ "${SUBSET}" == "splits" ]; then
    PATTERN_ARG="--include 'data/splits/**'"
else
    echo "Error: Unknown subset '${SUBSET}'"
    echo "Valid subsets: all, meta_data, processed, raw, splits"
    exit 1
fi

echo ""
echo ">> Downloading from HuggingFace..."

if [ "${SUBSET}" == "all" ]; then
    python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='dataset',
    local_dir='${OUTPUT_DIR}',
)
print(f'Downloaded to: {path}')
"
else
    python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='dataset',
    local_dir='${OUTPUT_DIR}',
    allow_patterns='data/${SUBSET}/**',
)
print(f'Downloaded to: {path}')
"
fi

echo ""
echo "============================================"
echo "  Download complete!"
echo "  Data saved to: ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Dataset structure:"
echo "  data/meta_data/    - Sample metadata (metadata_human_written.csv, metadata_ai_generated.csv)"
echo "  data/processed/    - Cleaned text data organized by model and language"
echo "  data/raw/          - Original source data"
echo "  data/splits/       - Train/val split definitions (JSON files with UUIDs)"
echo ""
echo "For usage instructions, see: https://huggingface.co/datasets/${REPO_ID}"
