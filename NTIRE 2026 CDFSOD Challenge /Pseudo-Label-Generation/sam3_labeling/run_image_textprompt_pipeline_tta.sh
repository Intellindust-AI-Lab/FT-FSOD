#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


GPU_ID=""
INPUT_PATH=""
TARGETS=()
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpuid)      GPU_ID="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --input)      INPUT_PATH="$2"; shift 2 ;;
        --targets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TARGETS+=("$1")
                shift
            done
            ;;
        --output)     OUTPUT_PATH="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -n "$GPU_ID" && -n "$MODEL_PATH" && -n "$INPUT_PATH" && ${#TARGETS[@]} -gt 0 && -n "$OUTPUT_PATH" ]] || {
    echo "Usage: $0 --gpuid GPU_ID --model-path MODEL_PATH --input INPUT_PATH --targets \"t1\" \"t2\" ... --output OUTPUT_PATH"
    exit 1
}

BPE_PATH="$MODEL_PATH/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH="$MODEL_PATH/sam3.1_multiplex.pt"

mkdir -p "$OUTPUT_PATH"
RAW_DIR="${OUTPUT_PATH}/_raw_annot"
mkdir -p "$RAW_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" python annotate_tta.py \
    --input "$INPUT_PATH" \
    --targets "${TARGETS[@]}" \
    --output "$RAW_DIR" \
    --bpe-path "$BPE_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --save-confidence \
    --tta-horizontal

LABELS_DIR="${OUTPUT_PATH}/labels"
mkdir -p "$LABELS_DIR"

python output_final_annotation.py \
    --annotation "$RAW_DIR" \
    --output "$LABELS_DIR" \
    --nms-iou 0.65 \
    --input-format yolo_center

echo "labels:    $LABELS_DIR"
