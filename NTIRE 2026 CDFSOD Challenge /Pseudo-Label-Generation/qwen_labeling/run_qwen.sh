# #!/bin/bash
DATASET_PATH=your_dataset_path
MODEL_PATH=your_model_path

for dataset in dataset1; do
    CUDA_VISIBLE_DEVICES=3,4 python detect_all_categories_${dataset}.py ${DATASET_PATH}/${dataset}/train/ -o ./annotations/${dataset}/train/labels_json -m ${MODEL_PATH}/Qwen3.5-35B-A3B
    CUDA_VISIBLE_DEVICES=3,4 python detect_all_categories_${dataset}.py ${DATASET_PATH}/${dataset}/test/ -o ./annotations/${dataset}/test/labels_json -m ${MODEL_PATH}/Qwen3.5-35B-A3B
done

for dataset in dataset1; do
    python coco_to_yolo.py ./annotations/${dataset}/train/labels_json -o ./annotations/${dataset}/train/labels
    python coco_to_yolo.py ./annotations/${dataset}/test/labels_json -o ./annotations/${dataset}/test/labels
done