

declare -A TARGETS_MAP
TARGETS_MAP["dataset1"]="holothurian echinus scallop starfish fish corals diver cuttlefish turtle jellyfish"
TARGETS_MAP["dataset2"]="car"
TARGETS_MAP["dataset3"]="dent scratch crack \"glass shatter\" \"lamp broken\" \"tire flat\""

GPU_ID=6
MODEL_PATH="your_model_path/sam3.1/"
INPUT_PATH="your_dataset_pathcdfsod_test_26/"
OUTPUT_PATH="./annotations/"
for dataset in dataset1 dataset2 dataset3; do
    targets=${TARGETS_MAP[$dataset]}
    echo "Processing dataset: ${dataset} with targets: ${targets}"
    for split in train test; do
        bash run_image_textprompt_pipeline_tta.sh \
            --gpuid ${GPU_ID} \
            --model-path ${MODEL_PATH} \
            --input ${INPUT_PATH}/${dataset}/${split} \
            --targets ${targets} \
            --output ./annotations/${dataset}/${split}
    done
done



for DATASET_NAME in dataset1 dataset2 dataset3; do  
  python yolo_to_coco.py --image-dir ${INPUT_PATH}/${DATASET_NAME}/train/ --label-dir ./annotations/${DATASET_NAME}/train/labels -o ./annotations/${DATASET_NAME}/train/coco.json
  python yolo_to_coco.py --image-dir ${INPUT_PATH}/${DATASET_NAME}/test/ --label-dir ./annotations/${DATASET_NAME}/test/labels -o ./annotations/${DATASET_NAME}/test/coco.json
done