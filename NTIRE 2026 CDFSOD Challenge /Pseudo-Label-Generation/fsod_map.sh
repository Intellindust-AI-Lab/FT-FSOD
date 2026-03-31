
DATASET_PATH=your_dataset_path

for DATASET_NAME in dataset1 dataset2 dataset3; do
    SUBDATASET_PATH=${DATASET_PATH}/${DATASET_NAME}
    for i in 1 5 10; do
      python fsod_map.py ./sam3_labeling/annotations/${DATASET_NAME}/train/coco.json ${SUBDATASET_PATH}/annotations/${i}_shot.json --metric fsod_map
    done
done



for DATASET_NAME in dataset1 dataset2 dataset3; do
    SUBDATASET_PATH=${DATASET_PATH}/${DATASET_NAME}
    for i in 1 5 10; do
      python fsod_map.py ./qwen_labeling/annotations/${DATASET_NAME}/train/coco.json ${SUBDATASET_PATH}/annotations/${i}_shot.json --metric fsod_map
    done
done