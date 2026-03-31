DATASET_PATH="your_dataset_pathcdfsod_test_26/"

for DATASET_NAME in dataset1 dataset2; do
  SUBDATASET_PATH=${DATASET_PATH}/${DATASET_NAME}
  for i in 1 5 10; do
    python merge_gt_to_pred.py \
      --gt-json ${SUBDATASET_PATH}/annotations/${i}_shot.json \
      --pred-label-dir ./sam3_labeling/annotations/${DATASET_NAME}/train/labels \
      --pred-image-dir ${SUBDATASET_PATH}/train \
      --pred-test-label-dir ./sam3_labeling/annotations/${DATASET_NAME}/test/labels \
      --pred-test-image-dir ${SUBDATASET_PATH}/test \
      --output ${SUBDATASET_PATH}/annotations/${i}_shot_sam3_conf08_strategy2.json \
      --score-threshold 0.8 \
      --iou-thresh 0.8 \
      --train-val-split \
      --split-strategy strategy2
  done
done


DATASET_NAME="dataset3"
SUBDATASET_PATH=${DATASET_PATH}/${DATASET_NAME}
for i in 1 5 10; do
  python merge_gt_to_pred.py \
    --gt-json ${SUBDATASET_PATH}/annotations/${i}_shot.json \
    --pred-label-dir ./qwen_labeling/annotations/${DATASET_NAME}/train/labels \
    --pred-image-dir ${SUBDATASET_PATH}/train \
    --pred-test-label-dir ./qwen_labeling/annotations/${DATASET_NAME}/test/labels \
    --pred-test-image-dir ${SUBDATASET_PATH}/test \
    --output ${SUBDATASET_PATH}/annotations/${i}_shot_qwen_conf08_strategy2.json \
    --score-threshold 0.8 \
    --iou-thresh 0.8 \
    --train-val-split \
    --split-strategy strategy2
done


