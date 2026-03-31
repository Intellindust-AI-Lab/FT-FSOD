#!/bin/bash

DATA_PATH="./exp_cdfsodchallenge26_results_trainval1/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for total_name in dataset1 dataset2 dataset3; do
    for dataset_name in 1shot 5shot 10shot; do
        echo "Converting ${total_name} ${dataset_name}..."
        python "${SCRIPT_DIR}/pickle_to_coco.py" \
            --data_path "${DATA_PATH}" \
            --total_name "${total_name}" \
            --dataset_name "${dataset_name}"
    done
done


DATA_PATH="./exp_cdfsodchallenge26_results_trainval2/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for total_name in dataset1 dataset2 dataset3; do
    for dataset_name in 1shot 5shot 10shot; do
        echo "Converting ${total_name} ${dataset_name}..."
        python "${SCRIPT_DIR}/pickle_to_coco.py" \
            --data_path "${DATA_PATH}" \
            --total_name "${total_name}" \
            --dataset_name "${dataset_name}"
    done
done
