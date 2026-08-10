#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CONFIG_DIR="configs_cdfsod/final_configs_bs4"
CKPT_DIR="exp_cdfosd_results/"
RESULT_OUTPUT_DIR="exp_cdfosd_results/"

GPUID="4"  # Set this to the GPU ID you want to use (e.g., "0", "1", "2", "3", etc.)
PORT="9999"

mkdir -p "${CKPT_DIR}"
mkdir -p "${RESULT_OUTPUT_DIR}"


for config_file in "${CONFIG_DIR}"/grounding_dino_swin-b_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_1shot.py" ]]; then
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-b_finetune_//' | sed 's/\.py$//')
        
        work_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_train.sh "${config_file}" 1 $PORT $GPUID --work-dir "${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinB_all_${dataset_name}"
        ckpt_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 1 $PORT $GPUID --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done



for config_file in "${CONFIG_DIR}"/grounding_dino_swin-b_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_5shot.py" ]]; then
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-b_finetune_//' | sed 's/\.py$//')
        
        work_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_train.sh "${config_file}" 1 $PORT $GPUID --work-dir "${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinB_all_${dataset_name}"
        ckpt_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 1 $PORT $GPUID --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done



for config_file in "${CONFIG_DIR}"/grounding_dino_swin-b_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_10shot.py" ]]; then
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-b_finetune_//' | sed 's/\.py$//')
        
        work_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_train.sh "${config_file}" 1 $PORT $GPUID --work-dir "${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinB_all_${dataset_name}"
        ckpt_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 1 $PORT $GPUID --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done
