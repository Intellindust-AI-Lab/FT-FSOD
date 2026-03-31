#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CONFIG_DIR="configs_rf100vlfsod/final_configs"
CKPT_DIR="exp_rf100vlfsod_results/"
RESULT_OUTPUT_DIR="exp_rf100vlfsod_results/"

mkdir -p "${CKPT_DIR}"
mkdir -p "${RESULT_OUTPUT_DIR}"


for config_file in "${CONFIG_DIR}"/grounding_dino_swin-l_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_10shot.py" ]]; then
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-l_finetune_//' | sed 's/\.py$//')
        
        work_dir="${CKPT_DIR}/swinL_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_train.sh "${config_file}" 2 9999 "0,1" --work-dir "${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinL_all_${dataset_name}"
        ckpt_dir="${CKPT_DIR}/swinL_all_${dataset_name}"
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 2 9999 "0,1" --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done

