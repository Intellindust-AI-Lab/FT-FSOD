#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CONFIG_DIR="configs_cdfsod/final_configs_cdmixed"
CKPT_DIR="exp_cdfosd_results"
RESULT_OUTPUT_DIR="exp_cdfosd_results_cdmixed"

# Ensure output directories exist
mkdir -p "${CKPT_DIR}"
mkdir -p "${RESULT_OUTPUT_DIR}"


# Loop over all config files
for config_file in "${CONFIG_DIR}"/grounding_dino_swin-l_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_1shot.py" ]]; then
        # Extract dataset name from config filename
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-l_finetune_//' | sed 's/\.py$//')
        
        # Build output directory path
        work_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinB_all_${dataset_name}"
        # Build checkpoint directory path
        ckpt_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        # Find the best checkpoint file
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 6 1204 "4,5,8,6,7,9" --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done




# Loop over all config files
for config_file in "${CONFIG_DIR}"/grounding_dino_swin-l_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_10shot.py" ]]; then
        # Extract dataset name from config filename
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-l_finetune_//' | sed 's/\.py$//')
        
        # Build output directory path
        work_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinB_all_${dataset_name}"
        # Build checkpoint directory path
        ckpt_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        # Find the best checkpoint file
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 6 1204 "4,5,8,6,7,9" --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done



# Loop over all config files
for config_file in "${CONFIG_DIR}"/grounding_dino_swin-l_finetune_*.py; do
    if [ -f "${config_file}" ] && [[ "$(basename "${config_file}")" == *"_5shot.py" ]]; then
        # Extract dataset name from config filename
        dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-l_finetune_//' | sed 's/\.py$//')
        
        # Build output directory path
        work_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        
        echo "Processing dataset: ${dataset_name}"
        echo "Config file: ${config_file}"
        echo "Output directory: ${work_dir}"

        work_dir_test="${RESULT_OUTPUT_DIR}/swinB_all_${dataset_name}"
        # Build checkpoint directory path
        ckpt_dir="${CKPT_DIR}/swinB_all_${dataset_name}"
        # Find the best checkpoint file
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1

        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 4 1204 "4,5,8,9" --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
        
        echo "Finished processing ${dataset_name}"
        echo "----------------------------------------"
    fi
done








