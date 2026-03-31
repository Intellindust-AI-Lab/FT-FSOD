#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CONFIG_DIR="configs_odinw/final_configs"
CKPT_DIR="exp_odinwfsod_results/"
RESULT_OUTPUT_DIR="exp_odinwfsod_results/"

mkdir -p "${CKPT_DIR}"
mkdir -p "${RESULT_OUTPUT_DIR}"

for shot_dir in "${CONFIG_DIR}"/{1shot,3shot,5shot,10shot}; do
    if [ ! -d "${shot_dir}" ]; then
        continue
    fi
    
    shot_name=$(basename "${shot_dir}")
    
    for seed_dir in "${shot_dir}"/${shot_name}_seed*; do
        if [ ! -d "${seed_dir}" ]; then
            continue
        fi
        
        seed_name=$(basename "${seed_dir}")
        echo "Processing seed directory: ${seed_name}"
        
        for config_file in "${seed_dir}"/grounding_dino_swin-l_finetune_*.py; do
            if [ -f "${config_file}" ]; then
                dataset_name=$(basename "${config_file}" | sed 's/grounding_dino_swin-l_finetune_//' | sed 's/\.py$//')
                
                work_dir="${CKPT_DIR}/${shot_name}/${seed_name}/swinL_all_${dataset_name}"
                
                echo "Processing dataset: ${dataset_name}"
                echo "Config file: ${config_file}"
                echo "Output directory: ${work_dir}"

                export NCCL_P2P_DISABLE=1
                export NCCL_IB_DISABLE=1

                ./tools/dist_train.sh "${config_file}" 2 8888 "6,7" --work-dir "${work_dir}"

                work_dir_test="${RESULT_OUTPUT_DIR}/${shot_name}/${seed_name}/swinL_all_${dataset_name}"
                ckpt_dir="${CKPT_DIR}/${shot_name}/${seed_name}/swinL_all_${dataset_name}"
                ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

                export NCCL_P2P_DISABLE=1
                export NCCL_IB_DISABLE=1

                ./tools/dist_test.sh "${config_file}" "${ckpt_path}" 2 8888 "6,7" --work-dir "${work_dir_test}" --out "${work_dir_test}/${dataset_name}.pkl"
                
                echo "Finished processing ${shot_name}/${seed_name}/${dataset_name}"
                echo "----------------------------------------"
            fi
        done
    done
done