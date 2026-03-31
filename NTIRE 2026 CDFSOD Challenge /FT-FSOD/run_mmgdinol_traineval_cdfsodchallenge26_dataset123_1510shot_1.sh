#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CONFIG_DIR="configs_cdfsod26/final_configs_trainval1"
CKPT_DIR="exp_cdfsodchallenge26_results_trainval1/"
RESULT_OUTPUT_DIR="exp_cdfsodchallenge26_results_trainval1/"
RESULT_OUTPUT_DIR_TTA="exp_cdfsodchallenge26_results_trainval1/tta/"
GPU_IDS="0,1"
NUM_GPUS=2
PORT_START=1111

mkdir -p "${CKPT_DIR}"
mkdir -p "${RESULT_OUTPUT_DIR}"


DATASETS=("dataset1" "dataset2" "dataset3")
SHOTS=("1shot" "5shot" "10shot")

for dataset in "${DATASETS[@]}"; do
    for shot in "${SHOTS[@]}"; do
        config_file="${CONFIG_DIR}/grounding_dino_swin-l_finetune_${dataset}_${shot}.py"

        if [ ! -f "${config_file}" ]; then
            echo "[WARN] Config not found: ${config_file} (skip)"
            continue
        fi

        echo "=========================================="
        echo "Processing ${dataset} - ${shot}"
        echo "Config file: ${config_file}"
        echo "=========================================="

        run_name="swinL_${dataset}_${shot}"
        work_dir="${CKPT_DIR}/${dataset}/${shot}/${run_name}"

        mkdir -p "${work_dir}"

        echo "[TRAIN] work_dir = ${work_dir}"
        ./tools/dist_train.sh "${config_file}" ${NUM_GPUS} ${PORT_START} "${GPU_IDS}" --work-dir "${work_dir}"

        
        work_dir_test="${RESULT_OUTPUT_DIR}/${dataset}/${shot}/${run_name}"
        mkdir -p "${work_dir_test}"

        ckpt_dir="${work_dir}"
        ckpt_path=$(find "${ckpt_dir}" -name "best_coco_bbox_mAP_iter_*.pth" | head -n 1)

        if [ -z "${ckpt_path}" ]; then
            echo "[WARN] No best checkpoint found in ${ckpt_dir}, skip test."
            continue
        fi

        echo "[TEST TTA] Using checkpoint: ${ckpt_path}"
        ./tools/dist_test.sh "${config_file}" "${ckpt_path}" ${NUM_GPUS} ${PORT_START} "${GPU_IDS}" \
            --work-dir "${work_dir_test}" \
            --out "${work_dir_test}/${dataset}_${shot}.pkl" \
            --tta

        echo "Finished processing ${dataset} ${shot}"
        echo "----------------------------------------"
    done
done