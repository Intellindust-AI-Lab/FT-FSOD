# A Closer Look at Cross-Domain Few-Shot Object Detection:  
Fine-Tuning Matters and Parallel Decoder Helps



[Xuanlong Yu](https://xuanlong-yu.github.io/)1, Youyang Sha1, [Longfei Liu](https://capsule2077.github.io/)1, [Xi Shen](https://xishen0220.github.io/)† 1, [Di Yang](https://walker1126.github.io/)† 2

1[Intellindust AI Lab](https://intellindust-ai-lab.github.io/)    2Suzhou Institute for Advanced Research, USTC

---

## 🔍 TL;DR

- We introduce a Hybrid Ensemble Decoder (HED) to improve query diversity and model robustness.
- We propose a simple fine-tuning recipe for cross-domain few-shot object detection, with plateau-aware scheduling and progressive optimization.
- This repository includes:
  - Training/evaluation code and configs for CD-FSOD, ODinW-13, and RF100-VL benchmarks.
  - A challenge subproject: [NTIRE 2026 CDFSOD Challenge](./NTIRE%202026%20CDFSOD%20Challenge%20/README.md), including pseudo-label annotation strategy and challenge-oriented pipeline details.



---

## 📦 Installation

```bash
# Create conda environment
conda create -n ft-fsod python=3.10 -y
conda activate ft-fsod

# Install PyTorch first (pick the command matching your CUDA)
# Example for CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install OpenMMLab dependencies
pip install -U openmim
mim install "mmengine"
mim install "mmcv==2.1.0"
mim install mmdet

# Common dependencies
pip install -r requirements.txt
```

**Deterministic hack (MMEngine):** to reduce few-shot training instability, we set `deterministic=True` in configs, which needs to patch MMEngine as follows:  

1) `mmengine/runner/runner.py` (`set_randomness`): add `warn_only: bool = True` and pass `warn_only=warn_only` to `set_random_seed`.  

2) `mmengine/runner/utils.py` (`set_random_seed`): add `warn_only: bool = True`, and change to `torch.use_deterministic_algorithms(True, warn_only=warn_only)`.  

Refs: [runner.py#L698](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L698), [runner.py#L719](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L719), [utils.py#L48](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/utils.py#L48), [utils.py#L90](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/utils.py#L90).

---

## 📍 Fine-tuning on Benchmarks

#### 1) Download bert-base-uncased, pre-trained weights and train models

- Download bert-base-uncased and nltk_data following this [instruction](bert-base-uncased)
- Download pre-trained weight from: [MMGDINO-B](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth) and [MMGDINO-L](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth)
- Adjust the dataset path and pre-trained weight path in `src_path.py`

#### 2) Run the following scripts to fine-tune the models on various benchmarks

- **CD-FSOD (1/5/10-shot train+eval):** `bash run_mmgdinob_traineval_cdfsod.sh`
- **ODinW-13 (1/3/5/10-shot, multi-seed):** `bash run_mmgdinol_traineval_odwin.sh`
- **RF100-VL (10-shot):** `bash run_mmgdinol_traineval_rf100vl.sh`

*Due to the instability of few-shot fine-tuning (even if the random seed is fixed), the results will be slightly different from the ones in the paper.*

#### 3) Run the following scripts to create CD-Mixed test set and evaluate your CD-FSOD fine-tuned models on it

- **Create CD-Mixed test set:** `bash create_cdmixed_set.sh`
- **CD-Mixed evaluation (1/5/10-shot):** `bash run_mmgdinob_eval_cdmixed.sh`

## 🏁 NTIRE 2026 CDFSOD Challenge

Challenge materials and instructions are in the same repository: [NTIRE 2026 CDFSOD Challenge](./NTIRE%202026%20CDFSOD%20Challenge%20/README.md).  
It includes our challenge-oriented pipeline, such as pseudo-label annotation strategy using FSOD-mAP as the evaluation metric for annotation selection, model fine-tuning and TTAs.

---

## 📊 Result Aggregation

CD-FSOD:

```bash
python analyze_results_cdfsod.py <experiment_directory_path>
# e.g.
python analyze_results_cdfsod.py exp_cdfosd_results
```

ODinW-13:

```bash
python analyze_results_odinw.py <experiment_directory_path>
# e.g.
python analyze_results_odinw.py exp_odinwfsod_results
```

RF100-VL:

```bash
python analyze_results_rf100.py <experiment_directory_path>
# e.g.
python analyze_results_rf100.py exp_rf100vlfsod_results
```

---

## 🙏 Acknowledgements

This project is built on top of the [OpenMMLab ecosystem](https://github.com/open-mmlab/mmdetection), [RF100-VL benchmark](https://github.com/roboflow/rf100-vl), [CD-FSOD benchmark](https://github.com/lovelyqian/CDFSOD-benchmark) and [OdinW-13 benchmark](https://github.com/microsoft/GLIP).

---

## 📚 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{yu2026acloser,
  title={A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps},
  author={Yu, Xuanlong and Sha, Youyang and Liu, Longfei and Shen, Xi and Yang, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 📄 License

This project is released under the [Apache 2.0 License](./LICENSE).