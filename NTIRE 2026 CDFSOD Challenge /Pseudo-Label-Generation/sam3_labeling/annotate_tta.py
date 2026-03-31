#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 object detection script (TTA, no translation).

Similar to annotate_with_translate_tta.py, but without a translation model:
- Use targets directly as English text queries
- Optional horizontal/vertical flip TTA
- Run inference on original + flipped views
- Map flipped boxes back to original coordinates and fuse per class with Soft-NMS
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
GLOBAL_COUNTER = 1


def get_image_files(input_path: str) -> List[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        raise ValueError(f"Unsupported image format: {input_path.suffix}")

    if input_path.is_dir():
        image_files: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        if not image_files:
            raise ValueError(f"No image files found in {input_path}")
        return sorted(image_files)

    raise ValueError(f"Path does not exist: {input_path}")


def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "Please set image first"
    h, w = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[h, w],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


def setup_model_and_transforms(bpe_path: str, checkpoint_path: str, confidence: float):
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=confidence,
        to_cpu=True,
    )
    return model, transform, postprocessor


def save_yolo_center_with_conf(
    image_path: Path,
    results: Dict[int, Dict[str, torch.Tensor]],
    output_dir: Path,
) -> int:
    """
    Save YOLO 6-column format: class_id score cx cy w h (normalized)
    """
    try:
        with Image.open(image_path) as im:
            img_width, img_height = im.size
    except Exception:
        img_width, img_height = 1e6, 1e6

    if img_width <= 0 or img_height <= 0:
        return 0

    output_file = output_dir / f"{image_path.stem}.txt"
    lines: List[str] = []

    for class_id, result in results.items():
        if "boxes" not in result or len(result["boxes"]) == 0:
            continue
        boxes = result["boxes"].cpu().numpy()
        scores = result.get("scores")
        scores_np = scores.cpu().numpy() if scores is not None else None

        for idx, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(float, box)
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(float(img_width), x_max)
            y_max = min(float(img_height), y_max)
            if x_min >= x_max or y_min >= y_max:
                continue

            cx = (x_min + x_max) / 2.0 / img_width
            cy = (y_min + y_max) / 2.0 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            score = float(scores_np[idx]) if scores_np is not None else 1.0
            lines.append(
                f"{class_id} {score:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(lines)


def _flip_boxes_h(boxes: np.ndarray, width: float) -> np.ndarray:
    """Flip boxes horizontally (xyxy)."""
    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1]
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3]
    new_x1 = width - x2
    new_x2 = width - x1
    out = boxes.copy()
    out[:, 0] = new_x1
    out[:, 2] = new_x2
    return out


def _flip_boxes_v(boxes: np.ndarray, height: float) -> np.ndarray:
    """Flip boxes vertically (xyxy)."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2]
    y2 = boxes[:, 3].copy()
    new_y1 = height - y2
    new_y2 = height - y1
    out = boxes.copy()
    out[:, 1] = new_y1
    out[:, 3] = new_y2
    return out


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
    sigma: float = 0.5,
    score_thresh: float = 1e-4,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple Soft-NMS (executed independently per class).
    boxes: [N,4] xyxy, scores: [N]
    Returns kept boxes and scores.
    """
    if boxes.size == 0:
        return boxes, scores

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    N = boxes.shape[0]
    idxs = np.arange(N)

    keep_boxes: List[np.ndarray] = []
    keep_scores: List[float] = []

    while len(idxs) > 0:
        # Highest-score box at current step
        max_idx = idxs[np.argmax(scores[idxs])]
        max_box = boxes[max_idx].copy()
        max_score = scores[max_idx]

        keep_boxes.append(max_box)
        keep_scores.append(float(max_score))

        # Compute IoU against it
        others = idxs[idxs != max_idx]
        if len(others) == 0:
            break

        xx1 = np.maximum(max_box[0], boxes[others, 0])
        yy1 = np.maximum(max_box[1], boxes[others, 1])
        xx2 = np.minimum(max_box[2], boxes[others, 2])
        yy2 = np.minimum(max_box[3], boxes[others, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_max = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        area_others = (boxes[others, 2] - boxes[others, 0]) * (
            boxes[others, 3] - boxes[others, 1]
        )
        iou = inter / (area_max + area_others - inter + 1e-6)

        if method == "linear":
            weight = np.ones_like(iou)
            weight[iou > iou_thresh] -= iou[iou > iou_thresh]
        else:  # gaussian
            weight = np.exp(-(iou * iou) / sigma)

        scores[others] = scores[others] * weight

        # Drop low-score boxes
        idxs = others[scores[others] >= score_thresh]

    if not keep_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.stack(keep_boxes, axis=0), np.array(keep_scores, dtype=np.float32)


def run_single_image_tta(
    img_path: Path,
    class_names: List[str],
    model,
    transform,
    postprocessor,
    device,
    tta_h: bool,
    tta_v: bool,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Run TTA inference for one image and apply per-class Soft-NMS in
    the original image coordinate system.
    Returns: {class_id: {'boxes': tensor[N,4], 'scores': tensor[N]}}
    """
    try:
        pil_orig = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Warning: failed to load image {img_path}: {e}")
        return {}

    w, h = pil_orig.size

    # Define TTA views
    tta_configs = [("orig", pil_orig)]
    if tta_h:
        tta_configs.append(("h", pil_orig.transpose(Image.FLIP_LEFT_RIGHT)))
    if tta_v:
        tta_configs.append(("v", pil_orig.transpose(Image.FLIP_TOP_BOTTOM)))
    if tta_h and tta_v:
        tta_configs.append(
            (
                "hv",
                pil_orig.transpose(Image.FLIP_LEFT_RIGHT).transpose(
                    Image.FLIP_TOP_BOTTOM
                ),
            )
        )

    # Aggregate all TTA results (in numpy first)
    agg_boxes: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(class_names))}
    agg_scores: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(class_names))}

    for tag, pil_img in tta_configs:
        datapoint = create_empty_datapoint()
        set_image(datapoint, pil_img)

        query_ids: Dict[int, int] = {}
        for class_id, class_name in enumerate(class_names):
            qid = add_text_prompt(datapoint, class_name)
            query_ids[class_id] = qid

        dp = transform(datapoint)
        batch = collate([dp], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)

        with torch.inference_mode():
            output = model(batch)
        processed = postprocessor.process_results(output, batch.find_metadatas)

        # Single image: only use query ids from this image
        for class_id, qid in query_ids.items():
            if qid not in processed:
                continue
            det = processed[qid]
            if "boxes" not in det or len(det["boxes"]) == 0:
                continue
            boxes = det["boxes"].cpu().numpy()
            scores = (
                det["scores"].cpu().numpy()
                if det.get("scores") is not None
                else np.ones((boxes.shape[0],), dtype=np.float32)
            )

            # Map boxes from current view back to original image
            if tag == "orig":
                boxes_back = boxes
            else:
                boxes_back = boxes
                if "h" in tag:
                    boxes_back = _flip_boxes_h(boxes_back, float(w))
                if "v" in tag:
                    boxes_back = _flip_boxes_v(boxes_back, float(h))

            agg_boxes[class_id].append(boxes_back)
            agg_scores[class_id].append(scores)

    # Run soft-nms per class and build final results
    final_results: Dict[int, Dict[str, torch.Tensor]] = {}
    for class_id in range(len(class_names)):
        if not agg_boxes[class_id]:
            continue
        boxes_cat = np.concatenate(agg_boxes[class_id], axis=0)
        scores_cat = np.concatenate(agg_scores[class_id], axis=0)

        boxes_n, scores_n = soft_nms(
            boxes_cat,
            scores_cat,
            iou_thresh=0.5,
            sigma=0.5,
            score_thresh=1e-4,
            method="linear",
        )
        if boxes_n.size == 0:
            continue

        final_results[class_id] = {
            "boxes": torch.from_numpy(boxes_n),
            "scores": torch.from_numpy(scores_n),
        }

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 object detection script (TTA, no translation) - single image or directory"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input image path or directory path",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        nargs="+",
        default=["object"],
        help="Target text list (English), space-separated, e.g. --targets person car dog",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for annotations (YOLO 6-column center format)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.01,
        help="Confidence threshold passed to SAM3 postprocessor (default: 0.01)",
    )
    parser.add_argument(
        "--bpe-path",
        type=str,
        default="assets/bpe_simple_vocab_16e6.txt.gz",
        help="BPE vocabulary path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="your_model_path/sam3.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--tta-horizontal",
        action="store_true",
        help="Enable horizontal flip TTA (original + horizontal flip)",
    )
    parser.add_argument(
        "--tta-vertical",
        action="store_true",
        help="Enable vertical flip TTA (original + vertical flip)",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Legacy compatibility flag (this script always outputs 6-column YOLO with score); ignored.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Use targets directly without translation
    class_names = args.targets.copy()

    # Image list
    print(f"Scanning input path: {args.input}")
    image_files = get_image_files(args.input)
    print(f"Found {len(image_files)} images")

    # Output directory + classes.txt
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Annotation results will be saved to: {output_dir}")

    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print(f"Class list written to: {classes_file}")

    if not Path(args.bpe_path).exists() or not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(
            f"BPE vocab file or checkpoint not found: {args.bpe_path} or {args.checkpoint_path}"
        )

    # Model
    model, transform, postprocessor = setup_model_and_transforms(
        args.bpe_path, args.checkpoint_path, args.confidence
    )
    model = model.to(device)
    model.eval()

    print(f"\nDetection targets: {class_names}")
    print(f"Number of images: {len(image_files)}")
    print(f"TTA horizontal: {args.tta_horizontal}, vertical: {args.tta_vertical}\n")

    total_detections = 0
    with tqdm(total=len(image_files), desc="Processing") as pbar:
        for img_path in image_files:
            per_img_results = run_single_image_tta(
                img_path=img_path,
                class_names=class_names,
                model=model,
                transform=transform,
                postprocessor=postprocessor,
                device=device,
                tta_h=args.tta_horizontal,
                tta_v=args.tta_vertical,
            )

            num_dets = save_yolo_center_with_conf(
                img_path,
                per_img_results,
                output_dir,
            )
            total_detections += num_dets
            pbar.update(1)

    print("\nDone!")
    print(f"Processed images: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Annotation files saved to: {output_dir}")


if __name__ == "__main__":
    main()
