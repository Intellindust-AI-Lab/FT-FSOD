#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge GT COCO JSON and prediction COCO JSON into a new GT-style COCO JSON.

Logic:
- Input:
  - gt_json: standard COCO GT annotations (dict with images / annotations / categories)
  - pred_json: COCO predictions
      * supports two formats:
        1) prediction list (default output of yolo_to_coco_pred.py)
        2) dict with "annotations" field
- Processing:
  1) Keep prediction boxes whose image_id exists in gt_json["images"]
  2) For each image_id:
     - Compare all GT and prediction annotations by IoU
     - For each GT box, remove prediction boxes with IoU >= iou_thresh and same category_id
  3) Final annotations = all GT annotations + filtered prediction annotations
     - Re-assign annotation ids from 1 to avoid id conflicts
- Output:
  - A GT-style COCO JSON: {"images": ..., "annotations": ..., "categories": ...}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image


def load_image_files(image_dir: Path) -> List[Path]:
    """Load image files in sorted filename order (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files: List[Path] = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _ensure_coco_gt(obj: Any) -> Dict:
    """
    Ensure GT COCO dict structure.
    If input is a list, treat it as annotations and wrap as a simple dict.
    """
    if isinstance(obj, dict):
        # Basic robustness checks
        obj.setdefault("images", [])
        obj.setdefault("annotations", [])
        obj.setdefault("categories", [])
        return obj
    elif isinstance(obj, list):
        return {
            "images": [],
            "annotations": obj,
            "categories": [],
        }
    else:
        raise ValueError("GT JSON must be dict or list.")


def _ensure_coco_pred(obj: Any) -> List[Dict]:
    """
    Normalize prediction JSON to an annotation list.
    Supports:
      - list[annotation]
      - {"annotations": [...]}
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        anns = obj.get("annotations", [])
        if not isinstance(anns, list):
            raise ValueError("pred_json['annotations'] must be a list.")
        return anns
    raise ValueError("Prediction JSON must be list or dict with 'annotations'.")


def parse_yolo_line(line: str) -> Tuple[int, float, float, float, float, float]:
    """
    Support:
      - 5 cols: class x y w h (score=1.0)
      - 6 cols: class score x y w h
    """
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    if len(parts) == 5:
        cls = int(parts[0])
        score = 1.0
        x, y, w, h = map(float, parts[1:5])
    elif len(parts) == 6:
        cls = int(parts[0])
        score = float(parts[1])
        x, y, w, h = map(float, parts[2:6])
    else:
        raise ValueError(f"unexpected column count: {len(parts)}")
    return cls, score, x, y, w, h


def _build_image_lookup(image_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Build:
    - file_name -> image_path
    - stem -> image_path
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            by_name[p.name] = p
            by_stem[p.stem] = p
    return by_name, by_stem


def build_pred_from_yolo(
    gt_coco: Dict,
    pred_label_dir: Path,
    pred_image_dir: Path,
    class_offset: int = 1,
    score_threshold: float = 0.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert YOLO label directory to COCO-style predictions aligned by file_name:
    - If label stem matches GT image file_name/stem -> use GT image_id
    - Otherwise assign new image_id (> max GT image_id)
    Returns:
    - pred_images: image records for all pred-involved images (matched + unmatched)
    - pred_anns: annotation list
    """
    gt_images = gt_coco.get("images", [])
    gt_name_to_id = {img.get("file_name"): img.get("id") for img in gt_images if "file_name" in img and "id" in img}
    gt_stem_to_id = {Path(img["file_name"]).stem: img["id"] for img in gt_images if "file_name" in img and "id" in img}
    gt_id_to_img = {img["id"]: img for img in gt_images if "id" in img}

    max_gt_img_id = max([img["id"] for img in gt_images], default=0)
    next_img_id = max_gt_img_id + 1

    img_by_name, img_by_stem = _build_image_lookup(pred_image_dir)

    pred_images_by_id: Dict[int, Dict] = {}
    pred_anns: List[Dict] = []
    ann_id = 1
    missing_images = 0
    pred_total = 0
    pred_filtered_by_score = 0

    label_files = sorted(
        [
            p for p in pred_label_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt" and p.name != "classes.txt"
        ]
    )

    for label_path in label_files:
        stem = label_path.stem
        image_path = img_by_stem.get(stem)
        if image_path is None:
            missing_images += 1
            continue

        file_name = image_path.name
        if file_name in gt_name_to_id:
            img_id = gt_name_to_id[file_name]
        elif stem in gt_stem_to_id:
            img_id = gt_stem_to_id[stem]
        else:
            img_id = next_img_id
            next_img_id += 1

        if img_id in gt_id_to_img:
            # Prefer original GT image metadata for consistency
            pred_images_by_id[img_id] = gt_id_to_img[img_id]
            width = int(gt_id_to_img[img_id].get("width", 0))
            height = int(gt_id_to_img[img_id].get("height", 0))
            if width <= 0 or height <= 0:
                with Image.open(image_path) as im:
                    width, height = im.size
        else:
            with Image.open(image_path) as im:
                width, height = im.size
            pred_images_by_id[img_id] = {
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }

        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cls, score, x, y, w, h = parse_yolo_line(line)
                except ValueError:
                    continue

                pred_total += 1
                if score < score_threshold:
                    pred_filtered_by_score += 1
                    continue

                x1 = (x - w / 2.0) * width
                y1 = (y - h / 2.0) * height
                bw = w * width
                bh = h * height

                x1 = max(0.0, min(float(x1), float(width)))
                y1 = max(0.0, min(float(y1), float(height)))
                bw = max(0.0, min(float(bw), float(width - x1)))
                bh = max(0.0, min(float(bh), float(height - y1)))
                if bw <= 0 or bh <= 0:
                    continue

                pred_anns.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls) + class_offset,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "score": float(score),
                    }
                )
                ann_id += 1

    if missing_images > 0:
        print(f"Warning: {missing_images} label files have no matched image in {pred_image_dir}.")

    if score_threshold > 0 and pred_total > 0:
        kept = pred_total - pred_filtered_by_score
        print(
            f"Score filter (--score-threshold={score_threshold}): "
            f"removed {pred_filtered_by_score}, kept {kept} / {pred_total} predictions."
        )

    pred_images = sorted(pred_images_by_id.values(), key=lambda d: d["id"])
    return pred_images, pred_anns


def bbox_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    COCO bbox: [x, y, w, h] -> (x1, y1, x2, y2)
    """
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou_bbox(b1: List[float], b2: List[float]) -> float:
    """
    IoU between two COCO bboxes [x, y, w, h].
    """
    x1_min, y1_min, x1_max, y1_max = bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = bbox_to_xyxy(b2)

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    if area1 <= 0 or area2 <= 0:
        return 0.0

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def merge_gt_and_pred(
    gt_coco: Dict,
    pred_images: List[Dict],
    pred_anns: List[Dict],
    iou_thresh: float = 0.7,
) -> Dict:
    """
    Merge GT and predictions into a new GT-style COCO JSON.

    - Keep prediction boxes whose image_id exists in GT
    - For each GT box, remove same-class predictions with IoU >= iou_thresh
    - Final annotations = GT annotations + filtered predictions (ids reassigned)
    """
    gt_images = gt_coco.get("images", [])
    gt_annotations = gt_coco.get("annotations", [])
    categories = gt_coco.get("categories", [])

    gt_image_ids = {img["id"] for img in gt_images}

    # Group GT and predictions by image_id
    gt_by_img: Dict[int, List[Dict]] = {}
    for ann in gt_annotations:
        img_id = ann["image_id"]
        if img_id not in gt_image_ids:
            # Keep GT annotations even if their image is missing from images;
            # they will not affect IoU-based filtering
            pass
        gt_by_img.setdefault(img_id, []).append(ann)

    # Group predictions by image_id
    pred_by_img: Dict[int, List[Dict]] = {}
    for ann in pred_anns:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        pred_by_img.setdefault(img_id, []).append(ann)

    # Mark prediction boxes to remove (prefer original id when available)
    to_delete_ids = set()
    for img_id, gt_anns in gt_by_img.items():
        preds = pred_by_img.get(img_id, [])
        if not preds:
            continue
        for gt_ann in gt_anns:
            gt_cat = gt_ann.get("category_id")
            gt_box = gt_ann.get("bbox", [])
            if not gt_box or len(gt_box) != 4:
                continue
            for pred_ann in preds:
                if pred_ann.get("category_id") != gt_cat:
                    continue
                pred_box = pred_ann.get("bbox", [])
                if not pred_box or len(pred_box) != 4:
                    continue
                iou = iou_bbox(gt_box, pred_box)
                if iou >= iou_thresh:
                    # Remove same-class prediction boxes with high overlap with GT
                    if "id" in pred_ann:
                        to_delete_ids.add(pred_ann["id"])

    # Print number of removed prediction boxes (only counted when id exists)
    if to_delete_ids:
        print(f"Removed {len(to_delete_ids)} prediction boxes due to IoU >= {iou_thresh}.")
    else:
        print("No prediction boxes removed by IoU filtering.")

    # Count pseudo labels kept after IoU filtering
    total_pred = 0
    kept_pred = 0
    for ann in pred_anns:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        total_pred += 1
        if "id" in ann and ann["id"] in to_delete_ids:
            continue
        kept_pred += 1
    print(f"Pseudo labels added (after IoU filtering): {kept_pred} (from {total_pred} predictions).")

    # Build merged annotations: GT first, then filtered predictions
    merged_annotations: List[Dict] = []

    for ann in gt_annotations:
        merged_annotations.append(dict(ann))  # shallow copy

    for ann in pred_anns:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        if "id" in ann and ann["id"] in to_delete_ids:
            continue
        merged_annotations.append(dict(ann))

    # Re-assign ids to avoid conflicts between original GT and prediction ids
    for new_id, ann in enumerate(merged_annotations, start=1):
        ann["id"] = new_id

    # Merge image metadata: GT images + images appearing only in predictions
    images_by_id: Dict[int, Dict] = {img["id"]: dict(img) for img in gt_images if "id" in img}
    for img in pred_images:
        img_id = img.get("id")
        if img_id is None:
            continue
        if img_id not in images_by_id:
            images_by_id[img_id] = dict(img)
    merged_images = sorted(images_by_id.values(), key=lambda d: d["id"])

    merged_coco = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }
    return merged_coco


def split_train_val_coco(
    merged_coco: Dict,
    gt_coco: Dict,
    val_ratio: float = 0.2,
) -> Tuple[Dict, Dict]:
    """
    Split train/val at image level:
    - Approximate 2:8 (val:train) split
    - Images with GT annotations will not be assigned to val
    """
    images = merged_coco.get("images", [])
    annotations = merged_coco.get("annotations", [])
    categories = merged_coco.get("categories", [])

    # Images in GT are forced into train (with or without annotations)
    gt_image_ids = {
        img.get("id")
        for img in gt_coco.get("images", [])
        if "id" in img
    }

    all_img_ids = [img["id"] for img in images]
    # Candidate val images: images not in GT images
    candidate_val_ids = [i for i in all_img_ids if i not in gt_image_ids]

    if not all_img_ids or not candidate_val_ids:
        # No val can be split: all images are GT images or image list is empty
        train_ids = set(all_img_ids)
        val_ids = set()
    else:
        target_val_count = int(round(len(all_img_ids) * val_ratio))
        target_val_count = max(0, min(target_val_count, len(candidate_val_ids)))
        # For deterministic behavior, choose the first N after sorting
        candidate_val_ids_sorted = sorted(candidate_val_ids)
        val_ids = set(candidate_val_ids_sorted[:target_val_count])
        train_ids = set(all_img_ids) - val_ids

    # Split images
    train_images = [img for img in images if img["id"] in train_ids]
    val_images = [img for img in images if img["id"] in val_ids]

    # Split annotations
    train_anns_raw = [ann for ann in annotations if ann.get("image_id") in train_ids]
    val_anns_raw = [ann for ann in annotations if ann.get("image_id") in val_ids]

    # Re-assign ids to keep continuous ids within each JSON
    train_annotations: List[Dict] = []
    for new_id, ann in enumerate(train_anns_raw, start=1):
        a = dict(ann)
        a["id"] = new_id
        train_annotations.append(a)

    val_annotations: List[Dict] = []
    for new_id, ann in enumerate(val_anns_raw, start=1):
        a = dict(ann)
        a["id"] = new_id
        val_annotations.append(a)

    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }
    return train_coco, val_coco


def build_val_from_test_pred(
    test_json_path: Path,
    categories: List[Dict],
    min_score: float = 0.8,
    test_image_dir: Path = None,
) -> Dict:
    """
    Build val set from test.json (COCO-style predictions):
    - Keep only predictions with score >= min_score
    - No GT, pseudo labels only
    """
    with test_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        images = obj.get("images", [])
    elif isinstance(obj, list):
        # Compatible with COCO prediction list format (e.g., default yolo_to_coco_pred.py output)
        images = []
    else:
        raise ValueError("test.json must be a COCO-style dict or prediction list.")

    anns = _ensure_coco_pred(obj)

    filtered: List[Dict] = []
    for ann in anns:
        if float(ann.get("score", 0.0)) >= min_score:
            filtered.append(dict(ann))

    # Keep only images that have annotations
    valid_img_ids = {ann["image_id"] for ann in filtered if "image_id" in ann}
    if images:
        images = [img for img in images if img.get("id") in valid_img_ids]
    elif test_image_dir is not None:
        # If test.json is a pure prediction list, rebuild images with
        # yolo_to_coco_pred.py-style indexing (sorted by filename, starting from 1)
        rebuilt_images: List[Dict] = []
        for img_id, img_path in enumerate(load_image_files(test_image_dir), start=1):
            if img_id not in valid_img_ids:
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            rebuilt_images.append(
                {
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": w,
                    "height": h,
                }
            )
        images = rebuilt_images
    else:
        # If no images and no test_image_dir, return empty images for valid structure
        images = []

    # Re-assign ids
    for new_id, ann in enumerate(filtered, start=1):
        ann["id"] = new_id

    return {
        "images": images,
        "annotations": filtered,
        "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge GT COCO JSON and prediction COCO JSON. "
            "GT annotations are kept, prediction annotations that overlap "
            "strongly (IoU >= threshold, same category) with GT are removed. "
            "Output is a GT-style COCO JSON."
        )
    )
    parser.add_argument(
        "--gt-json",
        type=str,
        required=True,
        help="Path to GT COCO JSON file.",
    )
    parser.add_argument(
        "--pred-label-dir",
        type=str,
        required=True,
        help="Path to YOLO prediction label directory (txt files).",
    )
    parser.add_argument(
        "--pred-image-dir",
        type=str,
        required=True,
        help="Path to image directory for YOLO labels (used for file_name and size).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output merged GT COCO JSON path.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.7,
        help="IoU threshold for removing prediction boxes (default: 0.7).",
    )
    parser.add_argument(
        "--class-offset",
        type=int,
        default=1,
        help="Add to YOLO class_id for COCO category_id (default: 1).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Filter predictions with score < threshold (default: 0.0).",
    )
    parser.add_argument(
        "--train-val-split",
        action="store_true",
        help=(
            "If set, split merged GT COCO into train/val (approx. 2:8). "
            "Images with GT annotations will NOT be assigned to val."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["original", "strategy1", "strategy2"],
        default="original",
        help=(
            "Train/val split strategy mode: "
            "original=use original split logic; "
            "strategy1=8:2 split (pseudo confidence already filtered by --score-threshold); "
            "strategy2=train uses merged COCO; val from test YOLO or test.json "
            "(filtered by --score-threshold)."
        ),
    )
    parser.add_argument(
        "--test-json",
        type=str,
        default=None,
        help="Optional when --split-strategy=strategy2: provide test.json (COCO-style dict or prediction list).",
    )
    parser.add_argument(
        "--pred-test-label-dir",
        type=str,
        default=None,
        help="Optional when --split-strategy=strategy2: YOLO prediction label directory (txt) for test set.",
    )
    parser.add_argument(
        "--pred-test-image-dir",
        type=str,
        default=None,
        help="Optional when --split-strategy=strategy2: test image directory (used with --pred-test-label-dir or list-style --test-json).",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt_json)
    pred_label_dir = Path(args.pred_label_dir)
    pred_image_dir = Path(args.pred_image_dir)
    out_path = Path(args.output)

    with gt_path.open("r", encoding="utf-8") as f:
        gt_obj = json.load(f)

    gt_coco = _ensure_coco_gt(gt_obj)
    pred_images, pred_anns = build_pred_from_yolo(
        gt_coco=gt_coco,
        pred_label_dir=pred_label_dir,
        pred_image_dir=pred_image_dir,
        class_offset=args.class_offset,
        score_threshold=args.score_threshold,
    )

    merged = merge_gt_and_pred(
        gt_coco=gt_coco,
        pred_images=pred_images,
        pred_anns=pred_anns,
        iou_thresh=args.iou_thresh,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.train_val_split:
        # Select pseudo-label handling and train/val split strategy by strategy
        if args.split_strategy == "original":
            merged_for_split = merged
            train_coco, val_coco = split_train_val_coco(
                merged_coco=merged_for_split,
                gt_coco=gt_coco,
                val_ratio=0.2,
            )
        elif args.split_strategy == "strategy1":
            train_coco, val_coco = split_train_val_coco(
                merged_coco=merged,
                gt_coco=gt_coco,
                val_ratio=0.2,
            )
        else:  # strategy2
            if not args.test_json and not args.pred_test_label_dir:
                raise ValueError(
                    "--split-strategy=strategy2 requires --test-json or --pred-test-label-dir."
                )
            train_coco = merged
            # Val under strategy2
            # Option A: build directly from test YOLO labels (recommended, consistent with train flow)
            if args.pred_test_label_dir:
                if not args.pred_test_image_dir:
                    raise ValueError(
                        "--pred-test-label-dir mode requires --pred-test-image-dir."
                    )
                val_pred_images, val_pred_anns = build_pred_from_yolo(
                    gt_coco={"images": [], "annotations": [], "categories": []},
                    pred_label_dir=Path(args.pred_test_label_dir),
                    pred_image_dir=Path(args.pred_test_image_dir),
                    class_offset=args.class_offset,
                    score_threshold=args.score_threshold,
                )
                for new_id, ann in enumerate(val_pred_anns, start=1):
                    ann["id"] = new_id
                val_coco = {
                    "images": val_pred_images,
                    "annotations": val_pred_anns,
                    "categories": train_coco.get("categories", []),
                }
                print(
                    f"Built val from test YOLO labels: {len(val_pred_images)} images, "
                    f"{len(val_pred_anns)} annotations (score >= {args.score_threshold})."
                )
            else:
                # Option B: build from test.json (supports dict and list)
                test_json_path = Path(args.test_json)
                test_img_dir = Path(args.pred_test_image_dir) if args.pred_test_image_dir else None
                val_coco = build_val_from_test_pred(
                    test_json_path=test_json_path,
                    categories=train_coco.get("categories", []),
                    min_score=args.score_threshold,
                    test_image_dir=test_img_dir,
                )

        print(
            f"Train/Val image count: "
            f"{len(train_coco.get('images', []))} train, "
            f"{len(val_coco.get('images', []))} val."
        )
        # Generate train/val filenames from output path
        if out_path.suffix:
            train_path = out_path.with_name(out_path.stem + "_train" + out_path.suffix)
            val_path = out_path.with_name(out_path.stem + "_val" + out_path.suffix)
        else:
            train_path = out_path.with_name(out_path.name + "_train")
            val_path = out_path.with_name(out_path.name + "_val")

        with train_path.open("w", encoding="utf-8") as f:
            json.dump(train_coco, f, ensure_ascii=False, indent=2)
        with val_path.open("w", encoding="utf-8") as f:
            json.dump(val_coco, f, ensure_ascii=False, indent=2)

        print(f"Saved train GT COCO JSON to: {train_path}")
        print(f"Saved val GT COCO JSON to: {val_path}")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Saved merged GT COCO JSON to: {out_path}")


if __name__ == "__main__":
    main()

