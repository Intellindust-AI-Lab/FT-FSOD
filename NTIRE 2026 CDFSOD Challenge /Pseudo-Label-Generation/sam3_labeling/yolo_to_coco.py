#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert YOLO prediction labels (txt) to COCO prediction JSON for COCO eval.

- category_id: For COCO eval typically starts from 1; YOLO class_id 0 -> category_id 1
  via --class-offset 1 (default). Categories are read from label-dir/classes.txt when present.
- Optional: --gt-label-json to align image IDs with a few-shot GT COCO JSON.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def load_image_files(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return files


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
        class_id = int(parts[0])
        score = 1.0
        x, y, w, h = map(float, parts[1:5])
    elif len(parts) == 6:
        class_id = int(parts[0])
        score = float(parts[1])
        x, y, w, h = map(float, parts[2:6])
    else:
        raise ValueError(f"unexpected column count: {len(parts)}")

    return class_id, score, x, y, w, h


def load_categories_from_label_dir(label_dir: Path, class_offset: int) -> List[Dict]:
    """
    Load categories from label_dir/classes.txt for COCO JSON.
    Line index i (0-based) -> category id = i + class_offset, name = line.strip().
    Returns list of {"id": int, "name": str}. If file missing or empty, return [].
    """
    classes_file = label_dir / "classes.txt"
    if not classes_file.exists():
        return []
    names = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    if not names:
        return []
    return [
        {"id": i + class_offset, "name": name}
        for i, name in enumerate(names)
    ]


def load_gt_image_id_map(gt_path: Path) -> Tuple[Dict[str, int], int]:
    """
    Load GT COCO JSON and return:
    - file_name -> image_id
    - max image_id in GT (or 0 if empty)
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    name_to_id: Dict[str, int] = {}
    max_id = 0
    for img in gt.get("images", []):
        fid = img["id"]
        fname = img.get("file_name", "")
        if fname:
            name_to_id[fname] = fid
        max_id = max(max_id, fid)
    return name_to_id, max_id


def align_image_ids_with_gt(
    coco: dict,
    gt_name_to_id: Dict[str, int],
    gt_max_id: int,
) -> dict:
    """
    Rewrite image ids in coco so that:
    - If file_name is in GT, use GT's image_id.
    - Otherwise assign max(gt_max_id, ...) + 1, +2, ... (order preserved by current image list).
    """
    images = coco["images"]
    annotations = coco["annotations"]

    old_id_to_new: Dict[int, int] = {}
    next_new_id = gt_max_id + 1

    for img in images:
        old_id = img["id"]
        fname = img.get("file_name", "")
        if fname in gt_name_to_id:
            new_id = gt_name_to_id[fname]
        else:
            new_id = next_new_id
            next_new_id += 1
        old_id_to_new[old_id] = new_id
        img["id"] = new_id

    for ann in annotations:
        ann["image_id"] = old_id_to_new.get(ann["image_id"], ann["image_id"])

    return coco


def yolo_to_coco_pred(
    image_dir: Path,
    label_dir: Path,
    output_json: Path,
    gt_label_json: Optional[Path] = None,
    class_offset: int = 1,
    score_threshold: float = 0.0,
) -> None:
    """
    Convert YOLO txt under label_dir to COCO prediction JSON.
    COCO category_id = YOLO class_id + class_offset (default 1 for COCO eval).
    Categories are read from label_dir/classes.txt when present.
    """
    image_paths = load_image_files(image_dir)
    if not image_paths:
        raise RuntimeError(f"no images found in {image_dir}")

    categories = load_categories_from_label_dir(label_dir, class_offset)
    if categories:
        print(f"Loaded {len(categories)} categories from {label_dir / 'classes.txt'}")

    gt_name_to_id: Dict[str, int] = {}
    gt_max_id = 0
    if gt_label_json is not None and gt_label_json.exists():
        gt_name_to_id, gt_max_id = load_gt_image_id_map(gt_label_json)
        print(f"Loaded GT: {len(gt_name_to_id)} images, max image_id = {gt_max_id}")

    images = []
    annotations = []
    ann_id = 1
    seen_cat_ids: set = set()

    for img_id, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as im:
            width, height = im.size

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cls, score, x, y, w, h = parse_yolo_line(line)
                except ValueError:
                    continue

                if score < score_threshold:
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

                cid = int(cls) + class_offset
                seen_cat_ids.add(cid)
                annotations.append(
                    {
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [x1, y1, bw, bh],
                        "score": float(score),
                    }
                )

    if not categories and seen_cat_ids:
        categories = [
            {"id": cid, "name": f"class_{cid - class_offset}"}
            for cid in sorted(seen_cat_ids)
        ]
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    if gt_label_json is not None and gt_label_json.exists():
        coco = align_image_ids_with_gt(coco, gt_name_to_id, gt_max_id)
        print("Aligned image IDs with GT COCO.")

    # Root node: list of predictions (COCO prediction format for eval)
    output_list = coco["annotations"]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
    print(f"Saved COCO prediction JSON to: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO predictions (txt) to COCO prediction JSON. "
        "Optionally align image IDs with a few-shot GT COCO JSON."
    )
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory.")
    parser.add_argument(
        "--label-dir",
        type=str,
        required=True,
        help="YOLO txt directory (same stem as image files).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--gt-label-json",
        type=str,
        default=None,
        help="Few-shot GT COCO JSON. Prediction image_id will match GT by file_name; "
        "remaining images get IDs from max(GT image_id)+1 onward.",
    )
    parser.add_argument(
        "--class-offset",
        type=int,
        default=1,
        help="Add to YOLO class_id for COCO category_id (default: 1 for COCO eval).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Filter predictions with score < threshold (default: 0.0).",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt_label_json) if args.gt_label_json else None

    yolo_to_coco_pred(
        image_dir=Path(args.image_dir),
        label_dir=Path(args.label_dir),
        output_json=Path(args.output),
        gt_label_json=gt_path,
        class_offset=args.class_offset,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
