#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path


def coco_bbox_to_yolo(bbox: list, img_width: float, img_height: float) -> tuple:
    x, y, w, h = bbox
    if img_width <= 0 or img_height <= 0:
        return None
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def convert_json_to_yolo(json_path: Path, out_dir: Path, class_offset: int = 0) -> bool:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    images = data.get("images", [])
    anns = data.get("annotations", [])
    if not images:
        return False

    img_info = images[0]
    width = float(img_info.get("width", 0) or 0)
    height = float(img_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return False

    lines = []
    for ann in anns:
        bbox = ann.get("bbox")
        cat_id = ann.get("category_id")
        score = float(ann.get("score", 1.0))
        if bbox is None or cat_id is None or len(bbox) != 4:
            continue

        yolo = coco_bbox_to_yolo(bbox, width, height)
        if yolo is None:
            continue

        x_c, y_c, w_n, h_n = yolo
        cls = int(cat_id) - 1 + class_offset  # COCO 1-indexed -> YOLO 0-indexed
        cls = max(0, cls)
        lines.append(f"{cls} {score:.6f} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    out_path = out_dir / (json_path.stem + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO per-image JSON directory to YOLO txt format (cls score x y w h)"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing COCO *.json files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory, default: {input_dir}_yolo",
    )
    parser.add_argument(
        "--class-offset",
        type=int,
        default=0,
        help="YOLO cls = COCO category_id - 1 + class_offset (default 0)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    out_dir = Path(args.output) if args.output else Path(str(in_dir) + "_yolo")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    converted = 0
    for p in json_files:
        if convert_json_to_yolo(p, out_dir, class_offset=args.class_offset):
            converted += 1

    print(f"Converted {converted}/{len(json_files)} JSON files to {out_dir}")


if __name__ == "__main__":
    main()
