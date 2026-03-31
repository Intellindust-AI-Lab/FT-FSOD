#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 annotation filtering script.
Read YOLO-format annotations, filter by confidence threshold,
and output final annotation files.

Features:
- Support single-file and batch (directory) modes
- Filter low-confidence annotations
- Keep YOLO-format output
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

from collections import defaultdict


def _box_iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Compute IoU of two boxes in (x_min, y_min, x_max, y_max)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    sa = (ax2 - ax1) * (ay2 - ay1)
    sb = (bx2 - bx1) * (by2 - by1)
    union = sa + sb - inter
    return inter / union if union > 0 else 0.0


def nms_annotations(
    annotations: List[Tuple[int, float, float, float, float, float]],
    iou_threshold: float = 0.65,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Run per-class NMS (descending by confidence inside each class).
    Boxes with IoU > iou_threshold are suppressed.
    Each item is (class_id, x_min, y_min, x_max, y_max, confidence).
    """
    if iou_threshold <= 0 or not annotations:
        return annotations
    by_class: dict[int, List[Tuple[float, float, float, float, float]]] = defaultdict(list)
    for (cid, x1, y1, x2, y2, conf) in annotations:
        by_class[cid].append((x1, y1, x2, y2, conf))
    out = []
    for cid in sorted(by_class.keys()):
        boxes = by_class[cid]
        boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)  # by confidence
        kept = []
        for (x1, y1, x2, y2, conf) in boxes_sorted:
            box = (x1, y1, x2, y2)
            suppress = False
            for k in kept:
                if _box_iou_xyxy(box, k[:4]) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append((x1, y1, x2, y2, conf))
        for (x1, y1, x2, y2, conf) in kept:
            out.append((cid, x1, y1, x2, y2, conf))
    return out


def read_yolo_annotations(
    txt_path: str,
    confidence_threshold: float = 0.0,
    verbose: bool = False,
    input_format: str = "auto",
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Read YOLO annotations and normalize internally to
    (class_id, x_min, y_min, x_max, y_max, confidence).

    Supported formats:
    - 5 columns: class_id x_min y_min x_max y_max (confidence=1.0),
      or class_id cx cy w h when input_format=yolo_center
    - 6 columns (new): class_id class_score x_min y_min x_max y_max
    - 6 columns (old): class_id x_min y_min x_max y_max confidence
    - 6 columns yolo_center: class_id class_score cx cy w h

    Args:
        input_format: 'auto' or 'yolo_center'. In 'yolo_center',
            6-column rows are parsed as class_id score cx cy w h.
    Returns:
        Annotation list, each item is
        (class_id, x_min, y_min, x_max, y_max, confidence)
    """
    annotations = []
    txt_path = Path(txt_path)
    total_count = 0
    filtered_count = 0

    if not txt_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {txt_path}")

    with open(txt_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) not in [5, 6]:
                if verbose:
                    print(f"Warning: invalid format at line {line_num}, skipped: {line}")
                continue

            try:
                if input_format == "yolo_center":
                    # Center format: class_id [score] cx cy w h
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        confidence = 1.0
                    else:
                        class_id = int(parts[0])
                        confidence = float(parts[1])
                        cx, cy, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2
                elif len(parts) == 5:
                    class_id = int(parts[0])
                    x_min, y_min, x_max, y_max = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    confidence = 1.0
                else:
                    p1, p5 = float(parts[1]), float(parts[5])
                    if 0 <= p5 <= 1 and p1 > 1:
                        class_id = int(parts[0])
                        x_min, y_min, x_max, y_max = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        confidence = p5
                    else:
                        class_id = int(parts[0])
                        confidence = p1
                        x_min, y_min, x_max, y_max = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                total_count += 1
                if confidence >= confidence_threshold:
                    annotations.append((class_id, x_min, y_min, x_max, y_max, confidence))
                else:
                    filtered_count += 1
            except (ValueError, IndexError) as e:
                if verbose:
                    print(f"Warning: parse error at line {line_num}, skipped: {line} - {e}")
                continue

    if verbose and confidence_threshold > 0.0 and total_count > 0:
        print(
            f"  Confidence filter: kept {len(annotations)} / {total_count} "
            f"(threshold >= {confidence_threshold:.3f}), filtered {filtered_count}"
        )

    return annotations


def detect_annotation_format(txt_path: str) -> str:
    """
    Detect annotation file format.
    
    Returns:
        '5col' or '6col_old' or '6col_new'
    """
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                return '5col'
            elif len(parts) == 6:
                p1, p5 = float(parts[1]), float(parts[5])
                if 0 <= p5 <= 1 and p1 > 1:
                    return '6col_old'  # class_id x_min y_min x_max y_max confidence
                else:
                    return '6col_new'  # class_id class_score x_min y_min x_max y_max
    return '5col'  # default


def cleanup_output_path(output_path: Path, verbose: bool = True):
    """
    Clean up output path (delete file or directory).
    
    Args:
        output_path: output path
        verbose: whether to print detailed logs
    """
    if output_path.exists():
        if output_path.is_file():
            output_path.unlink()
            if verbose:
                print(f"Removed old file: {output_path}")
        elif output_path.is_dir():
            shutil.rmtree(output_path)
            if verbose:
                print(f"Removed old directory: {output_path}")


def write_yolo_annotations(
    annotations: List[Tuple[int, float, float, float, float, float]],
    output_path: str,
    original_format: str = "6col_old",
    no_confidence: bool = False,
    input_format: str = "auto",
):
    """
    Write annotations to a YOLO-format file.
    When no_confidence=True, always output 5 columns:
    class_id cx cy w h (without confidence).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for annotation in annotations:
            class_id, x_min, y_min, x_max, y_max, confidence = annotation

            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            w = x_max - x_min
            h = y_max - y_min

            if no_confidence:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            elif input_format == "yolo_center":
                # Keep consistent with annotate_with_translate.py: 6-column center format
                f.write(f"{class_id} {confidence:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            elif original_format == "5col":
                f.write(f"{class_id} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")
            elif original_format == "6col_old":
                f.write(f"{class_id} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {confidence:.6f}\n")
            else:
                f.write(f"{class_id} {confidence:.6f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}\n")


def process_single(
    annotation_path: str,
    output_path: str,
    confidence_threshold: float = 0.0,
    verbose: bool = True,
    input_format: str = "auto",
    no_confidence: bool = False,
    nms_iou: float = 0.0,
):
    """Process one annotation file. With no_confidence=True output 5 columns."""
    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    output_path_obj = Path(output_path)
    cleanup_output_path(output_path_obj, verbose)

    if verbose:
        print(f"Processing: {annotation_path.name}")

    original_format = detect_annotation_format(str(annotation_path)) if input_format == "auto" else "6col_new"
    annotations = read_yolo_annotations(
        str(annotation_path),
        confidence_threshold,
        verbose=verbose,
        input_format=input_format,
    )
    if nms_iou > 0:
        annotations = nms_annotations(annotations, iou_threshold=nms_iou)
    write_yolo_annotations(
        annotations,
        output_path,
        original_format,
        no_confidence=no_confidence,
        input_format=input_format,
    )

    if verbose:
        print(f"  Output: {output_path}")
        print(f"  Kept {len(annotations)} annotations")


def process_batch(
    annotation_dir: str,
    output_dir: str,
    confidence_threshold: float = 0.0,
    verbose: bool = True,
    input_format: str = "auto",
    no_confidence: bool = False,
    nms_iou: float = 0.0,
    to_coco: bool = False,
    image_dir: str | None = None,
    class_names: list[str] | None = None,
):
    """Process annotation files in batch mode.

    With no_confidence=True output 5 columns: class_id cx cy w h.
    When to_coco=True, additionally generate coco_annotations.json in output_dir.
    """
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    cleanup_output_path(output_dir, verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = list(annotation_dir.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No annotation files found in {annotation_dir}")

    if verbose:
        print(f"Found {len(txt_files)} annotation files")
        if confidence_threshold > 0.0:
            print(f"Confidence threshold: {confidence_threshold:.3f}")
        if no_confidence:
            print("Output format: 5 columns class_id cx cy w h (without confidence)")
        print()

    success_count = 0
    total_annotations_before = 0
    total_annotations_after = 0

    # COCO caches
    coco_images = []
    coco_annotations = []
    coco_categories = {}
    ann_id = 1
    img_id_map = {}

    for txt_path in txt_files:
        try:
            output_path = output_dir / txt_path.name
            original_format = detect_annotation_format(str(txt_path)) if input_format == "auto" else "6col_new"
            annotations = read_yolo_annotations(
                str(txt_path),
                confidence_threshold,
                verbose=False,
                input_format=input_format,
            )
            if nms_iou > 0:
                annotations = nms_annotations(annotations, iou_threshold=nms_iou)
            all_annotations = read_yolo_annotations(
                str(txt_path),
                confidence_threshold=0.0,
                verbose=False,
                input_format=input_format,
            )
            total_annotations_before += len(all_annotations)
            total_annotations_after += len(annotations)
            write_yolo_annotations(
                annotations,
                str(output_path),
                original_format,
                no_confidence=no_confidence,
                input_format=input_format,
            )

            # Collect COCO annotations (from filtered annotations)
            if to_coco:
                from PIL import Image as _PILImage  # local import to avoid unnecessary dependency

                stem = txt_path.stem
                if stem not in img_id_map:
                    if image_dir is None:
                        raise ValueError("When to_coco=True, --image-dir is required to locate images.")
                    img_path = None
                    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                        cand = Path(image_dir) / f"{stem}{ext}"
                        if cand.exists():
                            img_path = cand
                            break
                    if img_path is None:
                        raise FileNotFoundError(
                            f"No matching image found for {stem} in image_dir={image_dir}"
                        )
                    with _PILImage.open(img_path) as im:
                        w, h = im.size
                    img_id = len(img_id_map) + 1
                    img_id_map[stem] = img_id
                    coco_images.append(
                        {
                            "id": img_id,
                            "file_name": img_path.name,
                            "width": w,
                            "height": h,
                        }
                    )
                else:
                    img_id = img_id_map[stem]
                    # Already added to coco_images

                # Build annotations
                for (cid, x_min, y_min, x_max, y_max, conf) in annotations:
                    coco_cid = int(cid) + 1
                    # Input uses relative coordinates (0-1), convert to absolute COCO xywh
                    img_info = next(img for img in coco_images if img["id"] == img_id)
                    iw, ih = img_info["width"], img_info["height"]
                    x = max(0.0, min(1.0, x_min)) * iw
                    y = max(0.0, min(1.0, y_min)) * ih
                    w = max(0.0, min(1.0, x_max - x_min)) * iw
                    h = max(0.0, min(1.0, y_max - y_min)) * ih
                    area = max(0.0, w * h)

                    coco_annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": coco_cid,
                            "bbox": [x, y, w, h],
                            "area": area,
                            "iscrowd": 0,
                            "score": float(conf),
                        }
                    )
                    ann_id += 1

                    if coco_cid not in coco_categories:
                        if class_names is not None and 0 <= int(cid) < len(class_names):
                            cname = class_names[int(cid)]
                        else:
                            cname = str(cid)
                        coco_categories[coco_cid] = {"id": coco_cid, "name": cname}

            success_count += 1
            if verbose:
                print(f"  {txt_path.name}: {len(all_annotations)} -> {len(annotations)} annotations")
        except Exception as e:
            if verbose:
                print(f"Error: failed to process {txt_path.name}: {e}")
            continue

    if to_coco and coco_images:
        import json

        coco_dict = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": sorted(coco_categories.values(), key=lambda c: c["id"]),
        }
        coco_path = output_dir / "coco_annotations.json"
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False)
        if verbose:
            print(f"\nGenerated COCO annotation file: {coco_path}")

    if verbose:
        print()
        print(f"Done! Successfully processed {success_count}/{len(txt_files)} files")
        print(f"Total annotations: {total_annotations_before} -> {total_annotations_after}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 annotation filter - filter YOLO annotations by confidence threshold"
    )
    parser.add_argument(
        '--annotation', '-a',
        type=str,
        required=True,
        help='Input annotation file path or annotation directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output annotation file path or output directory'
    )
    parser.add_argument(
        '--confidence-threshold', '-ct',
        type=float,
        default=0,
        help='Confidence threshold; annotations below this value are filtered (default: 0)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode (when input is a directory)'
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode, suppress verbose logs",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Output without confidence, only 5 columns: class_id cx cy w h",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["auto", "yolo_center"],
        default="auto",
        help="Input format: auto or yolo_center (6 columns: class_id score cx cy w h)",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.65,
        help="NMS IoU threshold; for same-class boxes with IoU above this value, keep higher-confidence box. Set 0 to disable NMS (default: 0.65)",
    )
    parser.add_argument(
        "--to-coco",
        action="store_true",
        help="In directory mode, additionally generate COCO-format coco_annotations.json (requires --image-dir)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="When --to-coco is enabled, provide image directory for width/height lookup (matched by annotation filename stem).",
    )
    parser.add_argument(
        "--classes-file",
        type=str,
        default=None,
        help="Optional class-name file, one class name per line (mapped to YOLO class_id), used for COCO category names.",
    )

    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    output_path = Path(args.output)
    verbose = not args.quiet

    if annotation_path.is_file():
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / annotation_path.name
        process_single(
            str(annotation_path),
            str(output_path),
            args.confidence_threshold,
            verbose,
            input_format=args.input_format,
            no_confidence=args.no_confidence,
            nms_iou=args.nms_iou,
        )
        if args.to_coco:
            raise ValueError(
                "Current implementation only supports COCO generation in directory batch mode (--batch). "
                "Please set --annotation to an annotation directory."
            )
    elif annotation_path.is_dir():
        if not args.batch:
            print("Warning: input is a directory but --batch is not set. Batch mode will be enabled automatically.")
        if output_path.exists() and output_path.is_file():
            raise ValueError("In batch mode, output path must be a directory, not a file")

        class_names = None
        if args.classes_file is not None:
            classes_path = Path(args.classes_file)
            if not classes_path.exists():
                raise FileNotFoundError(f"Classes file not found: {classes_path}")
            with open(classes_path, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f if line.strip()]

        process_batch(
            str(annotation_path),
            str(output_path),
            args.confidence_threshold,
            verbose,
            input_format=args.input_format,
            no_confidence=args.no_confidence,
            nms_iou=args.nms_iou,
            to_coco=args.to_coco,
            image_dir=args.image_dir,
            class_names=class_names,
        )
    else:
        raise ValueError(f"Annotation path does not exist: {annotation_path}")


if __name__ == "__main__":
    main()
