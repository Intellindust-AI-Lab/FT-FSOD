
import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration


ENABLE_THINKING = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORIES_COCO = [
    {"id": 1, "name": "holothurian"},
    {"id": 2, "name": "echinus"},
    {"id": 3, "name": "scallop"},
    {"id": 4, "name": "starfish"},
    {"id": 5, "name": "fish"},
    {"id": 6, "name": "corals"},
    {"id": 7, "name": "diver"},
    {"id": 8, "name": "cuttlefish"},
    {"id": 9, "name": "turtle"},
    {"id": 10, "name": "jellyfish"},
]
NAME_TO_CAT_ID = {c["name"]: c["id"] for c in CATEGORIES_COCO}
CATEGORY_NAMES = list(NAME_TO_CAT_ID.keys())

CATEGORIES_PROMPT = """Detect the following categories in the image:
[
    {"id": 1, "name": "holothurian"},
    {"id": 2, "name": "echinus"},
    {"id": 3, "name": "scallop"},
    {"id": 4, "name": "starfish"},
    {"id": 5, "name": "fish"},
    {"id": 6, "name": "corals"},
    {"id": 7, "name": "diver"},
    {"id": 8, "name": "cuttlefish"},
    {"id": 9, "name": "turtle"},
    {"id": 10, "name": "jellyfish"}
]
Please output the bounding box (bbox) of each target in the image, format: category name [x1, y1, x2, y2], where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner, using image pixel coordinates. Output one detection result per line. If a category does not appear, it can be omitted."""

DEFAULT_CONF = 0.85
NMS_IOU_THRESH = 0.65


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].lstrip("\n").strip()
    return text.strip()


def bbox_to_pixel(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    if all(0 <= v <= 1000 for v in [x1, y1, x2, y2]):
        return [x1 / 1000 * img_w, y1 / 1000 * img_h, x2 / 1000 * img_w, y2 / 1000 * img_h]
    if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
        return [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]
    return [x1, y1, x2, y2]


def flip_bbox_horizontal(bbox, img_w):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return [img_w - x2, y1, img_w - x1, y2]


def _parse_detections_from_output(output: str, img_w: int, img_h: int):
    bbox_pattern = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")
    detections = []
    lines = output.replace("\r", "\n").split("\n")
    current_category = None
    for line in lines:
        line_lower = line.strip().lower()
        for c in CATEGORY_NAMES:
            if c in line_lower:
                current_category = c
                break
        for m in bbox_pattern.finditer(line):
            x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            cat = current_category or "unknown"
            pixel_bbox = bbox_to_pixel([x1, y1, x2, y2], img_w, img_h)
            x1, y1, x2, y2 = min(pixel_bbox[0], pixel_bbox[2]), min(pixel_bbox[1], pixel_bbox[3]), max(pixel_bbox[0], pixel_bbox[2]), max(pixel_bbox[1], pixel_bbox[3])
            detections.append({"category": cat, "bbox": [x1, y1, x2, y2]})
            current_category = None
    if not detections:
        for m in bbox_pattern.finditer(output):
            x1, y1, x2, y2 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            pixel_bbox = bbox_to_pixel([x1, y1, x2, y2], img_w, img_h)
            x1, y1, x2, y2 = min(pixel_bbox[0], pixel_bbox[2]), min(pixel_bbox[1], pixel_bbox[3]), max(pixel_bbox[0], pixel_bbox[2]), max(pixel_bbox[1], pixel_bbox[3])
            detections.append({"category": "unknown", "bbox": [x1, y1, x2, y2]})
    return detections


def run_inference_one_image(model, processor, image_path: str, max_new_tokens: int = 8192):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": CATEGORIES_PROMPT},
            ],
        }
    ]
    apply_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not ENABLE_THINKING:
        apply_kwargs["enable_thinking"] = False
    inputs = processor.apply_chat_template(messages, **apply_kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return strip_thinking(output)


def nms_boxes(boxes, scores, iou_threshold: float):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-12)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


def fuse_detections(orig_detections, flip_back_detections, conf: float, iou_thresh: float):
    by_cat = {}
    for d in orig_detections:
        cat = d["category"]
        if cat not in by_cat:
            by_cat[cat] = {"boxes": [], "scores": []}
        by_cat[cat]["boxes"].append(d["bbox"])
        by_cat[cat]["scores"].append(conf)
    for d in flip_back_detections:
        cat = d["category"]
        if cat not in by_cat:
            by_cat[cat] = {"boxes": [], "scores": []}
        by_cat[cat]["boxes"].append(d["bbox"])
        by_cat[cat]["scores"].append(conf)

    fused = []
    for cat, data in by_cat.items():
        boxes = np.array(data["boxes"])
        scores = np.array(data["scores"])
        keep = nms_boxes(boxes, scores, iou_thresh)
        for idx in keep:
            fused.append({"category": cat, "bbox": boxes[idx].tolist(), "score": float(scores[idx])})
    return fused


def detections_to_coco(image_id: int, file_name: str, width: int, height: int, detections: list):
    ann_id = 1
    annotations = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
        cat_name = d.get("category", "unknown")
        category_id = NAME_TO_CAT_ID.get(cat_name, 0)
        if category_id == 0:
            continue
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
            "score": d.get("score", DEFAULT_CONF),
        })
        ann_id += 1
    return {
        "images": [{"id": image_id, "file_name": file_name, "width": width, "height": height}],
        "annotations": annotations,
        "categories": CATEGORIES_COCO,
    }


def process_one_image(model, processor, image_path: str, output_dir: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    file_name = os.path.basename(image_path)
    stem = Path(image_path).stem

    out_orig = run_inference_one_image(model, processor, image_path)
    det_orig = _parse_detections_from_output(out_orig, w, h)

    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    suf = Path(image_path).suffix
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as f:
        flip_path = f.name
    try:
        img_flip.save(flip_path)
        out_flip = run_inference_one_image(model, processor, flip_path)
        det_flip_raw = _parse_detections_from_output(out_flip, w, h)
        det_flip_back = [{"category": d["category"], "bbox": flip_bbox_horizontal(d["bbox"], w)} for d in det_flip_raw]
    finally:
        if os.path.isfile(flip_path):
            os.remove(flip_path)

    fused = fuse_detections(det_orig, det_flip_back, conf=DEFAULT_CONF, iou_thresh=NMS_IOU_THRESH)

    coco = detections_to_coco(image_id=1, file_name=file_name, width=w, height=h, detections=fused)
    out_path = os.path.join(output_dir, stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Directory inference: all images -> COCO JSON per image (orig + flip fusion).")
    parser.add_argument("image_path", type=str, help="Input image path (file or directory). If directory, traverse all images under it.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory path, COCO JSON will be written to this directory; if not specified, write to input path directory.")
    parser.add_argument("-m", "--model-path", type=str, default="./Qwen3.5-35B-A3B", help="Model path.")
    parser.add_argument("--skip-existing-in", type=str, default="dataset1_test", help="Exclude images with the same name .json in this directory (match by stem); default dataset1_test.")
    args = parser.parse_args()

    path = os.path.abspath(args.image_path)
    if os.path.isfile(path):
        image_files = [path]
        default_output_dir = os.path.dirname(path) or "."
    elif os.path.isdir(path):
        default_output_dir = path
        image_files = []
        for f in os.listdir(path):
            ext = Path(f).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(path, f))
        image_files.sort()
    else:
        raise SystemExit(f"Not a file or directory: {path}")

    skip_dir = os.path.abspath(args.skip_existing_in) if args.skip_existing_in else None
    if skip_dir and os.path.isdir(skip_dir):
        original_count = len(image_files)
        image_files = [
            p for p in image_files
            if not os.path.isfile(os.path.join(skip_dir, Path(p).stem + ".json"))
        ]
        skipped = original_count - len(image_files)
        print(f"Skip existing: {skipped} images already have JSON in {skip_dir}, {len(image_files)} remaining.")
    elif skip_dir:
        print(f"Warning: --skip-existing-in path is not a directory: {skip_dir}, ignoring.")

    output_dir = os.path.abspath(args.output) if args.output else default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not image_files:
        print(f"No images to process (all skipped or none found under {path}).")
        return

    print("Loading model and processor...")
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    total = len(image_files)
    start_time = time.perf_counter()
    for i, img_path in enumerate(image_files):
        print(f"[{i + 1}/{total}] {os.path.basename(img_path)}", end="")
        try:
            out_json = process_one_image(model, processor, img_path, output_dir)
            avg_per_image = (time.perf_counter() - start_time) / (i + 1)
            remaining = total - (i + 1)
            eta_sec = avg_per_image * remaining
            eta_min = eta_sec / 60
            print(f"  -> {out_json}")
            print(f"  ETA: {eta_min:.1f} min remaining ({remaining} images)")
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    main()
