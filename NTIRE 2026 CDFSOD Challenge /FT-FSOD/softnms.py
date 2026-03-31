#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Soft-NMS on merged COCO predictions from two json files."
    )
    parser.add_argument("--pred1", type=str, required=True, help="Path to first prediction json")
    parser.add_argument("--pred2", type=str, required=True, help="Path to second prediction json")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output json")
    parser.add_argument(
        "--method",
        type=str,
        choices=["linear", "gaussian"],
        default="gaussian",
        help="Soft-NMS method",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold (used in linear Soft-NMS)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Sigma (used in gaussian Soft-NMS)",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.001,
        help="Boxes with score below this threshold are removed",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=-1,
        help="Max kept boxes per (image_id, category_id). -1 means no limit",
    )
    return parser.parse_args()


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def soft_nms_single_group(boxes, scores, method, iou_thr, sigma, score_thr, max_per_group):
    # In-place variant with index swaps
    n = len(boxes)
    if n == 0:
        return []

    keep = []

    for i in range(n):
        # Pick the highest score box among [i, n)
        max_pos = i
        max_score = scores[i]
        for j in range(i + 1, n):
            if scores[j] > max_score:
                max_score = scores[j]
                max_pos = j

        if max_pos != i:
            boxes[i], boxes[max_pos] = boxes[max_pos], boxes[i]
            scores[i], scores[max_pos] = scores[max_pos], scores[i]

        cur_box = boxes[i]
        cur_score = scores[i]
        if cur_score < score_thr:
            continue

        keep.append((cur_box[:], cur_score))
        if max_per_group > 0 and len(keep) >= max_per_group:
            break

        # Decay following boxes
        for j in range(i + 1, n):
            ov = iou(cur_box, boxes[j])
            if method == "linear":
                if ov > iou_thr:
                    scores[j] *= (1.0 - ov)
            else:
                # gaussian
                scores[j] *= math.exp(-(ov * ov) / sigma)

    return keep


def load_pred_json(path):
    with Path(path).open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a COCO prediction list json")
    return data


def main():
    args = parse_args()
    pred1 = load_pred_json(args.pred1)
    pred2 = load_pred_json(args.pred2)
    merged = pred1 + pred2

    grouped = defaultdict(list)
    for item in merged:
        key = (int(item["image_id"]), int(item["category_id"]))
        grouped[key].append(
            {
                "bbox_xyxy": xywh_to_xyxy([float(v) for v in item["bbox"]]),
                "score": float(item["score"]),
            }
        )

    output = []
    for (image_id, category_id), dets in grouped.items():
        boxes = [d["bbox_xyxy"][:] for d in dets]
        scores = [d["score"] for d in dets]
        kept = soft_nms_single_group(
            boxes=boxes,
            scores=scores,
            method=args.method,
            iou_thr=args.iou_thr,
            sigma=args.sigma,
            score_thr=args.score_thr,
            max_per_group=args.max_per_group,
        )
        for box_xyxy, score in kept:
            output.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(v) for v in xyxy_to_xywh(box_xyxy)],
                    "score": float(min(1.0, max(0.0, score))),
                }
            )

    output.sort(key=lambda x: (x["image_id"], x["category_id"], -x["score"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f)

    print(f"Saved Soft-NMS predictions to: {out_path}")
    print(f"Input boxes: {len(pred1)} + {len(pred2)} = {len(merged)}")
    print(f"Output boxes: {len(output)}")


if __name__ == "__main__":
    main()
