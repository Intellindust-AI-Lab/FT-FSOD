#!/usr/bin/env python3
import pickle
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert pickle predictions to COCO format JSON.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="your_exp_results_path",
        help="Base path for experiment results",
    )
    parser.add_argument(
        "--total_name",
        type=str,
        required=True,
        help="Dataset group name (e.g. dataset1, dataset2, dataset3)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Shot config name (e.g. 1shot, 5shot, 10shot)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON path (default: same dir as pkl, with suffix _coco_preds.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path.rstrip("/") + "/"
    total_name = args.total_name
    dataset_name = args.dataset_name

    pkl_path = Path(
        data_path
        + f"{total_name}/{dataset_name}/swinL_{total_name}_{dataset_name}/{total_name}_{dataset_name}.pkl"
    )

    out_json_path = Path(args.output) if args.output else pkl_path.with_name(f"{total_name}_{dataset_name}_coco_preds.json")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    coco_preds = []

    for item in data:
        if "img_id" in item:
            image_id = item["img_id"]
        else:
            raise ValueError("No 'img_id' in item and no fallback implemented.")

        pred = item.get("pred_instances", {})
        scores = pred.get("scores", [])
        labels = pred.get("labels", [])
        bboxes = pred.get("bboxes", [])

        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().tolist()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().tolist()
        if hasattr(bboxes, "detach"):
            bboxes = bboxes.detach().cpu().tolist()

        for score, label, box in zip(scores, labels, bboxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            category_id = int(label) + 1

            coco_preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    with out_json_path.open("w") as f:
        json.dump(coco_preds, f)

    print(f"Saved COCO predictions to: {out_json_path}")
    print(f"Total detections: {len(coco_preds)}")


if __name__ == "__main__":
    main()