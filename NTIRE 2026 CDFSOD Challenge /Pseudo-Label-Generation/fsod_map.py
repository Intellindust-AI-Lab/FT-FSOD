import argparse
import json
import os
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_filename_to_id(gt_coco):
    return {img["file_name"]: img["id"] for img in gt_coco.get("images", [])}


def _collect_predictions_from_dir(pred_dir, gt_filename_to_id):
    """
    旧格式：目录下每张图一个 COCO-style json（包含 images/annotations/categories）。
    通过 file_name 对齐 GT，将 image_id 映射到 GT image_id。
    """
    pred_dir = Path(pred_dir)
    dets = []
    total_pred_files = 0
    used_pred_files = 0
    total_pred_anns = 0
    used_pred_anns = 0

    for p in sorted(pred_dir.glob("*.json")):
        total_pred_files += 1
        try:
            data = load_json(str(p))
        except Exception:
            continue

        images = data.get("images", [])
        anns = data.get("annotations", [])
        total_pred_anns += len(anns)
        if not images:
            continue

        file_name = images[0].get("file_name")
        if file_name not in gt_filename_to_id:
            continue

        gt_image_id = gt_filename_to_id[file_name]
        used_pred_files += 1

        for ann in anns:
            bbox = ann.get("bbox", None)
            category_id = ann.get("category_id", None)
            if bbox is None or category_id is None:
                continue
            if len(bbox) != 4:
                continue
            score = float(ann.get("score", 0.0))
            dets.append(
                {
                    "image_id": gt_image_id,
                    "category_id": int(category_id),
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                    "score": score,
                }
            )
            used_pred_anns += 1

    stats = {
        "total_pred_files": total_pred_files,
        "used_pred_files": used_pred_files,
        "total_pred_anns": total_pred_anns,
        "used_pred_anns": used_pred_anns,
    }
    return dets, stats


def _collect_predictions_from_file(pred_file, gt_img_ids_set):
    """
    新格式：一个 COCO prediction json 文件（list of detections，或 dict['annotations']）。
    直接使用其中的 image_id/category_id/bbox/score，仅保留 image_id 在 GT 内的预测。
    """
    data = load_json(pred_file)
    if isinstance(data, list):
        raw_dets = data
    else:
        raw_dets = data.get("annotations", [])

    dets = []
    total_pred_anns = 0
    used_pred_anns = 0

    for ann in raw_dets:
        total_pred_anns += 1
        img_id = ann.get("image_id", None)
        cat_id = ann.get("category_id", None)
        bbox = ann.get("bbox", None)
        if img_id is None or cat_id is None or bbox is None or len(bbox) != 4:
            continue
        if img_id not in gt_img_ids_set:
            continue
        score = float(ann.get("score", 0.0))
        dets.append(
            {
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ],
                "score": score,
            }
        )
        used_pred_anns += 1

    stats = {
        "total_pred_files": 1,
        "used_pred_files": 1,
        "total_pred_anns": total_pred_anns,
        "used_pred_anns": used_pred_anns,
    }
    return dets, stats


def collect_predictions_any(pred_path, gt_dict):
    """
    统一入口：
    - 如果 pred_path 是目录：按旧逻辑读取 per-image json。
    - 如果 pred_path 是文件：按 COCO prediction list 读取。
    """
    if os.path.isdir(pred_path):
        gt_filename_to_id = build_gt_filename_to_id(gt_dict)
        return _collect_predictions_from_dir(pred_path, gt_filename_to_id)
    elif os.path.isfile(pred_path):
        gt_img_ids_set = {img["id"] for img in gt_dict.get("images", [])}
        return _collect_predictions_from_file(pred_path, gt_img_ids_set)
    else:
        raise SystemExit(f"pred_path is neither a directory nor a file: {pred_path}")


def evaluate_coco_map(pred_path, gt_json):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise SystemExit(
            "pycocotools is required. Install with: pip install pycocotools"
        ) from e

    gt_json = os.path.abspath(gt_json)
    pred_path = os.path.abspath(pred_path)

    coco_gt = COCO(gt_json)
    gt_dict = load_json(gt_json)
    gt_img_ids = [img["id"] for img in gt_dict.get("images", [])]

    dets, info = collect_predictions_any(pred_path, gt_dict)

    if len(dets) == 0:
        print("No valid predictions matched GT image set. mAP will be 0.")
        coco_dt = coco_gt.loadRes([])
    else:
        coco_dt = coco_gt.loadRes(dets)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    # Crucial: evaluate only few-shot GT images; ignore extra predicted images.
    coco_eval.params.imgIds = gt_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCO stats:
    # 0: mAP@[.5:.95], 1: AP50, 2: AP75, 3: AP small, 4: AP medium, 5: AP large
    # 6: AR1, 7: AR10, 8: AR100, 9: AR small, 10: AR medium, 11: AR large
    s = coco_eval.stats
    print("\n=== Key Metrics ===")
    print(f"mAP (AP@[0.50:0.95]) : {s[0]:.4f}")
    print(f"AP50                : {s[1]:.4f}")
    print(f"AP75                : {s[2]:.4f}")
    print("\n=== Coverage Info ===")
    print(f"GT images                        : {len(gt_img_ids)}")
    print(f"Prediction json files found      : {info['total_pred_files']}")
    print(f"Prediction files used (GT only)  : {info['used_pred_files']}")
    print(f"Prediction anns found            : {info['total_pred_anns']}")
    print(f"Prediction anns used (GT only)   : {info['used_pred_anns']}")


def _bbox_xywh_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou_xywh(box1, box2):
    x1_min, y1_min, x1_max, y1_max = _bbox_xywh_to_xyxy(box1)
    x2_min, y2_min, x2_max, y2_max = _bbox_xywh_to_xyxy(box2)
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return inter / (area1 + area2 - inter)


def _compute_ap(recalls, precisions):
    """
    Standard interpolation: AP = integral over recall of p_interp(r).
    recalls, precisions: lists sorted by recall ascending.
    """
    if not recalls:
        return 0.0
    # Append sentinels
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    # Make precision non-increasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_fsod_ap(
    pred_path,
    gt_json,
    iou_match=0.5,
    iou_near=0.3,
):
    """
    GT-centered fsod AP:
    - 仅在 GT 邻域 (max IoU >= iou_near) 内评估；
    - 对保留下来的预测按分数排序，做标准 PR 曲线并积分得到 AP。
    """
    gt_json = os.path.abspath(gt_json)
    pred_path = os.path.abspath(pred_path)

    gt_dict = load_json(gt_json)
    dets, info = collect_predictions_any(pred_path, gt_dict)

    # Build GT structures: per (img_id, cat_id) list of boxes and matched flags
    gt_by_img_cat = {}
    for ann in gt_dict.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        bbox = ann["bbox"]
        key = (img_id, cat_id)
        gt_by_img_cat.setdefault(key, []).append(
            {"bbox": bbox, "matched": False}
        )

    npos = sum(len(v) for v in gt_by_img_cat.values())
    if npos == 0:
        print("No GT boxes in annotation; fsod AP is undefined (set to 0).")
        print("GT images                        :", len(gt_dict.get('images', [])))
        print("Prediction json files found      :", info["total_pred_files"])
        print("Prediction files used (GT only)  :", info["used_pred_files"])
        print("Prediction anns found            :", info["total_pred_anns"])
        print("Prediction anns used (GT only)   :", info["used_pred_anns"])
        return

    # Filter detections: keep only those that are near any GT (IoU >= iou_near)
    kept_dets = []
    ignored_dets = 0
    for d in dets:
        img_id = d["image_id"]
        cat_id = d["category_id"]
        bbox = d["bbox"]
        key = (img_id, cat_id)
        gts = gt_by_img_cat.get(key, [])
        best_iou = 0.0
        for g in gts:
            iou = _iou_xywh(bbox, g["bbox"])
            if iou > best_iou:
                best_iou = iou
        if best_iou >= iou_near:
            kept_dets.append(d)
        else:
            ignored_dets += 1

    if not kept_dets:
        print("No predictions in GT neighborhoods; fsod AP is 0.")
        print("GT boxes                        :", npos)
        print("Total predictions (GT-filtered) :", len(dets))
        print("Predictions kept (near GT)      :", 0)
        print("Predictions ignored (far from GT):", ignored_dets)
        return

    # Sort kept detections by score desc
    kept_dets.sort(key=lambda x: x["score"], reverse=True)

    tp = []
    fp = []
    # Reset matched flags
    for gts in gt_by_img_cat.values():
        for g in gts:
            g["matched"] = False

    for d in kept_dets:
        img_id = d["image_id"]
        cat_id = d["category_id"]
        bbox = d["bbox"]
        key = (img_id, cat_id)
        gts = gt_by_img_cat.get(key, [])
        best_iou = 0.0
        best_gt_idx = -1
        for idx, g in enumerate(gts):
            if g["matched"]:
                continue
            iou = _iou_xywh(bbox, g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_match and best_gt_idx >= 0:
            gts[best_gt_idx]["matched"] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # Compute precision-recall
    tp_cum = []
    fp_cum = []
    t_sum = 0
    f_sum = 0
    for t, f in zip(tp, fp):
        t_sum += t
        f_sum += f
        tp_cum.append(t_sum)
        fp_cum.append(f_sum)

    recalls = []
    precisions = []
    for t_c, f_c in zip(tp_cum, fp_cum):
        if t_c + f_c == 0:
            prec = 0.0
        else:
            prec = t_c / (t_c + f_c)
        rec = t_c / npos
        recalls.append(rec)
        precisions.append(prec)

    ap = _compute_ap(recalls, precisions)

    print("=== fsod (GT-centered) AP ===")
    print(f"IoU_match threshold          : {iou_match}")
    print(f"IoU_near  threshold (keep)   : {iou_near}")
    print(f"GT boxes (npos)              : {npos}")
    print(f"Total predictions (GT-filter): {len(dets)}")
    print(f"Predictions kept (near GT)   : {len(kept_dets)}")
    print(f"Predictions ignored (far GT) : {ignored_dets}")
    print(f"fsod AP (GT neighborhoods)  : {ap:.4f}")


def evaluate_fsod_map(
    pred_path,
    gt_json,
    iou_match_start=0.5,
    iou_near=0.3,
):
    """
    GT-centered fsod mAP:
    - 与 evaluate_fsod_ap 相同的 GT 邻域过滤方式；
    - 在一系列 IoU 阈值上（例如 0.50:0.05:0.95）计算 AP 再取平均。
    """
    gt_json = os.path.abspath(gt_json)
    pred_path = os.path.abspath(pred_path)

    gt_dict = load_json(gt_json)
    dets, info = collect_predictions_any(pred_path, gt_dict)

    gt_by_img_cat = {}
    for ann in gt_dict.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        bbox = ann["bbox"]
        key = (img_id, cat_id)
        gt_by_img_cat.setdefault(key, []).append(
            {"bbox": bbox, "matched": False}
        )

    npos = sum(len(v) for v in gt_by_img_cat.values())
    if npos == 0:
        print("No GT boxes in annotation; fsod mAP is undefined (set to 0).")
        print("GT images                        :", len(gt_dict.get("images", [])))
        print("Prediction json files found      :", info["total_pred_files"])
        print("Prediction files used (GT only)  :", info["used_pred_files"])
        print("Prediction anns found            :", info["total_pred_anns"])
        print("Prediction anns used (GT only)   :", info["used_pred_anns"])
        return

    kept_dets = []
    ignored_dets = 0
    for d in dets:
        img_id = d["image_id"]
        cat_id = d["category_id"]
        bbox = d["bbox"]
        key = (img_id, cat_id)
        gts = gt_by_img_cat.get(key, [])
        best_iou = 0.0
        for g in gts:
            iou = _iou_xywh(bbox, g["bbox"])
            if iou > best_iou:
                best_iou = iou
        if best_iou >= iou_near:
            kept_dets.append(d)
        else:
            ignored_dets += 1

    if not kept_dets:
        print("No predictions in GT neighborhoods; fsod mAP is 0.")
        print("GT boxes                        :", npos)
        print("Total predictions (GT-filtered) :", len(dets))
        print("Predictions kept (near GT)      :", 0)
        print("Predictions ignored (far from GT):", ignored_dets)
        return

    kept_dets.sort(key=lambda x: x["score"], reverse=True)

    # IoU 阈值列表：从 iou_match_start 到 0.95，步长 0.05，并裁剪到 [0.5, 0.95]
    start = min(max(iou_match_start, 0.5), 0.95)
    iou_thresholds = []
    t = start
    while t <= 0.95 + 1e-9:
        iou_thresholds.append(round(t, 2))
        t += 0.05

    aps = []
    for thr in iou_thresholds:
        # 重置 matched
        for gts in gt_by_img_cat.values():
            for g in gts:
                g["matched"] = False

        tp = []
        fp = []
        for d in kept_dets:
            img_id = d["image_id"]
            cat_id = d["category_id"]
            bbox = d["bbox"]
            key = (img_id, cat_id)
            gts = gt_by_img_cat.get(key, [])
            best_iou = 0.0
            best_gt_idx = -1
            for idx, g in enumerate(gts):
                if g["matched"]:
                    continue
                iou = _iou_xywh(bbox, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= thr and best_gt_idx >= 0:
                gts[best_gt_idx]["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = []
        fp_cum = []
        t_sum = 0
        f_sum = 0
        for t_v, f_v in zip(tp, fp):
            t_sum += t_v
            f_sum += f_v
            tp_cum.append(t_sum)
            fp_cum.append(f_sum)

        recalls = []
        precisions = []
        for t_c, f_c in zip(tp_cum, fp_cum):
            if t_c + f_c == 0:
                prec = 0.0
            else:
                prec = t_c / (t_c + f_c)
            rec = t_c / npos
            recalls.append(rec)
            precisions.append(prec)

        ap_thr = _compute_ap(recalls, precisions)
        aps.append(ap_thr)

    mean_ap = sum(aps) / len(aps) if aps else 0.0

    print("=== fsod (GT-centered) mAP ===")
    print(f"IoU_near  threshold (keep)   : {iou_near}")
    print(f"GT boxes (npos)              : {npos}")
    print(f"Total predictions (GT-filter): {len(dets)}")
    print(f"Predictions kept (near GT)   : {len(kept_dets)}")
    print(f"Predictions ignored (far GT) : {ignored_dets}")
    print("Per-threshold fsod AP:")
    for thr, ap_thr in zip(iou_thresholds, aps):
        print(f"  IoU={thr:.2f} : AP={ap_thr:.4f}")
    print(f"Mean fsod mAP ({iou_thresholds[0]:.2f}-{iou_thresholds[-1]:.2f}) : {mean_ap:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-image COCO prediction json folder against few-shot GT COCO annotation."
        )
    )
    parser.add_argument(
        "pred_json",
        type=str,
        help="预测结果：可以是 COCO prediction json（list 格式），也可以是单图 json 目录。",
    )
    parser.add_argument(
        "gt_coco_json",
        type=str,
        help="few-shot GT COCO annotation json（例如 1_shot.json）",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="coco",
        choices=["coco", "fsod_ap", "fsod_map"],
        help=(
            "评估指标类型："
            "'coco' 为标准 COCO mAP；"
            "'fsod_ap' 为单一 IoU 阈值下只在 GT 邻域上的局部 AP；"
            "'fsod_map' 为 IoU 区间 (start-0.95) 上局部 AP 的平均。"
        ),
    )
    parser.add_argument(
        "--iou_match",
        type=float,
        default=0.5,
        help=(
            "fsod_ap 模式下匹配 GT 的 IoU 阈值（默认 0.5）；"
            "fsod_map 模式下为 IoU 起始阈值（终止固定为 0.95，"
            "起始会被裁剪到 [0.5, 0.95]）。"
        ),
    )
    parser.add_argument(
        "--iou_near",
        type=float,
        default=0.3,
        help="fsod_ap 模式下，预测被认为“在 GT 邻域内”的最小 IoU（默认 0.3）。",
    )
    args = parser.parse_args()

    if not (os.path.isdir(args.pred_json) or os.path.isfile(args.pred_json)):
        raise SystemExit(f"pred_json is not a directory or file: {args.pred_json}")
    if not os.path.isfile(args.gt_coco_json):
        raise SystemExit(f"gt_coco_json not found: {args.gt_coco_json}")

    if args.metric == "coco":
        evaluate_coco_map(args.pred_json, args.gt_coco_json)
    elif args.metric == "fsod_ap":
        evaluate_fsod_ap(
            args.pred_json,
            args.gt_coco_json,
            iou_match=args.iou_match,
            iou_near=args.iou_near,
        )
    else:
        evaluate_fsod_map(
            args.pred_json,
            args.gt_coco_json,
            iou_match_start=args.iou_match,
            iou_near=args.iou_near,
        )


if __name__ == "__main__":
    main()
