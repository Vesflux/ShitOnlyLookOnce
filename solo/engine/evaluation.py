from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.core.features import *
from solo.data.dataset import *
from solo.models.matcher import *
from solo.engine.proposals import *
from solo.utils.cv_image import Image

def _truth_boxes_for_image(
    image_path: Path,
    annotations: str,
    labels_dir: str | Path | None,
    class_names: dict[int, str] | None,
) -> list[dict[str, Any]]:
    _annotation_path, annotations_payload = load_annotations_for_image(image_path, annotations, labels_dir, class_names)
    truth = []
    for item in annotations_payload:
        label = item.get("label") or item.get("class_id") or "unknown"
        truth.append({"label": str(label), "bbox": item["bbox"]})
    return truth

def _dedupe_truth_boxes(
    truth_boxes: list[dict[str, Any]],
    duplicate_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    duplicates = []
    for truth in truth_boxes:
        duplicate_of = None
        for kept_truth in kept:
            if truth["label"] == kept_truth["label"] and bbox_iou(truth["bbox"], kept_truth["bbox"]) >= duplicate_iou:
                duplicate_of = kept_truth
                break
        if duplicate_of is None:
            kept.append(truth)
            continue
        duplicates.append(
            {
                "label": truth["label"],
                "bbox": truth["bbox"],
                "duplicate_of": duplicate_of["bbox"],
                "iou": round(bbox_iou(truth["bbox"], duplicate_of["bbox"]), 6),
            }
        )
    return kept, duplicates

def _best_truth_match(detection: dict[str, Any], truth_boxes: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    same_label = [truth for truth in truth_boxes if truth["label"] == detection["label"]]
    candidates = same_label or truth_boxes
    if not candidates:
        return 0.0, None
    best_truth = None
    best_iou = 0.0
    for truth in candidates:
        iou = bbox_iou(detection["bbox"], truth["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_truth = truth
    return best_iou, best_truth

def _best_bbox_match(bbox: dict[str, Any], truth_boxes: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    if not truth_boxes:
        return 0.0, None
    best_truth = None
    best_iou = 0.0
    for truth in truth_boxes:
        iou = bbox_iou(bbox, truth["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_truth = truth
    return best_iou, best_truth

def _truth_key(truth: dict[str, Any], index: int) -> str:
    bbox = truth["bbox"]
    return f"{index}:{truth.get('label', '')}:{bbox['x1']}:{bbox['y1']}:{bbox['x2']}:{bbox['y2']}"

def _area_from_bbox_payload(bbox: dict[str, Any]) -> float:
    return max(0, float(bbox["width"])) * max(0, float(bbox["height"]))

def _image_group_name(image_path: str | Path) -> str:
    stem = Path(image_path).stem
    if "_" in stem:
        return f"{stem.split('_', 1)[0]}_*"
    if stem and stem[0].isdigit():
        return f"{stem[0]}*"
    return stem[:3] or "unknown"

def _blank_metric_bucket() -> dict[str, Any]:
    return {
        "images": 0,
        "truth": 0,
        "detections": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

def _finalize_metric_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    tp = int(bucket.get("tp", 0))
    fp = int(bucket.get("fp", 0))
    fn = int(bucket.get("fn", 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision or recall else 0.0
    bucket["precision"] = precision
    bucket["recall"] = recall
    bucket["f1"] = f1
    return bucket

def _image_detection_metrics(
    detections: list[dict[str, Any]],
    truth_boxes: list[dict[str, Any]],
    match_iou: float = DEFAULT_VAL_MATCH_IOU,
    strict_labels: bool = False,
) -> dict[str, Any]:
    matched_truth: set[int] = set()
    matches = []
    false_positives = []
    sorted_detections = sorted(
        detections,
        key=lambda item: item.get("adjusted_score", item.get("score", 0.0)),
        reverse=True,
    )
    for detection in sorted_detections:
        available_same_label = [
            index
            for index, truth in enumerate(truth_boxes)
            if index not in matched_truth and truth["label"] == detection["label"]
        ]
        available_any_label = [index for index, _truth in enumerate(truth_boxes) if index not in matched_truth]
        candidate_indexes = available_same_label if strict_labels else (available_same_label or available_any_label)
        best_index = None
        best_iou = 0.0
        for truth_index in candidate_indexes:
            iou = bbox_iou(detection["bbox"], truth_boxes[truth_index]["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = truth_index
        if best_index is not None and best_iou >= match_iou:
            matched_truth.add(best_index)
            matches.append(
                {
                    "label": detection["label"],
                    "score": detection.get("adjusted_score", detection.get("score")),
                    "iou": round(best_iou, 6),
                    "bbox": detection["bbox"],
                    "truth_bbox": truth_boxes[best_index]["bbox"],
                }
            )
            continue
        false_positives.append(
            {
                "label": detection.get("label"),
                "score": detection.get("adjusted_score", detection.get("score")),
                "best_iou": round(best_iou, 6),
                "bbox": detection.get("bbox"),
            }
        )

    false_negatives = [
        {
            "label": truth["label"],
            "bbox": truth["bbox"],
        }
        for index, truth in enumerate(truth_boxes)
        if index not in matched_truth
    ]
    bucket = _blank_metric_bucket()
    bucket.update(
        {
            "images": 1,
            "truth": len(truth_boxes),
            "detections": len(detections),
            "tp": len(matches),
            "fp": len(false_positives),
            "fn": len(false_negatives),
            "matches": matches,
            "false_positives": false_positives[:MAX_CORRECTION_REPORTS],
            "false_negatives": false_negatives[:MAX_CORRECTION_REPORTS],
        }
    )
    return _finalize_metric_bucket(bucket)

def _proposal_recall_for_image(
    image_path: str | Path,
    weight_index: dict[str, Any],
    truth_boxes: list[dict[str, Any]],
    proposal: str = "both",
    min_area: int = 12,
    max_area_ratio: float = 0.25,
    proposal_expand: float = 1.1,
    min_box_size: int = 4,
    max_box_size: int = 0,
    max_aspect_ratio: float = 8.0,
    window_sizes: list[int] | None = None,
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
    proposal_dedupe_iou: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    input_size: int = DEFAULT_INPUT_SIZE,
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = sorted(set(iou_thresholds or [0.1, DEFAULT_VAL_MATCH_IOU, 0.5]))
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        proposal_image, letterbox = letterbox_image(image, input_size)
        proposal_width, proposal_height = proposal_image.size
        needs_context_maps = proposal_refine
        proposal_maps = context_maps_for_image(proposal_image) if needs_context_maps else None
        proposals = generate_proposals(
            proposal_image,
            proposal=proposal,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            expand=proposal_expand,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            window_sizes=window_sizes,
            window_ratios=window_ratios,
            stride_ratio=stride_ratio,
            bbox_prior=weight_index.get("bbox_prior"),
            proposal_dedupe_iou=proposal_dedupe_iou,
            max_proposals=max_proposals,
            edge_proposals=edge_proposals,
            body_proposals=body_proposals,
            proposal_refine=proposal_refine,
            proposal_maps=proposal_maps,
        )
        proposal_boxes = []
        for proposal_item in proposals:
            scaled_bbox = _expand_bbox(_bbox_tuple(proposal_item["bbox"]), proposal_width, proposal_height, bbox_scale)
            if scaled_bbox is not None:
                source_bbox = _map_bbox_from_letterbox(scaled_bbox, letterbox)
                if source_bbox is not None:
                    proposal_boxes.append(_bbox_payload(source_bbox, width, height))

    best_ious = []
    for truth in truth_boxes:
        best_iou, _best_truth = _best_bbox_match(truth["bbox"], [{"label": "proposal", "bbox": bbox} for bbox in proposal_boxes])
        best_ious.append(best_iou)
    hits = {str(threshold): sum(1 for value in best_ious if value >= threshold) for threshold in thresholds}
    recalls = {
        str(threshold): hits[str(threshold)] / len(truth_boxes) if truth_boxes else 0.0
        for threshold in thresholds
    }
    return {
        "proposal_count": len(proposal_boxes),
        "truth": len(truth_boxes),
        "hits": hits,
        "recall": recalls,
        "best_iou_min": min(best_ious) if best_ious else 0.0,
        "best_iou_median": statistics.median(best_ious) if best_ious else 0.0,
        "best_iou_max": max(best_ious) if best_ious else 0.0,
    }

def evaluate_detection_results(
    results: list[dict[str, Any]],
    labels_dir: str | Path,
    annotations: str = "yolo",
    class_names_path: str | Path | None = None,
    match_iou: float = DEFAULT_VAL_MATCH_IOU,
    duplicate_iou: float = DEFAULT_VAL_DUPLICATE_IOU,
    weight_index: dict[str, Any] | None = None,
    proposal: str = "both",
    min_area: int = 12,
    max_area_ratio: float = 0.25,
    proposal_expand: float = 1.1,
    min_box_size: int = 4,
    max_box_size: int = 0,
    max_aspect_ratio: float = 8.0,
    window_sizes: list[int] | None = None,
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
    proposal_dedupe_iou: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    evaluate_proposals: bool = True,
    input_size: int = DEFAULT_INPUT_SIZE,
    strict_labels: bool = False,
) -> dict[str, Any]:
    class_names = load_class_names(class_names_path)
    summary = _blank_metric_bucket()
    groups: dict[str, dict[str, Any]] = {}
    image_reports = []
    duplicate_label_count = 0
    proposal_thresholds = sorted(set([0.1, match_iou, 0.5]))
    proposal_hits = {str(threshold): 0 for threshold in proposal_thresholds}
    proposal_recall_enabled = evaluate_proposals and weight_index is not None
    started = time.time()

    for index, result in enumerate(results, start=1):
        image_path = Path(result["image"])
        truth_boxes = _truth_boxes_for_image(image_path, annotations, labels_dir, class_names)
        truth_boxes, duplicate_truth = _dedupe_truth_boxes(truth_boxes, duplicate_iou)
        duplicate_label_count += len(duplicate_truth)
        metrics = _image_detection_metrics(
            result.get("detections", []),
            truth_boxes,
            match_iou=match_iou,
            strict_labels=strict_labels,
        )
        proposal_metrics = None
        if proposal_recall_enabled:
            proposal_metrics = _proposal_recall_for_image(
                image_path,
                weight_index,
                truth_boxes,
                proposal=proposal,
                min_area=min_area,
                max_area_ratio=max_area_ratio,
                proposal_expand=proposal_expand,
                min_box_size=min_box_size,
                max_box_size=max_box_size,
                max_aspect_ratio=max_aspect_ratio,
                window_sizes=window_sizes,
                window_ratios=window_ratios,
                stride_ratio=stride_ratio,
                bbox_scale=bbox_scale,
                proposal_dedupe_iou=proposal_dedupe_iou,
                max_proposals=max_proposals,
                edge_proposals=edge_proposals,
                body_proposals=body_proposals,
                proposal_refine=proposal_refine,
                input_size=input_size,
                iou_thresholds=proposal_thresholds,
            )
            for threshold in proposal_thresholds:
                proposal_hits[str(threshold)] += proposal_metrics["hits"][str(threshold)]

        for key in ("images", "truth", "detections", "tp", "fp", "fn"):
            summary[key] += metrics[key]
        group_name = _image_group_name(image_path)
        group = groups.setdefault(group_name, _blank_metric_bucket())
        for key in ("images", "truth", "detections", "tp", "fp", "fn"):
            group[key] += metrics[key]
        image_reports.append(
            {
                "image": str(image_path),
                "group": group_name,
                "metrics": {
                    key: metrics[key]
                    for key in ("truth", "detections", "tp", "fp", "fn", "precision", "recall", "f1")
                },
                "proposal_recall": proposal_metrics,
            }
        )
        proposal_text = ""
        if proposal_metrics:
            proposal_text = f" proposal_recall@{match_iou:g}={proposal_metrics['recall'][str(match_iou)]:.4f}"
        print(
            f"[eval] image {index}/{len(results)} {image_path.name} truth={metrics['truth']} "
            f"det={metrics['detections']} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']}"
            f"{proposal_text} elapsed={_elapsed(started)}",
            flush=True,
        )

    _finalize_metric_bucket(summary)
    for group in groups.values():
        _finalize_metric_bucket(group)
    proposal_recall = None
    if proposal_recall_enabled:
        proposal_recall = {
            str(threshold): proposal_hits[str(threshold)] / summary["truth"] if summary["truth"] else 0.0
            for threshold in proposal_thresholds
        }
    return {
        "enabled": True,
        "match_iou": match_iou,
        "duplicate_iou": duplicate_iou,
        "duplicate_label_count": duplicate_label_count,
        "summary": summary,
        "groups": dict(sorted(groups.items())),
        "proposal_recall": proposal_recall,
        "images": image_reports,
    }
__all__ = [
    '_truth_boxes_for_image',
    '_dedupe_truth_boxes',
    '_best_truth_match',
    '_best_bbox_match',
    '_truth_key',
    '_area_from_bbox_payload',
    '_image_group_name',
    '_blank_metric_bucket',
    '_finalize_metric_bucket',
    '_image_detection_metrics',
    '_proposal_recall_for_image',
    'evaluate_detection_results',
]
