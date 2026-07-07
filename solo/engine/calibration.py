from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.data.dataset import *
from solo.models.matcher import *
from solo.engine.detector import detect_image, _dedupe_truth_boxes, _image_detection_metrics, _image_paths

def _evaluate_threshold(
    positive_scores: list[float],
    negative_scores: list[float],
    total_truth: int,
    threshold: float,
    beta: float,
) -> dict[str, Any]:
    tp = sum(1 for score in positive_scores if score >= threshold)
    fp = sum(1 for score in negative_scores if score >= threshold)
    fn = max(0, total_truth - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta2 = beta * beta
    f_score = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision or recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }

def _learn_score_threshold(
    positive_scores: list[float],
    negative_scores: list[float],
    total_truth: int,
    beta: float,
    min_threshold: float,
    max_threshold: float,
) -> dict[str, Any]:
    candidates = sorted(
        {
            round(max(min_threshold, min(max_threshold, score)), 6)
            for score in positive_scores + negative_scores
            if min_threshold <= score <= max_threshold
        }
        | {min_threshold, max_threshold}
    )
    if not candidates:
        return _evaluate_threshold(positive_scores, negative_scores, total_truth, min_threshold, beta)

    best = None
    for threshold in candidates:
        item = _evaluate_threshold(positive_scores, negative_scores, total_truth, threshold, beta)
        if best is None:
            best = item
            continue
        if item["f_score"] > best["f_score"]:
            best = item
        elif item["f_score"] == best["f_score"] and item["recall"] > best["recall"]:
            best = item
        elif (
            item["f_score"] == best["f_score"]
            and item["recall"] == best["recall"]
            and item["threshold"] > best["threshold"]
        ):
            best = item
    return best or _evaluate_threshold(positive_scores, negative_scores, total_truth, min_threshold, beta)

def self_calibrate_detection(
    train_images: str | Path | None,
    weight_index: dict[str, Any],
    annotations: str = "yolo",
    train_labels_dir: str | Path | None = None,
    class_names_path: str | Path | None = None,
    proposal: str = "color",
    score_floor: float = 0.0,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    proposal_expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
    samples: int = 0,
    match_iou: float = DEFAULT_VAL_MATCH_IOU,
    beta: float = 2.0,
    min_threshold: float = 0.98,
    max_threshold: float = DEFAULT_MAX_CALIBRATED_SCORE_THRESHOLD,
    negative_penalty: float = DEFAULT_NEGATIVE_PENALTY,
    min_negative_margin: float = DEFAULT_MIN_NEGATIVE_MARGIN,
    bbox_prior_mode: str = DEFAULT_BBOX_PRIOR_MODE,
    match_mode: str = DEFAULT_MATCH_MODE,
    channel_mode: str = DEFAULT_CHANNEL_MODE,
    channel_top_k: int = DEFAULT_CHANNEL_TOP_K,
    proposal_dedupe_iou: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    min_detection_width: int = DEFAULT_MIN_DETECTION_WIDTH,
    min_detection_height: int = DEFAULT_MIN_DETECTION_HEIGHT,
    min_detection_area: int = DEFAULT_MIN_DETECTION_AREA,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    min_refine_edge_gain: float = DEFAULT_MIN_REFINE_EDGE_GAIN,
    refine_boxes: bool = DEFAULT_REFINE_BOXES,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
    refine_edge_weight: float = DEFAULT_REFINE_EDGE_WEIGHT,
    refine_edge_gain: float = DEFAULT_REFINE_EDGE_GAIN,
    refine_rematch_top_k: int = DEFAULT_REFINE_REMATCH_TOP_K,
    refine_require_rematch: bool = DEFAULT_REFINE_REQUIRE_REMATCH,
    nms_containment_threshold: float = DEFAULT_NMS_CONTAINMENT_THRESHOLD,
    cluster_nms_center_distance: float = DEFAULT_CLUSTER_NMS_CENTER_DISTANCE,
    cluster_nms_containment: float = DEFAULT_CLUSTER_NMS_CONTAINMENT,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    box_quality_weight: float = DEFAULT_BOX_QUALITY_WEIGHT,
    min_box_quality: float = DEFAULT_MIN_BOX_QUALITY,
    context_weight: float = DEFAULT_CONTEXT_WEIGHT,
    min_context_quality: float = DEFAULT_MIN_CONTEXT_QUALITY,
    context_expand: bool = DEFAULT_CONTEXT_EXPAND,
    fragmentation_weight: float = DEFAULT_FRAGMENTATION_WEIGHT,
    max_fragmentation: float = DEFAULT_MAX_FRAGMENTATION,
    accelerator: str = DEFAULT_ACCELERATOR,
    second_stage_rescoring: bool = DEFAULT_SECOND_STAGE_RESCORING,
    second_stage_threshold: float = DEFAULT_SECOND_STAGE_THRESHOLD,
    second_stage_margin_weight: float = DEFAULT_SECOND_STAGE_MARGIN_WEIGHT,
    second_stage_support_weight: float = DEFAULT_SECOND_STAGE_SUPPORT_WEIGHT,
    second_stage_proposal_weight: float = DEFAULT_SECOND_STAGE_PROPOSAL_WEIGHT,
    second_stage_quality_weight: float = DEFAULT_SECOND_STAGE_QUALITY_WEIGHT,
    second_stage_sky_region: float = DEFAULT_SECOND_STAGE_SKY_REGION,
    second_stage_sky_penalty: float = DEFAULT_SECOND_STAGE_SKY_PENALTY,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> dict[str, Any]:
    if not train_images:
        return {"enabled": False}

    train_paths = _image_paths(train_images)
    if samples:
        train_paths = train_paths[:samples]
    class_names = load_class_names(class_names_path)
    positive_scores = []
    negative_scores = []
    margin_positive = []
    margin_negative = []
    total_truth = 0
    hard_negative_reports = []
    processed = 0
    started = time.time()
    print(
        f"[self-calibrate] enabled train={len(train_paths)} samples={samples or 'all'} "
        f"match_iou={match_iou} beta={beta}",
        flush=True,
    )

    for image_index, image_path in enumerate(train_paths, start=1):
        truth_boxes = _truth_boxes_for_image(image_path, annotations, train_labels_dir, class_names)
        truth_boxes, _duplicates = _dedupe_truth_boxes(truth_boxes, DEFAULT_VAL_DUPLICATE_IOU)
        if not truth_boxes:
            continue
        total_truth += len(truth_boxes)
        result = detect_image(
            image_path,
            weight_index,
            proposal=proposal,
            score_threshold=score_floor,
            nms_threshold=nms_threshold,
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
            negative_penalty=negative_penalty,
            min_negative_margin=min_negative_margin,
            excluded_source_keys={_source_key(image_path)},
            bbox_prior_mode=bbox_prior_mode,
            match_mode=match_mode,
            channel_mode=channel_mode,
            channel_top_k=channel_top_k,
            proposal_dedupe_iou=proposal_dedupe_iou,
            max_proposals=max_proposals,
            edge_proposals=edge_proposals,
            body_proposals=body_proposals,
            proposal_refine=proposal_refine,
            min_detection_width=min_detection_width,
            min_detection_height=min_detection_height,
            min_detection_area=min_detection_area,
            max_detections=max_detections,
            min_refine_edge_gain=min_refine_edge_gain,
            refine_boxes=refine_boxes,
            refine_top_k=refine_top_k,
            refine_edge_weight=refine_edge_weight,
            refine_edge_gain=refine_edge_gain,
            refine_rematch_top_k=refine_rematch_top_k,
            refine_require_rematch=refine_require_rematch,
            nms_containment_threshold=nms_containment_threshold,
            cluster_nms_center_distance=cluster_nms_center_distance,
            cluster_nms_containment=cluster_nms_containment,
            structure_weight=structure_weight,
            box_quality_weight=box_quality_weight,
            min_box_quality=min_box_quality,
            context_weight=context_weight,
            min_context_quality=min_context_quality,
            context_expand=context_expand,
            fragmentation_weight=fragmentation_weight,
            max_fragmentation=max_fragmentation,
            accelerator=accelerator,
            second_stage_rescoring=second_stage_rescoring,
            second_stage_threshold=second_stage_threshold,
            second_stage_margin_weight=second_stage_margin_weight,
            second_stage_support_weight=second_stage_support_weight,
            second_stage_proposal_weight=second_stage_proposal_weight,
            second_stage_quality_weight=second_stage_quality_weight,
            second_stage_sky_region=second_stage_sky_region,
            second_stage_sky_penalty=second_stage_sky_penalty,
            input_size=input_size,
        )
        processed += 1
        image_positive = 0
        image_negative = 0
        for detection in result["detections"]:
            best_iou, _best_truth = _best_truth_match(detection, truth_boxes)
            score = detection.get("adjusted_score", detection["score"])
            margin = detection.get("negative_margin", 0.0)
            if best_iou >= match_iou:
                positive_scores.append(score)
                margin_positive.append(margin)
                image_positive += 1
                continue
            negative_scores.append(score)
            margin_negative.append(margin)
            image_negative += 1
            if len(hard_negative_reports) < MAX_CORRECTION_REPORTS:
                hard_negative_reports.append(
                    {
                        "image": str(image_path),
                        "label": detection["label"],
                        "score": detection["score"],
                        "adjusted_score": score,
                        "negative_margin": margin,
                        "best_iou": round(best_iou, 6),
                        "bbox": detection["bbox"],
                    }
                )
        print(
            f"[self-calibrate] image {image_index}/{len(train_paths)} {image_path.name} "
            f"truth={len(truth_boxes)} positive={image_positive} hard_negative={image_negative} "
            f"elapsed={_elapsed(started)}",
            flush=True,
        )

    learned = _learn_score_threshold(
        positive_scores,
        negative_scores,
        total_truth,
        beta=beta,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    positive_margin_threshold = statistics.median(margin_positive) if margin_positive else min_negative_margin
    negative_margin_threshold = statistics.median(margin_negative) if margin_negative else min_negative_margin
    learned_margin = min_negative_margin
    if margin_positive and margin_negative:
        learned_margin = min(positive_margin_threshold, max(min_negative_margin, negative_margin_threshold))
    print(
        f"[self-calibrate] done processed={processed} threshold={learned['threshold']:.6f} "
        f"precision={learned['precision']:.4f} recall={learned['recall']:.4f} "
        f"f{beta:g}={learned['f_score']:.4f} hard_negatives={len(negative_scores)} "
        f"elapsed={_elapsed(started)}",
        flush=True,
    )
    return {
        "enabled": True,
        "processed": processed,
        "total_truth": total_truth,
        "positive_scores": len(positive_scores),
        "negative_scores": len(negative_scores),
        "score_threshold": learned["threshold"],
        "precision": learned["precision"],
        "recall": learned["recall"],
        "f_score": learned["f_score"],
        "beta": beta,
        "tp": learned["tp"],
        "fp": learned["fp"],
        "fn": learned["fn"],
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "learned_negative_margin": learned_margin,
        "positive_margin_median": positive_margin_threshold,
        "negative_margin_median": negative_margin_threshold,
        "hard_negative_reports": hard_negative_reports,
    }

def calibrate_detection(
    train_images: str | Path | None,
    val_images: str | Path | None,
    weight_index: dict[str, Any],
    annotations: str = "yolo",
    train_labels_dir: str | Path | None = None,
    val_labels_dir: str | Path | None = None,
    class_names_path: str | Path | None = None,
    proposal: str = "color",
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    proposal_expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
    max_samples: int = 0,
    val_match_iou: float = DEFAULT_VAL_MATCH_IOU,
    val_duplicate_iou: float = DEFAULT_VAL_DUPLICATE_IOU,
    missing_label_score: float = DEFAULT_MISSING_LABEL_SCORE,
    max_calibrated_score_threshold: float = DEFAULT_MAX_CALIBRATED_SCORE_THRESHOLD,
    calibration_score_slack: float = DEFAULT_CALIBRATION_SCORE_SLACK,
    negative_penalty: float = DEFAULT_NEGATIVE_PENALTY,
    min_negative_margin: float = DEFAULT_MIN_NEGATIVE_MARGIN,
    bbox_prior_mode: str = DEFAULT_BBOX_PRIOR_MODE,
    match_mode: str = DEFAULT_MATCH_MODE,
    channel_mode: str = DEFAULT_CHANNEL_MODE,
    channel_top_k: int = DEFAULT_CHANNEL_TOP_K,
    proposal_dedupe_iou: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    min_detection_width: int = DEFAULT_MIN_DETECTION_WIDTH,
    min_detection_height: int = DEFAULT_MIN_DETECTION_HEIGHT,
    min_detection_area: int = DEFAULT_MIN_DETECTION_AREA,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    min_refine_edge_gain: float = DEFAULT_MIN_REFINE_EDGE_GAIN,
    refine_boxes: bool = DEFAULT_REFINE_BOXES,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
    refine_edge_weight: float = DEFAULT_REFINE_EDGE_WEIGHT,
    refine_edge_gain: float = DEFAULT_REFINE_EDGE_GAIN,
    refine_rematch_top_k: int = DEFAULT_REFINE_REMATCH_TOP_K,
    refine_require_rematch: bool = DEFAULT_REFINE_REQUIRE_REMATCH,
    nms_containment_threshold: float = DEFAULT_NMS_CONTAINMENT_THRESHOLD,
    cluster_nms_center_distance: float = DEFAULT_CLUSTER_NMS_CENTER_DISTANCE,
    cluster_nms_containment: float = DEFAULT_CLUSTER_NMS_CONTAINMENT,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    box_quality_weight: float = DEFAULT_BOX_QUALITY_WEIGHT,
    min_box_quality: float = DEFAULT_MIN_BOX_QUALITY,
    context_weight: float = DEFAULT_CONTEXT_WEIGHT,
    min_context_quality: float = DEFAULT_MIN_CONTEXT_QUALITY,
    context_expand: bool = DEFAULT_CONTEXT_EXPAND,
    fragmentation_weight: float = DEFAULT_FRAGMENTATION_WEIGHT,
    max_fragmentation: float = DEFAULT_MAX_FRAGMENTATION,
    accelerator: str = DEFAULT_ACCELERATOR,
    second_stage_rescoring: bool = DEFAULT_SECOND_STAGE_RESCORING,
    second_stage_threshold: float = DEFAULT_SECOND_STAGE_THRESHOLD,
    second_stage_margin_weight: float = DEFAULT_SECOND_STAGE_MARGIN_WEIGHT,
    second_stage_support_weight: float = DEFAULT_SECOND_STAGE_SUPPORT_WEIGHT,
    second_stage_proposal_weight: float = DEFAULT_SECOND_STAGE_PROPOSAL_WEIGHT,
    second_stage_quality_weight: float = DEFAULT_SECOND_STAGE_QUALITY_WEIGHT,
    second_stage_sky_region: float = DEFAULT_SECOND_STAGE_SKY_REGION,
    second_stage_sky_penalty: float = DEFAULT_SECOND_STAGE_SKY_PENALTY,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> dict[str, Any]:
    if not train_images or not val_images:
        return {
            "enabled": False,
            "score_threshold": score_threshold,
            "bbox_scale": bbox_scale,
            "nms_threshold": nms_threshold,
            "max_calibrated_score_threshold": max_calibrated_score_threshold,
            "calibration_score_slack": calibration_score_slack,
        }

    train_paths = _image_paths(train_images)
    val_paths = _image_paths(val_images)
    if not val_paths:
        return {
            "enabled": False,
            "score_threshold": score_threshold,
            "bbox_scale": bbox_scale,
            "nms_threshold": nms_threshold,
            "max_calibrated_score_threshold": max_calibrated_score_threshold,
            "calibration_score_slack": calibration_score_slack,
        }

    ratio = max(1, round(len(train_paths) / len(val_paths))) if train_paths else 1
    class_names = load_class_names(class_names_path)
    val_iter = iter(val_paths)
    score_samples = []
    scale_samples = []
    missing_label_scores = []
    duplicate_predictions = 0
    duplicate_truth_hits = 0
    duplicate_val_labels = 0
    false_negative_count = 0
    correction_reports = []
    processed_val = 0
    started = time.time()
    print(
        f"[calibrate] enabled train={len(train_paths)} val={len(val_paths)} "
        f"ratio=1 val per {ratio} train samples limit={max_samples or 'all'}",
        flush=True,
    )

    for train_index, _train_image in enumerate(train_paths, start=1):
        if train_index % ratio != 0:
            continue
        try:
            val_image = next(val_iter)
        except StopIteration:
            break
        truth_boxes = _truth_boxes_for_image(val_image, annotations, val_labels_dir, class_names)
        truth_boxes, duplicate_truth = _dedupe_truth_boxes(truth_boxes, val_duplicate_iou)
        duplicate_val_labels += len(duplicate_truth)
        for duplicate in duplicate_truth:
            if len(correction_reports) < MAX_CORRECTION_REPORTS:
                correction_reports.append(
                    {
                        "type": "possible_duplicate_label",
                        "image": str(val_image),
                        **duplicate,
                    }
                )
        if not truth_boxes:
            print(
                f"[calibrate] train_step={train_index}/{len(train_paths)} val={val_image.name} skipped=no_truth "
                f"elapsed={_elapsed(started)}",
                flush=True,
            )
            continue
        result = detect_image(
            val_image,
            weight_index,
            proposal=proposal,
            score_threshold=max(0.0, score_threshold - 0.2),
            nms_threshold=nms_threshold,
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
            negative_penalty=negative_penalty,
            min_negative_margin=min_negative_margin,
            bbox_prior_mode=bbox_prior_mode,
            match_mode=match_mode,
            channel_mode=channel_mode,
            channel_top_k=channel_top_k,
            proposal_dedupe_iou=proposal_dedupe_iou,
            max_proposals=max_proposals,
            edge_proposals=edge_proposals,
            body_proposals=body_proposals,
            proposal_refine=proposal_refine,
            min_detection_width=min_detection_width,
            min_detection_height=min_detection_height,
            min_detection_area=min_detection_area,
            max_detections=max_detections,
            min_refine_edge_gain=min_refine_edge_gain,
            refine_boxes=refine_boxes,
            refine_top_k=refine_top_k,
            refine_edge_weight=refine_edge_weight,
            refine_edge_gain=refine_edge_gain,
            refine_rematch_top_k=refine_rematch_top_k,
            refine_require_rematch=refine_require_rematch,
            nms_containment_threshold=nms_containment_threshold,
            cluster_nms_center_distance=cluster_nms_center_distance,
            cluster_nms_containment=cluster_nms_containment,
            structure_weight=structure_weight,
            box_quality_weight=box_quality_weight,
            min_box_quality=min_box_quality,
            context_weight=context_weight,
            min_context_quality=min_context_quality,
            context_expand=context_expand,
            fragmentation_weight=fragmentation_weight,
            max_fragmentation=max_fragmentation,
            accelerator=accelerator,
            second_stage_rescoring=second_stage_rescoring,
            second_stage_threshold=second_stage_threshold,
            second_stage_margin_weight=second_stage_margin_weight,
            second_stage_support_weight=second_stage_support_weight,
            second_stage_proposal_weight=second_stage_proposal_weight,
            second_stage_quality_weight=second_stage_quality_weight,
            second_stage_sky_region=second_stage_sky_region,
            second_stage_sky_penalty=second_stage_sky_penalty,
            input_size=input_size,
        )
        processed_val += 1
        truth_hit_counts = {_truth_key(truth, index): 0 for index, truth in enumerate(truth_boxes)}
        image_missing_labels = 0
        image_duplicate_predictions = len(duplicate_truth)
        for detection in result["detections"]:
            best_iou, best_truth = _best_truth_match(detection, truth_boxes)
            if best_iou >= val_match_iou:
                score_samples.append(detection.get("adjusted_score", detection["score"]))
                if best_truth is not None:
                    truth_index = truth_boxes.index(best_truth)
                    truth_key = _truth_key(best_truth, truth_index)
                    truth_hit_counts[truth_key] += 1
                    detection_area = _area_from_bbox_payload(detection["bbox"])
                    truth_area = _area_from_bbox_payload(best_truth["bbox"])
                    if detection_area > 0 and truth_area > 0:
                        scale_samples.append(math.sqrt(truth_area / detection_area))
                continue

            if detection.get("adjusted_score", detection["score"]) >= missing_label_score:
                missing_label_scores.append(detection.get("adjusted_score", detection["score"]))
                image_missing_labels += 1
                if len(correction_reports) < MAX_CORRECTION_REPORTS:
                    correction_reports.append(
                        {
                            "type": "possible_missing_label",
                            "image": str(val_image),
                            "label": detection["label"],
                            "score": detection["score"],
                            "adjusted_score": detection.get("adjusted_score", detection["score"]),
                            "negative_margin": detection.get("negative_margin"),
                            "best_iou": round(best_iou, 6),
                            "bbox": detection["bbox"],
                        }
                    )
        for truth_index, truth in enumerate(truth_boxes):
            truth_key = _truth_key(truth, truth_index)
            hits = truth_hit_counts[truth_key]
            if hits == 0:
                false_negative_count += 1
                if len(correction_reports) < MAX_CORRECTION_REPORTS:
                    correction_reports.append(
                        {
                            "type": "possible_missed_detection",
                            "image": str(val_image),
                            "label": truth["label"],
                            "bbox": truth["bbox"],
                        }
                    )
            elif hits > 1:
                duplicate_truth_hits += hits - 1
                image_duplicate_predictions += hits - 1
        for left_index, left in enumerate(result["detections"]):
            for right in result["detections"][left_index + 1 :]:
                if left["label"] == right["label"] and bbox_iou(left["bbox"], right["bbox"]) >= val_duplicate_iou:
                    duplicate_predictions += 1
                    image_duplicate_predictions += 1
                    if len(correction_reports) < MAX_CORRECTION_REPORTS:
                        correction_reports.append(
                            {
                                "type": "possible_duplicate_detection",
                                "image": str(val_image),
                                "label": left["label"],
                                "left_score": left["score"],
                                "right_score": right["score"],
                                "iou": round(bbox_iou(left["bbox"], right["bbox"]), 6),
                                "left_bbox": left["bbox"],
                                "right_bbox": right["bbox"],
                            }
                        )
        print(
            f"[calibrate] train_step={train_index}/{len(train_paths)} val={processed_val}/{len(val_paths)} "
            f"{val_image.name} detections={len(result['detections'])} "
            f"score_samples={len(score_samples)} scale_samples={len(scale_samples)} "
            f"missing={image_missing_labels} duplicates={image_duplicate_predictions} "
            f"elapsed={_elapsed(started)}",
            flush=True,
        )
        if max_samples and processed_val >= max_samples:
            break

    calibrated_threshold = score_threshold
    if score_samples:
        calibrated_threshold = max(
            0.0,
            min(max_calibrated_score_threshold, statistics.median(score_samples) - calibration_score_slack),
        )
    calibrated_bbox_scale = bbox_scale
    if scale_samples:
        calibrated_bbox_scale = max(0.2, min(5.0, bbox_scale * statistics.median(scale_samples)))
    calibrated_nms_threshold = nms_threshold
    if duplicate_predictions or duplicate_truth_hits:
        calibrated_nms_threshold = max(0.05, min(nms_threshold, val_duplicate_iou * 0.5))
    print(
        f"[calibrate] done processed_val={processed_val} "
        f"score_threshold={calibrated_threshold:.6f} bbox_scale={calibrated_bbox_scale:.4f} "
        f"nms_threshold={calibrated_nms_threshold:.4f} missing_labels={len(missing_label_scores)} "
        f"duplicate_predictions={duplicate_predictions} duplicate_val_labels={duplicate_val_labels} "
        f"false_negatives={false_negative_count} "
        f"elapsed={_elapsed(started)}",
        flush=True,
    )

    return {
        "enabled": True,
        "ratio": ratio,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "processed_val": processed_val,
        "score_samples": len(score_samples),
        "scale_samples": len(scale_samples),
        "possible_missing_labels": len(missing_label_scores),
        "possible_missed_detections": false_negative_count,
        "duplicate_predictions": duplicate_predictions,
        "duplicate_truth_hits": duplicate_truth_hits,
        "duplicate_val_labels": duplicate_val_labels,
        "correction_reports": correction_reports,
        "score_threshold": calibrated_threshold,
        "bbox_scale": calibrated_bbox_scale,
        "nms_threshold": calibrated_nms_threshold,
        "max_calibrated_score_threshold": max_calibrated_score_threshold,
        "calibration_score_slack": calibration_score_slack,
    }
__all__ = [
    '_evaluate_threshold',
    '_learn_score_threshold',
    'self_calibrate_detection',
    'calibrate_detection',
]
