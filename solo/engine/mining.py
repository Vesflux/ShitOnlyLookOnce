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
from solo.core.features import *
from solo.data.dataset import *
from solo.data.dataloader import *
from solo.models.matcher import *
from solo.engine.detector import *

def _metric_score(metrics: dict[str, Any], recall_weight: float = 1.25) -> float:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    if precision <= 0 and recall <= 0:
        return 0.0
    beta2 = recall_weight * recall_weight
    return (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision or recall else 0.0

def run_mining_rounds(
    train_images: str | Path,
    train_labels_dir: str | Path,
    val_images: str | Path,
    val_labels_dir: str | Path,
    output_dir: str | Path,
    rounds: int = DEFAULT_MINING_ROUNDS,
    annotations: str = "yolo",
    class_names_path: str | Path | None = None,
    size: int = 8,
    qua: int = 8,
    nab: float = DEFAULT_NAB,
    pt_size: int = 16,
    kernel: str = DEFAULT_KERNEL,
    field: str = DEFAULT_FIELD,
    max_radius: int = 16,
    normalize: bool = True,
    normalize_each_step: bool = True,
    crop_mode: str = DEFAULT_CROP_MODE,
    feature_mode: str = "multi",
    channels: list[str] | str | None = None,
    prototype_count: int = DEFAULT_PROTOTYPE_COUNT,
    negative_samples_per_image: int = 3,
    negative_ratio: float = 0.25,
    negative_iou: float = 0.03,
    negative_seed: int = 42,
    proposal: str = "both",
    score_threshold: float = 0.88,
    nms_threshold: float = 0.25,
    min_area: int = 12,
    max_area_ratio: float = 0.25,
    proposal_expand: float = 1.1,
    min_box_size: int = 4,
    max_box_size: int = 0,
    max_aspect_ratio: float = 8.0,
    window_sizes: list[int] | None = None,
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.65,
    bbox_scale: float = 1.0,
    bbox_prior_mode: str = "soft",
    match_mode: str = DEFAULT_MATCH_MODE,
    channel_mode: str = DEFAULT_CHANNEL_MODE,
    channel_top_k: int = DEFAULT_CHANNEL_TOP_K,
    proposal_dedupe_iou: float = 0.80,
    max_proposals: int = 160,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    input_size: int = DEFAULT_INPUT_SIZE,
    nms_containment_threshold: float = DEFAULT_NMS_CONTAINMENT_THRESHOLD,
    cluster_nms_center_distance: float = DEFAULT_CLUSTER_NMS_CENTER_DISTANCE,
    cluster_nms_containment: float = DEFAULT_CLUSTER_NMS_CONTAINMENT,
    val_match_iou: float = DEFAULT_VAL_MATCH_IOU,
    val_duplicate_iou: float = DEFAULT_VAL_DUPLICATE_IOU,
    hard_mine_score_threshold: float = DEFAULT_HARD_MINE_SCORE_THRESHOLD,
    hard_positive_weight: float = DEFAULT_HARD_POSITIVE_WEIGHT,
    hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
    hard_negative_max_iou: float = DEFAULT_HARD_NEGATIVE_MIN_IOU,
    mined_prototype_count: int = 0,
    max_hard_positives: int = 300,
    max_hard_negatives: int = 120,
    structure_mode: str = DEFAULT_STRUCTURE_MODE,
    structure_grid: int = DEFAULT_STRUCTURE_GRID,
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
    compact_weights: bool = DEFAULT_COMPACT_WEIGHTS,
    weight_precision: int = DEFAULT_WEIGHT_PRECISION,
    workers: int = 1,
) -> dict[str, Any]:
    if rounds < 0:
        raise ValueError("rounds must be 0 or greater")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_channels = parse_channels(channels)
    started = time.time()
    print(
        f"[rounds] start rounds={rounds} train={Path(train_images).resolve()} val={Path(val_images).resolve()} "
        f"output={output_path.resolve()}",
        flush=True,
    )

    base_weight_path = output_path / "round_00_weights.json"
    get_image_pt(
        train_images,
        size=size,
        qua=qua,
        nab=nab,
        pt_size=pt_size,
        kernel=kernel,
        field=field,
        max_radius=max_radius,
        normalize=normalize,
        normalize_each_step=normalize_each_step,
        annotations=annotations,
        labels_dir=train_labels_dir,
        class_names_path=class_names_path,
        crop_mode=crop_mode,
        save_path=base_weight_path,
        print_pt=False,
        negative_samples_per_image=negative_samples_per_image,
        negative_ratio=negative_ratio,
        negative_iou=negative_iou,
        negative_seed=negative_seed,
        feature_mode=feature_mode,
        channels=selected_channels,
        prototype_count=prototype_count,
        structure_mode=structure_mode,
        structure_grid=structure_grid,
        accelerator=accelerator,
        workers=workers,
        compact_weights=False,
        weight_precision=weight_precision,
    )

    round_summaries = []
    best_round = None
    best_score = -1.0
    current_weight_path = base_weight_path

    for round_index in range(0, rounds + 1):
        round_label = f"round_{round_index:02d}"
        detection_report_path = output_path / f"{round_label}_val_detection.json"
        weight_index = prepare_weight_index([current_weight_path], accelerator=accelerator)
        detection_result = run_detection(
            val_images,
            [current_weight_path],
            output_path=detection_report_path,
            draw_dir=None,
            proposal=proposal,
            score_threshold=score_threshold,
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
            bbox_prior_mode=bbox_prior_mode,
            match_mode=match_mode,
            channel_mode=channel_mode,
            channel_top_k=channel_top_k,
            proposal_dedupe_iou=proposal_dedupe_iou,
            max_proposals=max_proposals,
            edge_proposals=edge_proposals,
            body_proposals=body_proposals,
            proposal_refine=proposal_refine,
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
            eval_labels_dir=val_labels_dir,
            class_names_path=class_names_path,
            annotation_format=annotations,
            val_match_iou=val_match_iou,
            val_duplicate_iou=val_duplicate_iou,
            evaluate_proposals=False,
        )
        evaluation = detection_result["metadata"].get("evaluation", {})
        summary = evaluation.get("summary", {})
        score = _metric_score(summary)
        round_summary = {
            "round": round_index,
            "weights": str(current_weight_path),
            "report": str(detection_report_path),
            "metrics": summary,
            "proposal_recall": evaluation.get("proposal_recall"),
            "score": score,
        }
        round_summaries.append(round_summary)
        print(
            f"[rounds] {round_label} precision={summary.get('precision', 0.0):.4f} "
            f"recall={summary.get('recall', 0.0):.4f} f1={summary.get('f1', 0.0):.4f} "
            f"score={score:.4f} proposals={evaluation.get('proposal_recall')} elapsed={_elapsed(started)}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_round = round_summary
        if round_index >= rounds:
            break

        next_weight_path = output_path / f"round_{round_index + 1:02d}_weights.json"
        mining_result = mine_val_hard_examples(
            [current_weight_path],
            next_weight_path,
            val_images,
            val_labels_dir,
            annotations=annotations,
            class_names_path=class_names_path,
            proposal=proposal,
            score_threshold=max(0.0, score_threshold - 0.2),
            false_positive_score=hard_mine_score_threshold,
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
            val_match_iou=val_match_iou,
            bbox_prior_mode=bbox_prior_mode,
            match_mode=match_mode,
            channel_mode=channel_mode,
            channel_top_k=channel_top_k,
            proposal_dedupe_iou=proposal_dedupe_iou,
            max_proposals=max_proposals,
            edge_proposals=edge_proposals,
            body_proposals=body_proposals,
            proposal_refine=proposal_refine,
            nms_containment_threshold=nms_containment_threshold,
            cluster_nms_center_distance=cluster_nms_center_distance,
            cluster_nms_containment=cluster_nms_containment,
            input_size=input_size,
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
            hard_positive_weight=hard_positive_weight,
            hard_negative_weight=hard_negative_weight,
            hard_negative_max_iou=hard_negative_max_iou,
            mined_prototype_count=mined_prototype_count,
            max_hard_positives=max_hard_positives,
            max_hard_negatives=max_hard_negatives,
            detection_results=detection_result["results"],
            compact_weights=compact_weights if round_index + 1 >= rounds else False,
            weight_precision=weight_precision,
        )
        round_summary["mining"] = mining_result
        current_weight_path = next_weight_path

    summary_path = output_path / "mining_rounds_summary.json"
    payload = {
        "rounds": rounds,
        "best_round": best_round,
        "summaries": round_summaries,
        "elapsed_seconds": round(time.time() - started, 4),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[rounds] done best_round={best_round.get('round') if best_round else None} "
        f"summary={summary_path.resolve()} elapsed={_elapsed(started)}",
        flush=True,
    )
    return {"summary_path": str(summary_path), **payload}

def mine_val_hard_examples(
    weight_paths: list[str | Path],
    output_path: str | Path,
    val_images: str | Path,
    val_labels_dir: str | Path,
    annotations: str = "yolo",
    class_names_path: str | Path | None = None,
    proposal: str = "both",
    score_threshold: float = 0.0,
    false_positive_score: float = DEFAULT_HARD_MINE_SCORE_THRESHOLD,
    nms_threshold: float = 0.25,
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
    val_match_iou: float = DEFAULT_VAL_MATCH_IOU,
    bbox_prior_mode: str = DEFAULT_BBOX_PRIOR_MODE,
    match_mode: str = DEFAULT_MATCH_MODE,
    channel_mode: str = DEFAULT_CHANNEL_MODE,
    channel_top_k: int = DEFAULT_CHANNEL_TOP_K,
    proposal_dedupe_iou: float = DEFAULT_PROPOSAL_DEDUPE_IOU,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    edge_proposals: bool = DEFAULT_EDGE_PROPOSALS,
    body_proposals: bool = DEFAULT_BODY_PROPOSALS,
    proposal_refine: bool = DEFAULT_PROPOSAL_REFINE,
    input_size: int = DEFAULT_INPUT_SIZE,
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
    hard_positive_weight: float = DEFAULT_HARD_POSITIVE_WEIGHT,
    hard_negative_weight: float = DEFAULT_HARD_NEGATIVE_WEIGHT,
    hard_negative_max_iou: float = DEFAULT_HARD_NEGATIVE_MIN_IOU,
    mined_prototype_count: int = 0,
    max_hard_positives: int = 0,
    max_hard_negatives: int = 0,
    detection_results: list[dict[str, Any]] | None = None,
    compact_weights: bool = DEFAULT_COMPACT_WEIGHTS,
    weight_precision: int = DEFAULT_WEIGHT_PRECISION,
) -> dict[str, Any]:
    if not weight_paths:
        raise ValueError("at least one source weight file is required for hard mining")
    base_payload = load_weights(weight_paths[0])
    mined_weights = list(base_payload.get("weights", []))
    if not mined_weights:
        raise ValueError(
            "hard mining requires a full weight file with training samples; "
            "retrain with --full-weights or use --mine-rounds so intermediate weights stay expandable"
        )
    config = dict(base_payload.get("config", {}))
    weight_index = prepare_weight_index(weight_paths, accelerator=accelerator)
    class_names = load_class_names(class_names_path)
    val_paths = _image_paths(val_images)
    result_lookup = {
        _source_key(result.get("image")): result
        for result in detection_results or []
        if result.get("image")
    }
    positives_added = 0
    negatives_added = 0
    started = time.time()
    print(f"[mine] enabled val={len(val_paths)} output={output_path}", flush=True)

    for image_index, image_path in enumerate(val_paths, start=1):
        truth_boxes = _truth_boxes_for_image(image_path, annotations, val_labels_dir, class_names)
        if not truth_boxes:
            continue
        result = result_lookup.get(_source_key(image_path))
        if result is None:
            result = detect_image(
                image_path,
                weight_index,
                proposal=proposal,
                score_threshold=score_threshold,
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
        matched_truth_keys = set()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            for detection_index, detection in enumerate(result["detections"]):
                best_iou, best_truth = _best_truth_match(detection, truth_boxes)
                if best_iou >= val_match_iou and best_truth is not None:
                    matched_truth_keys.add(_truth_key(best_truth, truth_boxes.index(best_truth)))
                    continue
                score = detection.get("adjusted_score", detection["score"])
                if score < false_positive_score:
                    continue
                if best_iou > hard_negative_max_iou:
                    continue
                if max_hard_negatives and negatives_added >= max_hard_negatives:
                    continue
                bbox = _bbox_tuple(detection["bbox"])
                crop = _prepare_crop_for_features(image.crop(bbox), _crop_mode_from_config(config))
                features = extract_image_features(
                    crop,
                    size=config.get("size", 2),
                    qua=config.get("qua", 8),
                    nab=config.get("nab", DEFAULT_NAB),
                    pt_size=config.get("pt_size", 8),
                    kernel=config.get("kernel", DEFAULT_KERNEL),
                    field=config.get("field", DEFAULT_FIELD),
                    max_radius=config.get("max_radius", DEFAULT_MAX_RADIUS),
                    normalize=config.get("normalize", True),
                    normalize_each_step=config.get("normalize_each_step", True),
                    feature_mode=config.get("feature_mode", DEFAULT_FEATURE_MODE),
                    channels=config.get("channels") or None,
                    structure_mode=config.get("structure_mode", DEFAULT_STRUCTURE_MODE),
                    structure_grid=int(config.get("structure_grid", DEFAULT_STRUCTURE_GRID)),
                    stats_version=_stats_version_from_config(config),
                )
                mined_weights.append(
                    {
                        "name": f"{image_path.stem}_hard_negative_{detection_index}{image_path.suffix}",
                        "source_image": str(image_path),
                        "annotation_path": str(Path(val_labels_dir) / f"{image_path.stem}.txt"),
                        "annotation": {
                            "format": "hard_negative",
                            "class_id": None,
                            "label": DEFAULT_NEGATIVE_LABEL,
                            "bbox": _bbox_payload(bbox, image_width, image_height),
                        },
                        "crop_mode": _crop_mode_from_config(config),
                        "negative": True,
                        "sample_weight": hard_negative_weight,
                        "mining": {"type": "hard_negative", "score": score, "best_iou": round(best_iou, 6)},
                        "pt": features["pt"],
                        "features": features,
                    }
                )
                negatives_added += 1

            for truth_index, truth in enumerate(truth_boxes):
                truth_key = _truth_key(truth, truth_index)
                if truth_key in matched_truth_keys:
                    continue
                if max_hard_positives and positives_added >= max_hard_positives:
                    continue
                bbox = _bbox_tuple(truth["bbox"])
                crop = _prepare_crop_for_features(image.crop(bbox), _crop_mode_from_config(config))
                features = extract_image_features(
                    crop,
                    size=config.get("size", 2),
                    qua=config.get("qua", 8),
                    nab=config.get("nab", DEFAULT_NAB),
                    pt_size=config.get("pt_size", 8),
                    kernel=config.get("kernel", DEFAULT_KERNEL),
                    field=config.get("field", DEFAULT_FIELD),
                    max_radius=config.get("max_radius", DEFAULT_MAX_RADIUS),
                    normalize=config.get("normalize", True),
                    normalize_each_step=config.get("normalize_each_step", True),
                    feature_mode=config.get("feature_mode", DEFAULT_FEATURE_MODE),
                    channels=config.get("channels") or None,
                    structure_mode=config.get("structure_mode", DEFAULT_STRUCTURE_MODE),
                    structure_grid=int(config.get("structure_grid", DEFAULT_STRUCTURE_GRID)),
                    stats_version=_stats_version_from_config(config),
                )
                mined_weights.append(
                    {
                        "name": f"{image_path.stem}_hard_positive_{truth_index}{image_path.suffix}",
                        "source_image": str(image_path),
                        "annotation_path": str(Path(val_labels_dir) / f"{image_path.stem}.txt"),
                        "annotation": {
                            "format": "hard_positive",
                            "class_id": None,
                            "label": truth["label"],
                            "bbox": truth["bbox"],
                        },
                        "crop_mode": _crop_mode_from_config(config),
                        "sample_weight": hard_positive_weight,
                        "mining": {"type": "hard_positive"},
                        "pt": features["pt"],
                        "features": features,
                    }
                )
                positives_added += 1
        print(
            f"[mine] image {image_index}/{len(val_paths)} {image_path.name} "
            f"hard_positive={positives_added} hard_negative={negatives_added} elapsed={_elapsed(started)}",
            flush=True,
        )
        if max_hard_positives and positives_added >= max_hard_positives:
            if max_hard_negatives and negatives_added >= max_hard_negatives:
                break

    config["mining"] = {
        "source_weights": [str(path) for path in weight_paths],
        "val_images": str(val_images),
        "val_labels_dir": str(val_labels_dir),
        "hard_positive_weight": hard_positive_weight,
        "hard_negative_weight": hard_negative_weight,
        "hard_negative_max_iou": hard_negative_max_iou,
        "mined_prototype_count": mined_prototype_count,
        "hard_positive_count": positives_added,
        "hard_negative_count": negatives_added,
    }
    if mined_prototype_count > 0:
        config["prototype_count"] = max(int(config.get("prototype_count", DEFAULT_PROTOTYPE_COUNT)), mined_prototype_count)
    save_weights(mined_weights, output_path, config=config, compact=compact_weights, precision=weight_precision)
    print(
        f"[mine] done total_weights={len(mined_weights)} hard_positive={positives_added} "
        f"hard_negative={negatives_added} elapsed={_elapsed(started)}",
        flush=True,
    )
    return {
        "output_path": str(output_path),
        "total_weights": len(mined_weights),
        "hard_positive_count": positives_added,
        "hard_negative_count": negatives_added,
    }
__all__ = [
    '_metric_score',
    'run_mining_rounds',
    'mine_val_hard_examples',
]
