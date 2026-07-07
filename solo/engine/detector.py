from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.utils.visualization import *
from solo.core.features import *
from solo.core.accelerator import *
from solo.data.dataset import *
from solo.data.dataloader import *
from solo.models.matcher import *
from solo.engine.proposals import *
from solo.engine.rescore import *
from solo.engine.refinement import *
from solo.engine.evaluation import *
from solo.engine.image_detector import detect_image

def _parse_window_sizes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(value.strip()) for value in raw.split(",") if value.strip()]

def _parse_window_ratios(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    return [float(value.strip()) for value in raw.split(",") if value.strip()]

def run_detection(
    image_path: str | Path,
    weight_paths: list[str | Path],
    output_path: str | Path = DEFAULT_DETECTION_RESULTS,
    draw_dir: str | Path | None = None,
    hide_labels: bool = False,
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
    train_images: str | Path | None = None,
    val_images: str | Path | None = None,
    train_labels_dir: str | Path | None = None,
    val_labels_dir: str | Path | None = None,
    eval_labels_dir: str | Path | None = None,
    class_names_path: str | Path | None = None,
    annotation_format: str = "yolo",
    calibration_samples: int = 0,
    val_match_iou: float = DEFAULT_VAL_MATCH_IOU,
    val_duplicate_iou: float = DEFAULT_VAL_DUPLICATE_IOU,
    missing_label_score: float = DEFAULT_MISSING_LABEL_SCORE,
    max_calibrated_score_threshold: float = DEFAULT_MAX_CALIBRATED_SCORE_THRESHOLD,
    calibration_score_slack: float = DEFAULT_CALIBRATION_SCORE_SLACK,
    negative_penalty: float = DEFAULT_NEGATIVE_PENALTY,
    min_negative_margin: float = DEFAULT_MIN_NEGATIVE_MARGIN,
    self_calibrate: bool = False,
    self_calibration_samples: int = 0,
    self_calibration_beta: float = 2.0,
    self_calibration_min_threshold: float = 0.98,
    val_updates_threshold: bool = False,
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
    evaluate_proposals: bool = True,
    input_size: int = DEFAULT_INPUT_SIZE,
    backend: str = DEFAULT_DETECTOR_BACKEND,
    taichi_backend: str = DEFAULT_TAICHI_BACKEND,
    neural_device: str = DEFAULT_NEURAL_DEVICE,
    neural_image_size: int = DEFAULT_NEURAL_IMAGE_SIZE,
    neural_label: str = "0",
    neural_max_candidates: int = DEFAULT_NEURAL_MAX_CANDIDATES,
    neural_soft_nms_mode: str = DEFAULT_NEURAL_SOFT_NMS_MODE,
) -> dict[str, Any]:
    from solo.engine.calibration import calibrate_detection, self_calibrate_detection

    _validate_detector_backend(backend)
    if backend == "taichi":
        from solo.taichi_detector.pipeline import detect_images_taichi

        if len(weight_paths) != 1:
            raise ValueError("--backend taichi expects exactly one Taichi detector JSON in --detect")
        started = time.time()
        threshold = None if score_threshold == DEFAULT_SCORE_THRESHOLD else score_threshold
        active_nms = None if nms_threshold == DEFAULT_NMS_THRESHOLD else nms_threshold
        results, evaluation = detect_images_taichi(
            image_path,
            weight_paths[0],
            labels_dir=eval_labels_dir,
            output_path=output_path,
            draw_dir=draw_dir,
            hide_labels=hide_labels,
            score_threshold=threshold,
            nms_threshold=active_nms,
            max_detections=max_detections,
            backend=taichi_backend,
            annotation_format=annotation_format,
            class_names_path=class_names_path,
            match_iou=val_match_iou,
            duplicate_iou=val_duplicate_iou,
            min_detection_width=min_detection_width,
            min_detection_height=min_detection_height,
            min_detection_area=min_detection_area,
        )
        metadata = {
            "backend": "taichi",
            "weights": [str(path) for path in weight_paths],
            "score_threshold": threshold,
            "nms_threshold": active_nms,
            "taichi_backend": taichi_backend,
            "min_detection_width": min_detection_width,
            "min_detection_height": min_detection_height,
            "min_detection_area": min_detection_area,
            "max_detections": max_detections,
            "evaluation": evaluation,
            "total_images": len(results),
            "total_detections": sum(len(result["detections"]) for result in results),
        }
        report_path = Path(output_path)
        print(f"[detect] done backend=taichi images={len(results)} elapsed={_elapsed(started)}", flush=True)
        return {"metadata": metadata, "results": results, "report_path": str(report_path)}

    if backend == "neural":
        from solo.neural.inference import detect_images_neural

        if len(weight_paths) != 1:
            raise ValueError("--backend neural expects exactly one .pt checkpoint in --detect")
        started = time.time()
        threshold = score_threshold
        if threshold == DEFAULT_SCORE_THRESHOLD:
            threshold = -1.0
        active_nms = nms_threshold
        if active_nms == DEFAULT_NMS_THRESHOLD:
            active_nms = -1.0
        results, neural_metadata = detect_images_neural(
            image_path,
            weight_paths[0],
            device=neural_device,
            score_threshold=threshold,
            nms_threshold=active_nms,
            image_size=neural_image_size,
            label=neural_label,
            max_candidates=neural_max_candidates,
            soft_nms_mode=neural_soft_nms_mode,
            min_detection_width=min_detection_width,
            min_detection_height=min_detection_height,
            min_detection_area=min_detection_area,
            max_detections=max_detections,
        )
        if draw_dir is not None:
            for result in results:
                path = Path(result["image"])
                draw_path = Path(draw_dir) / path.name
                result["draw_path"] = str(draw_detections(path, result["detections"], draw_path, hide_labels=hide_labels))

        evaluation = {"enabled": False}
        if eval_labels_dir is not None:
            evaluation = evaluate_detection_results(
                results,
                eval_labels_dir,
                annotations=annotation_format,
                class_names_path=class_names_path,
                match_iou=val_match_iou,
                duplicate_iou=val_duplicate_iou,
                weight_index=None,
                evaluate_proposals=False,
                strict_labels=True,
            )

        metadata = {
            "backend": "neural",
            "weights": [str(path) for path in weight_paths],
            "score_threshold": neural_metadata.get("score_threshold", threshold),
            "nms_threshold": neural_metadata.get("nms_threshold", active_nms),
            "neural_device": neural_device,
            "neural_image_size": neural_image_size,
            "neural_label": neural_label,
            "neural_max_candidates": neural_max_candidates,
            "neural_soft_nms_mode": neural_soft_nms_mode,
            "neural": neural_metadata,
            "min_detection_width": min_detection_width,
            "min_detection_height": min_detection_height,
            "min_detection_area": min_detection_area,
            "max_detections": max_detections,
            "evaluation": evaluation,
            "total_images": len(results),
            "total_detections": sum(len(result["detections"]) for result in results),
        }
        report_path = save_detection_report(results, output_path, metadata)
        print(f"[detect] done backend=neural images={len(results)} elapsed={_elapsed(started)}", flush=True)
        return {"metadata": metadata, "results": results, "report_path": str(report_path)}

    weight_index = prepare_weight_index(weight_paths, accelerator=accelerator)
    print(f"[detect] loaded_weights={len(weight_index['entries'])}", flush=True)
    self_calibration = {"enabled": False}
    if self_calibrate:
        self_calibration = self_calibrate_detection(
            train_images,
            weight_index,
            annotations=annotation_format,
            train_labels_dir=train_labels_dir,
            class_names_path=class_names_path,
            proposal=proposal,
            score_floor=self_calibration_min_threshold,
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
            samples=self_calibration_samples,
            match_iou=val_match_iou,
            beta=self_calibration_beta,
            min_threshold=self_calibration_min_threshold,
            max_threshold=max_calibrated_score_threshold,
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
            accelerator=accelerator,
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
        if self_calibration.get("enabled"):
            score_threshold = self_calibration.get("score_threshold", score_threshold)
            min_negative_margin = self_calibration.get("learned_negative_margin", min_negative_margin)
    calibration = calibrate_detection(
        train_images,
        val_images,
        weight_index,
        annotations=annotation_format,
        train_labels_dir=train_labels_dir,
        val_labels_dir=val_labels_dir,
        class_names_path=class_names_path,
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
        max_samples=calibration_samples,
        val_match_iou=val_match_iou,
        val_duplicate_iou=val_duplicate_iou,
        missing_label_score=missing_label_score,
        max_calibrated_score_threshold=max_calibrated_score_threshold,
        calibration_score_slack=calibration_score_slack,
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
        accelerator=accelerator,
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
    if val_updates_threshold:
        score_threshold = calibration.get("score_threshold", score_threshold)
    bbox_scale = calibration.get("bbox_scale", bbox_scale)
    nms_threshold = calibration.get("nms_threshold", nms_threshold)

    results = []
    paths = _image_paths(image_path)
    started = time.time()
    for index, path in enumerate(paths, start=1):
        print(f"[detect] image {index}/{len(paths)} {path.name}", flush=True)
        result = detect_image(
            path,
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
            accelerator=accelerator,
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
        if draw_dir is not None:
            draw_path = Path(draw_dir) / path.name
            result["draw_path"] = str(draw_detections(path, result["detections"], draw_path, hide_labels=hide_labels))
        results.append(result)
        print(
            f"[detect] image {index}/{len(paths)} {path.name} proposals={result['proposal_count']} "
            f"detections={len(result['detections'])} elapsed={result['elapsed_seconds']}s",
            flush=True,
        )

    evaluation = {"enabled": False}
    if eval_labels_dir is not None:
        evaluation = evaluate_detection_results(
            results,
            eval_labels_dir,
            annotations=annotation_format,
            class_names_path=class_names_path,
            match_iou=val_match_iou,
            duplicate_iou=val_duplicate_iou,
            weight_index=weight_index,
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
            evaluate_proposals=evaluate_proposals,
            input_size=input_size,
        )

    metadata = {
        "backend": "solo",
        "weights": [str(path) for path in weight_paths],
        "proposal": proposal,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "bbox_scale": bbox_scale,
        "bbox_prior_mode": bbox_prior_mode,
        "match_mode": match_mode,
        "channel_mode": channel_mode,
        "channel_top_k": channel_top_k,
        "proposal_dedupe_iou": proposal_dedupe_iou,
        "max_proposals": max_proposals,
        "edge_proposals": edge_proposals,
        "body_proposals": body_proposals,
        "proposal_refine": proposal_refine,
        "accelerator": accelerator,
        "accelerator_status": accelerator_status(),
        "compact_weights": bool(weight_index.get("compact")),
        "second_stage_rescoring": second_stage_rescoring,
        "second_stage_threshold": second_stage_threshold,
        "second_stage_margin_weight": second_stage_margin_weight,
        "second_stage_support_weight": second_stage_support_weight,
        "second_stage_proposal_weight": second_stage_proposal_weight,
        "second_stage_quality_weight": second_stage_quality_weight,
        "second_stage_sky_region": second_stage_sky_region,
        "second_stage_sky_penalty": second_stage_sky_penalty,
        "min_detection_width": min_detection_width,
        "min_detection_height": min_detection_height,
        "min_detection_area": min_detection_area,
        "max_detections": max_detections,
        "min_refine_edge_gain": min_refine_edge_gain,
        "refine_boxes": refine_boxes,
        "refine_top_k": refine_top_k,
        "refine_edge_weight": refine_edge_weight,
        "refine_edge_gain": refine_edge_gain,
        "refine_rematch_top_k": refine_rematch_top_k,
        "refine_require_rematch": refine_require_rematch,
        "nms_containment_threshold": nms_containment_threshold,
        "cluster_nms_center_distance": cluster_nms_center_distance,
        "cluster_nms_containment": cluster_nms_containment,
        "structure_weight": structure_weight,
        "box_quality_weight": box_quality_weight,
        "min_box_quality": min_box_quality,
        "context_weight": context_weight,
        "min_context_quality": min_context_quality,
        "context_expand": context_expand,
        "fragmentation_weight": fragmentation_weight,
        "max_fragmentation": max_fragmentation,
        "box_quality": weight_index.get("box_quality"),
        "input_size": input_size,
        "window_ratios": window_ratios,
        "negative_penalty": negative_penalty,
        "min_negative_margin": min_negative_margin,
        "self_calibration": self_calibration,
        "calibration": calibration,
        "evaluation": evaluation,
        "total_images": len(results),
        "total_detections": sum(len(result["detections"]) for result in results),
    }
    report_path = save_detection_report(results, output_path, metadata)
    print(f"[detect] done images={len(results)} elapsed={_elapsed(started)}", flush=True)
    return {"metadata": metadata, "results": results, "report_path": str(report_path)}
__all__ = [
    '_edge_alignment_score',
    '_candidate_refine_boxes',
    '_snap_bbox_to_edges',
    '_match_crop',
    '_detection_from_match',
    'refine_detections',
    'detect_image',
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
    '_parse_window_sizes',
    '_parse_window_ratios',
    'run_detection',
]
