from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from solo.config import *
from solo.core.features import *
from solo.data.dataset import _crop_mode_from_config, _prepare_crop_for_features, _stats_version_from_config
from solo.engine.proposals import generate_proposals
from solo.engine.proposal_diagnostics import proposal_diagnostics, shape_multiplier_from_diagnostics
from solo.engine.refinement import _context_expand_bbox, _match_crop, refine_detections
from solo.engine.rescore import second_stage_rescore_detections
from solo.models.matcher import _channel_selection, bbox_prior_score, match_pt
from solo.utils.bbox import *
from solo.utils.cv_image import Image


def detect_image(
    image_path: str | Path,
    weight_index: dict[str, Any],
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
    negative_penalty: float = DEFAULT_NEGATIVE_PENALTY,
    min_negative_margin: float = DEFAULT_MIN_NEGATIVE_MARGIN,
    excluded_source_keys: set[str] | None = None,
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
    _validate_bbox_prior_mode(bbox_prior_mode)
    _validate_match_mode(match_mode)
    _validate_channel_mode(channel_mode)
    _validate_accelerator(accelerator)
    path = Path(image_path)
    config = weight_index["config"]
    started = time.time()
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        proposal_image, letterbox = letterbox_image(image, input_size)
        proposal_width, proposal_height = proposal_image.size
        needs_context_maps = context_weight > 0 or min_context_quality > 0 or proposal_refine
        context_maps = context_maps_for_image(proposal_image) if needs_context_maps else None
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
            proposal_maps=context_maps,
        )
        detections = []
        negative_suppressed_count = 0
        quality_suppressed_count = 0
        context_suppressed_count = 0
        fragmentation_suppressed_count = 0
        shape_suppressed_count = 0
        score_suppressed_count = 0
        quality_enabled = _box_quality_enabled(box_quality_weight, min_box_quality)
        fragmentation_enabled = fragmentation_weight > 0 or max_fragmentation > 0
        for proposal_item in proposals:
            bbox = _bbox_tuple(proposal_item["bbox"])
            scaled_bbox = _expand_bbox(bbox, proposal_width, proposal_height, bbox_scale)
            if scaled_bbox is None:
                continue
            source_bbox = _map_bbox_from_letterbox(scaled_bbox, letterbox)
            if source_bbox is None:
                continue
            crop = _prepare_crop_for_features(proposal_image.crop(scaled_bbox), _crop_mode_from_config(config))
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
                structure_mode=_feature_structure_mode(config, quality_enabled),
                structure_grid=int(config.get("structure_grid", DEFAULT_STRUCTURE_GRID)),
                stats_version=_stats_version_from_config(config),
            )
            channel_weights = _channel_selection(
                features.get("channel_quality"),
                weight_index,
                channel_mode=channel_mode,
                channel_top_k=channel_top_k,
            )
            match = match_pt(
                features["vector"],
                weight_index,
                excluded_source_keys=excluded_source_keys,
                match_mode=match_mode,
                channel_weights=channel_weights,
                structure_vector=(features.get("structure") or {}).get("vector"),
                structure_weight=structure_weight,
                accelerator=accelerator,
            )
            positive_score = match.get("positive_score", match["score"])
            positive_distance = match.get("positive_distance", match["distance"])
            negative_score = match.get("negative_score")
            negative_distance = match.get("negative_distance")
            negative_margin = positive_score - negative_score if negative_score is not None else positive_score
            if negative_margin < min_negative_margin:
                negative_suppressed_count += 1
                continue

            label = match.get("positive_label", match["label"])
            adjusted_score = positive_score
            if negative_score is not None:
                adjusted_score = positive_score - max(0.0, -negative_margin) * negative_penalty
            prior_score = bbox_prior_score(
                _bbox_payload(scaled_bbox, proposal_width, proposal_height),
                weight_index.get("bbox_prior"),
                mode=bbox_prior_mode,
            )
            adjusted_score *= prior_score
            shape_diagnostics = proposal_item.get("diagnostics") or proposal_diagnostics(
                _bbox_payload(scaled_bbox, proposal_width, proposal_height),
                maps=context_maps,
                image_width=proposal_width,
                image_height=proposal_height,
            )
            shape_multiplier = shape_multiplier_from_diagnostics(shape_diagnostics)
            adjusted_score *= shape_multiplier
            if shape_multiplier <= 0.24 and negative_margin < max(min_negative_margin * 4.0, 0.03):
                shape_suppressed_count += 1
                continue
            box_quality = _box_quality_payload_from_features(features)
            box_quality_raw, box_quality_score = _box_quality_score(box_quality, weight_index)
            if box_quality_score is not None and box_quality_score < min_box_quality:
                quality_suppressed_count += 1
                continue
            quality_multiplier = _box_quality_multiplier(box_quality_score, box_quality_weight)
            adjusted_score *= quality_multiplier
            context_quality = context_quality_for_bbox_from_maps(context_maps, scaled_bbox) if context_maps is not None else None
            context_expanded_from = None
            context_expanded = False
            if context_expand and context_quality is not None and float(context_quality.get("partialness", 0.0) or 0.0) >= 0.35:
                expanded_bbox = _context_expand_bbox(scaled_bbox, context_quality, proposal_width, proposal_height)
                if expanded_bbox is not None:
                    expanded_match, expanded_channels, expanded_box_quality = _match_crop(
                        proposal_image.crop(expanded_bbox),
                        weight_index,
                        config,
                        channel_mode,
                        channel_top_k,
                        match_mode,
                        structure_weight=structure_weight,
                        box_quality_weight=box_quality_weight,
                        min_box_quality=min_box_quality,
                        accelerator=accelerator,
                        excluded_source_keys=excluded_source_keys,
                    )
                    expanded_positive = float(expanded_match.get("positive_score", expanded_match["score"]))
                    expanded_negative = expanded_match.get("negative_score")
                    expanded_margin = (
                        expanded_positive - float(expanded_negative)
                        if expanded_negative is not None
                        else expanded_positive
                    )
                    if (
                        expanded_match.get("positive_label", expanded_match["label"]) == label
                        and expanded_positive >= positive_score - 0.015
                        and expanded_margin >= negative_margin - 0.015
                    ):
                        expanded_source_bbox = _map_bbox_from_letterbox(expanded_bbox, letterbox)
                        if expanded_source_bbox is not None:
                            context_expanded_from = scaled_bbox
                            scaled_bbox = expanded_bbox
                            source_bbox = expanded_source_bbox
                            match = expanded_match
                            channel_weights = {
                                item["name"]: float(item["weight"])
                                for item in expanded_channels
                            }
                            box_quality = expanded_box_quality
                            positive_score = expanded_positive
                            positive_distance = expanded_match.get("positive_distance", expanded_match["distance"])
                            negative_score = expanded_match.get("negative_score")
                            negative_distance = expanded_match.get("negative_distance")
                            negative_margin = expanded_margin
                            label = expanded_match.get("positive_label", expanded_match["label"])
                            adjusted_score = positive_score
                            if negative_score is not None:
                                adjusted_score = positive_score - max(0.0, -negative_margin) * negative_penalty
                            prior_score = bbox_prior_score(
                                _bbox_payload(scaled_bbox, proposal_width, proposal_height),
                                weight_index.get("bbox_prior"),
                                mode=bbox_prior_mode,
                            )
                            adjusted_score *= prior_score
                            shape_diagnostics = proposal_diagnostics(
                                _bbox_payload(scaled_bbox, proposal_width, proposal_height),
                                maps=context_maps,
                                image_width=proposal_width,
                                image_height=proposal_height,
                            )
                            shape_multiplier = shape_multiplier_from_diagnostics(shape_diagnostics)
                            adjusted_score *= shape_multiplier
                            box_quality_raw, box_quality_score = _box_quality_score(box_quality, weight_index)
                            quality_multiplier = _box_quality_multiplier(box_quality_score, box_quality_weight)
                            adjusted_score *= quality_multiplier
                            context_quality = context_quality_for_bbox_from_maps(context_maps, scaled_bbox)
                            context_expanded = True
            context_score = float((context_quality or {}).get("quality", 1.0))
            if context_quality is not None and context_score < min_context_quality:
                context_suppressed_count += 1
                continue
            context_multiplier = (1.0 - _clamp01(context_weight)) + _clamp01(context_weight) * _clamp01(context_score)
            adjusted_score *= context_multiplier

            fragmentation_quality = fragmentation_quality_from_box_quality(box_quality) if fragmentation_enabled else None
            fragmentation_score = float((fragmentation_quality or {}).get("fragmentation", 0.0))
            if fragmentation_quality is not None and max_fragmentation > 0 and fragmentation_score > max_fragmentation:
                fragmentation_suppressed_count += 1
                continue
            fragmentation_multiplier = 1.0 - _clamp01(fragmentation_weight) * _clamp01(fragmentation_score)
            adjusted_score *= fragmentation_multiplier

            if adjusted_score < score_threshold:
                score_suppressed_count += 1
                continue
            detections.append(
                {
                    "label": label,
                    "score": round(positive_score, 6),
                    "adjusted_score": round(adjusted_score, 6),
                    "distance": round(positive_distance, 8),
                    "appearance_score": round(match["positive_appearance_score"], 6)
                    if match.get("positive_appearance_score") is not None
                    else None,
                    "structure_score": round(match["positive_structure_score"], 6)
                    if match.get("positive_structure_score") is not None
                    else None,
                    "structure_distance": round(match["positive_structure_distance"], 8)
                    if match.get("positive_structure_distance") is not None
                    else None,
                    "positive_score": round(positive_score, 6),
                    "positive_distance": round(positive_distance, 8),
                    "positive_appearance_score": round(match["positive_appearance_score"], 6)
                    if match.get("positive_appearance_score") is not None
                    else None,
                    "positive_structure_score": round(match["positive_structure_score"], 6)
                    if match.get("positive_structure_score") is not None
                    else None,
                    "negative_score": round(negative_score, 6) if negative_score is not None else None,
                    "negative_distance": round(negative_distance, 8) if negative_distance is not None else None,
                    "negative_appearance_score": round(match["negative_appearance_score"], 6)
                    if match.get("negative_appearance_score") is not None
                    else None,
                    "negative_structure_score": round(match["negative_structure_score"], 6)
                    if match.get("negative_structure_score") is not None
                    else None,
                    "negative_margin": round(negative_margin, 8),
                    "positive_prototype_count": match.get("positive_prototype_count"),
                    "negative_prototype_count": match.get("negative_prototype_count"),
                    "positive_prototype_cluster": match.get("positive_prototype_cluster"),
                    "negative_prototype_cluster": match.get("negative_prototype_cluster"),
                    "positive_prototype_kind": match.get("positive_prototype_kind"),
                    "negative_prototype_kind": match.get("negative_prototype_kind"),
                    "used_channels": [
                        {"name": channel, "weight": round(weight, 6)}
                        for channel, weight in sorted(channel_weights.items(), key=lambda item: item[1], reverse=True)
                    ],
                    "channel_mode": channel_mode,
                    "proposal_prior_score": round(prior_score, 6),
                    "shape_multiplier": round(shape_multiplier, 6),
                    "shape_diagnostics": shape_diagnostics,
                    "box_quality_score": round(box_quality_score, 6) if box_quality_score is not None else None,
                    "box_quality_raw_score": round(box_quality_raw, 6) if box_quality_raw is not None else None,
                    "box_quality_multiplier": round(quality_multiplier, 6),
                    "box_quality": box_quality,
                    "context_quality_score": round(context_score, 6) if context_quality is not None else None,
                    "context_quality_multiplier": round(context_multiplier, 6),
                    "context_quality": context_quality,
                    "context_expanded": context_expanded,
                    "context_expanded_from": _bbox_payload(context_expanded_from, proposal_width, proposal_height)
                    if context_expanded_from is not None
                    else None,
                    "fragmentation_score": round(fragmentation_score, 6) if fragmentation_quality is not None else None,
                    "fragmentation_multiplier": round(fragmentation_multiplier, 6),
                    "fragmentation_quality": fragmentation_quality,
                    "bbox": _bbox_payload(source_bbox, width, height),
                    "proposal_bbox": _bbox_payload(scaled_bbox, proposal_width, proposal_height),
                    "proposal": proposal_item.get("proposal", proposal),
                    "proposal_refined": bool(proposal_item.get("refined", False)),
                    "proposal_refined_from": proposal_item.get("proposal_refined_from"),
                    "proposal_refine_iou_with_source": proposal_item.get("refine_iou_with_source"),
                    "proposal_expanded_from": proposal_item.get("proposal_expanded_from"),
                    "weight_path": match["weight_path"],
                    "weight_index": match["weight_index"],
                    "weight_source": match["weight_source"],
                }
            )

    if refine_boxes:
        detections = refine_detections(
            detections,
            proposal_image,
            letterbox,
            weight_index,
            config,
            bbox_prior_mode=bbox_prior_mode,
            match_mode=match_mode,
            channel_mode=channel_mode,
            channel_top_k=channel_top_k,
            negative_penalty=negative_penalty,
            top_k=refine_top_k,
            edge_weight=refine_edge_weight,
            edge_gain_threshold=refine_edge_gain,
            rematch_top_k=refine_rematch_top_k,
            require_rematch=refine_require_rematch,
            structure_weight=structure_weight,
            box_quality_weight=box_quality_weight,
            min_box_quality=min_box_quality,
            context_weight=context_weight,
            min_context_quality=min_context_quality,
            context_expand=context_expand,
            fragmentation_weight=fragmentation_weight,
            max_fragmentation=max_fragmentation,
            accelerator=accelerator,
        )
    detections, second_stage_report = second_stage_rescore_detections(
        detections,
        enabled=second_stage_rescoring,
        threshold=second_stage_threshold,
        margin_weight=second_stage_margin_weight,
        support_weight=second_stage_support_weight,
        proposal_weight=second_stage_proposal_weight,
        quality_weight=second_stage_quality_weight,
        sky_region=second_stage_sky_region,
        sky_penalty=second_stage_sky_penalty,
    )
    final_detections = nms(
        detections,
        nms_threshold,
        containment_threshold=nms_containment_threshold,
        cluster_center_distance=cluster_nms_center_distance,
        cluster_containment_threshold=cluster_nms_containment,
    )
    final_detections, postprocess_suppressed = filter_detections(
        final_detections,
        min_width=min_detection_width,
        min_height=min_detection_height,
        min_area=min_detection_area,
        max_detections=max_detections,
        min_refine_edge_gain=min_refine_edge_gain,
    )
    return {
        "image": str(path),
        "width": width,
        "height": height,
        "letterbox": letterbox,
        "proposal_count": len(proposals),
        "raw_detection_count": len(detections),
        "negative_suppressed_count": negative_suppressed_count,
        "quality_suppressed_count": quality_suppressed_count,
        "context_suppressed_count": context_suppressed_count,
        "fragmentation_suppressed_count": fragmentation_suppressed_count,
        "shape_suppressed_count": shape_suppressed_count,
        "score_suppressed_count": score_suppressed_count,
        "second_stage": second_stage_report,
        "postprocess_suppressed": postprocess_suppressed,
        "detections": final_detections,
        "elapsed_seconds": round(time.time() - started, 4),
    }


__all__ = ["detect_image"]
