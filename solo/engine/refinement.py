from __future__ import annotations

from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.core.features import *
from solo.data.dataset import *
from solo.models.matcher import *
from solo.utils.cv_image import Image

def _edge_alignment_score(image: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    gray = image.convert("L")
    pixels = gray.load()
    width, height = gray.size

    def sample_vertical(x: int) -> float:
        x = max(1, min(width - 2, x))
        values = []
        for y in range(max(1, y1), min(height - 1, y2)):
            values.append(abs(int(pixels[x + 1, y]) - int(pixels[x - 1, y])) / 255)
        return sum(values) / len(values) if values else 0.0

    def sample_horizontal(y: int) -> float:
        y = max(1, min(height - 2, y))
        values = []
        for x in range(max(1, x1), min(width - 1, x2)):
            values.append(abs(int(pixels[x, y + 1]) - int(pixels[x, y - 1])) / 255)
        return sum(values) / len(values) if values else 0.0

    return (sample_vertical(x1) + sample_vertical(x2 - 1) + sample_horizontal(y1) + sample_horizontal(y2 - 1)) / 4

def _candidate_refine_boxes(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    shifts = [0.0, -0.10, 0.10, -0.22, 0.22, -0.34, 0.34]
    scales = [0.58, 0.66, 0.75, 0.88, 1.0, 1.12, 1.28]
    aspect_scales = [
        (1.0, 1.0),
        (1.18, 0.88),
        (0.88, 1.18),
        (1.35, 0.78),
        (0.78, 1.35),
        (1.25, 0.66),
        (1.45, 0.72),
        (1.55, 0.95),
        (1.85, 1.02),
    ]
    candidates = []
    for shift_y in shifts:
        for shift_x in shifts:
            shifted_x = center_x + shift_x * width
            shifted_y = center_y + shift_y * height
            for scale in scales:
                for aspect_w, aspect_h in aspect_scales:
                    candidate_width = width * scale * aspect_w
                    candidate_height = height * scale * aspect_h
                    candidate = _clamp_bbox(
                        (
                            shifted_x - candidate_width / 2,
                            shifted_y - candidate_height / 2,
                            shifted_x + candidate_width / 2,
                            shifted_y + candidate_height / 2,
                        ),
                        image_width,
                        image_height,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return list(dict.fromkeys(candidates))

def _snap_bbox_to_edges(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    search_ratio: float = 0.18,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    search_x = max(2, round(width * search_ratio))
    search_y = max(2, round(height * search_ratio))
    gray = image.convert("L")
    pixels = gray.load()
    image_width, image_height = gray.size

    def vertical_strength(x: int) -> float:
        x = max(1, min(image_width - 2, x))
        values = [
            abs(int(pixels[x + 1, y]) - int(pixels[x - 1, y])) / 255
            for y in range(max(1, y1), min(image_height - 1, y2))
        ]
        return sum(values) / len(values) if values else 0.0

    def horizontal_strength(y: int) -> float:
        y = max(1, min(image_height - 2, y))
        values = [
            abs(int(pixels[x, y + 1]) - int(pixels[x, y - 1])) / 255
            for x in range(max(1, x1), min(image_width - 1, x2))
        ]
        return sum(values) / len(values) if values else 0.0

    left_candidates = range(max(0, x1 - search_x), min(x2 - 2, x1 + search_x) + 1)
    right_candidates = range(max(x1 + 2, x2 - search_x), min(image_width, x2 + search_x) + 1)
    top_candidates = range(max(0, y1 - search_y), min(y2 - 2, y1 + search_y) + 1)
    bottom_candidates = range(max(y1 + 2, y2 - search_y), min(image_height, y2 + search_y) + 1)
    snapped = (
        max(left_candidates, key=vertical_strength),
        max(top_candidates, key=horizontal_strength),
        max(right_candidates, key=lambda x: vertical_strength(x - 1)),
        max(bottom_candidates, key=lambda y: horizontal_strength(y - 1)),
    )
    return _clamp_bbox(snapped, image_width, image_height)

def _context_expand_bbox(
    bbox: tuple[int, int, int, int],
    context_quality: dict[str, Any] | None,
    image_width: int,
    image_height: int,
    step_ratio: float = 0.35,
) -> tuple[int, int, int, int] | None:
    if not context_quality:
        return None
    partialness = float(context_quality.get("partialness", 0.0) or 0.0)
    if partialness < 0.35:
        return None
    side_partialness = context_quality.get("side_partialness") or {}
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    left = x1 - width * step_ratio if float(side_partialness.get("left", 0.0) or 0.0) >= 0.30 else x1
    right = x2 + width * step_ratio if float(side_partialness.get("right", 0.0) or 0.0) >= 0.30 else x2
    top = y1 - height * step_ratio if float(side_partialness.get("top", 0.0) or 0.0) >= 0.30 else y1
    bottom = y2 + height * step_ratio if float(side_partialness.get("bottom", 0.0) or 0.0) >= 0.30 else y2
    expanded = _clamp_bbox((left, top, right, bottom), image_width, image_height)
    if expanded is None or expanded == bbox:
        return None
    return expanded

def _match_crop(
    crop: Image.Image,
    weight_index: dict[str, Any],
    config: dict[str, Any],
    channel_mode: str,
    channel_top_k: int,
    match_mode: str,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    box_quality_weight: float = DEFAULT_BOX_QUALITY_WEIGHT,
    min_box_quality: float = DEFAULT_MIN_BOX_QUALITY,
    excluded_source_keys: set[str] | None = None,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, Any] | None]:
    quality_enabled = _box_quality_enabled(box_quality_weight, min_box_quality)
    crop = _prepare_crop_for_features(crop, _crop_mode_from_config(config))
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
    used_channels = [
        {"name": channel, "weight": round(weight, 6)}
        for channel, weight in sorted(channel_weights.items(), key=lambda item: item[1], reverse=True)
    ]
    return match, used_channels, _box_quality_payload_from_features(features)

def _detection_from_match(
    match: dict[str, Any],
    used_channels: list[dict[str, float]],
    source_bbox: tuple[int, int, int, int],
    proposal_bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    proposal_width: int,
    proposal_height: int,
    weight_index: dict[str, Any],
    bbox_prior_mode: str,
    proposal_name: str,
    channel_mode: str,
    negative_penalty: float = DEFAULT_NEGATIVE_PENALTY,
    box_quality: dict[str, Any] | None = None,
    box_quality_weight: float = DEFAULT_BOX_QUALITY_WEIGHT,
    min_box_quality: float = DEFAULT_MIN_BOX_QUALITY,
    context_quality: dict[str, Any] | None = None,
    context_weight: float = DEFAULT_CONTEXT_WEIGHT,
    min_context_quality: float = DEFAULT_MIN_CONTEXT_QUALITY,
    context_expand: bool = DEFAULT_CONTEXT_EXPAND,
    fragmentation_quality: dict[str, Any] | None = None,
    fragmentation_weight: float = DEFAULT_FRAGMENTATION_WEIGHT,
    max_fragmentation: float = DEFAULT_MAX_FRAGMENTATION,
) -> dict[str, Any] | None:
    positive_score = match.get("positive_score", match["score"])
    positive_distance = match.get("positive_distance", match["distance"])
    negative_score = match.get("negative_score")
    negative_distance = match.get("negative_distance")
    negative_margin = positive_score - negative_score if negative_score is not None else positive_score
    label = match.get("positive_label", match["label"])
    adjusted_score = positive_score
    if negative_score is not None:
        adjusted_score = positive_score - max(0.0, -negative_margin) * negative_penalty
    prior_score = bbox_prior_score(
        _bbox_payload(proposal_bbox, proposal_width, proposal_height),
        weight_index.get("bbox_prior"),
        mode=bbox_prior_mode,
    )
    adjusted_score *= prior_score
    box_quality_raw, box_quality_score = _box_quality_score(box_quality, weight_index)
    if box_quality_score is not None and box_quality_score < min_box_quality:
        return None
    quality_multiplier = _box_quality_multiplier(box_quality_score, box_quality_weight)
    adjusted_score *= quality_multiplier
    context_score = float((context_quality or {}).get("quality", 1.0))
    if context_quality is not None and context_score < min_context_quality:
        return None
    context_multiplier = (1.0 - _clamp01(context_weight)) + _clamp01(context_weight) * _clamp01(context_score)
    adjusted_score *= context_multiplier
    fragmentation_score = float((fragmentation_quality or {}).get("fragmentation", 0.0))
    if fragmentation_quality is not None and max_fragmentation > 0 and fragmentation_score > max_fragmentation:
        return None
    fragmentation_multiplier = 1.0 - _clamp01(fragmentation_weight) * _clamp01(fragmentation_score)
    adjusted_score *= fragmentation_multiplier
    return {
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
        "used_channels": used_channels,
        "channel_mode": channel_mode,
        "proposal_prior_score": round(prior_score, 6),
        "box_quality_score": round(box_quality_score, 6) if box_quality_score is not None else None,
        "box_quality_raw_score": round(box_quality_raw, 6) if box_quality_raw is not None else None,
        "box_quality_multiplier": round(quality_multiplier, 6),
        "box_quality": box_quality,
        "context_quality_score": round(context_score, 6) if context_quality is not None else None,
        "context_quality_multiplier": round(context_multiplier, 6),
        "context_quality": context_quality,
        "fragmentation_score": round(fragmentation_score, 6) if fragmentation_quality is not None else None,
        "fragmentation_multiplier": round(fragmentation_multiplier, 6),
        "fragmentation_quality": fragmentation_quality,
        "bbox": _bbox_payload(source_bbox, image_width, image_height),
        "proposal_bbox": _bbox_payload(proposal_bbox, proposal_width, proposal_height),
        "proposal": proposal_name,
        "weight_path": match["weight_path"],
        "weight_index": match["weight_index"],
        "weight_source": match["weight_source"],
    }

def refine_detections(
    detections: list[dict[str, Any]],
    proposal_image: Image.Image,
    letterbox: dict[str, Any],
    weight_index: dict[str, Any],
    config: dict[str, Any],
    bbox_prior_mode: str,
    match_mode: str,
    channel_mode: str,
    channel_top_k: int,
    negative_penalty: float = DEFAULT_NEGATIVE_PENALTY,
    top_k: int = DEFAULT_REFINE_TOP_K,
    edge_weight: float = DEFAULT_REFINE_EDGE_WEIGHT,
    edge_gain_threshold: float = DEFAULT_REFINE_EDGE_GAIN,
    rematch_top_k: int = DEFAULT_REFINE_REMATCH_TOP_K,
    require_rematch: bool = DEFAULT_REFINE_REQUIRE_REMATCH,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    box_quality_weight: float = DEFAULT_BOX_QUALITY_WEIGHT,
    min_box_quality: float = DEFAULT_MIN_BOX_QUALITY,
    context_weight: float = DEFAULT_CONTEXT_WEIGHT,
    min_context_quality: float = DEFAULT_MIN_CONTEXT_QUALITY,
    context_expand: bool = DEFAULT_CONTEXT_EXPAND,
    fragmentation_weight: float = DEFAULT_FRAGMENTATION_WEIGHT,
    max_fragmentation: float = DEFAULT_MAX_FRAGMENTATION,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not detections:
        return detections
    width = int(letterbox.get("source_width", proposal_image.size[0]))
    height = int(letterbox.get("source_height", proposal_image.size[1]))
    proposal_width, proposal_height = proposal_image.size
    refined = []
    refine_targets = sorted(detections, key=lambda item: item.get("adjusted_score", item["score"]), reverse=True)[:top_k]
    refine_target_ids = {id(detection) for detection in refine_targets}
    untouched = [detection for detection in detections if id(detection) not in refine_target_ids]
    for detection in refine_targets:
        original_bbox = _bbox_tuple(detection["proposal_bbox"])
        best_detection = detection
        original_adjusted_score = float(detection.get("adjusted_score", detection["score"]))
        best_edge_score = _edge_alignment_score(proposal_image, original_bbox)
        best_score = original_adjusted_score + best_edge_score * edge_weight
        snapped_bbox = _snap_bbox_to_edges(proposal_image, original_bbox)
        candidate_bboxes = _candidate_refine_boxes(original_bbox, proposal_width, proposal_height)
        if snapped_bbox is not None:
            candidate_bboxes.insert(0, snapped_bbox)
        edge_candidates = []
        for candidate_bbox in candidate_bboxes:
            if candidate_bbox == original_bbox:
                continue
            source_bbox = _map_bbox_from_letterbox(candidate_bbox, letterbox)
            if source_bbox is None:
                continue
            edge_score = _edge_alignment_score(proposal_image, candidate_bbox)
            edge_gain = edge_score - best_edge_score
            if not require_rematch and candidate_bbox == snapped_bbox and edge_gain > edge_gain_threshold:
                snapped_detection = dict(detection)
                snapped_detection["bbox"] = _bbox_payload(source_bbox, width, height)
                snapped_detection["proposal_bbox"] = _bbox_payload(candidate_bbox, proposal_width, proposal_height)
                snapped_detection["proposal"] = f"{detection.get('proposal', 'proposal')}_edge_refined"
                snapped_detection["refined_from"] = detection["bbox"]
                snapped_detection["refine_edge_gain"] = round(edge_gain, 6)
                snapped_detection["refine_mode"] = "edge_snap"
                best_detection = snapped_detection
                best_score = max(best_score, original_adjusted_score + edge_score * edge_weight)
                best_edge_score = edge_score
                continue
            edge_candidates.append((edge_score, edge_gain, candidate_bbox, source_bbox))

        if rematch_top_k <= 0:
            refined.append(best_detection)
            continue

        edge_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        for edge_score, edge_gain, candidate_bbox, source_bbox in edge_candidates[:rematch_top_k]:
            crop = proposal_image.crop(candidate_bbox)
            match, used_channels, box_quality = _match_crop(
                crop,
                weight_index,
                config,
                channel_mode=channel_mode,
                channel_top_k=channel_top_k,
                match_mode=match_mode,
                structure_weight=structure_weight,
                box_quality_weight=box_quality_weight,
                min_box_quality=min_box_quality,
                accelerator=accelerator,
            )
            candidate_detection = _detection_from_match(
                match,
                used_channels,
                source_bbox,
                candidate_bbox,
                width,
                height,
                proposal_width,
                proposal_height,
                weight_index,
                bbox_prior_mode,
                f"{detection.get('proposal', 'proposal')}_refined",
                channel_mode,
                negative_penalty,
                box_quality,
                box_quality_weight,
                min_box_quality,
                context_quality_for_bbox(proposal_image, candidate_bbox)
                if context_weight > 0 or min_context_quality > 0
                else None,
                context_weight,
                min_context_quality,
                context_expand,
                fragmentation_quality_from_box_quality(box_quality)
                if fragmentation_weight > 0 or max_fragmentation > 0
                else None,
                fragmentation_weight,
                max_fragmentation,
            )
            if candidate_detection is None or candidate_detection["label"] != detection["label"]:
                continue
            original_area = max(1.0, float(detection["proposal_bbox"].get("width", 0)) * float(detection["proposal_bbox"].get("height", 0)))
            candidate_area = max(1.0, float(candidate_detection["proposal_bbox"].get("width", 0)) * float(candidate_detection["proposal_bbox"].get("height", 0)))
            if candidate_area < original_area * 0.68 and edge_gain < edge_gain_threshold * 1.8:
                continue
            adjusted_score = float(candidate_detection.get("adjusted_score", candidate_detection["score"]))
            score = adjusted_score + edge_score * edge_weight
            score_floor = original_adjusted_score - max(0.002, edge_weight * 0.5)
            margin_floor = float(detection.get("negative_margin", 0.0)) - max(0.01, edge_weight * 0.25)
            if require_rematch:
                score_floor = original_adjusted_score - max(0.001, edge_weight * 0.25)
                margin_floor = float(detection.get("negative_margin", 0.0)) - max(0.005, edge_weight * 0.10)
            candidate_margin = float(candidate_detection.get("negative_margin") or 0.0)
            if candidate_margin < margin_floor:
                continue
            if score > best_score or (edge_gain > edge_gain_threshold and adjusted_score >= score_floor):
                candidate_detection["refined_from"] = detection["bbox"]
                candidate_detection["refine_edge_gain"] = round(edge_gain, 6)
                candidate_detection["refine_mode"] = "confirmed_edge" if require_rematch else "rematch"
                best_detection = candidate_detection
                best_score = score
                best_edge_score = edge_score
        refined.append(best_detection)
    return refined + untouched
__all__ = [
    '_edge_alignment_score',
    '_candidate_refine_boxes',
    '_snap_bbox_to_edges',
    '_context_expand_bbox',
    '_match_crop',
    '_detection_from_match',
    'refine_detections',
]
