from __future__ import annotations

import math
from typing import Any

from solo.config import *
from solo.utils.bbox import *

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default

def _score_from_margin(detection: dict[str, Any]) -> float:
    positive = _safe_float(detection.get("positive_score"), _safe_float(detection.get("score"), 0.0))
    negative = detection.get("negative_score")
    if negative is None:
        return 1.0
    margin = positive - _safe_float(negative, 0.0)
    return max(0.0, min(1.0, margin / max(positive * 0.35, 1e-6)))

def _support_score(detection: dict[str, Any]) -> float:
    count = _safe_float(detection.get("positive_prototype_count"), 1.0)
    return max(0.0, min(1.0, math.log1p(max(0.0, count)) / math.log(10)))

def _proposal_score(detection: dict[str, Any], margin: float | None = None, quality: float | None = None) -> float:
    proposal = str(detection.get("proposal") or "")
    margin = _score_from_margin(detection) if margin is None else margin
    quality = _quality_score(detection) if quality is None else quality
    if proposal == "anchor_refined":
        if margin < DEFAULT_SECOND_STAGE_MIN_ANCHOR_MARGIN or quality < DEFAULT_SECOND_STAGE_MIN_ANCHOR_QUALITY:
            return 0.32
        return 0.92
    if "edge_refined" in proposal or proposal.endswith("_refined"):
        if margin < DEFAULT_SECOND_STAGE_MIN_ANCHOR_MARGIN * 0.5 and quality < DEFAULT_SECOND_STAGE_MIN_ANCHOR_QUALITY:
            return 0.55
        return 0.94
    if proposal in {"objectness", "color"}:
        return 0.82
    if proposal == "edge_component":
        return 0.70
    if proposal == "horizontal_body":
        return 0.66
    if proposal == "anchor":
        return 0.54 if margin < DEFAULT_SECOND_STAGE_MIN_ANCHOR_MARGIN else 0.66
    if proposal == "sliding":
        return 0.48
    return 0.60

def _quality_score(detection: dict[str, Any]) -> float:
    values = []
    if detection.get("box_quality_score") is not None:
        values.append(_safe_float(detection.get("box_quality_score"), 1.0))
    if detection.get("context_quality_score") is not None:
        values.append(_safe_float(detection.get("context_quality_score"), 1.0))
    if detection.get("fragmentation_score") is not None:
        values.append(1.0 - _safe_float(detection.get("fragmentation_score"), 0.0))
    if not values:
        return 1.0
    return max(0.0, min(1.0, sum(values) / len(values)))

def _vertical_prior_score(
    detection: dict[str, Any],
    sky_region: float = DEFAULT_SECOND_STAGE_SKY_REGION,
    sky_penalty: float = DEFAULT_SECOND_STAGE_SKY_PENALTY,
) -> float:
    if sky_region <= 0 or sky_penalty <= 0:
        return 1.0
    proposal = str(detection.get("proposal") or "")
    if "anchor" not in proposal:
        return 1.0
    bbox = detection.get("bbox") or {}
    normalized = bbox.get("normalized") or {}
    y_center = (float(normalized.get("y1", 0.0)) + float(normalized.get("y2", 0.0))) / 2
    height = float(normalized.get("y2", 0.0)) - float(normalized.get("y1", 0.0))
    if y_center <= 0:
        return 1.0
    if y_center < sky_region and height < sky_region * 0.80:
        depth = (sky_region - y_center) / max(sky_region, 1e-6)
        return max(0.05, 1.0 - _clamp01(sky_penalty) * (0.55 + depth * 0.45))
    return 1.0

def _shape_prior_score(detection: dict[str, Any]) -> float:
    diagnostics = detection.get("shape_diagnostics") or {}
    try:
        thin_vertical = float(diagnostics.get("thin_vertical_score", 0.0) or 0.0)
        road = float(diagnostics.get("road_score", 0.0) or 0.0)
        flat_road = float(diagnostics.get("flat_road_score", 0.0) or 0.0)
        vertical_fragment = float(diagnostics.get("vertical_fragment_score", 0.0) or 0.0)
        vehicle = float(diagnostics.get("vehicle_shape_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 1.0
    penalty = _clamp01(thin_vertical * 0.72 + road * 0.58 + flat_road * 0.88 + vertical_fragment * 0.72)
    bonus = vehicle * 0.08
    return max(0.08, min(1.08, 1.0 - penalty + bonus))

def second_stage_rescore_detections(
    detections: list[dict[str, Any]],
    enabled: bool = DEFAULT_SECOND_STAGE_RESCORING,
    threshold: float = DEFAULT_SECOND_STAGE_THRESHOLD,
    margin_weight: float = DEFAULT_SECOND_STAGE_MARGIN_WEIGHT,
    support_weight: float = DEFAULT_SECOND_STAGE_SUPPORT_WEIGHT,
    proposal_weight: float = DEFAULT_SECOND_STAGE_PROPOSAL_WEIGHT,
    quality_weight: float = DEFAULT_SECOND_STAGE_QUALITY_WEIGHT,
    sky_region: float = DEFAULT_SECOND_STAGE_SKY_REGION,
    sky_penalty: float = DEFAULT_SECOND_STAGE_SKY_PENALTY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return detections, {"enabled": False, "suppressed": 0}
    kept = []
    suppressed = 0
    total_aux_weight = max(0.0, margin_weight) + max(0.0, support_weight) + max(0.0, proposal_weight) + max(0.0, quality_weight)
    base_weight = max(0.0, 1.0 - total_aux_weight)
    normalizer = max(1e-9, base_weight + total_aux_weight)
    for detection in detections:
        adjusted = _safe_float(detection.get("adjusted_score"), _safe_float(detection.get("score"), 0.0))
        margin = _score_from_margin(detection)
        support = _support_score(detection)
        quality = _quality_score(detection)
        proposal = _proposal_score(detection, margin=margin, quality=quality)
        vertical_prior = _vertical_prior_score(detection, sky_region=sky_region, sky_penalty=sky_penalty)
        shape_prior = _shape_prior_score(detection)
        second_score = (
            adjusted * base_weight
            + margin * max(0.0, margin_weight)
            + support * max(0.0, support_weight)
            + proposal * max(0.0, proposal_weight)
            + quality * max(0.0, quality_weight)
        ) / normalizer
        second_score *= vertical_prior
        second_score *= shape_prior
        item = dict(detection)
        item["second_stage_score"] = round(second_score, 6)
        item["second_stage_features"] = {
            "base_score": round(adjusted, 6),
            "margin_score": round(margin, 6),
            "support_score": round(support, 6),
            "proposal_score": round(proposal, 6),
            "quality_score": round(quality, 6),
            "vertical_prior_score": round(vertical_prior, 6),
            "shape_prior_score": round(shape_prior, 6),
        }
        if threshold > 0 and second_score < threshold:
            suppressed += 1
            continue
        kept.append(item)
    kept.sort(key=lambda item: item.get("second_stage_score", item.get("adjusted_score", item.get("score", 0.0))), reverse=True)
    return kept, {
        "enabled": True,
        "threshold": threshold,
        "suppressed": suppressed,
        "input": len(detections),
        "output": len(kept),
        "weights": {
            "base": base_weight,
            "margin": margin_weight,
            "support": support_weight,
            "proposal": proposal_weight,
            "quality": quality_weight,
            "sky_region": sky_region,
            "sky_penalty": sky_penalty,
            "min_anchor_margin": DEFAULT_SECOND_STAGE_MIN_ANCHOR_MARGIN,
            "min_anchor_quality": DEFAULT_SECOND_STAGE_MIN_ANCHOR_QUALITY,
        },
    }

__all__ = [
    'second_stage_rescore_detections',
]
