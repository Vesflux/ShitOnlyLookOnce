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

import cv2
import numpy as np

from solo.config import *
from solo.utils.cv_image import Image

def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))

def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)

def _clamp_bbox(bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    left = max(0, min(image_width, round(min(x1, x2))))
    top = max(0, min(image_height, round(min(y1, y2))))
    right = max(0, min(image_width, round(max(x1, x2))))
    bottom = max(0, min(image_height, round(max(y1, y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom

def _clamp_bbox_float(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    max_x = float(max(0, image_width))
    max_y = float(max(0, image_height))
    left = max(0.0, min(max_x, min(float(x1), float(x2))))
    top = max(0.0, min(max_y, min(float(y1), float(y2))))
    right = max(0.0, min(max_x, max(float(x1), float(x2))))
    bottom = max(0.0, min(max_y, max(float(y1), float(y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom

def _bbox_payload(bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> dict[str, Any]:
    left, top, right, bottom = bbox
    return {
        "x1": left,
        "y1": top,
        "x2": right,
        "y2": bottom,
        "width": right - left,
        "height": bottom - top,
        "normalized": {
            "x1": left / image_width,
            "y1": top / image_height,
            "x2": right / image_width,
            "y2": bottom / image_height,
        },
    }

def letterbox_image(image: Image.Image, input_size: int = DEFAULT_INPUT_SIZE) -> tuple[Image.Image, dict[str, Any]]:
    if input_size <= 0:
        width, height = image.size
        return image, {
            "enabled": False,
            "input_size": 0,
            "source_width": width,
            "source_height": height,
            "scale": 1.0,
            "pad_x": 0,
            "pad_y": 0,
            "resized_width": width,
            "resized_height": height,
        }
    if input_size < 16:
        raise ValueError("input_size must be 0 or at least 16")

    source_width, source_height = image.size
    scale = min(input_size / source_width, input_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.resize((resized_width, resized_height))
    boxed = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    boxed.paste(resized, (pad_x, pad_y))
    return boxed, {
        "enabled": True,
        "input_size": input_size,
        "source_width": source_width,
        "source_height": source_height,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }

def _map_bbox_from_letterbox(
    bbox: tuple[int, int, int, int],
    letterbox: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    if not letterbox.get("enabled"):
        return bbox
    scale = float(letterbox.get("scale", 1.0))
    if scale <= 0:
        return None
    pad_x = float(letterbox.get("pad_x", 0.0))
    pad_y = float(letterbox.get("pad_y", 0.0))
    mapped = (
        (bbox[0] - pad_x) / scale,
        (bbox[1] - pad_y) / scale,
        (bbox[2] - pad_x) / scale,
        (bbox[3] - pad_y) / scale,
    )
    return _clamp_bbox(
        mapped,
        int(letterbox.get("source_width", 0)),
        int(letterbox.get("source_height", 0)),
    )

def _map_bbox_to_letterbox(
    bbox: dict[str, Any],
    letterbox: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    source_width = int(letterbox.get("source_width", 0))
    source_height = int(letterbox.get("source_height", 0))
    if not letterbox.get("enabled"):
        return _clamp_bbox(
            _bbox_tuple(bbox),
            source_width or int(bbox.get("width", 0)),
            source_height or int(bbox.get("height", 0)),
        )
    scale = float(letterbox.get("scale", 1.0))
    pad_x = float(letterbox.get("pad_x", 0.0))
    pad_y = float(letterbox.get("pad_y", 0.0))
    mapped = (
        float(bbox["x1"]) * scale + pad_x,
        float(bbox["y1"]) * scale + pad_y,
        float(bbox["x2"]) * scale + pad_x,
        float(bbox["y2"]) * scale + pad_y,
    )
    return _clamp_bbox(
        mapped,
        int(letterbox.get("input_size", source_width)),
        int(letterbox.get("input_size", source_height)),
    )

def _bbox_tuple(box: dict[str, Any]) -> tuple[int, int, int, int]:
    return round(float(box["x1"])), round(float(box["y1"])), round(float(box["x2"])), round(float(box["y2"]))

def _bbox_float_tuple(box: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])

def _bbox_tuple_from_annotation(annotation: dict[str, Any]) -> tuple[int, int, int, int]:
    return _bbox_tuple(annotation["bbox"])

def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_float_tuple(a)
    bx1, by1, bx2, by2 = _bbox_float_tuple(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator else 0.0

def bbox_diou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_float_tuple(a)
    bx1, by1, bx2, by2 = _bbox_float_tuple(b)
    iou = bbox_iou(a, b)
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    center_distance_sq = (acx - bcx) * (acx - bcx) + (acy - bcy) * (acy - bcy)
    enclosing_left = min(ax1, bx1)
    enclosing_top = min(ay1, by1)
    enclosing_right = max(ax2, bx2)
    enclosing_bottom = max(ay2, by2)
    diagonal_sq = (enclosing_right - enclosing_left) ** 2 + (enclosing_bottom - enclosing_top) ** 2
    if diagonal_sq <= 1e-9:
        return iou
    return iou - center_distance_sq / diagonal_sq

def bbox_containment(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_float_tuple(a)
    bx1, by1, bx2, by2 = _bbox_float_tuple(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller_area = min(area_a, area_b)
    return intersection / smaller_area if smaller_area else 0.0

def bbox_center_distance_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_float_tuple(a)
    bx1, by1, bx2, by2 = _bbox_float_tuple(b)
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    diagonal = math.sqrt(max(ax2 - ax1, bx2 - bx1, 1) ** 2 + max(ay2 - ay1, by2 - by1, 1) ** 2)
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / max(diagonal, 1e-6)

def _nms_score(detection: dict[str, Any]) -> float:
    base = float(detection.get("second_stage_score", detection.get("adjusted_score", detection["score"])))
    quality = detection.get("box_quality_score")
    if quality is not None:
        base *= 0.96 + _clamp01(float(quality)) * 0.04
    context = detection.get("context_quality_score")
    if context is not None:
        base *= 0.97 + _clamp01(float(context)) * 0.03
    margin = float(detection.get("negative_margin") or 0.0)
    base *= 0.98 + _clamp01(margin * 8.0) * 0.02
    return base

def _cluster_nms_overlaps(
    detection: dict[str, Any],
    kept: dict[str, Any],
    center_distance_threshold: float,
    containment_threshold: float,
) -> bool:
    if detection.get("label") != kept.get("label"):
        return False
    containment = bbox_containment(detection["bbox"], kept["bbox"])
    if containment_threshold > 0 and containment >= containment_threshold:
        return True
    if center_distance_threshold <= 0:
        return False
    center_distance = bbox_center_distance_ratio(detection["bbox"], kept["bbox"])
    if center_distance > center_distance_threshold:
        return False
    area_a = max(1.0, float(detection["bbox"].get("width", 0)) * float(detection["bbox"].get("height", 0)))
    area_b = max(1.0, float(kept["bbox"].get("width", 0)) * float(kept["bbox"].get("height", 0)))
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    if area_ratio < 0.18:
        return containment >= max(0.35, containment_threshold * 0.75)
    return bbox_iou(detection["bbox"], kept["bbox"]) >= 0.08 or containment >= 0.35

def _bbox_iou_tuple(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator else 0.0

def nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.35,
    containment_threshold: float = DEFAULT_NMS_CONTAINMENT_THRESHOLD,
    cluster_center_distance: float = DEFAULT_CLUSTER_NMS_CENTER_DISTANCE,
    cluster_containment_threshold: float = DEFAULT_CLUSTER_NMS_CONTAINMENT,
    diou_threshold: float = 0.0,
    class_aware: bool = True,
) -> list[dict[str, Any]]:
    kept = []
    for detection in sorted(
        detections,
        key=_nms_score,
        reverse=True,
    ):
        overlaps_existing = False
        for item in kept:
            same_label = detection.get("label") == item.get("label")
            if class_aware and not same_label:
                continue
            iou = bbox_iou(detection["bbox"], item["bbox"])
            if iou > iou_threshold:
                overlaps_existing = True
                break
            if diou_threshold > 0 and bbox_diou(detection["bbox"], item["bbox"]) > diou_threshold:
                overlaps_existing = True
                break
            containment = bbox_containment(detection["bbox"], item["bbox"])
            if containment_threshold > 0 and containment > containment_threshold:
                overlaps_existing = True
                break
            if _cluster_nms_overlaps(
                detection,
                item,
                center_distance_threshold=cluster_center_distance,
                containment_threshold=cluster_containment_threshold,
            ):
                overlaps_existing = True
                break
        if not overlaps_existing:
            kept.append(detection)
    return kept

def soft_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.35,
    sigma: float = 0.50,
    score_threshold: float = 0.001,
    containment_threshold: float = DEFAULT_NMS_CONTAINMENT_THRESHOLD,
    diou_threshold: float = 0.0,
    class_aware: bool = True,
) -> list[dict[str, Any]]:
    pending = [dict(item) for item in detections]
    kept: list[dict[str, Any]] = []
    sigma = max(float(sigma), 1e-6)
    while pending:
        best_index = max(range(len(pending)), key=lambda index: _nms_score(pending[index]))
        best = pending.pop(best_index)
        if _nms_score(best) < score_threshold:
            break
        kept.append(best)
        survivors: list[dict[str, Any]] = []
        for detection in pending:
            same_label = detection.get("label") == best.get("label")
            if class_aware and not same_label:
                survivors.append(detection)
                continue
            iou = bbox_iou(detection["bbox"], best["bbox"])
            containment = bbox_containment(detection["bbox"], best["bbox"])
            suppress_like_duplicate = containment_threshold > 0 and containment > containment_threshold
            suppress_like_duplicate = suppress_like_duplicate or (diou_threshold > 0 and bbox_diou(detection["bbox"], best["bbox"]) > diou_threshold)
            if iou <= iou_threshold and not suppress_like_duplicate:
                survivors.append(detection)
                continue
            decay_source = max(iou, containment if suppress_like_duplicate else 0.0)
            decay = math.exp(-((decay_source * decay_source) / sigma))
            updated = dict(detection)
            updated_score = float(updated.get("adjusted_score", updated.get("score", 0.0))) * decay
            updated["score"] = round(updated_score, 6)
            updated["adjusted_score"] = round(updated_score, 6)
            updated["soft_nms_decay"] = round(decay, 6)
            if updated_score >= score_threshold:
                survivors.append(updated)
        pending = survivors
    return kept

def _bbox_area(box: dict[str, Any]) -> float:
    return max(0.0, float(box["x2"]) - float(box["x1"])) * max(0.0, float(box["y2"]) - float(box["y1"]))

def _bbox_union_payload(group: list[dict[str, Any]], image_width: int, image_height: int) -> dict[str, Any]:
    clamped = _clamp_bbox_float(
        (
            min(float(item["bbox"]["x1"]) for item in group),
            min(float(item["bbox"]["y1"]) for item in group),
            max(float(item["bbox"]["x2"]) for item in group),
            max(float(item["bbox"]["y2"]) for item in group),
        ),
        image_width,
        image_height,
    )
    if clamped is None:
        return _bbox_payload((0.0, 0.0, 1.0, 1.0), max(1, image_width), max(1, image_height))
    return _bbox_payload(clamped, image_width, image_height)

def _horizontal_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    if float(a["x2"]) < float(b["x1"]):
        return float(b["x1"]) - float(a["x2"])
    if float(b["x2"]) < float(a["x1"]):
        return float(a["x1"]) - float(b["x2"])
    return 0.0

def _vertical_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = min(float(a["y2"]), float(b["y2"])) - max(float(a["y1"]), float(b["y1"]))
    if overlap <= 0.0:
        return 0.0
    return overlap / max(1.0, min(float(a["height"]), float(b["height"])))

def _bbox_center(box: dict[str, Any]) -> tuple[float, float]:
    return (float(box["x1"]) + float(box["x2"])) * 0.5, (float(box["y1"]) + float(box["y2"])) * 0.5

def _looks_like_separate_instances(
    detection: dict[str, Any],
    group: list[dict[str, Any]],
    group_bbox: dict[str, Any],
    union_bbox: dict[str, Any],
) -> bool:
    box = detection["bbox"]
    width = max(1.0, float(box["width"]))
    height = max(1.0, float(box["height"]))
    group_width = max(1.0, float(group_bbox["width"]))
    group_height = max(1.0, float(group_bbox["height"]))
    union_width = max(1.0, float(union_bbox["width"]))
    union_height = max(1.0, float(union_bbox["height"]))
    height_ratio = min(height, group_height) / max(height, group_height)
    gap = _horizontal_gap(box, group_bbox)
    center_x, _center_y = _bbox_center(box)
    group_center_x, _group_center_y = _bbox_center(group_bbox)
    center_spread = abs(center_x - group_center_x)
    largest_area = max([_bbox_area(item["bbox"]) for item in group] + [_bbox_area(box)] + [1.0])
    growth = (union_width * union_height) / largest_area
    union_aspect = union_width / union_height
    already_vehicle_like = (
        width / height >= 1.05
        and group_width / group_height >= 1.05
        and min(width, group_width) >= 18.0
        and min(height, group_height) >= 10.0
    )
    distant_centers = center_spread >= max(18.0, min(width, group_width) * 0.70)
    clear_gap = gap >= max(10.0, min(width, group_width) * 0.22)
    large_union = growth >= 1.72 or union_width / max(width, group_width) >= 1.52
    if already_vehicle_like and height_ratio >= 0.58 and distant_centers and clear_gap and large_union:
        return True
    return len(group) >= 2 and union_aspect > 3.8 and growth > 1.55 and center_spread > min(width, group_width) * 0.55

def _distance_fusion_compatible(
    detection: dict[str, Any],
    group: list[dict[str, Any]],
    group_bbox: dict[str, Any],
    union_bbox: dict[str, Any],
    image_width: int,
    image_height: int,
    max_gap_ratio: float,
    min_vertical_overlap: float,
    max_area_ratio: float,
    max_growth: float,
) -> bool:
    if detection.get("label") != group[0].get("label"):
        return False
    box = detection["bbox"]
    width = max(1.0, float(box["width"]))
    height = max(1.0, float(box["height"]))
    group_width = max(1.0, float(group_bbox["width"]))
    group_height = max(1.0, float(group_bbox["height"]))
    union_width = max(1.0, float(union_bbox["width"]))
    union_height = max(1.0, float(union_bbox["height"]))

    height_ratio = min(height, group_height) / max(height, group_height)
    if height_ratio < 0.42:
        return False
    if _vertical_overlap_ratio(box, group_bbox) < min_vertical_overlap:
        return False
    _center_x, center_y = _bbox_center(box)
    _group_center_x, group_center_y = _bbox_center(group_bbox)
    if abs(center_y - group_center_y) > max(height, group_height) * 0.46:
        return False

    gap = _horizontal_gap(box, group_bbox)
    allowed_gap = max(12.0, min(width, group_width) * max_gap_ratio, max(width, group_width) * 0.20)
    if gap > allowed_gap:
        return False
    if union_height / max(height, group_height) > 1.40:
        return False
    union_aspect = union_width / union_height
    if union_aspect < 1.12 or union_aspect > 7.6:
        return False

    image_area = max(1.0, float(image_width * image_height))
    if (union_width * union_height) / image_area > max_area_ratio:
        return False
    if union_width > float(image_width) * 0.84 and union_height > float(image_height) * 0.36:
        return False
    largest_area = max([_bbox_area(item["bbox"]) for item in group] + [_bbox_area(box)] + [1.0])
    if (union_width * union_height) / largest_area > max_growth:
        return False
    if _looks_like_separate_instances(detection, group, group_bbox, union_bbox):
        return False
    if bbox_iou(box, group_bbox) > 0.02 or bbox_containment(box, group_bbox) > 0.10:
        return True
    return gap <= max(10.0, min(width, group_width) * 0.34)

def distance_box_fusion(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    max_gap_ratio: float = 0.48,
    min_vertical_overlap: float = 0.34,
    max_area_ratio: float = 0.30,
    max_growth: float = 2.45,
    proposal_name: str = "distance_box_fused",
) -> tuple[list[dict[str, Any]], int]:
    if len(detections) < 2:
        return detections, 0
    groups: list[list[dict[str, Any]]] = []
    for detection in sorted(detections, key=_nms_score, reverse=True):
        target: list[dict[str, Any]] | None = None
        for group in groups:
            group_bbox = _bbox_union_payload(group, image_width, image_height)
            union_bbox = _bbox_union_payload([*group, detection], image_width, image_height)
            if _distance_fusion_compatible(
                detection,
                group,
                group_bbox,
                union_bbox,
                image_width,
                image_height,
                max_gap_ratio,
                min_vertical_overlap,
                max_area_ratio,
                max_growth,
            ):
                target = group
                break
        if target is None:
            groups.append([detection])
        else:
            target.append(detection)

    fused: list[dict[str, Any]] = []
    fused_count = 0
    for group in groups:
        if len(group) == 1:
            fused.append(group[0])
            continue
        best = max(group, key=_nms_score)
        item = dict(best)
        item["bbox"] = _bbox_union_payload(group, image_width, image_height)
        item["score"] = round(max(float(member.get("score", 0.0)) for member in group), 6)
        item["adjusted_score"] = round(max(float(member.get("adjusted_score", member.get("score", 0.0))) for member in group), 6)
        item["proposal"] = proposal_name
        item["fused_count"] = len(group)
        item["fused_components"] = [
            {
                "bbox": member["bbox"],
                "score": round(float(member.get("adjusted_score", member.get("score", 0.0))), 6),
                "proposal": member.get("proposal"),
            }
            for member in sorted(group, key=lambda member: float(member["bbox"]["x1"]))
        ]
        fused_count += len(group) - 1
        fused.append(item)
    return fused, fused_count

def _fusion_components_should_split(parent: dict[str, Any], components: list[dict[str, Any]]) -> bool:
    if len(components) < 2:
        return False
    parent_box = parent["bbox"]
    parent_width = max(1.0, float(parent_box["width"]))
    parent_height = max(1.0, float(parent_box["height"]))
    parent_area = parent_width * parent_height
    boxes = [item.get("bbox") for item in components if isinstance(item.get("bbox"), dict)]
    if len(boxes) < 2:
        return False
    largest_area = max(max(1.0, _bbox_area(box)) for box in boxes)
    if parent_area / largest_area < 1.52:
        return False
    ordered = sorted(boxes, key=lambda box: _bbox_center(box)[0])
    for left, right in zip(ordered, ordered[1:]):
        left_width = max(1.0, float(left["width"]))
        right_width = max(1.0, float(right["width"]))
        left_height = max(1.0, float(left["height"]))
        right_height = max(1.0, float(right["height"]))
        left_aspect = left_width / left_height
        right_aspect = right_width / right_height
        if min(left_width, right_width) < 18.0 or min(left_height, right_height) < 10.0:
            continue
        if left_aspect < 1.05 or right_aspect < 1.05:
            continue
        gap = _horizontal_gap(left, right)
        if gap < max(10.0, min(left_width, right_width) * 0.22):
            continue
        center_spread = abs(_bbox_center(left)[0] - _bbox_center(right)[0])
        if center_spread < max(18.0, min(left_width, right_width) * 0.70):
            continue
        if _vertical_overlap_ratio(left, right) < 0.42:
            continue
        return True
    return False

def split_overmerged_fusions(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    proposal_suffix: str = "split_component",
) -> tuple[list[dict[str, Any]], int]:
    split: list[dict[str, Any]] = []
    split_count = 0
    for detection in detections:
        components = detection.get("fused_components")
        if not isinstance(components, list) or not _fusion_components_should_split(detection, components):
            split.append(detection)
            continue
        split_count += 1
        for component in components:
            bbox = component.get("bbox") if isinstance(component, dict) else None
            if not isinstance(bbox, dict):
                continue
            clamped = _clamp_bbox_float(_bbox_float_tuple(bbox), image_width, image_height)
            if clamped is None:
                continue
            score = round(float(component.get("score", detection.get("adjusted_score", detection.get("score", 0.0)))), 6)
            item = dict(detection)
            item["bbox"] = _bbox_payload(clamped, image_width, image_height)
            item["score"] = score
            item["adjusted_score"] = score
            item["proposal"] = f"{detection.get('proposal', 'fusion')}_{proposal_suffix}"
            item["split_from_fused"] = detection.get("proposal", "fusion")
            item.pop("fused_components", None)
            item.pop("fused_count", None)
            split.append(item)
    return split, split_count

def _edge_maps_from_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(image)
    if source.ndim == 2:
        gray = source.astype(np.float32)
        if gray.max(initial=0.0) > 1.5:
            gray = gray / 255.0
    else:
        if source.shape[2] == 1:
            gray = source[:, :, 0].astype(np.float32)
            if gray.max(initial=0.0) > 1.5:
                gray = gray / 255.0
        else:
            gray = cv2.cvtColor(source[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return grad_x, grad_y

def _mean_vertical_edge(grad_x: np.ndarray, x: int, y1: int, y2: int) -> float:
    height, width = grad_x.shape[:2]
    x = max(0, min(width - 1, int(round(x))))
    top = max(0, min(height, int(round(y1))))
    bottom = max(0, min(height, int(round(y2))))
    if bottom <= top:
        return 0.0
    return float(np.mean(grad_x[top:bottom, x]))

def _mean_horizontal_edge(grad_y: np.ndarray, y: int, x1: int, x2: int) -> float:
    height, width = grad_y.shape[:2]
    y = max(0, min(height - 1, int(round(y))))
    left = max(0, min(width, int(round(x1))))
    right = max(0, min(width, int(round(x2))))
    if right <= left:
        return 0.0
    return float(np.mean(grad_y[y, left:right]))

def _edge_alignment_score_from_maps(
    bbox: tuple[int, int, int, int],
    grad_x: np.ndarray,
    grad_y: np.ndarray,
) -> float:
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    trim_y = max(0, round((y2 - y1) * 0.12))
    trim_x = max(0, round((x2 - x1) * 0.12))
    top = y1 + trim_y
    bottom = y2 - trim_y
    left = x1 + trim_x
    right = x2 - trim_x
    if bottom <= top:
        top, bottom = y1, y2
    if right <= left:
        left, right = x1, x2
    return (
        _mean_vertical_edge(grad_x, x1, top, bottom)
        + _mean_vertical_edge(grad_x, x2 - 1, top, bottom)
        + _mean_horizontal_edge(grad_y, y1, left, right)
        + _mean_horizontal_edge(grad_y, y2 - 1, left, right)
    ) / 4.0

def snap_bbox_to_edges(
    image: np.ndarray,
    bbox: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    search_px: int = 5,
    min_gain: float = 0.015,
    min_area_ratio: float = 0.55,
    max_area_ratio: float = 1.12,
    edge_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], float]:
    original = _clamp_bbox(_bbox_float_tuple(bbox), image_width, image_height)
    if original is None:
        return bbox, 0.0
    x1, y1, x2, y2 = original
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 8 or box_height < 6:
        return bbox, 0.0
    grad_x, grad_y = edge_maps or _edge_maps_from_image(image)
    search_x = max(1, int(search_px))
    search_y = max(1, int(search_px))
    trim_y = max(0, round(box_height * 0.12))
    trim_x = max(0, round(box_width * 0.12))
    vertical_top = y1 + trim_y
    vertical_bottom = y2 - trim_y
    horizontal_left = x1 + trim_x
    horizontal_right = x2 - trim_x
    if vertical_bottom <= vertical_top:
        vertical_top, vertical_bottom = y1, y2
    if horizontal_right <= horizontal_left:
        horizontal_left, horizontal_right = x1, x2

    left_candidates = range(max(0, x1 - search_x), min(x2 - 3, x1 + search_x) + 1)
    right_candidates = range(max(x1 + 3, x2 - search_x), min(image_width, x2 + search_x) + 1)
    top_candidates = range(max(0, y1 - search_y), min(y2 - 3, y1 + search_y) + 1)
    bottom_candidates = range(max(y1 + 3, y2 - search_y), min(image_height, y2 + search_y) + 1)
    if not left_candidates or not right_candidates or not top_candidates or not bottom_candidates:
        return bbox, 0.0

    snapped = (
        max(left_candidates, key=lambda x: _mean_vertical_edge(grad_x, x, vertical_top, vertical_bottom)),
        max(top_candidates, key=lambda y: _mean_horizontal_edge(grad_y, y, horizontal_left, horizontal_right)),
        max(right_candidates, key=lambda x: _mean_vertical_edge(grad_x, x - 1, vertical_top, vertical_bottom)),
        max(bottom_candidates, key=lambda y: _mean_horizontal_edge(grad_y, y - 1, horizontal_left, horizontal_right)),
    )
    clamped = _clamp_bbox(snapped, image_width, image_height)
    if clamped is None or clamped == original:
        return bbox, 0.0

    original_area = max(1.0, float(box_width * box_height))
    snapped_area = max(1.0, float((clamped[2] - clamped[0]) * (clamped[3] - clamped[1])))
    area_ratio = snapped_area / original_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return bbox, 0.0

    original_score = _edge_alignment_score_from_maps(original, grad_x, grad_y)
    snapped_score = _edge_alignment_score_from_maps(clamped, grad_x, grad_y)
    gain = snapped_score - original_score
    if gain < min_gain:
        return bbox, 0.0
    return _bbox_payload(clamped, image_width, image_height), float(gain)

def snap_detections_to_edges(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    search_px: int = 5,
    min_gain: float = 0.015,
) -> tuple[list[dict[str, Any]], int]:
    if not detections:
        return detections, 0
    edge_maps = _edge_maps_from_image(image)
    snapped_detections: list[dict[str, Any]] = []
    snapped_count = 0
    for detection in detections:
        bbox = detection.get("bbox")
        if not isinstance(bbox, dict):
            snapped_detections.append(detection)
            continue
        snapped_bbox, gain = snap_bbox_to_edges(
            image,
            bbox,
            image_width,
            image_height,
            search_px=search_px,
            min_gain=min_gain,
            edge_maps=edge_maps,
        )
        if gain <= 0.0:
            snapped_detections.append(detection)
            continue
        item = dict(detection)
        item["bbox"] = snapped_bbox
        item["edge_snapped_from"] = bbox
        item["edge_snap_gain"] = round(gain, 6)
        item["proposal"] = f"{item.get('proposal', 'detection')}_edge_snapped"
        snapped_detections.append(item)
        snapped_count += 1
    return snapped_detections, snapped_count

def filter_detections(
    detections: list[dict[str, Any]],
    min_width: int = DEFAULT_MIN_DETECTION_WIDTH,
    min_height: int = DEFAULT_MIN_DETECTION_HEIGHT,
    min_area: int = DEFAULT_MIN_DETECTION_AREA,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    min_refine_edge_gain: float = DEFAULT_MIN_REFINE_EDGE_GAIN,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept = []
    suppressed = {
        "small_width": 0,
        "small_height": 0,
        "small_area": 0,
        "weak_refine_edge": 0,
        "max_detections": 0,
    }
    for detection in sorted(detections, key=_nms_score, reverse=True):
        bbox = detection.get("bbox") or {}
        width = int(bbox.get("width", 0))
        height = int(bbox.get("height", 0))
        area = width * height
        if min_width > 0 and width < min_width:
            suppressed["small_width"] += 1
            continue
        if min_height > 0 and height < min_height:
            suppressed["small_height"] += 1
            continue
        if min_area > 0 and area < min_area:
            suppressed["small_area"] += 1
            continue
        if min_refine_edge_gain > 0 and float(detection.get("refine_edge_gain") or 0.0) < min_refine_edge_gain:
            suppressed["weak_refine_edge"] += 1
            continue
        kept.append(detection)

    if max_detections > 0 and len(kept) > max_detections:
        suppressed["max_detections"] = len(kept) - max_detections
        kept = kept[:max_detections]
    return kept, suppressed
__all__ = [
    '_clamp01',
    '_smoothstep',
    '_clamp_bbox',
    '_clamp_bbox_float',
    '_bbox_payload',
    'letterbox_image',
    '_map_bbox_from_letterbox',
    '_map_bbox_to_letterbox',
    '_bbox_tuple',
    '_bbox_float_tuple',
    '_bbox_tuple_from_annotation',
    'bbox_iou',
    'bbox_diou',
    'bbox_containment',
    'bbox_center_distance_ratio',
    'distance_box_fusion',
    'split_overmerged_fusions',
    'snap_bbox_to_edges',
    'snap_detections_to_edges',
    '_nms_score',
    '_cluster_nms_overlaps',
    '_bbox_iou_tuple',
    'nms',
    'soft_nms',
    'filter_detections',
]
