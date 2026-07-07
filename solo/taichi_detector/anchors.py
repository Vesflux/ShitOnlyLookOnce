from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from solo.utils.bbox import _bbox_payload, _clamp_bbox, bbox_iou


DEFAULT_PYRAMID_SCALES = (1.0, 0.75, 0.50, 0.35)
DEFAULT_ANCHOR_SIZES = (18, 24, 32, 44, 60, 82, 112, 152)
DEFAULT_ANCHOR_RATIOS = (0.65, 0.85, 1.0, 1.2, 1.35, 1.8, 2.4, 3.2)
DEFAULT_ANCHOR_STRIDE_RATIO = 0.58


def parse_float_list(raw: str | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return fallback
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    return values or fallback


def parse_int_list(raw: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return fallback
    values = tuple(int(float(item.strip())) for item in raw.split(",") if item.strip())
    return values or fallback


def generate_pyramid_anchors(
    image_width: int,
    image_height: int,
    scales: tuple[float, ...] = DEFAULT_PYRAMID_SCALES,
    sizes: tuple[int, ...] = DEFAULT_ANCHOR_SIZES,
    ratios: tuple[float, ...] = DEFAULT_ANCHOR_RATIOS,
    stride_ratio: float = DEFAULT_ANCHOR_STRIDE_RATIO,
    min_box_size: int = 6,
    max_anchors: int = 7000,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for scale in scales:
        if scale <= 0:
            continue
        scaled_width = max(1, round(image_width * scale))
        scaled_height = max(1, round(image_height * scale))
        for base_size in sizes:
            for ratio in ratios:
                if ratio <= 0:
                    continue
                anchor_w = base_size * math.sqrt(ratio)
                anchor_h = base_size / math.sqrt(ratio)
                if anchor_w < min_box_size or anchor_h < min_box_size:
                    continue
                if anchor_w > scaled_width * 0.95 or anchor_h > scaled_height * 0.95:
                    continue
                stride = max(6, round(min(anchor_w, anchor_h) * stride_ratio))
                max_x = max(0, scaled_width - round(anchor_w))
                max_y = max(0, scaled_height - round(anchor_h))
                x_positions = _scan_positions(max_x, stride)
                y_positions = _scan_positions(max_y, stride)
                for y in y_positions:
                    for x in x_positions:
                        mapped = _clamp_bbox(
                            (
                                x / scale,
                                y / scale,
                                (x + anchor_w) / scale,
                                (y + anchor_h) / scale,
                            ),
                            image_width,
                            image_height,
                        )
                        if mapped is not None and mapped not in seen:
                            seen.add(mapped)
                            anchors.append(
                                {
                                    "bbox": _bbox_payload(mapped, image_width, image_height),
                                    "source": "pyramid_anchor",
                                    "pyramid_scale": scale,
                                    "anchor_size": base_size,
                                    "anchor_ratio": ratio,
                                }
                            )
    if len(anchors) <= max_anchors:
        return anchors
    return sorted(anchors, key=lambda item: _anchor_rank(item, image_width, image_height), reverse=True)[:max_anchors]


def _scan_positions(max_position: int, stride: int) -> list[int]:
    if max_position <= 0:
        return [0]
    positions = list(range(0, max_position + 1, max(1, stride)))
    if positions[-1] != max_position:
        positions.append(max_position)
    return positions


def _anchor_rank(anchor: dict[str, Any], image_width: int, image_height: int) -> float:
    bbox = anchor["bbox"]
    width = max(1.0, float(bbox["width"]))
    height = max(1.0, float(bbox["height"]))
    aspect = width / height
    area_ratio = (width * height) / max(1.0, float(image_width * image_height))
    center_y = (float(bbox["y1"]) + float(bbox["y2"])) * 0.5 / max(1.0, float(image_height))
    aspect_score = max(
        math.exp(-abs(math.log(aspect / 1.25))) * 0.82,
        math.exp(-abs(math.log(aspect / 1.8))),
        math.exp(-abs(math.log(aspect / 2.45))) * 0.92,
    )
    small_anchor_enabled = int(anchor.get("anchor_size", 999)) <= 14
    area_targets = [
        (0.018, 1.00),
        (0.070, 0.90),
        (0.155, 0.82),
    ]
    if small_anchor_enabled:
        area_targets = [(0.0015, 0.86), (0.0060, 0.94), *area_targets]
    area_score = max(math.exp(-abs(math.log(max(area_ratio, 1e-5) / target))) * weight for target, weight in area_targets)
    road_score = 1.0 - abs(center_y - 0.62) / 0.62
    edge_bonus = 0.0
    if int(bbox["x1"]) <= 2 or int(bbox["x2"]) >= image_width - 2:
        edge_bonus += 0.08
    if int(bbox["y2"]) >= image_height - 2:
        edge_bonus += 0.04
    small_bonus = 0.08 if small_anchor_enabled and area_ratio <= 0.008 and center_y >= 0.28 else 0.0
    return aspect_score * 0.40 + area_score * 0.32 + max(0.0, road_score) * 0.20 + edge_bonus + small_bonus


def max_iou_with_truth(bbox: dict[str, Any], truth_boxes: list[dict[str, Any]]) -> float:
    if not truth_boxes:
        return 0.0
    return max(bbox_iou(bbox, truth["bbox"]) for truth in truth_boxes)


def quick_anchor_hardness(image: np.ndarray, bbox: dict[str, Any]) -> float:
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    edge = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))))
    texture = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1] / 255.0))
    return edge * 0.45 + texture * 0.35 + saturation * 0.20


def _integral_2d(values: np.ndarray) -> np.ndarray:
    integral = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.float32)
    integral[1:, 1:] = np.cumsum(np.cumsum(values.astype(np.float32), axis=0), axis=1)
    return integral


def build_anchor_hardness_index(planes: np.ndarray) -> dict[str, np.ndarray]:
    gray = planes[:, :, 0].astype(np.float32)
    saturation = planes[:, :, 1].astype(np.float32)
    gradient = planes[:, :, 4].astype(np.float32)
    return {
        "gray": _integral_2d(gray),
        "gray_sq": _integral_2d(gray * gray),
        "saturation": _integral_2d(saturation),
        "gradient": _integral_2d(gradient),
    }


def _mean_from_integral(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    left = max(0, min(width, x1))
    top = max(0, min(height, y1))
    right = max(0, min(width, x2))
    bottom = max(0, min(height, y2))
    if right <= left or bottom <= top:
        return 0.0
    area = float((right - left) * (bottom - top))
    total = float(integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left])
    return total / max(1.0, area)


def quick_anchor_hardness_from_index(index: dict[str, np.ndarray], bbox: dict[str, Any]) -> float:
    x1 = round(float(bbox["x1"]))
    y1 = round(float(bbox["y1"]))
    x2 = round(float(bbox["x2"]))
    y2 = round(float(bbox["y2"]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray_mean = _mean_from_integral(index["gray"], x1, y1, x2, y2)
    gray_sq_mean = _mean_from_integral(index["gray_sq"], x1, y1, x2, y2)
    texture = math.sqrt(max(0.0, gray_sq_mean - gray_mean * gray_mean))
    edge = _mean_from_integral(index["gradient"], x1, y1, x2, y2)
    saturation = _mean_from_integral(index["saturation"], x1, y1, x2, y2)
    return edge * 0.45 + texture * 0.35 + saturation * 0.20


__all__ = [
    "build_anchor_hardness_index",
    "DEFAULT_ANCHOR_RATIOS",
    "DEFAULT_ANCHOR_SIZES",
    "DEFAULT_ANCHOR_STRIDE_RATIO",
    "DEFAULT_PYRAMID_SCALES",
    "generate_pyramid_anchors",
    "max_iou_with_truth",
    "parse_float_list",
    "parse_int_list",
    "quick_anchor_hardness",
    "quick_anchor_hardness_from_index",
]
