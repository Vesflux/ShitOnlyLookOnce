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
from solo.core.ops import *
from solo.utils.cv_image import Image

def _edge_values(gray_values: list[float], source_size: int) -> list[float]:
    edges = []
    for y in range(source_size):
        for x in range(source_size):
            center = gray_values[y * source_size + x]
            right = gray_values[y * source_size + min(source_size - 1, x + 1)]
            down = gray_values[min(source_size - 1, y + 1) * source_size + x]
            edges.append(min(1.0, abs(center - right) + abs(center - down)))
    return edges

def _local_contrast_values(values: list[float], source_size: int) -> list[float]:
    contrasts = []
    for y in range(source_size):
        for x in range(source_size):
            neighbors = []
            for ny in range(max(0, y - 1), min(source_size, y + 2)):
                for nx in range(max(0, x - 1), min(source_size, x + 2)):
                    neighbors.append(values[ny * source_size + nx])
            center = values[y * source_size + x]
            local_mean = sum(neighbors) / len(neighbors)
            contrasts.append(min(1.0, abs(center - local_mean) * 2.0))
    return contrasts

def _texture_values(values: list[float], source_size: int) -> list[float]:
    textures = []
    for y in range(source_size):
        for x in range(source_size):
            neighbors = []
            for ny in range(max(0, y - 1), min(source_size, y + 2)):
                for nx in range(max(0, x - 1), min(source_size, x + 2)):
                    neighbors.append(values[ny * source_size + nx])
            mean = sum(neighbors) / len(neighbors)
            variance = sum((value - mean) ** 2 for value in neighbors) / len(neighbors)
            textures.append(min(1.0, math.sqrt(variance) * 3.0))
    return textures

def _lab_b_values(rgb_pixels: list[tuple[int, int, int]]) -> list[float]:
    values = []
    for red, green, blue in rgb_pixels:
        # Lightweight Lab-b proxy: positive values mean yellow/blue separation without pulling in heavy color libraries.
        b_proxy = ((red / 255) + (green / 255)) * 0.5 - (blue / 255)
        values.append(max(0.0, min(1.0, (b_proxy + 1.0) / 2.0)))
    return values

def _channel_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "range": 0.0, "variance": 0.0}
    mean = sum(values) / len(values)
    minimum = min(values)
    maximum = max(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "range": maximum - minimum,
        "variance": variance,
    }

def _channel_layout_from_names(channels: list[str], pt_size: int, stats_length: int = 10) -> dict[str, dict[str, int]]:
    layout: dict[str, dict[str, int]] = {}
    offset = 0
    channel_length = pt_size * pt_size
    for channel in channels:
        layout[channel] = {"start": offset, "end": offset + channel_length, "length": channel_length}
        offset += channel_length
    layout["stats"] = {"start": offset, "end": offset + stats_length, "length": stats_length}
    return layout

def _feature_geometry_stats(
    width: int,
    height: int,
    red_values: list[float],
    green_values: list[float],
    blue_values: list[float],
    hue_values: list[float],
    sat_values: list[float],
    val_values: list[float],
    yellow_mask: list[float],
    edge_values: list[float],
    local_contrast_values: list[float],
    texture_values: list[float],
) -> list[float]:
    aspect = width / max(1, height)
    inverse_aspect = height / max(1, width)
    long_side = max(width, height, 1)
    short_side = max(1, min(width, height))
    square_fill = (width * height) / max(1, long_side * long_side)
    log_aspect = abs(math.log(max(aspect, 1e-6)))
    wide_score = _smoothstep(1.25, 3.0, aspect)
    tall_score = _smoothstep(1.25, 3.0, inverse_aspect)
    elongation = 1.0 - short_side / long_side
    return [
        sum(red_values) / len(red_values),
        sum(green_values) / len(green_values),
        sum(blue_values) / len(blue_values),
        sum(hue_values) / len(hue_values),
        sum(sat_values) / len(sat_values),
        sum(val_values) / len(val_values),
        sum(yellow_mask) / len(yellow_mask),
        sum(edge_values) / len(edge_values),
        sum(local_contrast_values) / len(local_contrast_values),
        sum(texture_values) / len(texture_values),
        min(1.0, aspect / 4),
        min(1.0, inverse_aspect / 4),
        min(1.0, log_aspect / 2.0),
        wide_score,
        tall_score,
        _clamp01(elongation),
        _clamp01(square_fill),
    ]

def _legacy_geometry_stats(
    width: int,
    height: int,
    red_values: list[float],
    green_values: list[float],
    blue_values: list[float],
    hue_values: list[float],
    sat_values: list[float],
    val_values: list[float],
    yellow_mask: list[float],
    edge_values: list[float],
) -> list[float]:
    aspect = width / max(1, height)
    return [
        sum(red_values) / len(red_values),
        sum(green_values) / len(green_values),
        sum(blue_values) / len(blue_values),
        sum(hue_values) / len(hue_values),
        sum(sat_values) / len(sat_values),
        sum(val_values) / len(val_values),
        sum(yellow_mask) / len(yellow_mask),
        sum(edge_values) / len(edge_values),
        min(1.0, aspect / 4),
        min(1.0, (1 / max(aspect, 1e-6)) / 4),
    ]

def _feature_quality_from_stats(stats: dict[str, dict[str, float]], channels: list[str]) -> dict[str, float]:
    qualities = {}
    for channel in channels:
        payload = stats.get(channel, {})
        channel_range = float(payload.get("range", 0.0))
        variance = float(payload.get("variance", 0.0))
        qualities[channel] = max(0.05, min(1.0, channel_range * 0.65 + math.sqrt(max(0.0, variance)) * 1.5))
    return qualities

def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number

def _score_summary(values: list[float]) -> dict[str, float | int]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {"count": 0}
    count = len(clean)

    def percentile(position: float) -> float:
        index = (count - 1) * position
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return clean[int(index)]
        weight = index - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight

    return {
        "count": count,
        "min": clean[0],
        "max": clean[-1],
        "mean": sum(clean) / count,
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
    }

def _box_quality_payload_from_features(features: dict[str, Any] | None) -> dict[str, Any] | None:
    if not features:
        return None
    structure_payload = features.get("structure") or {}
    quality = structure_payload.get("quality") if isinstance(structure_payload, dict) else None
    return quality if isinstance(quality, dict) else None

def _box_quality_raw_score(quality: dict[str, Any] | None) -> float | None:
    if not quality:
        return None
    score = _float_or_none(quality.get("score"))
    if score is None:
        return None
    return _clamp01(score)

def _box_quality_score(
    quality: dict[str, Any] | None,
    weight_index: dict[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    raw_score = _box_quality_raw_score(quality)
    if raw_score is None:
        return None, None
    prior = ((weight_index or {}).get("box_quality") or {}).get("positive") or {}
    count = int(prior.get("count") or 0)
    if count < 3:
        return raw_score, raw_score
    p10 = _float_or_none(prior.get("p10"))
    p50 = _float_or_none(prior.get("p50"))
    p90 = _float_or_none(prior.get("p90"))
    if p10 is None or p50 is None:
        return raw_score, raw_score
    low = max(0.0, p10 * 0.65)
    high = max(low + 1e-6, p50)
    learned_score = _smoothstep(low, high, raw_score)
    if p90 is not None and p90 > p50:
        learned_score = min(1.0, learned_score + _smoothstep(p50, p90, raw_score) * 0.10)
    return raw_score, _clamp01(raw_score * 0.70 + learned_score * 0.30)

def _box_quality_multiplier(box_quality_score: float | None, box_quality_weight: float) -> float:
    if box_quality_score is None or box_quality_weight <= 0:
        return 1.0
    weight = _clamp01(box_quality_weight)
    return (1.0 - weight) + weight * _clamp01(box_quality_score)

def _box_quality_enabled(box_quality_weight: float, min_box_quality: float) -> bool:
    return box_quality_weight > 0 or min_box_quality > 0

def _feature_structure_mode(config: dict[str, Any], box_quality_enabled: bool = False) -> str:
    structure_mode = str(config.get("structure_mode", DEFAULT_STRUCTURE_MODE))
    if structure_mode == "none" and box_quality_enabled:
        return "grid"
    return structure_mode

def _box_quality_prior_from_scores(
    positive_scores: list[float],
    negative_scores: list[float],
) -> dict[str, Any]:
    return {
        "version": "centered_box_quality_v1",
        "positive": _score_summary(positive_scores),
        "negative": _score_summary(negative_scores),
    }

def _cell_mean(values: list[float], source_size: int, x1: int, y1: int, x2: int, y2: int) -> float:
    total = 0.0
    count = 0
    for y in range(y1, max(y1 + 1, y2)):
        offset = min(source_size - 1, y) * source_size
        for x in range(x1, max(x1 + 1, x2)):
            total += values[offset + min(source_size - 1, x)]
            count += 1
    return total / max(1, count)

def _cell_std(values: list[float], source_size: int, x1: int, y1: int, x2: int, y2: int, mean: float) -> float:
    total = 0.0
    count = 0
    for y in range(y1, max(y1 + 1, y2)):
        offset = min(source_size - 1, y) * source_size
        for x in range(x1, max(x1 + 1, x2)):
            delta = values[offset + min(source_size - 1, x)] - mean
            total += delta * delta
            count += 1
    return math.sqrt(total / max(1, count))

def _orientation_histogram(gray_values: list[float], source_size: int, x1: int, y1: int, x2: int, y2: int) -> list[float]:
    bins = [0.0, 0.0, 0.0, 0.0]
    for y in range(max(1, y1), min(source_size - 1, y2)):
        offset = y * source_size
        for x in range(max(1, x1), min(source_size - 1, x2)):
            gx = gray_values[offset + x + 1] - gray_values[offset + x - 1]
            gy = gray_values[(y + 1) * source_size + x] - gray_values[(y - 1) * source_size + x]
            magnitude = math.sqrt(gx * gx + gy * gy)
            if magnitude <= 0:
                continue
            angle = abs(math.atan2(gy, gx))
            if angle < math.pi / 8 or angle >= math.pi * 7 / 8:
                bins[0] += magnitude
            elif angle < math.pi * 3 / 8:
                bins[1] += magnitude
            elif angle < math.pi * 5 / 8:
                bins[2] += magnitude
            else:
                bins[3] += magnitude
    total = sum(bins)
    if total <= 0:
        return bins
    return [value / total for value in bins]

def _ring_region_mean(
    values: list[float],
    source_size: int,
    outer_start: int,
    outer_end: int,
    inner_start: int,
    inner_end: int,
) -> float:
    total = 0.0
    count = 0
    for y in range(max(0, outer_start), min(source_size, outer_end)):
        for x in range(max(0, outer_start), min(source_size, outer_end)):
            if inner_start <= x < inner_end and inner_start <= y < inner_end:
                continue
            total += values[y * source_size + x]
            count += 1
    return total / max(1, count)

def _extract_box_quality(
    edge_values: list[float],
    local_contrast_values: list[float],
    texture_values: list[float],
    source_size: int,
    qua: int = 8,
) -> dict[str, float]:
    center_start = max(0, round(source_size * 0.25))
    center_end = min(source_size, round(source_size * 0.75))
    core_start = max(0, round(source_size * 0.33))
    core_end = min(source_size, round(source_size * 0.67))
    inner_start = max(0, round(source_size * 0.17))
    inner_end = min(source_size, round(source_size * 0.83))

    center_edge = _cell_mean(edge_values, source_size, center_start, center_start, center_end, center_end)
    center_contrast = _cell_mean(
        local_contrast_values,
        source_size,
        center_start,
        center_start,
        center_end,
        center_end,
    )
    center_texture = _cell_mean(texture_values, source_size, center_start, center_start, center_end, center_end)
    core_edge = _cell_mean(edge_values, source_size, core_start, core_start, core_end, core_end)
    inner_edge = _cell_mean(edge_values, source_size, inner_start, inner_start, inner_end, inner_end)
    mean_edge = sum(edge_values) / max(1, len(edge_values))
    mean_texture = sum(texture_values) / max(1, len(texture_values))
    mean_contrast = sum(local_contrast_values) / max(1, len(local_contrast_values))
    border_edge = _ring_region_mean(edge_values, source_size, 0, source_size, inner_start, inner_end)
    border_texture = _ring_region_mean(texture_values, source_size, 0, source_size, inner_start, inner_end)
    center_support = _clamp01(center_edge * 0.45 + center_contrast * 0.30 + center_texture * 0.25)
    core_support = _clamp01(core_edge * 0.55 + center_contrast * 0.25 + center_texture * 0.20)
    objectness = _clamp01(center_support * 0.70 + core_support * 0.30)
    edge_border_ratio = border_edge / max(center_edge, 1e-6)
    texture_border_ratio = border_texture / max(center_texture, 1e-6)
    center_edge_ratio = center_edge / max(mean_edge, 1e-6)
    center_texture_ratio = center_texture / max(mean_texture, 1e-6)
    center_contrast_ratio = center_contrast / max(mean_contrast, 1e-6)

    border_penalty = 1.0 - _smoothstep(1.15, 2.20, edge_border_ratio) * 0.55
    texture_penalty = 1.0 - _smoothstep(1.20, 2.30, texture_border_ratio) * 0.35
    weak_center_penalty = 0.55 + _smoothstep(0.08, 0.24, objectness) * 0.45
    center_balance = _clamp01(
        center_edge_ratio * 0.45 + center_texture_ratio * 0.25 + center_contrast_ratio * 0.30
    )
    balance_bonus = 0.70 + _smoothstep(0.75, 1.20, center_balance) * 0.30
    score = _clamp01(objectness * border_penalty * texture_penalty * weak_center_penalty * balance_bonus)

    return {
        "score": round(score, qua),
        "objectness": round(objectness, qua),
        "center_support": round(center_support, qua),
        "core_support": round(core_support, qua),
        "center_edge": round(center_edge, qua),
        "core_edge": round(core_edge, qua),
        "inner_edge": round(inner_edge, qua),
        "border_edge": round(border_edge, qua),
        "mean_edge": round(mean_edge, qua),
        "center_texture": round(center_texture, qua),
        "border_texture": round(border_texture, qua),
        "mean_texture": round(mean_texture, qua),
        "center_contrast": round(center_contrast, qua),
        "mean_contrast": round(mean_contrast, qua),
        "edge_border_ratio": round(edge_border_ratio, qua),
        "texture_border_ratio": round(texture_border_ratio, qua),
        "center_edge_ratio": round(center_edge_ratio, qua),
        "center_texture_ratio": round(center_texture_ratio, qua),
        "center_contrast_ratio": round(center_contrast_ratio, qua),
        "border_penalty": round(border_penalty, qua),
        "texture_penalty": round(texture_penalty, qua),
        "weak_center_penalty": round(weak_center_penalty, qua),
        "center_balance": round(center_balance, qua),
    }

def extract_structure_features(
    gray_values: list[float],
    sat_values: list[float],
    edge_values: list[float],
    local_contrast_values: list[float],
    texture_values: list[float],
    source_size: int,
    grid: int = DEFAULT_STRUCTURE_GRID,
    qua: int = 8,
) -> dict[str, Any]:
    grid = max(1, int(grid))
    vector: list[float] = []
    cell_payloads = []
    for grid_y in range(grid):
        y1 = math.floor(grid_y * source_size / grid)
        y2 = max(y1 + 1, math.floor((grid_y + 1) * source_size / grid))
        for grid_x in range(grid):
            x1 = math.floor(grid_x * source_size / grid)
            x2 = max(x1 + 1, math.floor((grid_x + 1) * source_size / grid))
            gray_mean = _cell_mean(gray_values, source_size, x1, y1, x2, y2)
            edge_mean = _cell_mean(edge_values, source_size, x1, y1, x2, y2)
            sat_mean = _cell_mean(sat_values, source_size, x1, y1, x2, y2)
            contrast_mean = _cell_mean(local_contrast_values, source_size, x1, y1, x2, y2)
            texture_mean = _cell_mean(texture_values, source_size, x1, y1, x2, y2)
            gray_std = _cell_std(gray_values, source_size, x1, y1, x2, y2, gray_mean)
            dark_ratio = _cell_mean([1.0 if value > 0.62 else 0.0 for value in gray_values], source_size, x1, y1, x2, y2)
            orientations = _orientation_histogram(gray_values, source_size, x1, y1, x2, y2)
            cell_vector = [
                gray_mean,
                gray_std,
                edge_mean,
                sat_mean,
                contrast_mean,
                texture_mean,
                dark_ratio,
                *orientations,
            ]
            vector.extend(cell_vector)
            cell_payloads.append(
                {
                    "x": grid_x,
                    "y": grid_y,
                    "gray": gray_mean,
                    "edge": edge_mean,
                    "saturation": sat_mean,
                    "contrast": contrast_mean,
                    "texture": texture_mean,
                }
            )

    half = source_size // 2
    left_gray = _cell_mean(gray_values, source_size, 0, 0, half, source_size)
    right_gray = _cell_mean(gray_values, source_size, half, 0, source_size, source_size)
    left_edge = _cell_mean(edge_values, source_size, 0, 0, half, source_size)
    right_edge = _cell_mean(edge_values, source_size, half, 0, source_size, source_size)
    top_edge = _cell_mean(edge_values, source_size, 0, 0, source_size, half)
    bottom_edge = _cell_mean(edge_values, source_size, 0, half, source_size, source_size)
    center_start = source_size // 4
    center_end = source_size - center_start
    center_edge = _cell_mean(edge_values, source_size, center_start, center_start, center_end, center_end)
    border_edge = _ring_region_mean(
        edge_values,
        source_size,
        0,
        source_size,
        max(0, round(source_size * 0.17)),
        min(source_size, round(source_size * 0.83)),
    )
    quality = _extract_box_quality(edge_values, local_contrast_values, texture_values, source_size, qua=qua)
    summary = [
        abs(left_gray - right_gray),
        abs(left_edge - right_edge),
        abs(top_edge - bottom_edge),
        center_edge,
        border_edge,
        sum(edge_values) / max(1, len(edge_values)),
        sum(texture_values) / max(1, len(texture_values)),
        sum(local_contrast_values) / max(1, len(local_contrast_values)),
    ]
    vector.extend(summary)
    return {
        "mode": "grid",
        "version": "spatial_grid_v1",
        "grid": grid,
        "length": len(vector),
        "vector": [round(max(0.0, min(1.0, value)), qua) for value in vector],
        "summary": {
            "left_right_gray_delta": round(abs(left_gray - right_gray), qua),
            "left_right_edge_delta": round(abs(left_edge - right_edge), qua),
            "top_bottom_edge_delta": round(abs(top_edge - bottom_edge), qua),
            "center_edge": round(center_edge, qua),
            "border_edge": round(border_edge, qua),
            "mean_edge": round(sum(edge_values) / max(1, len(edge_values)), qua),
        },
        "quality": quality,
        "cells": cell_payloads,
    }

def extract_image_features(
    image: Image.Image,
    size: int = 2,
    qua: int = 8,
    nab: float = DEFAULT_NAB,
    pt_size: int = 8,
    kernel: str = DEFAULT_KERNEL,
    field: str = DEFAULT_FIELD,
    max_radius: int = DEFAULT_MAX_RADIUS,
    normalize: bool = True,
    normalize_each_step: bool = True,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    channels: list[str] | str | None = None,
    structure_mode: str = DEFAULT_STRUCTURE_MODE,
    structure_grid: int = DEFAULT_STRUCTURE_GRID,
    stats_version: str = DEFAULT_FEATURE_STATS_VERSION,
) -> dict[str, Any]:
    _validate_feature_mode(feature_mode)
    _validate_structure_mode(structure_mode)
    selected_channels = parse_channels(channels)
    if feature_mode == "gray":
        structure = None
        if structure_mode != "none":
            structure_source_size = max(8, pt_size)
            rgb = image.resize((structure_source_size, structure_source_size)).convert("RGB")
            hsv = rgb.convert("HSV")
            rgb_pixels = list(rgb.getdata())
            hsv_pixels = list(hsv.getdata())
            gray_values = [
                1 - (red * 0.299 + green * 0.587 + blue * 0.114) / 255
                for red, green, blue in rgb_pixels
            ]
            sat_values = [sat / 255 for _hue, sat, _val in hsv_pixels]
            edge_values = _edge_values(gray_values, structure_source_size)
            local_contrast_values = _local_contrast_values(gray_values, structure_source_size)
            texture_values = _texture_values(gray_values, structure_source_size)
            structure = extract_structure_features(
                gray_values,
                sat_values,
                edge_values,
                local_contrast_values,
                texture_values,
                structure_source_size,
                grid=structure_grid,
                qua=qua,
            )
        gray_pt = image_to_pt(
            image,
            size=size,
            qua=qua,
            nab=nab,
            pt_size=pt_size,
            kernel=kernel,
            field=field,
            max_radius=max_radius,
            normalize=normalize,
            normalize_each_step=normalize_each_step,
        )
        vector = _flatten_pt(gray_pt)
        channel_layout = _channel_layout_from_names(["gray"], pt_size, stats_length=0)
        return {
            "mode": feature_mode,
            "vector": vector,
            "pt": gray_pt,
            "channels": ["gray"],
            "channel_layout": channel_layout,
            "channel_stats": {"gray": _channel_stats(vector)},
            "channel_quality": {"gray": 1.0},
            "stats_version": "none",
            "stats": [],
            "structure": structure,
        }

    source_size = pt_size
    rgb = image.resize((source_size, source_size)).convert("RGB")
    hsv = rgb.convert("HSV")
    rgb_pixels = list(rgb.getdata())
    hsv_pixels = list(hsv.getdata())

    gray_values = []
    hue_values = []
    sat_values = []
    val_values = []
    yellow_mask = []
    red_values = []
    green_values = []
    blue_values = []
    for (red, green, blue), (hue, sat, val) in zip(rgb_pixels, hsv_pixels):
        gray = 1 - (red * 0.299 + green * 0.587 + blue * 0.114) / 255
        saturation = sat / 255
        value = val / 255
        hue_norm = hue / 255
        yellowish = 1.0 if 0.08 <= hue_norm <= 0.22 and saturation >= 0.18 and value >= 0.25 else 0.0
        gray_values.append(gray)
        hue_values.append(hue_norm)
        sat_values.append(saturation)
        val_values.append(value)
        yellow_mask.append(yellowish)
        red_values.append(red / 255)
        green_values.append(green / 255)
        blue_values.append(blue / 255)

    edge_values = _edge_values(gray_values, source_size)
    lab_b_values = _lab_b_values(rgb_pixels)
    local_contrast_values = _local_contrast_values(gray_values, source_size)
    texture_values = _texture_values(gray_values, source_size)
    channel_values = {
        "gray": gray_values,
        "saturation": sat_values,
        "edge": edge_values,
        "hue": hue_values,
        "lab_b": lab_b_values,
        "local_contrast": local_contrast_values,
        "texture": texture_values,
        "yellow_mask": yellow_mask,
    }
    channel_pts = {
        channel: _channel_to_fast_pt(
            channel_values[channel],
            source_size,
            qua,
            nab,
            pt_size,
            kernel,
            field,
            max_radius,
            normalize,
        )
        for channel in selected_channels
    }

    width = int(image.info.get("solo_source_width", image.size[0]))
    height = int(image.info.get("solo_source_height", image.size[1]))
    if stats_version == LEGACY_FEATURE_STATS_VERSION:
        stats = _legacy_geometry_stats(
            width,
            height,
            red_values,
            green_values,
            blue_values,
            hue_values,
            sat_values,
            val_values,
            yellow_mask,
            edge_values,
        )
    else:
        stats_version = DEFAULT_FEATURE_STATS_VERSION
        stats = _feature_geometry_stats(
            width,
            height,
            red_values,
            green_values,
            blue_values,
            hue_values,
            sat_values,
            val_values,
            yellow_mask,
            edge_values,
            local_contrast_values,
            texture_values,
        )
    vector = []
    for channel in selected_channels:
        vector.extend(_flatten_pt(channel_pts[channel]))
    vector.extend([round(value, qua) for value in stats])
    channel_stats = {channel: _channel_stats(channel_values[channel]) for channel in selected_channels}
    channel_quality = _feature_quality_from_stats(channel_stats, selected_channels)
    structure = None
    if structure_mode != "none":
        structure = extract_structure_features(
            gray_values,
            sat_values,
            edge_values,
            local_contrast_values,
            texture_values,
            source_size,
            grid=structure_grid,
            qua=qua,
        )
    return {
        "mode": feature_mode,
        "vector": vector,
        "pt": channel_pts.get("gray", next(iter(channel_pts.values()))),
        "channels": selected_channels,
        "channel_layout": _channel_layout_from_names(selected_channels, pt_size, stats_length=len(stats)),
        "channel_stats": channel_stats,
        "channel_quality": channel_quality,
        "stats_version": stats_version,
        "stats": stats,
        "structure": structure,
    }

from solo.core.proposal_features import *

__all__ = [
    '_edge_values',
    '_local_contrast_values',
    '_texture_values',
    '_lab_b_values',
    '_channel_stats',
    '_channel_layout_from_names',
    '_feature_geometry_stats',
    '_legacy_geometry_stats',
    '_feature_quality_from_stats',
    '_float_or_none',
    '_score_summary',
    '_box_quality_payload_from_features',
    '_box_quality_raw_score',
    '_box_quality_score',
    '_box_quality_multiplier',
    '_box_quality_enabled',
    '_feature_structure_mode',
    '_box_quality_prior_from_scores',
    '_cell_mean',
    '_cell_std',
    '_orientation_histogram',
    '_ring_region_mean',
    '_extract_box_quality',
    'extract_structure_features',
    'extract_image_features',
    '_proposal_passes_shape_filters',
    '_objectness_maps',
    '_edge_values_square_or_rect',
    '_edge_values_square',
    '_local_contrast_values_rect',
    '_texture_values_rect',
    '_mean_in_bbox',
    '_region_stats',
    'context_maps_for_image',
    'context_quality_for_bbox_from_maps',
    'context_quality_for_bbox',
    'fragmentation_quality_from_box_quality',
    '_expand_bbox',
    '_project_values',
    '_trim_projection',
    'refine_proposal_bbox',
    '_component_proposals_from_mask',
    '_scale_work_proposals_to_source',
    'edge_component_proposals',
    'horizontal_body_proposals',
    'color_proposals',
    'sliding_proposals',
    'anchor_proposals',
]
