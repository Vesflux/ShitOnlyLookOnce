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
from solo.data.dataloader import *
from solo.core.features import *
from solo.core.accelerator import *
from solo.models.clustering import *

def prepare_weight_index(
    weight_paths: list[str | Path],
    accelerator: str = DEFAULT_ACCELERATOR,
) -> dict[str, Any]:
    if not weight_paths:
        raise ValueError("at least one weight file is required for detection")
    _validate_accelerator(accelerator)

    entries = []
    configs = []
    channel_layout = {}
    channel_ranges = {}
    channel_reliability = {}
    box_quality = {}
    prototype_count = DEFAULT_PROTOTYPE_COUNT
    structure_mode = DEFAULT_STRUCTURE_MODE
    structure_grid = DEFAULT_STRUCTURE_GRID
    structure_length = 0
    cached_prototypes: list[dict[str, Any]] = []
    cached_prototypes_complete = True
    loaded_compact_payload = False
    for weight_path in weight_paths:
        payload = load_weights(weight_path)
        loaded_compact_payload = loaded_compact_payload or bool(payload.get("compact"))
        config = payload.get("config", {})
        configs.append({"path": str(weight_path), "config": config})
        channel_layout = channel_layout or payload.get("channel_layout") or config.get("channel_layout") or {}
        channel_ranges.update(payload.get("channel_ranges") or {})
        channel_reliability.update(payload.get("channel_reliability") or {})
        if not box_quality:
            box_quality = payload.get("box_quality") or config.get("box_quality") or {}
        prototype_count = int(payload.get("prototype_count") or config.get("prototype_count") or prototype_count)
        structure_mode = str(config.get("structure_mode") or structure_mode)
        structure_grid = int(config.get("structure_grid") or structure_grid)
        payload_prototypes = payload.get("prototypes") or []
        if not payload_prototypes:
            cached_prototypes_complete = False
        for prototype in payload_prototypes:
            item = dict(prototype)
            item["weight_path"] = str(weight_path)
            exemplars = []
            for exemplar in item.get("exemplars") or []:
                exemplar_item = dict(exemplar)
                exemplar_item.setdefault("weight_path", str(weight_path))
                exemplar_item.setdefault("weight_index", item.get("weight_index", 0))
                exemplar_item.setdefault("weight_source", item.get("weight_source"))
                exemplars.append(exemplar_item)
            if exemplars:
                item["exemplars"] = exemplars
            cached_prototypes.append(item)
        for item_index, compact_entry in enumerate(payload.get("compact_entries", [])):
            pt = compact_entry.get("pt") or []
            if not pt:
                continue
            entry = dict(compact_entry)
            entry["weight_path"] = str(weight_path)
            entry["index"] = int(entry.get("index", item_index))
            entry["pt"] = [float(value) for value in pt]
            entry["sample_weight"] = float(entry.get("sample_weight", 1.0))
            structure_vector = entry.get("structure_pt")
            entry["structure_pt"] = [float(value) for value in structure_vector] if structure_vector else None
            entry["source_key"] = entry.get("source_key") or _source_key(entry.get("source"))
            entries.append(entry)
        for item_index, item in enumerate(payload.get("weights", [])):
            pt = item.get("pt")
            features = item.get("features") or {}
            feature_vector = features.get("vector")
            structure_payload = features.get("structure") or {}
            structure_vector = structure_payload.get("vector") if structure_payload else None
            if not pt and not feature_vector:
                continue
            annotation = item.get("annotation") or {}
            label = annotation.get("label") or item.get("label") or annotation.get("class_id") or "unknown"
            flat = [float(value) for value in feature_vector] if feature_vector else [float(value) for row in pt for value in row]
            is_negative = bool(item.get("negative")) or str(label) == DEFAULT_NEGATIVE_LABEL
            source = item.get("source_image") or item.get("path") or item.get("name")
            entries.append(
                {
                    "weight_path": str(weight_path),
                    "index": item_index,
                    "label": str(label),
                    "negative": is_negative,
                    "pt": flat,
                    "sample_weight": float(item.get("sample_weight", 1.0)),
                    "feature_mode": features.get("mode") or config.get("feature_mode", DEFAULT_FEATURE_MODE),
                    "channel_quality": features.get("channel_quality") or {},
                    "channel_layout": features.get("channel_layout") or {},
                    "structure_pt": [float(value) for value in structure_vector] if structure_vector else None,
                    "source": source,
                    "source_key": _source_key(source),
                }
            )
    base_config = configs[0]["config"]
    if entries:
        vector_length = len(entries[0]["pt"])
        entries = [entry for entry in entries if len(entry["pt"]) == vector_length]
    else:
        vector_lengths = [len(prototype.get("pt") or []) for prototype in cached_prototypes if prototype.get("pt")]
        vector_length = vector_lengths[0] if vector_lengths else int(configs[0]["config"].get("vector_length", 0) or 0)
    if vector_length <= 0:
        raise ValueError("weight files do not contain usable pt entries or prototypes")
    if not channel_layout:
        channels = parse_channels(base_config.get("channels") or None)
        stats_length = max(0, vector_length - len(channels) * int(base_config.get("pt_size", 8)) ** 2)
        channel_layout = _channel_layout_from_names(channels, int(base_config.get("pt_size", 8)), stats_length)
    if not channel_reliability:
        channel_reliability = {channel: 1.0 for channel in channel_layout if channel != "stats"}
    structure_lengths = [len(entry.get("structure_pt") or []) for entry in entries if entry.get("structure_pt")]
    if not structure_lengths:
        structure_lengths = [
            len(prototype.get("structure_pt") or [])
            for prototype in cached_prototypes
            if prototype.get("structure_pt")
        ]
    structure_length = structure_lengths[0] if structure_lengths else 0
    if structure_length:
        entries = [
            entry
            for entry in entries
            if not entry.get("structure_pt") or len(entry.get("structure_pt") or []) == structure_length
        ]
    bbox_prior = (
        base_config.get("bbox_prior")
        or _first_payload_bbox_prior(weight_paths)
        or _bbox_prior_from_payloads(weight_paths)
    )
    prototypes = [
        prototype
        for prototype in cached_prototypes
        if len(prototype.get("pt") or []) == vector_length
    ] if cached_prototypes_complete else []
    if not prototypes:
        prototypes = _build_prototypes(
            entries,
            vector_length,
            prototype_count=prototype_count,
            accelerator=accelerator,
        )
    return {
        "entries": entries,
        "configs": configs,
        "config": base_config,
        "bbox_prior": bbox_prior,
        "prototypes": prototypes,
        "vector_length": vector_length,
        "channel_layout": channel_layout,
        "channel_ranges": channel_ranges,
        "channel_reliability": channel_reliability,
        "box_quality": box_quality,
        "prototype_count": prototype_count,
        "structure_mode": structure_mode,
        "structure_grid": structure_grid,
        "structure_length": structure_length,
        "compact": loaded_compact_payload,
    }

def _first_payload_bbox_prior(weight_paths: list[str | Path]) -> dict[str, Any] | None:
    for weight_path in weight_paths:
        payload = load_weights(weight_path)
        prior = payload.get("bbox_prior")
        if prior and prior.get("enabled"):
            return prior
    return None

def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

def _bbox_prior_from_payloads(weight_paths: list[str | Path]) -> dict[str, Any]:
    widths = []
    heights = []
    areas = []
    aspects = []
    normalized_widths = []
    normalized_heights = []
    normalized_areas = []
    for weight_path in weight_paths:
        payload = load_weights(weight_path)
        for item in payload.get("weights", []):
            if item.get("negative"):
                continue
            bbox = (item.get("annotation") or {}).get("bbox") or {}
            width = float(bbox.get("width", 0))
            height = float(bbox.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
            aspects.append(max(width / height, height / width))
            normalized = bbox.get("normalized") or {}
            normalized_width = float(normalized.get("x2", 0)) - float(normalized.get("x1", 0))
            normalized_height = float(normalized.get("y2", 0)) - float(normalized.get("y1", 0))
            if normalized_width > 0 and normalized_height > 0:
                normalized_widths.append(normalized_width)
                normalized_heights.append(normalized_height)
                normalized_areas.append(normalized_width * normalized_height)
    if not widths:
        return {"enabled": False}
    return {
        "enabled": True,
        "width": {"p05": _percentile(widths, 0.05), "p95": _percentile(widths, 0.95)},
        "height": {"p05": _percentile(heights, 0.05), "p95": _percentile(heights, 0.95)},
        "area": {"p05": _percentile(areas, 0.05), "p95": _percentile(areas, 0.95)},
        "normalized_width": {
            "p05": _percentile(normalized_widths, 0.05),
            "p95": _percentile(normalized_widths, 0.95),
        },
        "normalized_height": {
            "p05": _percentile(normalized_heights, 0.05),
            "p95": _percentile(normalized_heights, 0.95),
        },
        "normalized_area": {
            "p05": _percentile(normalized_areas, 0.05),
            "p95": _percentile(normalized_areas, 0.95),
        },
        "aspect": {"p95": _percentile(aspects, 0.95)},
        "templates": _bbox_templates(widths, heights),
        "normalized_templates": _bbox_templates(normalized_widths, normalized_heights, minimum=0.001),
    }

def _bbox_templates(
    widths: list[float],
    heights: list[float],
    minimum: float = 4,
) -> list[dict[str, float]]:
    if not widths or not heights:
        return []
    pairs = [(width, height) for width, height in zip(widths, heights) if width > 0 and height > 0]
    if not pairs:
        return []

    width_values = [pair[0] for pair in pairs]
    height_values = [pair[1] for pair in pairs]
    area_values = [pair[0] * pair[1] for pair in pairs]
    aspect_values = [pair[0] / pair[1] for pair in pairs]
    templates: list[tuple[int, int]] = []

    def add_template(width: float, height: float) -> None:
        if minimum < 1:
            rounded = (round(max(minimum, width), 5), round(max(minimum, height), 5))
        else:
            rounded = (max(round(minimum), round(width)), max(round(minimum), round(height)))
        if rounded not in templates:
            templates.append(rounded)

    for percentile in (0.15, 0.35, 0.5, 0.65, 0.85):
        add_template(_percentile(width_values, percentile), _percentile(height_values, percentile))

    for area_percentile in (0.25, 0.5, 0.75):
        area = max(1.0, _percentile(area_values, area_percentile))
        for aspect_percentile in (0.25, 0.5, 0.75):
            aspect = max(0.05, _percentile(aspect_values, aspect_percentile))
            add_template(math.sqrt(area * aspect), math.sqrt(area / aspect))

    return [{"width": width, "height": height} for width, height in templates[:12]]

def _range_score(value: float, low: float, high: float) -> float:
    if low <= 0 or high <= 0 or high <= low:
        return 1.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.1, value / low)
    return max(0.1, high / value)

def bbox_prior_score(bbox: dict[str, Any], prior: dict[str, Any] | None, mode: str = DEFAULT_BBOX_PRIOR_MODE) -> float:
    _validate_bbox_prior_mode(mode)
    if mode == "none" or not prior or not prior.get("enabled"):
        return 1.0
    width = float(bbox.get("width", 0))
    height = float(bbox.get("height", 0))
    area = width * height
    aspect = max(width / max(1.0, height), height / max(1.0, width))
    normalized = bbox.get("normalized") or {}
    normalized_width = float(normalized.get("x2", 0)) - float(normalized.get("x1", 0))
    normalized_height = float(normalized.get("y2", 0)) - float(normalized.get("y1", 0))
    normalized_area = normalized_width * normalized_height
    width_score = _range_score(width, prior["width"]["p05"], prior["width"]["p95"])
    height_score = _range_score(height, prior["height"]["p05"], prior["height"]["p95"])
    area_score = _range_score(area, prior["area"]["p05"], prior["area"]["p95"])
    if normalized_width > 0 and prior.get("normalized_width", {}).get("p95", 0) > 0:
        width_score = max(
            width_score,
            _range_score(
                normalized_width,
                prior["normalized_width"]["p05"],
                prior["normalized_width"]["p95"],
            ),
        )
    if normalized_height > 0 and prior.get("normalized_height", {}).get("p95", 0) > 0:
        height_score = max(
            height_score,
            _range_score(
                normalized_height,
                prior["normalized_height"]["p05"],
                prior["normalized_height"]["p95"],
            ),
        )
    if normalized_area > 0 and prior.get("normalized_area", {}).get("p95", 0) > 0:
        area_score = max(
            area_score,
            _range_score(
                normalized_area,
                prior["normalized_area"]["p05"],
                prior["normalized_area"]["p95"],
            ),
        )
    score = min(
        width_score,
        height_score,
        area_score,
        _range_score(aspect, 1.0, prior["aspect"]["p95"]),
    )
    if mode == "hard" and score < 0.4:
        return 0.0
    return score

def _mse(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)

def _channel_selection(
    feature_quality: dict[str, float] | None,
    weight_index: dict[str, Any],
    channel_mode: str = DEFAULT_CHANNEL_MODE,
    channel_top_k: int = DEFAULT_CHANNEL_TOP_K,
) -> dict[str, float]:
    _validate_channel_mode(channel_mode)
    layout = weight_index.get("channel_layout") or {}
    channels = [channel for channel in layout if channel != "stats"]
    reliability = weight_index.get("channel_reliability") or {}
    feature_quality = feature_quality or {}
    weights = {}
    for channel in channels:
        reliable = float(reliability.get(channel, 1.0))
        quality = float(feature_quality.get(channel, 1.0))
        if channel_mode == "fixed":
            value = reliable
        elif channel_mode == "adaptive":
            value = quality
        else:
            value = reliable * quality
        weights[channel] = max(0.01, value)
    if channel_top_k > 0 and len(weights) > channel_top_k:
        keep = {channel for channel, _value in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:channel_top_k]}
        weights = {channel: value for channel, value in weights.items() if channel in keep}
    stats_layout = layout.get("stats") or {}
    if int(stats_layout.get("length", 0)) > 0:
        weights["stats"] = max(0.01, DEFAULT_STATS_WEIGHT)
    total = sum(weights.values())
    if total <= 0:
        return {channel: 1.0 / max(1, len(channels)) for channel in channels}
    return {channel: value / total for channel, value in weights.items()}

def _weighted_mse_by_channel(
    left: list[float],
    right: list[float],
    weight_index: dict[str, Any],
    channel_weights: dict[str, float] | None = None,
) -> float:
    if not channel_weights:
        return _mse(left, right)
    layout = weight_index.get("channel_layout") or {}
    total = 0.0
    used_weight = 0.0
    for channel, weight in channel_weights.items():
        item = layout.get(channel)
        if not item:
            continue
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        if end <= start or end > len(left) or end > len(right):
            continue
        total += weight * _mse(left[start:end], right[start:end])
        used_weight += weight
    if used_weight <= 0:
        return _mse(left, right)
    return total / used_weight

def _structure_distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return _mse(left, right)

def _merge_structure_into_candidate(
    candidate: dict[str, Any],
    structure_distance: float | None,
    structure_weight: float,
) -> dict[str, Any]:
    if structure_distance is None or structure_weight <= 0:
        return candidate
    structure_score = 1 / (1 + structure_distance)
    appearance_score = float(candidate["score"])
    blended_score = appearance_score * (1 - structure_weight) + structure_score * structure_weight
    candidate["appearance_score"] = appearance_score
    candidate["structure_distance"] = structure_distance
    candidate["structure_score"] = structure_score
    candidate["score"] = blended_score
    candidate["distance"] = (1 / max(blended_score, 1e-9)) - 1
    return candidate

def match_pt(
    pt: list[list[float]] | list[float],
    weight_index: dict[str, Any],
    excluded_source_keys: set[str] | None = None,
    match_mode: str = DEFAULT_MATCH_MODE,
    channel_weights: dict[str, float] | None = None,
    structure_vector: list[float] | None = None,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> dict[str, Any]:
    _validate_match_mode(match_mode)
    _validate_accelerator(accelerator)
    if not pt:
        raise ValueError("pt cannot be empty")
    first = pt[0]
    if isinstance(first, list):
        vector = [float(value) for row in pt for value in row]  # type: ignore[union-attr]
    else:
        vector = [float(value) for value in pt]  # type: ignore[union-attr]

    if match_mode == "prototype" or not weight_index.get("entries"):
        return match_prototype(
            vector,
            weight_index,
            excluded_source_keys=excluded_source_keys,
            channel_weights=channel_weights,
            structure_vector=structure_vector,
            structure_weight=structure_weight,
            accelerator=accelerator,
        )

    best = None
    best_positive = None
    best_negative = None
    compatible_entries = [
        entry
        for entry in weight_index["entries"]
        if len(entry["pt"]) == len(vector)
        and not (excluded_source_keys and entry.get("source_key") in excluded_source_keys)
    ]
    distances = (
        batch_weighted_mse(
            vector,
            [entry["pt"] for entry in compatible_entries],
            weight_index,
            channel_weights=channel_weights,
            accelerator=accelerator,
        )
        if compatible_entries and accelerator != "cpu"
        else None
    )
    for entry_index, entry in enumerate(compatible_entries):
        distance = (
            distances[entry_index]
            if distances is not None
            else _weighted_mse_by_channel(vector, entry["pt"], weight_index, channel_weights)
        )
        candidate = {
            "label": entry["label"],
            "negative": entry.get("negative", False),
            "distance": distance,
            "score": 1 / (1 + distance),
            "weight_path": entry["weight_path"],
            "weight_index": entry["index"],
            "weight_source": entry["source"],
        }
        candidate = _merge_structure_into_candidate(
            candidate,
            _structure_distance(structure_vector, entry.get("structure_pt")),
            structure_weight,
        )
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        if candidate["negative"]:
            if best_negative is None or candidate["score"] > best_negative["score"]:
                best_negative = candidate
        else:
            if best_positive is None or candidate["score"] > best_positive["score"]:
                best_positive = candidate
    if best is None:
        raise ValueError("no compatible weight entries were found")
    if best_positive is not None:
        best["positive_label"] = best_positive["label"]
        best["positive_distance"] = best_positive["distance"]
        best["positive_score"] = best_positive["score"]
        best["positive_appearance_score"] = best_positive.get("appearance_score")
        best["positive_structure_score"] = best_positive.get("structure_score")
        best["positive_structure_distance"] = best_positive.get("structure_distance")
    if best_negative is not None:
        best["negative_distance"] = best_negative["distance"]
        best["negative_score"] = best_negative["score"]
        best["negative_appearance_score"] = best_negative.get("appearance_score")
        best["negative_structure_score"] = best_negative.get("structure_score")
        best["negative_structure_distance"] = best_negative.get("structure_distance")
    return best

def _prototype_vector(
    prototype: dict[str, Any],
    excluded_source_keys: set[str] | None,
) -> tuple[list[float] | None, int]:
    count = int(prototype.get("count", 0))
    vector_sum = prototype.get("sum")
    if not excluded_source_keys:
        return prototype.get("pt"), count
    if not vector_sum:
        return prototype.get("pt"), count
    adjusted_sum = [float(value) for value in vector_sum]
    adjusted_count = count
    adjusted_weight = float(prototype.get("weight", count))
    for source_key in excluded_source_keys:
        source_payload = prototype.get("source_sums", {}).get(source_key)
        if not source_payload:
            continue
        adjusted_count -= int(source_payload.get("count", 0))
        adjusted_weight -= float(source_payload.get("weight", source_payload.get("count", 0)))
        for index, value in enumerate(source_payload.get("sum", [])):
            adjusted_sum[index] -= float(value)
    if adjusted_count <= 0 or adjusted_weight <= 0:
        return None, 0
    return [value / adjusted_weight for value in adjusted_sum], adjusted_count

def match_prototype(
    vector: list[float],
    weight_index: dict[str, Any],
    excluded_source_keys: set[str] | None = None,
    channel_weights: dict[str, float] | None = None,
    structure_vector: list[float] | None = None,
    structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> dict[str, Any]:
    _validate_accelerator(accelerator)
    best = None
    best_positive = None
    best_negative = None
    prepared: list[tuple[dict[str, Any], list[float], int]] = []
    for prototype in weight_index.get("prototypes", []):
        prototype_vector, prototype_count = _prototype_vector(prototype, excluded_source_keys)
        if not prototype_vector or len(prototype_vector) != len(vector):
            continue
        prepared.append((prototype, prototype_vector, prototype_count))
        if excluded_source_keys:
            continue
        for exemplar in prototype.get("exemplars") or []:
            exemplar_vector = exemplar.get("pt") or []
            if len(exemplar_vector) != len(vector):
                continue
            prepared.append((exemplar, exemplar_vector, int(exemplar.get("count", 1) or 1)))
    distances = (
        batch_weighted_mse(
            vector,
            [item[1] for item in prepared],
            weight_index,
            channel_weights=channel_weights,
            accelerator=accelerator,
        )
        if prepared and accelerator != "cpu" and not excluded_source_keys
        else None
    )
    for prepared_index, (prototype, prototype_vector, prototype_count) in enumerate(prepared):
        distance = (
            distances[prepared_index]
            if distances is not None
            else _weighted_mse_by_channel(vector, prototype_vector, weight_index, channel_weights)
        )
        candidate = {
            "label": prototype["label"],
            "negative": prototype.get("negative", False),
            "distance": distance,
            "score": 1 / (1 + distance),
            "weight_path": prototype["weight_path"],
            "weight_index": prototype["weight_index"],
            "weight_source": prototype["weight_source"],
            "prototype_count": prototype_count,
            "prototype_cluster": prototype.get("cluster_index"),
            "prototype_kind": prototype.get("prototype_kind", "prototype"),
        }
        candidate = _merge_structure_into_candidate(
            candidate,
            _structure_distance(structure_vector, prototype.get("structure_pt")),
            structure_weight,
        )
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        if candidate["negative"]:
            if best_negative is None or candidate["score"] > best_negative["score"]:
                best_negative = candidate
        else:
            if best_positive is None or candidate["score"] > best_positive["score"]:
                best_positive = candidate
    if best is None:
        raise ValueError("no compatible weight prototypes were found")
    if best_positive is not None:
        best["positive_label"] = best_positive["label"]
        best["positive_distance"] = best_positive["distance"]
        best["positive_score"] = best_positive["score"]
        best["positive_appearance_score"] = best_positive.get("appearance_score")
        best["positive_structure_score"] = best_positive.get("structure_score")
        best["positive_structure_distance"] = best_positive.get("structure_distance")
        best["positive_prototype_count"] = best_positive.get("prototype_count")
        best["positive_prototype_cluster"] = best_positive.get("prototype_cluster")
        best["positive_prototype_kind"] = best_positive.get("prototype_kind")
    if best_negative is not None:
        best["negative_distance"] = best_negative["distance"]
        best["negative_score"] = best_negative["score"]
        best["negative_appearance_score"] = best_negative.get("appearance_score")
        best["negative_structure_score"] = best_negative.get("structure_score")
        best["negative_structure_distance"] = best_negative.get("structure_distance")
        best["negative_prototype_count"] = best_negative.get("prototype_count")
        best["negative_prototype_cluster"] = best_negative.get("prototype_cluster")
        best["negative_prototype_kind"] = best_negative.get("prototype_kind")
    return best
__all__ = [
    'prepare_weight_index',
    '_percentile',
    '_bbox_prior_from_payloads',
    '_bbox_templates',
    '_range_score',
    'bbox_prior_score',
    '_mse',
    '_channel_selection',
    '_weighted_mse_by_channel',
    '_structure_distance',
    '_merge_structure_into_candidate',
    'match_pt',
    '_prototype_vector',
    'match_prototype',
]
