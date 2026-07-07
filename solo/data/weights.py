from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from solo.config import (
    DEFAULT_ACCELERATOR,
    DEFAULT_COMPACT_EXEMPLARS,
    DEFAULT_COMPACT_SAMPLE_LIMIT,
    DEFAULT_COMPACT_WEIGHTS,
    DEFAULT_FEATURE_STATS_VERSION,
    DEFAULT_NEGATIVE_LABEL,
    DEFAULT_PROTOTYPE_COUNT,
    DEFAULT_WEIGHT_PATH,
    DEFAULT_WEIGHT_PRECISION,
    parse_channels,
)
from solo.core.features import (
    _box_quality_payload_from_features,
    _box_quality_prior_from_scores,
    _box_quality_raw_score,
)
from solo.data.compact import build_compact_entries
from solo.data.dataset import _source_key
from solo.models.clustering import _build_prototypes


def save_weights(
    weights: list[dict[str, Any]],
    path: str | Path = DEFAULT_WEIGHT_PATH,
    config: dict[str, Any] | None = None,
    compact: bool | None = None,
    precision: int | None = None,
    compact_exemplars: int | None = None,
    compact_sample_limit: int | None = None,
) -> Path:
    weight_path = Path(path)
    config_payload = dict(config or {})
    compact_enabled = DEFAULT_COMPACT_WEIGHTS if compact is None else bool(compact)
    precision_value = int(config_payload.get("weight_precision", DEFAULT_WEIGHT_PRECISION) if precision is None else precision)
    exemplar_count = int(
        config_payload.get("compact_exemplars", DEFAULT_COMPACT_EXEMPLARS)
        if compact_exemplars is None
        else compact_exemplars
    )
    sample_limit = int(
        config_payload.get("compact_sample_limit", DEFAULT_COMPACT_SAMPLE_LIMIT)
        if compact_sample_limit is None
        else compact_sample_limit
    )
    if not compact_enabled:
        exemplar_count = 0
        sample_limit = 0
    config_payload["compact_weights"] = compact_enabled
    config_payload["weight_precision"] = precision_value
    config_payload["compact_exemplars"] = max(0, exemplar_count)
    config_payload["compact_sample_limit"] = max(0, sample_limit)
    metadata = build_weight_metadata(weights, config_payload)
    compact_prototypes = _compact_prototypes(metadata.get("prototypes", []), precision_value) if compact_enabled else metadata.get("prototypes", [])
    if compact_enabled and exemplar_count > 0:
        compact_prototypes = _attach_compact_exemplars(compact_prototypes, weights, exemplar_count, precision_value)
    for prototype in compact_prototypes:
        prototype["weight_path"] = str(weight_path)
    metadata["prototypes"] = compact_prototypes
    compact_entries = build_compact_entries(weights, sample_limit, precision_value) if compact_enabled else []
    payload_weights = [] if compact_enabled else weights
    payload = {
        "version": 3 if compact_enabled else 2,
        "config": config_payload,
        **metadata,
        "compact": compact_enabled,
        "compact_entries": compact_entries,
        "weights": payload_weights,
        "source_weight_count": len(weights),
    }
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return weight_path


def _round_float(value: Any, precision: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, precision)


def _round_vector(values: list[Any] | None, precision: int) -> list[float] | None:
    if values is None:
        return None
    return [_round_float(value, precision) for value in values]


def _compact_prototypes(prototypes: list[dict[str, Any]], precision: int) -> list[dict[str, Any]]:
    compacted = []
    for prototype in prototypes:
        item = {
            "label": prototype.get("label"),
            "negative": bool(prototype.get("negative", False)),
            "cluster_index": prototype.get("cluster_index"),
            "count": int(prototype.get("count", 0)),
            "weight": _round_float(prototype.get("weight", prototype.get("count", 0)), precision),
            "pt": _round_vector(prototype.get("pt") or [], precision) or [],
            "structure_pt": _round_vector(prototype.get("structure_pt"), precision),
            "structure_length": int(prototype.get("structure_length", 0) or 0),
            "weight_index": prototype.get("weight_index", 0),
            "weight_source": prototype.get("weight_source"),
        }
        compacted.append(item)
    return compacted


def _mse(left: list[float], right: list[float]) -> float:
    length = min(len(left), len(right))
    if length == 0:
        return float("inf")
    return sum((left[index] - right[index]) ** 2 for index in range(length)) / length


def _compact_exemplar_from_entry(
    entry: dict[str, Any],
    prototype: dict[str, Any],
    exemplar_index: int,
    precision: int,
) -> dict[str, Any]:
    structure_pt = _round_vector(entry.get("structure_pt"), precision)
    return {
        "label": entry.get("label"),
        "negative": bool(entry.get("negative", False)),
        "cluster_index": prototype.get("cluster_index"),
        "exemplar_index": exemplar_index,
        "count": 1,
        "weight": _round_float(entry.get("sample_weight", 1.0), precision),
        "pt": _round_vector(entry.get("pt") or [], precision) or [],
        "structure_pt": structure_pt,
        "structure_length": len(structure_pt or []),
        "weight_index": entry.get("index", 0),
        "weight_source": entry.get("source"),
        "source_key": entry.get("source_key"),
        "prototype_kind": "exemplar",
    }


def _attach_compact_exemplars(
    prototypes: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    exemplar_count: int,
    precision: int,
) -> list[dict[str, Any]]:
    if exemplar_count <= 0 or not prototypes or not weights:
        return prototypes
    entries, _vector_length, _structure_length = _prototype_entries_from_weights(weights)
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((str(entry.get("label")), bool(entry.get("negative"))), []).append(entry)
    for prototype in prototypes:
        prototype_vector = prototype.get("pt") or []
        if not prototype_vector:
            continue
        candidates = grouped.get((str(prototype.get("label")), bool(prototype.get("negative"))), [])
        ranked = sorted(
            (
                (_mse([float(value) for value in entry.get("pt", [])], prototype_vector), entry)
                for entry in candidates
                if len(entry.get("pt") or []) == len(prototype_vector)
            ),
            key=lambda item: (item[0], item[1].get("source_key", ""), item[1].get("index", 0)),
        )
        prototype["exemplars"] = [
            _compact_exemplar_from_entry(entry, prototype, exemplar_index, precision)
            for exemplar_index, (_distance, entry) in enumerate(ranked[:exemplar_count])
        ]
    return prototypes


def _prototype_entries_from_weights(
    weights: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    entries = []
    vector_length = 0
    for item_index, item in enumerate(weights):
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
        if not flat:
            continue
        if vector_length == 0:
            vector_length = len(flat)
        if len(flat) != vector_length:
            continue
        is_negative = bool(item.get("negative")) or str(label) == DEFAULT_NEGATIVE_LABEL
        source = item.get("source_image") or item.get("path") or item.get("name")
        entries.append(
            {
                "weight_path": "",
                "index": item_index,
                "label": str(label),
                "negative": is_negative,
                "pt": flat,
                "sample_weight": float(item.get("sample_weight", 1.0)),
                "structure_pt": [float(value) for value in structure_vector] if structure_vector else None,
                "source": source,
                "source_key": _source_key(source),
            }
        )
    structure_lengths = [len(entry.get("structure_pt") or []) for entry in entries if entry.get("structure_pt")]
    structure_length = structure_lengths[0] if structure_lengths else 0
    if structure_length:
        entries = [
            entry
            for entry in entries
            if not entry.get("structure_pt") or len(entry.get("structure_pt") or []) == structure_length
        ]
    return entries, vector_length, structure_length


def build_weight_metadata(weights: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    channel_layout = {}
    channels = parse_channels(config.get("channels") or None)
    channel_ranges: dict[str, dict[str, float]] = {}
    positive_stats: dict[str, list[float]] = {channel: [] for channel in channels}
    negative_stats: dict[str, list[float]] = {channel: [] for channel in channels}
    positive_box_quality: list[float] = []
    negative_box_quality: list[float] = []
    first_features = None
    for item in weights:
        features = item.get("features") or {}
        if first_features is None and features:
            first_features = features
        item_layout = features.get("channel_layout") or {}
        if item_layout:
            channel_layout = item_layout
        item_stats = features.get("channel_stats") or {}
        is_negative = bool(item.get("negative")) or (item.get("annotation") or {}).get("label") == DEFAULT_NEGATIVE_LABEL
        quality_score = _box_quality_raw_score(_box_quality_payload_from_features(features))
        if quality_score is not None:
            target_quality = negative_box_quality if is_negative else positive_box_quality
            target_quality.append(quality_score)
        for channel in channels:
            stats = item_stats.get(channel) or {}
            channel_ranges.setdefault(channel, {"min": 1.0, "max": 0.0, "range_min": 1.0, "range_max": 0.0})
            if stats:
                channel_ranges[channel]["min"] = min(channel_ranges[channel]["min"], float(stats.get("min", 1.0)))
                channel_ranges[channel]["max"] = max(channel_ranges[channel]["max"], float(stats.get("max", 0.0)))
                channel_ranges[channel]["range_min"] = min(
                    channel_ranges[channel]["range_min"], float(stats.get("range", 1.0))
                )
                channel_ranges[channel]["range_max"] = max(
                    channel_ranges[channel]["range_max"], float(stats.get("range", 0.0))
                )
                target = negative_stats if is_negative else positive_stats
                target[channel].append(float(stats.get("mean", 0.0)))

    if not channel_layout and first_features:
        channel_layout = first_features.get("channel_layout") or {}

    channel_reliability = {}
    for channel in channels:
        positives = positive_stats.get(channel) or []
        negatives = negative_stats.get(channel) or []
        if positives and negatives:
            mean_gap = abs(sum(positives) / len(positives) - sum(negatives) / len(negatives))
            spread = statistics.pvariance(positives + negatives) if len(positives) + len(negatives) > 1 else 0.0
            channel_reliability[channel] = max(0.1, min(2.0, 0.5 + mean_gap * 4.0 + math.sqrt(spread)))
        else:
            channel_reliability[channel] = 1.0

    for channel, payload in channel_ranges.items():
        if payload["min"] > payload["max"]:
            channel_ranges[channel] = {"min": 0.0, "max": 1.0, "range_min": 0.0, "range_max": 1.0}

    prototype_count = int(config.get("prototype_count", DEFAULT_PROTOTYPE_COUNT))
    prototype_entries, vector_length, structure_length = _prototype_entries_from_weights(weights)
    prototypes = (
        _build_prototypes(
            prototype_entries,
            vector_length,
            prototype_count=prototype_count,
            accelerator=str(config.get("accelerator") or DEFAULT_ACCELERATOR),
        )
        if prototype_entries and vector_length > 0
        else []
    )

    return {
        "channel_layout": channel_layout,
        "channel_ranges": channel_ranges,
        "channel_reliability": channel_reliability,
        "box_quality": _box_quality_prior_from_scores(positive_box_quality, negative_box_quality),
        "bbox_prior": _bbox_prior_from_weights(weights),
        "prototype_count": prototype_count,
        "prototypes": prototypes,
        "vector_length": vector_length,
        "structure_length": structure_length,
    }


def _metadata_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _metadata_bbox_templates(
    widths: list[float],
    heights: list[float],
    minimum: float = 4,
) -> list[dict[str, float]]:
    pairs = [(width, height) for width, height in zip(widths, heights) if width > 0 and height > 0]
    if not pairs:
        return []
    width_values = [pair[0] for pair in pairs]
    height_values = [pair[1] for pair in pairs]
    area_values = [pair[0] * pair[1] for pair in pairs]
    aspect_values = [pair[0] / pair[1] for pair in pairs]
    templates: list[tuple[float, float]] = []

    def add_template(width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            return
        if minimum < 1:
            rounded = (round(max(minimum, width), 5), round(max(minimum, height), 5))
        else:
            rounded = (float(max(round(minimum), round(width))), float(max(round(minimum), round(height))))
        if rounded not in templates:
            templates.append(rounded)

    for percentile in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
        add_template(_metadata_percentile(width_values, percentile), _metadata_percentile(height_values, percentile))

    for area_percentile in (0.15, 0.30, 0.50, 0.70, 0.85):
        area = max(1.0, _metadata_percentile(area_values, area_percentile))
        for aspect_percentile in (0.15, 0.35, 0.50, 0.65, 0.85):
            aspect = max(0.05, _metadata_percentile(aspect_values, aspect_percentile))
            add_template(math.sqrt(area * aspect), math.sqrt(area / aspect))

    return [{"width": width, "height": height} for width, height in templates[:24]]


def _bbox_prior_from_weights(weights: list[dict[str, Any]]) -> dict[str, Any]:
    widths = []
    heights = []
    areas = []
    aspects = []
    normalized_widths = []
    normalized_heights = []
    normalized_areas = []
    for item in weights:
        annotation = item.get("annotation") or {}
        label = annotation.get("label") or item.get("label")
        if bool(item.get("negative")) or str(label) == DEFAULT_NEGATIVE_LABEL:
            continue
        bbox = annotation.get("bbox") or {}
        width = float(bbox.get("width", 0) or 0)
        height = float(bbox.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        aspects.append(max(width / height, height / width))
        normalized = bbox.get("normalized") or {}
        normalized_width = float(normalized.get("x2", 0) or 0) - float(normalized.get("x1", 0) or 0)
        normalized_height = float(normalized.get("y2", 0) or 0) - float(normalized.get("y1", 0) or 0)
        if normalized_width > 0 and normalized_height > 0:
            normalized_widths.append(normalized_width)
            normalized_heights.append(normalized_height)
            normalized_areas.append(normalized_width * normalized_height)
    if not widths:
        return {"enabled": False}
    return {
        "enabled": True,
        "width": {"p05": _metadata_percentile(widths, 0.05), "p95": _metadata_percentile(widths, 0.95)},
        "height": {"p05": _metadata_percentile(heights, 0.05), "p95": _metadata_percentile(heights, 0.95)},
        "area": {"p05": _metadata_percentile(areas, 0.05), "p95": _metadata_percentile(areas, 0.95)},
        "normalized_width": {
            "p05": _metadata_percentile(normalized_widths, 0.05),
            "p95": _metadata_percentile(normalized_widths, 0.95),
        },
        "normalized_height": {
            "p05": _metadata_percentile(normalized_heights, 0.05),
            "p95": _metadata_percentile(normalized_heights, 0.95),
        },
        "normalized_area": {
            "p05": _metadata_percentile(normalized_areas, 0.05),
            "p95": _metadata_percentile(normalized_areas, 0.95),
        },
        "aspect": {"p95": _metadata_percentile(aspects, 0.95)},
        "templates": _metadata_bbox_templates(widths, heights),
        "normalized_templates": _metadata_bbox_templates(normalized_widths, normalized_heights, minimum=0.001),
    }


def load_weights(path: str | Path = DEFAULT_WEIGHT_PATH) -> dict[str, Any]:
    weight_path = Path(path)
    payload = json.loads(weight_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return {"version": 0, "config": {}, "weights": payload}
    if not isinstance(payload, dict) or "weights" not in payload:
        raise ValueError(f"invalid weight file: {weight_path}")
    return payload


def get_weight_by_name(weights_payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in weights_payload.get("weights", []):
        if item.get("name") == name:
            return item
    return None


__all__ = [
    "save_weights",
    "_round_float",
    "_round_vector",
    "_compact_prototypes",
    "_prototype_entries_from_weights",
    "build_weight_metadata",
    "_metadata_percentile",
    "_metadata_bbox_templates",
    "_bbox_prior_from_weights",
    "load_weights",
    "get_weight_by_name",
]
