from __future__ import annotations

import math
from typing import Any

from solo.config import DEFAULT_NEGATIVE_LABEL
from solo.data.dataset import _source_key


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


def _flat_vector_from_weight(item: dict[str, Any]) -> list[float]:
    features = item.get("features") or {}
    feature_vector = features.get("vector")
    if feature_vector:
        return [float(value) for value in feature_vector]
    pt = item.get("pt") or []
    if pt and isinstance(pt[0], list):
        return [float(value) for row in pt for value in row]
    return [float(value) for value in pt]


def compact_entry_from_weight(item: dict[str, Any], item_index: int, precision: int) -> dict[str, Any] | None:
    vector = _flat_vector_from_weight(item)
    if not vector:
        return None
    features = item.get("features") or {}
    structure_payload = features.get("structure") or {}
    structure_vector = structure_payload.get("vector") if structure_payload else None
    annotation = item.get("annotation") or {}
    label = annotation.get("label") or item.get("label") or annotation.get("class_id") or "unknown"
    is_negative = bool(item.get("negative")) or str(label) == DEFAULT_NEGATIVE_LABEL
    source = item.get("source_image") or item.get("path") or item.get("name")
    rounded_structure = _round_vector(structure_vector, precision)
    return {
        "index": item_index,
        "label": str(label),
        "negative": is_negative,
        "pt": _round_vector(vector, precision) or [],
        "sample_weight": _round_float(item.get("sample_weight", 1.0), precision),
        "structure_pt": rounded_structure,
        "source": source,
        "source_key": _source_key(source),
    }


def _select_evenly_spaced(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if len(entries) <= limit:
        return entries
    if limit == 1:
        return [entries[0]]
    indexes = [round(index * (len(entries) - 1) / (limit - 1)) for index in range(limit)]
    selected = []
    seen = set()
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        selected.append(entries[index])
    return selected


def build_compact_entries(weights: list[dict[str, Any]], limit: int, precision: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    entries = [
        entry
        for index, item in enumerate(weights)
        for entry in [compact_entry_from_weight(item, index, precision)]
        if entry is not None
    ]
    if len(entries) <= limit:
        return entries

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((entry["label"], bool(entry.get("negative"))), []).append(entry)
    for group_entries in grouped.values():
        group_entries.sort(key=lambda item: (item.get("source_key") or "", int(item.get("index", 0))))

    quotas = {
        key: max(1, round(limit * len(group_entries) / max(1, len(entries))))
        for key, group_entries in grouped.items()
    }
    while sum(quotas.values()) > limit:
        key = max(quotas, key=lambda item: quotas[item])
        if quotas[key] <= 1:
            break
        quotas[key] -= 1
    while sum(quotas.values()) < limit:
        candidates = [
            key
            for key, group_entries in grouped.items()
            if quotas.get(key, 0) < len(group_entries)
        ]
        if not candidates:
            break
        key = max(candidates, key=lambda item: len(grouped[item]) - quotas.get(item, 0))
        quotas[key] += 1

    selected = []
    for key, group_entries in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        selected.extend(_select_evenly_spaced(group_entries, quotas.get(key, 0)))
    return selected[:limit]


__all__ = [
    "build_compact_entries",
    "compact_entry_from_weight",
]
