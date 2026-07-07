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
from solo.core.accelerator import *

def _mse(left: list[float], right: list[float]) -> float:
    length = min(len(left), len(right))
    if length == 0:
        return float("inf")
    return sum((left[index] - right[index]) ** 2 for index in range(length)) / length

def _build_prototypes(
    entries: list[dict[str, Any]],
    vector_length: int,
    prototype_count: int = DEFAULT_PROTOTYPE_COUNT,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, bool], dict[str, Any]] = {}
    for entry in entries:
        key = (entry["label"], bool(entry.get("negative")))
        group = groups.setdefault(
            key,
            {
                "label": entry["label"],
                "negative": bool(entry.get("negative")),
                "entries": [],
            },
        )
        group["entries"].append(entry)

    prototypes = []
    for group in groups.values():
        group_entries = group["entries"]
        cluster_count = max(1, min(int(prototype_count), len(group_entries)))
        clusters = _cluster_entries(group_entries, vector_length, cluster_count, accelerator=accelerator)
        for cluster_index, cluster_entries in enumerate(clusters):
            if not cluster_entries:
                continue
            prototype = _prototype_from_entries(
                cluster_entries,
                vector_length,
                label=group["label"],
                negative=group["negative"],
                cluster_index=cluster_index,
            )
            prototypes.append(prototype)
    return prototypes

def _cluster_entries(
    entries: list[dict[str, Any]],
    vector_length: int,
    cluster_count: int,
    iterations: int = 8,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[list[dict[str, Any]]]:
    if cluster_count <= 1 or len(entries) <= 1:
        return [entries]
    seeds = []
    sorted_entries = sorted(entries, key=lambda item: (item.get("source_key", ""), item.get("index", 0)))
    for index in range(cluster_count):
        seed_index = round(index * (len(sorted_entries) - 1) / max(1, cluster_count - 1))
        seeds.append(list(sorted_entries[seed_index]["pt"]))
    assignments = [0] * len(entries)
    for _iteration in range(iterations):
        changed = False
        vectors = [entry["pt"] for entry in entries]
        distance_columns = [
            batch_mse(seed, vectors, accelerator=accelerator)
            if accelerator != "cpu"
            else None
            for seed in seeds
        ]
        for entry_index, entry in enumerate(entries):
            if any(column is None for column in distance_columns):
                distances = [_mse(entry["pt"], seed) for seed in seeds]
            else:
                distances = [float(column[entry_index]) for column in distance_columns if column is not None]
            cluster_index = distances.index(min(distances))
            if assignments[entry_index] != cluster_index:
                changed = True
                assignments[entry_index] = cluster_index
        clusters = [[] for _ in range(cluster_count)]
        for entry, cluster_index in zip(entries, assignments):
            clusters[cluster_index].append(entry)
        for cluster_index, cluster_entries in enumerate(clusters):
            if not cluster_entries:
                continue
            total_weight = sum(float(entry.get("sample_weight", 1.0)) for entry in cluster_entries)
            centroid = [0.0] * vector_length
            for entry in cluster_entries:
                sample_weight = float(entry.get("sample_weight", 1.0))
                for value_index, value in enumerate(entry["pt"]):
                    centroid[value_index] += value * sample_weight
            seeds[cluster_index] = [value / max(total_weight, 1e-9) for value in centroid]
        if not changed:
            break
    clusters = [[] for _ in range(cluster_count)]
    for entry, cluster_index in zip(entries, assignments):
        clusters[cluster_index].append(entry)
    return [cluster for cluster in clusters if cluster]

def _average_entry_vectors(entries: list[dict[str, Any]], key: str, length: int) -> tuple[list[float] | None, list[float] | None]:
    usable = [entry for entry in entries if entry.get(key) and len(entry[key]) == length]
    if not usable:
        return None, None
    total_weight = sum(float(entry.get("sample_weight", 1.0)) for entry in usable)
    vector_sum = [0.0] * length
    for entry in usable:
        sample_weight = float(entry.get("sample_weight", 1.0))
        for index, value in enumerate(entry[key]):
            vector_sum[index] += float(value) * sample_weight
    return [value / max(total_weight, 1e-9) for value in vector_sum], vector_sum

def _prototype_from_entries(
    entries: list[dict[str, Any]],
    vector_length: int,
    label: str,
    negative: bool,
    cluster_index: int,
) -> dict[str, Any]:
    total_weight = sum(float(entry.get("sample_weight", 1.0)) for entry in entries)
    vector_sum = [0.0] * vector_length
    source_sums: dict[str, dict[str, Any]] = {}
    for entry in entries:
        sample_weight = float(entry.get("sample_weight", 1.0))
        source_key = entry.get("source_key") or ""
        source_sum = source_sums.setdefault(
            source_key,
            {"count": 0, "weight": 0.0, "sum": [0.0] * vector_length, "source": entry.get("source")},
        )
        source_sum["count"] += 1
        source_sum["weight"] += sample_weight
        for index, value in enumerate(entry["pt"]):
            weighted_value = value * sample_weight
            vector_sum[index] += weighted_value
            source_sum["sum"][index] += weighted_value
    first = entries[0]
    structure_length = max((len(entry.get("structure_pt") or []) for entry in entries), default=0)
    structure_pt = None
    structure_sum = None
    if structure_length > 0:
        structure_pt, structure_sum = _average_entry_vectors(entries, "structure_pt", structure_length)
    return {
        "label": label,
        "negative": negative,
        "cluster_index": cluster_index,
        "count": len(entries),
        "weight": total_weight,
        "pt": [value / max(total_weight, 1e-9) for value in vector_sum],
        "sum": vector_sum,
        "source_sums": source_sums,
        "structure_pt": structure_pt,
        "structure_sum": structure_sum,
        "structure_length": structure_length,
        "weight_path": first["weight_path"],
        "weight_index": first["index"],
        "weight_source": first["source"],
    }
__all__ = [
    '_mse',
    '_build_prototypes',
    '_cluster_entries',
    '_average_entry_vectors',
    '_prototype_from_entries',
]
