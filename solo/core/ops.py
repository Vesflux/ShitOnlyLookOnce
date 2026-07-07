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
from solo.utils.cv_image import Image

def _add_zero_border(pt: list[list[float]]) -> list[list[float]]:
    width = len(pt[0])
    return [[0.0 for _ in range(width + 2)]] + [[0.0] + row + [0.0] for row in pt] + [
        [0.0 for _ in range(width + 2)]
    ]

def _integral_image(pt: list[list[float]]) -> list[list[float]]:
    width = len(pt[0])
    integral = [[0.0 for _ in range(width + 1)]]
    for row in pt:
        running = 0.0
        integral_row = [0.0]
        previous_row = integral[-1]
        for x, value in enumerate(row):
            running += value
            integral_row.append(running + previous_row[x + 1])
        integral.append(integral_row)
    return integral

def _rect_sum(integral: list[list[float]], x1: int, y1: int, x2: int, y2: int) -> float:
    return integral[y2 + 1][x2 + 1] - integral[y1][x2 + 1] - integral[y2 + 1][x1] + integral[y1][x1]

def _clamped_square_sum(integral: list[list[float]], x: int, y: int, radius: int, width: int, height: int) -> tuple[float, int]:
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width - 1, x + radius)
    y2 = min(height - 1, y + radius)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    return _rect_sum(integral, x1, y1, x2, y2), area

def _ring_sum_and_count(
    integral: list[list[float]], x: int, y: int, radius: int, width: int, height: int
) -> tuple[float, int]:
    outer_sum, outer_count = _clamped_square_sum(integral, x, y, radius, width, height)
    inner_sum, inner_count = _clamped_square_sum(integral, x, y, radius - 1, width, height)
    return outer_sum - inner_sum, outer_count - inner_count

def _axis_sum_and_count(pt: list[list[float]], x: int, y: int, radius: int) -> tuple[float, int]:
    values = []
    for cell_x, cell_y in ((x - radius, y), (x + radius, y), (x, y - radius), (x, y + radius)):
        if 0 <= cell_y < len(pt) and 0 <= cell_x < len(pt[0]):
            values.append(pt[cell_y][cell_x])
    return sum(values), len(values)

def _global_radius_for_cell(x: int, y: int, width: int, height: int, max_radius: int) -> int:
    full_radius = max(x, y, width - 1 - x, height - 1 - y)
    return full_radius if max_radius == 0 else min(max_radius, full_radius)

def _compress_once_global(
    pt: list[list[float]], qua: int, nab: float, kernel: str = DEFAULT_KERNEL, max_radius: int = DEFAULT_MAX_RADIUS
) -> list[list[float]]:
    width = len(pt[0])
    height = len(pt)
    integral = _integral_image(pt)
    compressed = []

    for y in range(1, height - 1):
        row = []
        for x in range(1, width - 1):
            weighted_sum = pt[y][x]
            denominator = 1.0
            radius_limit = _global_radius_for_cell(x, y, width, height, max_radius)

            for radius in range(1, radius_limit + 1):
                ring_sum, ring_count = _ring_sum_and_count(integral, x, y, radius, width, height)
                if ring_count == 0:
                    continue

                ring_weight = nab**radius
                if kernel == "legacy":
                    weighted_sum += ring_sum * ring_weight
                    denominator += ring_count * ring_weight
                    continue

                axis_sum, axis_count = _axis_sum_and_count(pt, x, y, radius)
                off_axis_sum = ring_sum - axis_sum
                off_axis_count = ring_count - axis_count
                weighted_sum += axis_sum * ring_weight + off_axis_sum * ring_weight * 0.5
                denominator += axis_count * ring_weight + off_axis_count * ring_weight * 0.5

            row.append(round(weighted_sum / denominator, qua))
        compressed.append(row)

    return compressed

def _compress_once(
    pt: list[list[float]],
    qua: int,
    nab: float,
    kernel: str = DEFAULT_KERNEL,
    field: str = DEFAULT_FIELD,
    max_radius: int = DEFAULT_MAX_RADIUS,
) -> list[list[float]]:
    if field == "global":
        return _compress_once_global(pt, qua, nab, kernel, max_radius)

    denominator = _kernel_denominator(nab, kernel)
    compressed = []
    for y in range(1, len(pt) - 1):
        row = []
        for x in range(1, len(pt[0]) - 1):
            if kernel == "legacy":
                value = (
                    sum(pt[y - 1][x - 1 : x + 2]) * nab
                    + pt[y][x - 1] * nab
                    + pt[y][x]
                    + pt[y][x + 1] * nab
                    + sum(pt[y + 1][x - 1 : x + 2]) * nab
                ) / denominator
            else:
                diagonal_weight = nab / 2
                value = (
                    pt[y - 1][x - 1] * diagonal_weight
                    + pt[y - 1][x] * nab
                    + pt[y - 1][x + 1] * diagonal_weight
                    + pt[y][x - 1] * nab
                    + pt[y][x]
                    + pt[y][x + 1] * nab
                    + pt[y + 1][x - 1] * diagonal_weight
                    + pt[y + 1][x] * nab
                    + pt[y + 1][x + 1] * diagonal_weight
                ) / denominator
            row.append(round(value, qua))
        compressed.append(row)
    return compressed

def normalize_pt(pt: list[list[float]], qua: int = 8) -> list[list[float]]:
    flat = [value for row in pt for value in row]
    min_pt = min(flat)
    max_pt = max(flat)
    if max_pt == min_pt:
        return [[0.0 for _ in row] for row in pt]
    return [[round((value - min_pt) / (max_pt - min_pt), qua) for value in row] for row in pt]

def letptsmall(
    pt: list[list[float]],
    weight: int = 4,
    qua: int = 8,
    nab: float = DEFAULT_NAB,
    kernel: str = DEFAULT_KERNEL,
    field: str = DEFAULT_FIELD,
    max_radius: int = DEFAULT_MAX_RADIUS,
    normalize: bool = True,
    normalize_each_step: bool = True,
) -> list[list[float]]:
    _validate_pt(pt, weight, nab, kernel, field, max_radius)
    compressed = [[float(value) for value in row] for row in pt]
    compressed = _add_zero_border(compressed)

    while len(compressed) > weight:
        compressed = _compress_once(compressed, qua, nab, kernel, field, max_radius)
        if normalize and normalize_each_step:
            compressed = normalize_pt(compressed, qua)

    if normalize and not normalize_each_step:
        compressed = normalize_pt(compressed, qua)
    return compressed

def get_image_single_pt(
    path: str | Path,
    size: int = 2,
    qua: int = 8,
    nab: float = DEFAULT_NAB,
    pt_size: int = 8,
    kernel: str = DEFAULT_KERNEL,
    field: str = DEFAULT_FIELD,
    max_radius: int = DEFAULT_MAX_RADIUS,
    normalize: bool = True,
    normalize_each_step: bool = True,
) -> list[list[float]]:
    _validate_config(size, qua, nab, pt_size, kernel, field, max_radius)

    source_size = 16 * size
    image_path = Path(path)
    with Image.open(image_path) as image:
        return image_to_pt(
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

def image_to_pt(
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
) -> list[list[float]]:
    _validate_config(size, qua, nab, pt_size, kernel, field, max_radius)

    source_size = 16 * size
    gray = image.resize((source_size, source_size)).convert("L")
    pixels = list(gray.getdata())

    pt = []
    for y in range(source_size):
        row = []
        for x in range(source_size):
            row.append(round(1 - pixels[y * source_size + x] / 255, qua))
        pt.append(row)

    return letptsmall(
        pt,
        weight=pt_size,
        qua=qua,
        nab=nab,
        kernel=kernel,
        field=field,
        max_radius=max_radius,
        normalize=normalize,
        normalize_each_step=normalize_each_step,
    )

def _values_to_grid_pt(
    values: list[float],
    source_size: int,
    pt_size: int,
    qua: int,
    normalize: bool,
) -> list[list[float]]:
    if len(values) != source_size * source_size:
        raise ValueError("channel values do not match source size")
    grid = []
    for grid_y in range(pt_size):
        y1 = math.floor(grid_y * source_size / pt_size)
        y2 = max(y1 + 1, math.floor((grid_y + 1) * source_size / pt_size))
        row = []
        for grid_x in range(pt_size):
            x1 = math.floor(grid_x * source_size / pt_size)
            x2 = max(x1 + 1, math.floor((grid_x + 1) * source_size / pt_size))
            total = 0.0
            count = 0
            for y in range(y1, min(source_size, y2)):
                offset = y * source_size
                for x in range(x1, min(source_size, x2)):
                    total += values[offset + x]
                    count += 1
            row.append(round(total / max(1, count), qua))
        grid.append(row)
    return normalize_pt(grid, qua) if normalize else grid

def _contextualize_pt(
    pt: list[list[float]],
    qua: int,
    nab: float,
    kernel: str,
    field: str,
    max_radius: int,
    normalize: bool,
) -> list[list[float]]:
    if field not in {"local", "global"}:
        raise ValueError("field must be 'local' or 'global'")
    if len(pt) <= 1:
        return pt

    height = len(pt)
    width = len(pt[0])
    integral = _integral_image(pt)
    contextualized = []
    for y in range(height):
        row = []
        for x in range(width):
            weighted_sum = pt[y][x]
            denominator = 1.0
            if field == "local":
                radius_limit = 1
            else:
                radius_limit = _global_radius_for_cell(x, y, width, height, max_radius)
            for radius in range(1, radius_limit + 1):
                ring_sum, ring_count = _ring_sum_and_count(integral, x, y, radius, width, height)
                if ring_count == 0:
                    continue
                ring_weight = nab**radius
                if kernel == "legacy":
                    weighted_sum += ring_sum * ring_weight
                    denominator += ring_count * ring_weight
                    continue
                axis_sum, axis_count = _axis_sum_and_count(pt, x, y, radius)
                off_axis_sum = ring_sum - axis_sum
                off_axis_count = ring_count - axis_count
                weighted_sum += axis_sum * ring_weight + off_axis_sum * ring_weight * 0.5
                denominator += axis_count * ring_weight + off_axis_count * ring_weight * 0.5
            row.append(round(weighted_sum / denominator, qua))
        contextualized.append(row)
    return normalize_pt(contextualized, qua) if normalize else contextualized

def _channel_to_fast_pt(
    values: list[float],
    source_size: int,
    qua: int,
    nab: float,
    pt_size: int,
    kernel: str,
    field: str,
    max_radius: int,
    normalize: bool,
) -> list[list[float]]:
    grid = _values_to_grid_pt(values, source_size, pt_size, qua, normalize)
    return _contextualize_pt(grid, qua, nab, kernel, field, max_radius, normalize)

def _flatten_pt(pt: list[list[float]]) -> list[float]:
    return [float(value) for row in pt for value in row]
__all__ = [
    '_add_zero_border',
    '_integral_image',
    '_rect_sum',
    '_clamped_square_sum',
    '_ring_sum_and_count',
    '_axis_sum_and_count',
    '_global_radius_for_cell',
    '_compress_once_global',
    '_compress_once',
    'normalize_pt',
    'letptsmall',
    'get_image_single_pt',
    'image_to_pt',
    '_values_to_grid_pt',
    '_contextualize_pt',
    '_channel_to_fast_pt',
    '_flatten_pt',
]
