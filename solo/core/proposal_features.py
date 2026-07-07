from __future__ import annotations

import math
from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.utils.cv_image import Image
def _proposal_passes_shape_filters(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_area: int,
    max_area_ratio: float,
    min_box_size: int,
    max_box_size: int,
    max_aspect_ratio: float,
) -> bool:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < min_box_size or box_height < min_box_size:
        return False
    if max_box_size and (box_width > max_box_size or box_height > max_box_size):
        return False
    box_area = box_width * box_height
    if box_area < min_area:
        return False
    if max_area_ratio > 0 and box_area > image_width * image_height * max_area_ratio:
        return False
    aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
    return not (max_aspect_ratio and aspect > max_aspect_ratio)

def _objectness_maps(image: Image.Image) -> tuple[list[float], int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    hsv = rgb.convert("HSV")
    rgb_pixels = list(rgb.getdata())
    hsv_pixels = list(hsv.getdata())
    gray_values = [
        (red * 0.299 + green * 0.587 + blue * 0.114) / 255
        for red, green, blue in rgb_pixels
    ]
    edge_values = _edge_values_square_or_rect(gray_values, width, height)
    contrast_values = _local_contrast_values_rect(gray_values, width, height)
    texture_values = _texture_values_rect(gray_values, width, height)
    saturation_values = [sat / 255 for _hue, sat, _val in hsv_pixels]
    mean_gray = sum(gray_values) / max(1, len(gray_values))
    objectness = []
    for gray, edge, contrast, texture, saturation in zip(
        gray_values,
        edge_values,
        contrast_values,
        texture_values,
        saturation_values,
    ):
        brightness_deviation = abs(gray - mean_gray)
        value = (
            edge * 0.34
            + contrast * 0.26
            + texture * 0.20
            + min(1.0, brightness_deviation * 2.5) * 0.14
            + saturation * 0.06
        )
        objectness.append(_clamp01(value))
    return objectness, width, height

def _edge_values_square_or_rect(values: list[float], width: int, height: int) -> list[float]:
    return _edge_values_square(values, width, height)

def _edge_values_square(values: list[float], width: int, height: int) -> list[float]:
    edges = []
    for y in range(height):
        for x in range(width):
            center = values[y * width + x]
            right = values[y * width + min(width - 1, x + 1)]
            down = values[min(height - 1, y + 1) * width + x]
            edges.append(min(1.0, abs(center - right) + abs(center - down)))
    return edges

def _local_contrast_values_rect(values: list[float], width: int, height: int) -> list[float]:
    contrasts = []
    for y in range(height):
        for x in range(width):
            neighbors = []
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbors.append(values[ny * width + nx])
            center = values[y * width + x]
            local_mean = sum(neighbors) / len(neighbors)
            contrasts.append(min(1.0, abs(center - local_mean) * 2.0))
    return contrasts

def _texture_values_rect(values: list[float], width: int, height: int) -> list[float]:
    textures = []
    for y in range(height):
        for x in range(width):
            neighbors = []
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbors.append(values[ny * width + nx])
            mean = sum(neighbors) / len(neighbors)
            variance = sum((value - mean) ** 2 for value in neighbors) / len(neighbors)
            textures.append(min(1.0, math.sqrt(variance) * 3.0))
    return textures

def _mean_in_bbox(values: list[float], width: int, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    total = 0.0
    count = 0
    for y in range(y1, y2):
        offset = y * width
        for x in range(x1, x2):
            total += values[offset + x]
            count += 1
    return total / max(1, count)

def _region_stats(values: list[float], width: int, bbox: tuple[int, int, int, int]) -> dict[str, float]:
    x1, y1, x2, y2 = bbox
    region_values = []
    for y in range(max(0, y1), max(0, y2)):
        offset = y * width
        for x in range(max(0, x1), max(0, x2)):
            region_values.append(values[offset + x])
    if not region_values:
        return {"mean": 0.0, "std": 0.0, "high_ratio": 0.0}
    mean_value = sum(region_values) / len(region_values)
    variance = sum((value - mean_value) ** 2 for value in region_values) / len(region_values)
    high_ratio = sum(1 for value in region_values if value >= mean_value) / len(region_values)
    return {"mean": mean_value, "std": math.sqrt(variance), "high_ratio": high_ratio}

def context_maps_for_image(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    hsv = rgb.convert("HSV")
    gray_values = [
        (red * 0.299 + green * 0.587 + blue * 0.114) / 255
        for red, green, blue in rgb.getdata()
    ]
    saturation_values = [sat / 255 for _hue, sat, _val in hsv.getdata()]
    edge_values = _edge_values_square_or_rect(gray_values, width, height)
    texture_values = _texture_values_rect(gray_values, width, height)
    return {
        "width": width,
        "height": height,
        "gray": gray_values,
        "saturation": saturation_values,
        "edge": edge_values,
        "texture": texture_values,
    }

def context_quality_for_bbox_from_maps(
    maps: dict[str, Any],
    bbox: tuple[int, int, int, int],
    expand_ratio: float = 0.35,
    qua: int = 8,
) -> dict[str, Any]:
    width = int(maps["width"])
    height = int(maps["height"])
    edge_values = maps["edge"]
    texture_values = maps["texture"]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = max(2, round(box_width * expand_ratio))
    pad_y = max(2, round(box_height * expand_ratio))
    outer = _clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
    if outer is None:
        outer = bbox

    inside_edge = _region_stats(edge_values, width, bbox)
    inside_texture = _region_stats(texture_values, width, bbox)
    ox1, oy1, ox2, oy2 = outer
    ring_edges = []
    ring_textures = []
    for y in range(oy1, oy2):
        offset = y * width
        for x in range(ox1, ox2):
            if x1 <= x < x2 and y1 <= y < y2:
                continue
            index = offset + x
            ring_edges.append(edge_values[index])
            ring_textures.append(texture_values[index])

    if not ring_edges:
        outside_edge = {"mean": 0.0, "std": 0.0, "high_ratio": 0.0}
        outside_texture = {"mean": 0.0, "std": 0.0, "high_ratio": 0.0}
    else:
        outside_edge_mean = sum(ring_edges) / len(ring_edges)
        outside_texture_mean = sum(ring_textures) / len(ring_textures)
        outside_edge = {
            "mean": outside_edge_mean,
            "std": math.sqrt(sum((value - outside_edge_mean) ** 2 for value in ring_edges) / len(ring_edges)),
            "high_ratio": sum(1 for value in ring_edges if value >= outside_edge_mean) / len(ring_edges),
        }
        outside_texture = {
            "mean": outside_texture_mean,
            "std": math.sqrt(sum((value - outside_texture_mean) ** 2 for value in ring_textures) / len(ring_textures)),
            "high_ratio": sum(1 for value in ring_textures if value >= outside_texture_mean) / len(ring_textures),
        }

    left_edge = _mean_in_bbox(edge_values, width, (x1, y1, min(width, x1 + max(1, box_width // 6)), y2))
    right_edge = _mean_in_bbox(edge_values, width, (max(0, x2 - max(1, box_width // 6)), y1, x2, y2))
    top_edge = _mean_in_bbox(edge_values, width, (x1, y1, x2, min(height, y1 + max(1, box_height // 6))))
    bottom_edge = _mean_in_bbox(edge_values, width, (x1, max(0, y2 - max(1, box_height // 6)), x2, y2))
    boundary_edge = (left_edge + right_edge + top_edge + bottom_edge) / 4
    side_regions = {
        "left": (ox1, y1, x1, y2),
        "right": (x2, y1, ox2, y2),
        "top": (x1, oy1, x2, y1),
        "bottom": (x1, y2, x2, oy2),
    }
    boundary_edges = {
        "left": left_edge,
        "right": right_edge,
        "top": top_edge,
        "bottom": bottom_edge,
    }
    side_continuation = {}
    side_partialness = {}
    for side, side_bbox in side_regions.items():
        side_stats = _region_stats(edge_values, width, side_bbox)
        continuation = side_stats["mean"] / max(inside_edge["mean"], 1e-6)
        boundary_ratio = boundary_edges[side] / max(inside_edge["mean"], 1e-6)
        partialness = _smoothstep(0.72, 1.25, continuation) * (1.0 - _smoothstep(0.85, 1.45, boundary_ratio))
        side_continuation[side] = continuation
        side_partialness[side] = partialness

    continuation_ratio = outside_edge["mean"] / max(inside_edge["mean"], 1e-6)
    texture_continuation_ratio = outside_texture["mean"] / max(inside_texture["mean"], 1e-6)
    boundary_to_inner_ratio = boundary_edge / max(inside_edge["mean"], 1e-6)
    boundary_support = _smoothstep(0.75, 1.25, boundary_to_inner_ratio)
    context_penalty = max(
        _smoothstep(0.72, 1.25, continuation_ratio),
        _smoothstep(0.80, 1.35, texture_continuation_ratio) * 0.8,
        max(side_partialness.values(), default=0.0),
    )
    partialness = _clamp01(context_penalty * (1.0 - boundary_support * 0.55))
    quality = _clamp01(1.0 - partialness * 0.85)

    return {
        "quality": round(quality, qua),
        "multiplier": round(quality, qua),
        "partialness": round(partialness, qua),
        "continuation_ratio": round(continuation_ratio, qua),
        "texture_continuation_ratio": round(texture_continuation_ratio, qua),
        "boundary_to_inner_ratio": round(boundary_to_inner_ratio, qua),
        "inside_edge": round(inside_edge["mean"], qua),
        "outside_edge": round(outside_edge["mean"], qua),
        "inside_texture": round(inside_texture["mean"], qua),
        "outside_texture": round(outside_texture["mean"], qua),
        "boundary_edge": round(boundary_edge, qua),
        "side_continuation": {side: round(value, qua) for side, value in side_continuation.items()},
        "side_partialness": {side: round(value, qua) for side, value in side_partialness.items()},
    }

def context_quality_for_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    expand_ratio: float = 0.35,
    qua: int = 8,
) -> dict[str, Any]:
    return context_quality_for_bbox_from_maps(context_maps_for_image(image), bbox, expand_ratio=expand_ratio, qua=qua)

def fragmentation_quality_from_box_quality(box_quality: dict[str, Any] | None, qua: int = 8) -> dict[str, Any] | None:
    if not box_quality:
        return None
    center_texture = float(box_quality.get("center_texture", 0.0) or 0.0)
    mean_texture = float(box_quality.get("mean_texture", 0.0) or 0.0)
    center_edge = float(box_quality.get("center_edge", 0.0) or 0.0)
    mean_edge = float(box_quality.get("mean_edge", 0.0) or 0.0)
    center_balance = float(box_quality.get("center_balance", 0.0) or 0.0)
    texture_ratio = center_texture / max(mean_texture, 1e-6)
    edge_ratio = center_edge / max(mean_edge, 1e-6)
    high_texture = _smoothstep(1.05, 1.85, texture_ratio)
    high_edge = _smoothstep(1.05, 1.85, edge_ratio)
    weak_shape = 1.0 - _smoothstep(0.65, 1.10, center_balance)
    fragmentation = _clamp01(high_texture * 0.45 + high_edge * 0.35 + weak_shape * 0.20)
    quality = _clamp01(1.0 - fragmentation)
    return {
        "fragmentation": round(fragmentation, qua),
        "quality": round(quality, qua),
        "texture_ratio": round(texture_ratio, qua),
        "edge_ratio": round(edge_ratio, qua),
        "weak_shape": round(weak_shape, qua),
    }

def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    scale: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    if scale <= 0:
        raise ValueError("bbox scale must be greater than 0")
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale
    return _clamp_bbox(
        (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2),
        image_width,
        image_height,
    )

def _project_values(
    values: list[float],
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
    axis: str,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    if axis == "x":
        projection = []
        for x in range(x1, x2):
            total = 0.0
            count = 0
            for y in range(y1, y2):
                total += values[y * width + x]
                count += 1
            projection.append(total / max(1, count))
        return projection
    projection = []
    for y in range(y1, y2):
        offset = y * width
        total = 0.0
        count = 0
        for x in range(x1, x2):
            total += values[offset + x]
            count += 1
        projection.append(total / max(1, count))
    return projection

def _trim_projection(
    projection: list[float],
    start: int,
    threshold: float,
    min_span: int,
) -> tuple[int, int]:
    if not projection:
        return start, start
    left = 0
    right = len(projection) - 1
    while left < right and projection[left] < threshold:
        left += 1
    while right > left and projection[right] < threshold:
        right -= 1
    if right - left + 1 < min_span:
        return start, start + len(projection)
    return start + left, start + right + 1

def refine_proposal_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    search_ratio: float = 0.18,
    min_box_size: int = 4,
    maps: dict[str, Any] | None = None,
) -> tuple[int, int, int, int] | None:
    maps = maps or context_maps_for_image(image)
    width = int(maps["width"])
    height = int(maps["height"])
    edge_values = maps["edge"]
    texture_values = maps["texture"]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = max(2, round(box_width * search_ratio))
    pad_y = max(2, round(box_height * search_ratio))
    search = _clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
    if search is None:
        return bbox
    sx1, sy1, sx2, sy2 = search
    combined = [edge * 0.72 + texture * 0.28 for edge, texture in zip(edge_values, texture_values)]
    proj_x = _project_values(combined, width, height, search, "x")
    proj_y = _project_values(combined, width, height, search, "y")
    mean_x = sum(proj_x) / max(1, len(proj_x))
    mean_y = sum(proj_y) / max(1, len(proj_y))
    threshold_x = max(mean_x * 0.72, max(proj_x or [0.0]) * 0.18)
    threshold_y = max(mean_y * 0.72, max(proj_y or [0.0]) * 0.18)
    rx1, rx2 = _trim_projection(proj_x, sx1, threshold_x, min_box_size)
    ry1, ry2 = _trim_projection(proj_y, sy1, threshold_y, min_box_size)
    refined = _clamp_bbox((rx1, ry1, rx2, ry2), width, height)
    if refined is None:
        return bbox
    return refined

def _component_proposals_from_mask(
    mask: bytearray,
    scores: list[float],
    width: int,
    height: int,
    min_area: int,
    max_area_ratio: float,
    expand: float,
    min_box_size: int,
    max_box_size: int,
    max_aspect_ratio: float,
    proposal_name: str,
) -> list[dict[str, Any]]:
    seen = bytearray(width * height)
    proposals = []
    for start_y in range(height):
        for start_x in range(width):
            start_index = start_y * width + start_x
            if not mask[start_index] or seen[start_index]:
                continue
            stack = [(start_x, start_y)]
            seen[start_index] = 1
            xs = []
            ys = []
            score_total = 0.0
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                score_total += scores[y * width + x]
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    row_offset = ny * width
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        next_index = row_offset + nx
                        if mask[next_index] and not seen[next_index]:
                            seen[next_index] = 1
                            stack.append((nx, ny))
            area = len(xs)
            bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            if not _proposal_passes_shape_filters(
                bbox,
                width,
                height,
                min_area,
                max_area_ratio,
                min_box_size,
                max_box_size,
                max_aspect_ratio,
            ):
                continue
            expanded = _expand_bbox(bbox, width, height, expand)
            if expanded is None:
                continue
            proposals.append(
                {
                    "bbox": _bbox_payload(expanded, width, height),
                    "proposal": proposal_name,
                    "area": area,
                    "objectness": round(score_total / max(1, area), 6),
                }
            )
    return proposals

def _scale_work_proposals_to_source(
    proposals: list[dict[str, Any]],
    work_scale: float,
    source_width: int,
    source_height: int,
) -> list[dict[str, Any]]:
    if work_scale >= 1.0:
        return proposals
    scaled = []
    scale_area = max(work_scale * work_scale, 1e-6)
    for proposal in proposals:
        bbox = _bbox_tuple(proposal["bbox"])
        source_bbox = _clamp_bbox(
            (
                bbox[0] / work_scale,
                bbox[1] / work_scale,
                bbox[2] / work_scale,
                bbox[3] / work_scale,
            ),
            source_width,
            source_height,
        )
        if source_bbox is None:
            continue
        item = dict(proposal)
        item["bbox"] = _bbox_payload(source_bbox, source_width, source_height)
        if "area" in item:
            item["area"] = round(float(item["area"]) / scale_area)
        item["work_scale"] = round(work_scale, 6)
        scaled.append(item)
    return scaled

def edge_component_proposals(
    image: Image.Image,
    min_area: int = 80,
    max_area_ratio: float = 0.25,
    expand: float = 1.12,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 8.0,
    maps: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_width, source_height = image.size
    work_scale = 1.0
    if maps is None:
        work_scale = min(1.0, DEFAULT_OBJECTNESS_WORK_SIZE / max(source_width, source_height, 1))
        if work_scale < 1.0:
            work_width = max(16, round(source_width * work_scale))
            work_height = max(16, round(source_height * work_scale))
            image = image.resize((work_width, work_height))
            min_area = max(4, round(min_area * work_scale * work_scale))
            max_box_size = round(max_box_size * work_scale) if max_box_size else 0
            min_box_size = max(3, round(min_box_size * work_scale))
        maps = context_maps_for_image(image)
    width = int(maps["width"])
    height = int(maps["height"])
    edge_values = maps["edge"]
    texture_values = maps["texture"]
    scores = [edge * 0.78 + texture * 0.22 for edge, texture in zip(edge_values, texture_values)]
    mean_score = sum(scores) / max(1, len(scores))
    variance = sum((value - mean_score) ** 2 for value in scores) / max(1, len(scores))
    threshold = max(0.045, min(0.30, mean_score + math.sqrt(variance) * 0.55))
    seed = bytearray(1 if score >= threshold else 0 for score in scores)
    mask = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if not seed[index]:
                continue
            for ny in range(max(0, y - 1), min(height, y + 2)):
                row_offset = ny * width
                for nx in range(max(0, x - 2), min(width, x + 3)):
                    mask[row_offset + nx] = 1
    proposals = _component_proposals_from_mask(
        mask,
        scores,
        width,
        height,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        expand=expand,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        max_aspect_ratio=max_aspect_ratio,
        proposal_name="edge_component",
    )
    return _scale_work_proposals_to_source(proposals, work_scale, source_width, source_height)

def horizontal_body_proposals(
    image: Image.Image,
    min_area: int = 80,
    max_area_ratio: float = 0.35,
    expand: float = 1.08,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 10.0,
    maps: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_width, source_height = image.size
    work_scale = 1.0
    if maps is None:
        work_scale = min(1.0, DEFAULT_OBJECTNESS_WORK_SIZE / max(source_width, source_height, 1))
        if work_scale < 1.0:
            work_width = max(16, round(source_width * work_scale))
            work_height = max(16, round(source_height * work_scale))
            image = image.resize((work_width, work_height))
            min_area = max(4, round(min_area * work_scale * work_scale))
            max_box_size = round(max_box_size * work_scale) if max_box_size else 0
            min_box_size = max(3, round(min_box_size * work_scale))
        maps = context_maps_for_image(image)
    width = int(maps["width"])
    height = int(maps["height"])
    edge_values = maps["edge"]
    texture_values = maps["texture"]
    gray_values = maps["gray"]
    scores = [edge * 0.62 + texture * 0.22 + abs(gray - 0.5) * 0.16 for edge, texture, gray in zip(edge_values, texture_values, gray_values)]
    row_scores = []
    for y in range(height):
        row = scores[y * width : (y + 1) * width]
        row_scores.append(sum(row) / max(1, len(row)))
    mean_row = sum(row_scores) / max(1, len(row_scores))
    max_row = max(row_scores or [0.0])
    row_threshold = max(mean_row * 1.05, max_row * 0.32)
    bands: list[tuple[int, int]] = []
    y = 0
    while y < height:
        while y < height and row_scores[y] < row_threshold:
            y += 1
        start = y
        while y < height and row_scores[y] >= row_threshold:
            y += 1
        end = y
        if end - start >= max(3, min_box_size // 2):
            pad_y = max(2, round((end - start) * 0.65))
            bands.append((max(0, start - pad_y), min(height, end + pad_y)))

    proposals = []
    for y1, y2 in bands:
        if y2 <= y1:
            continue
        projection = _project_values(scores, width, height, (0, y1, width, y2), "x")
        if not projection:
            continue
        mean_col = sum(projection) / len(projection)
        max_col = max(projection)
        threshold = max(mean_col * 1.02, max_col * 0.24)
        x = 0
        while x < width:
            while x < width and projection[x] < threshold:
                x += 1
            start = x
            while x < width and projection[x] >= threshold:
                x += 1
            end = x
            if end - start < min_box_size:
                continue
            pad_x = max(2, round((end - start) * 0.12))
            bbox = _clamp_bbox((start - pad_x, y1, end + pad_x, y2), width, height)
            if bbox is None:
                continue
            box_width = bbox[2] - bbox[0]
            box_height = bbox[3] - bbox[1]
            if box_width < box_height * 0.85:
                continue
            if not _proposal_passes_shape_filters(
                bbox,
                width,
                height,
                min_area,
                max_area_ratio,
                min_box_size,
                max_box_size,
                max_aspect_ratio,
            ):
                continue
            expanded = _expand_bbox(bbox, width, height, expand)
            if expanded is None:
                continue
            proposals.append(
                {
                    "bbox": _bbox_payload(expanded, width, height),
                    "proposal": "horizontal_body",
                    "objectness": round(sum(projection[start:end]) / max(1, end - start), 6),
                }
            )
    return _scale_work_proposals_to_source(proposals, work_scale, source_width, source_height)

def color_proposals(
    image: Image.Image,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
) -> list[dict[str, Any]]:
    source_width, source_height = image.size
    work_scale = min(1.0, DEFAULT_OBJECTNESS_WORK_SIZE / max(source_width, source_height, 1))
    if work_scale < 1.0:
        work_width = max(16, round(source_width * work_scale))
        work_height = max(16, round(source_height * work_scale))
        work_image = image.resize((work_width, work_height))
        min_area = max(4, round(min_area * work_scale * work_scale))
        max_box_size = round(max_box_size * work_scale) if max_box_size else 0
        min_box_size = max(3, round(min_box_size * work_scale))
    else:
        work_image = image
    objectness, width, height = _objectness_maps(work_image)
    mean_score = sum(objectness) / max(1, len(objectness))
    variance = sum((value - mean_score) ** 2 for value in objectness) / max(1, len(objectness))
    threshold = max(0.08, min(0.38, mean_score + math.sqrt(variance) * 0.65))
    seed = bytearray(1 if score >= threshold else 0 for score in objectness)
    mask = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if not seed[index]:
                continue
            for ny in range(max(0, y - 1), min(height, y + 2)):
                row_offset = ny * width
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    mask[row_offset + nx] = 1

    seen = bytearray(width * height)
    proposals = []
    for start_y in range(height):
        for start_x in range(width):
            index = start_y * width + start_x
            if not mask[index] or seen[index]:
                continue
            stack = [(start_x, start_y)]
            seen[index] = 1
            xs = []
            ys = []
            score_total = 0.0
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                score_total += objectness[y * width + x]
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    row_offset = ny * width
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        next_index = row_offset + nx
                        if mask[next_index] and not seen[next_index]:
                            seen[next_index] = 1
                            stack.append((nx, ny))

            area = len(xs)
            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            bbox = (x1, y1, x2, y2)
            if not _proposal_passes_shape_filters(
                bbox,
                width,
                height,
                min_area,
                max_area_ratio,
                min_box_size,
                max_box_size,
                max_aspect_ratio,
            ):
                continue
            expanded = _expand_bbox(bbox, width, height, expand)
            if expanded is None:
                continue
            if work_scale < 1.0:
                source_bbox = _clamp_bbox(
                    (
                        expanded[0] / work_scale,
                        expanded[1] / work_scale,
                        expanded[2] / work_scale,
                        expanded[3] / work_scale,
                    ),
                    source_width,
                    source_height,
                )
            else:
                source_bbox = expanded
            if source_bbox is None:
                continue
            ex1, ey1, ex2, ey2 = source_bbox
            proposals.append(
                {
                    "bbox": _bbox_payload((ex1, ey1, ex2, ey2), source_width, source_height),
                    "proposal": "objectness",
                    "area": round(area / max(work_scale * work_scale, 1e-6)),
                    "objectness": round(score_total / max(1, area), 6),
                }
            )
    return proposals

def sliding_proposals(
    image: Image.Image,
    window_sizes: list[int],
    window_ratios: list[float] | None = None,
    stride_ratio: float = 0.5,
    window_templates: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    width, height = image.size
    proposals = []
    templates: list[tuple[float, float]] = []
    for window_size in window_sizes:
        if window_size > 0:
            templates.append((window_size, window_size))
    for ratio in window_ratios or []:
        if ratio <= 0:
            continue
        window_size = max(4, round(min(width, height) * ratio))
        templates.append((window_size, window_size))
    for template_width, template_height in window_templates or []:
        if template_width > 0 and template_height > 0:
            templates.append((template_width, template_height))
    templates = list(dict.fromkeys(templates))

    for template_width, template_height in templates:
        if template_width <= 0 or template_height <= 0:
            continue
        window_width = min(template_width, width)
        window_height = min(template_height, height)
        stride_x = max(1, int(window_width * stride_ratio))
        stride_y = max(1, int(window_height * stride_ratio))
        y_values = list(range(0, max(1, height - window_height + 1), stride_y))
        x_values = list(range(0, max(1, width - window_width + 1), stride_x))
        if y_values[-1] != height - window_height:
            y_values.append(height - window_height)
        if x_values[-1] != width - window_width:
            x_values.append(width - window_width)
        for y in y_values:
            for x in x_values:
                bbox = (x, y, x + window_width, y + window_height)
                proposals.append(
                    {
                        "bbox": _bbox_payload(bbox, width, height),
                        "proposal": "sliding",
                        "window_width": window_width,
                        "window_height": window_height,
                    }
                )
    return proposals

def anchor_proposals(
    image: Image.Image,
    anchor_templates: list[tuple[int, int]],
    stride_ratio: float = 0.5,
    max_positions_per_template: int = 160,
) -> list[dict[str, Any]]:
    width, height = image.size
    proposals = []
    for template_width, template_height in dict.fromkeys(anchor_templates):
        if template_width <= 0 or template_height <= 0:
            continue
        window_width = min(template_width, width)
        window_height = min(template_height, height)
        stride_x = max(1, int(window_width * stride_ratio))
        stride_y = max(1, int(window_height * stride_ratio))
        if max_positions_per_template > 0:
            estimated_x = math.ceil(max(1, width - window_width + 1) / stride_x) + 1
            estimated_y = math.ceil(max(1, height - window_height + 1) / stride_y) + 1
            estimated_positions = estimated_x * estimated_y
            if estimated_positions > max_positions_per_template:
                scale = math.sqrt(estimated_positions / max_positions_per_template)
                stride_x = max(stride_x, round(stride_x * scale))
                stride_y = max(stride_y, round(stride_y * scale))
        y_values = list(range(0, max(1, height - window_height + 1), stride_y))
        x_values = list(range(0, max(1, width - window_width + 1), stride_x))
        if y_values[-1] != height - window_height:
            y_values.append(height - window_height)
        if x_values[-1] != width - window_width:
            x_values.append(width - window_width)
        for y in y_values:
            for x in x_values:
                bbox = (x, y, x + window_width, y + window_height)
                proposals.append(
                    {
                        "bbox": _bbox_payload(bbox, width, height),
                        "proposal": "anchor",
                        "window_width": window_width,
                        "window_height": window_height,
                    }
                )
    return proposals
__all__ = [
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
