from __future__ import annotations

from typing import Any

from solo.utils.bbox import _bbox_tuple, _clamp01, _smoothstep


def _bbox_geometry(
    bbox: dict[str, Any] | tuple[int, int, int, int],
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, float]:
    if isinstance(bbox, dict):
        left, top, right, bottom = _bbox_tuple(bbox)
        normalized = bbox.get("normalized") or {}
        image_width = image_width or max(1, round((right - left) / max(float(normalized.get("x2", 0.0)) - float(normalized.get("x1", 0.0)), 1e-6)))
        image_height = image_height or max(1, round((bottom - top) / max(float(normalized.get("y2", 0.0)) - float(normalized.get("y1", 0.0)), 1e-6)))
    else:
        left, top, right, bottom = bbox
        image_width = image_width or max(1, right)
        image_height = image_height or max(1, bottom)
        normalized = {}

    box_width = max(1.0, float(right - left))
    box_height = max(1.0, float(bottom - top))
    source_width = max(1.0, float(image_width or 1))
    source_height = max(1.0, float(image_height or 1))
    norm_width = float(normalized.get("x2", right / source_width)) - float(normalized.get("x1", left / source_width))
    norm_height = float(normalized.get("y2", bottom / source_height)) - float(normalized.get("y1", top / source_height))
    center_x = (float(normalized.get("x1", left / source_width)) + float(normalized.get("x2", right / source_width))) / 2
    center_y = (float(normalized.get("y1", top / source_height)) + float(normalized.get("y2", bottom / source_height))) / 2
    wide_aspect = box_width / box_height
    tall_aspect = box_height / box_width
    return {
        "width": box_width,
        "height": box_height,
        "area": box_width * box_height,
        "wide_aspect": wide_aspect,
        "tall_aspect": tall_aspect,
        "norm_width": max(0.0, norm_width),
        "norm_height": max(0.0, norm_height),
        "norm_area": max(0.0, norm_width * norm_height),
        "center_x": center_x,
        "center_y": center_y,
    }


def _integral_for_values(values: list[float], width: int, height: int) -> list[float]:
    integral_width = width + 1
    integral = [0.0] * ((height + 1) * integral_width)
    for y in range(height):
        row_sum = 0.0
        source_offset = y * width
        previous_offset = y * integral_width
        current_offset = (y + 1) * integral_width
        for x in range(width):
            row_sum += values[source_offset + x]
            integral[current_offset + x + 1] = integral[previous_offset + x + 1] + row_sum
    return integral


def _integral_mean(
    integral: list[float],
    width: int,
    height: int,
    bbox: tuple[int, int, int, int],
) -> float:
    if not integral or width <= 0 or height <= 0:
        return 0.0
    left, top, right, bottom = bbox
    left = max(0, min(width, left))
    right = max(left, min(width, right))
    top = max(0, min(height, top))
    bottom = max(top, min(height, bottom))
    area = max(1, (right - left) * (bottom - top))
    integral_width = width + 1
    total = (
        integral[bottom * integral_width + right]
        - integral[top * integral_width + right]
        - integral[bottom * integral_width + left]
        + integral[top * integral_width + left]
    )
    return total / area


def _mean_region_from_maps(maps: dict[str, Any], channel: str, bbox: tuple[int, int, int, int]) -> float:
    values = maps.get(channel) or []
    width = int(maps.get("width") or 0)
    height = int(maps.get("height") or 0)
    if not values or width <= 0 or height <= 0:
        return 0.0
    expected_length = width * height
    if len(values) < expected_length:
        return 0.0
    integrals = maps.setdefault("_integral_maps", {})
    integral = integrals.get(channel)
    if integral is None:
        integral = _integral_for_values(values, width, height)
        integrals[channel] = integral
    return _integral_mean(integral, width, height, bbox)


def proposal_diagnostics(
    bbox: dict[str, Any] | tuple[int, int, int, int],
    maps: dict[str, Any] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, float]:
    if maps is not None:
        image_width = int(maps.get("width") or image_width or 1)
        image_height = int(maps.get("height") or image_height or 1)
    geometry = _bbox_geometry(bbox, image_width=image_width, image_height=image_height)
    if isinstance(bbox, dict):
        bbox_tuple = _bbox_tuple(bbox)
    else:
        bbox_tuple = bbox

    edge_mean = texture_mean = saturation_mean = gray_mean = 0.0
    if maps is not None:
        edge_mean = _mean_region_from_maps(maps, "edge", bbox_tuple)
        texture_mean = _mean_region_from_maps(maps, "texture", bbox_tuple)
        saturation_mean = _mean_region_from_maps(maps, "saturation", bbox_tuple)
        gray_mean = _mean_region_from_maps(maps, "gray", bbox_tuple)

    narrow_score = 1.0 - _smoothstep(0.055, 0.16, geometry["norm_width"])
    thin_vertical_score = _smoothstep(2.4, 5.4, geometry["tall_aspect"]) * narrow_score
    lower_region_score = _smoothstep(0.58, 0.86, geometry["center_y"])
    low_detail_score = 1.0 - _smoothstep(0.035, 0.12, edge_mean * 0.65 + texture_mean * 0.35)
    low_saturation_score = 1.0 - _smoothstep(0.08, 0.30, saturation_mean)
    large_patch_score = _smoothstep(0.025, 0.18, geometry["norm_area"])
    road_score = lower_region_score * low_detail_score * (0.70 + low_saturation_score * 0.30) * large_patch_score
    flat_score = _smoothstep(1.75, 3.10, geometry["wide_aspect"]) * (
        1.0 - _smoothstep(5.0, 7.5, geometry["wide_aspect"])
    )
    flat_road_score = (
        lower_region_score
        * flat_score
        * (1.0 - _smoothstep(0.16, 0.36, saturation_mean))
        * (1.0 - _smoothstep(0.085, 0.18, edge_mean * 0.55 + texture_mean * 0.45))
        * _smoothstep(0.0035, 0.018, geometry["norm_area"])
    )
    upper_region_score = 1.0 - _smoothstep(0.30, 0.48, geometry["center_y"])
    small_fragment_score = 1.0 - _smoothstep(0.003, 0.018, geometry["norm_area"])
    vertical_fragment_score = (
        upper_region_score
        * small_fragment_score
        * (1.0 - _smoothstep(1.05, 1.45, geometry["wide_aspect"]))
        * _smoothstep(1.12, 1.90, geometry["tall_aspect"])
    )
    vehicle_shape_score = (
        _smoothstep(0.95, 1.45, geometry["wide_aspect"])
        * (1.0 - _smoothstep(5.2, 8.0, geometry["wide_aspect"]))
        * (1.0 - _smoothstep(1.7, 2.7, geometry["tall_aspect"]))
    )
    partial_vehicle_score = (
        vehicle_shape_score
        * _smoothstep(0.35, 0.78, edge_mean * 0.65 + texture_mean * 0.35)
        * (1.0 - _smoothstep(0.18, 0.42, geometry["norm_area"]))
    )
    penalty = _clamp01(
        thin_vertical_score * 0.70
        + road_score * 0.55
        + flat_road_score * 0.72
        + vertical_fragment_score * 0.65
    )
    multiplier = max(0.12, 1.0 - penalty)
    return {
        **{key: round(value, 6) for key, value in geometry.items()},
        "edge_mean": round(edge_mean, 6),
        "texture_mean": round(texture_mean, 6),
        "saturation_mean": round(saturation_mean, 6),
        "gray_mean": round(gray_mean, 6),
        "thin_vertical_score": round(_clamp01(thin_vertical_score), 6),
        "road_score": round(_clamp01(road_score), 6),
        "flat_road_score": round(_clamp01(flat_road_score), 6),
        "vertical_fragment_score": round(_clamp01(vertical_fragment_score), 6),
        "vehicle_shape_score": round(_clamp01(vehicle_shape_score), 6),
        "partial_vehicle_score": round(_clamp01(partial_vehicle_score), 6),
        "shape_penalty": round(penalty, 6),
        "shape_multiplier": round(multiplier, 6),
    }


def shape_multiplier_from_diagnostics(diagnostics: dict[str, Any] | None) -> float:
    if not diagnostics:
        return 1.0
    try:
        return max(0.12, min(1.08, float(diagnostics.get("shape_multiplier", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def proposal_rank_bonus(diagnostics: dict[str, Any] | None) -> float:
    if not diagnostics:
        return 0.0
    vehicle = float(diagnostics.get("vehicle_shape_score", 0.0) or 0.0)
    partial = float(diagnostics.get("partial_vehicle_score", 0.0) or 0.0)
    penalty = float(diagnostics.get("shape_penalty", 0.0) or 0.0)
    flat_road = float(diagnostics.get("flat_road_score", 0.0) or 0.0)
    vertical_fragment = float(diagnostics.get("vertical_fragment_score", 0.0) or 0.0)
    return vehicle * 0.10 + partial * 0.08 - penalty * 0.28 - flat_road * 0.16 - vertical_fragment * 0.14


__all__ = [
    "proposal_diagnostics",
    "shape_multiplier_from_diagnostics",
    "proposal_rank_bonus",
]
