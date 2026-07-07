from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from solo.taichi_detector.runtime import initialize_taichi

_RESIZE_KERNEL_CACHE: dict[int, object] = {}
_GAUSSIAN_3X3 = np.asarray(
    [
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ],
    dtype=np.float32,
) / 16.0


def gaussian_blur_3x3(image: np.ndarray) -> np.ndarray:
    if min(image.shape[:2]) < 2:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(cv2.filter2D(image, -1, _GAUSSIAN_3X3, borderType=cv2.BORDER_REPLICATE))


def _get_resize_kernel(ti):
    cache_key = id(ti)
    cached = _RESIZE_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    @ti.kernel
    def resize_area_f32(
        source: ti.types.ndarray(dtype=ti.f32, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=3),
        source_height: ti.i32,
        source_width: ti.i32,
        output_height: ti.i32,
        output_width: ti.i32,
    ):
        for y, x, channel in ti.ndrange(output_height, output_width, 3):
            y0f = ti.cast(y * source_height, ti.f32) / ti.cast(output_height, ti.f32)
            y1f = ti.cast((y + 1) * source_height, ti.f32) / ti.cast(output_height, ti.f32)
            x0f = ti.cast(x * source_width, ti.f32) / ti.cast(output_width, ti.f32)
            x1f = ti.cast((x + 1) * source_width, ti.f32) / ti.cast(output_width, ti.f32)
            y0 = ti.max(0, ti.min(source_height - 1, ti.cast(ti.floor(y0f), ti.i32)))
            y1 = ti.max(y0 + 1, ti.min(source_height, ti.cast(ti.ceil(y1f), ti.i32)))
            x0 = ti.max(0, ti.min(source_width - 1, ti.cast(ti.floor(x0f), ti.i32)))
            x1 = ti.max(x0 + 1, ti.min(source_width, ti.cast(ti.ceil(x1f), ti.i32)))
            total = 0.0
            weight_total = 0.0
            for sy in range(y0, y1):
                wy = ti.max(0.0, ti.min(ti.cast(sy + 1, ti.f32), y1f) - ti.max(ti.cast(sy, ti.f32), y0f))
                for sx in range(x0, x1):
                    wx = ti.max(0.0, ti.min(ti.cast(sx + 1, ti.f32), x1f) - ti.max(ti.cast(sx, ti.f32), x0f))
                    weight = wx * wy
                    total += source[sy, sx, channel] * weight
                    weight_total += weight
            value = ti.round(total / ti.max(1e-6, weight_total))
            output[y, x, channel] = ti.max(0.0, ti.min(255.0, value))

    _RESIZE_KERNEL_CACHE[cache_key] = resize_area_f32
    return resize_area_f32


def resize_image_taichi(image: np.ndarray, width: int, height: int, backend: str = "auto") -> np.ndarray | None:
    if backend == "cpu" or width <= 0 or height <= 0:
        return None
    runtime = initialize_taichi(backend)
    if not runtime.get("available") or runtime.get("arch") == "cpu":
        return None
    try:
        ti = runtime["ti"]
        source = np.ascontiguousarray(image.astype(np.float32))
        output = np.zeros((height, width, 3), dtype=np.float32)
        kernel = _get_resize_kernel(ti)
        kernel(source, output, source.shape[0], source.shape[1], height, width)
    except Exception:
        return None
    return np.clip(output, 0, 255).astype(np.uint8)


def resize_image_area(image: np.ndarray, width: int, height: int, backend: str = "auto") -> np.ndarray:
    if width == image.shape[1] and height == image.shape[0]:
        return np.ascontiguousarray(image)
    source = gaussian_blur_3x3(image) if width < image.shape[1] or height < image.shape[0] else image
    resized = resize_image_taichi(source, width, height, backend=backend)
    if resized is not None:
        return np.ascontiguousarray(resized)
    return np.ascontiguousarray(cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA))


def build_image_pyramid(
    image: np.ndarray,
    scales: tuple[float, ...],
    backend: str = "auto",
    min_side: int = 16,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    levels: list[dict[str, Any]] = []
    seen_sizes: set[tuple[int, int]] = set()
    for raw_scale in scales:
        scale = float(raw_scale)
        if scale <= 0.0:
            continue
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        if min(target_width, target_height) < min_side:
            continue
        size_key = (target_width, target_height)
        if size_key in seen_sizes:
            continue
        seen_sizes.add(size_key)
        level_image = (
            np.ascontiguousarray(image)
            if target_width == width and target_height == height
            else resize_image_area(image, target_width, target_height, backend=backend)
        )
        levels.append(
            {
                "image": level_image,
                "scale": scale,
                "scale_x": target_width / max(1.0, float(width)),
                "scale_y": target_height / max(1.0, float(height)),
                "width": target_width,
                "height": target_height,
            }
        )
    if not levels:
        levels.append(
            {
                "image": np.ascontiguousarray(image),
                "scale": 1.0,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "width": width,
                "height": height,
            }
        )
    levels.sort(key=lambda item: float(item["scale"]), reverse=True)
    return levels


__all__ = ["build_image_pyramid", "gaussian_blur_3x3", "resize_image_area", "resize_image_taichi"]
