from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from solo.config import IMAGE_SUFFIXES
from solo.taichi_detector.feature_layout import FEATURE_CHANNELS, FEATURE_SIZE, feature_dimension
from solo.utils.bbox import _bbox_payload, _clamp_bbox

LOW_CONTRAST_RANGE_THRESHOLD = 30.0 / 255.0
LOW_CONTRAST_GRADIENT_THRESHOLD = 0.035
LOW_CONTRAST_STD_THRESHOLD = 0.055


def image_paths(path: str | Path) -> list[Path]:
    image_path = Path(path)
    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {image_path}")
        return [image_path]
    if not image_path.exists():
        raise FileNotFoundError(f"image path not found: {image_path}")
    paths = sorted(item for item in image_path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no image files found in: {image_path}")
    return paths


def read_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    expanded = _clamp_bbox(
        (
            x1 - width * padding,
            y1 - height * padding,
            x2 + width * padding,
            y2 + height * padding,
        ),
        image_width,
        image_height,
    )
    return expanded or bbox


def _safe_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _quality_features(target_stats: np.ndarray, context_stats: np.ndarray) -> np.ndarray:
    target_gray_std = float(target_stats[1])
    target_sat = float(target_stats[2])
    target_value = float(target_stats[4])
    target_grad = float(target_stats[6])
    context_sat = float(context_stats[2])
    context_grad = float(context_stats[6])

    target_grayness = 1.0 - _smoothstep(0.08, 0.24, target_sat)
    target_flatness = 1.0 - _smoothstep(0.035, 0.16, target_grad + target_gray_std * 0.35)
    target_darkness = 1.0 - _smoothstep(0.32, 0.62, target_value)
    target_shadow = max(0.0, min(1.0, target_grayness * (target_darkness * 0.55 + target_flatness * 0.45)))

    context_gray_std = float(context_stats[1])
    context_value = float(context_stats[4])
    context_grayness = 1.0 - _smoothstep(0.08, 0.24, context_sat)
    context_flatness = 1.0 - _smoothstep(0.035, 0.16, context_grad + context_gray_std * 0.35)
    context_darkness = 1.0 - _smoothstep(0.32, 0.62, context_value)
    context_shadow = max(0.0, min(1.0, context_grayness * (context_darkness * 0.55 + context_flatness * 0.45)))

    edge_ratio = min(3.0, (target_grad + 1e-4) / (context_grad + 1e-4)) / 3.0
    sat_delta = max(0.0, min(1.0, (target_sat - context_sat + 1.0) * 0.5))
    return np.asarray(
        [
            target_grayness,
            target_flatness,
            target_darkness,
            target_shadow,
            context_grayness,
            context_shadow,
            edge_ratio,
            sat_delta,
        ],
        dtype=np.float32,
    )


def _is_low_contrast_view(channels: np.ndarray, stats: np.ndarray) -> bool:
    gray_range = float(np.max(channels[0]) - np.min(channels[0]))
    value_range = float(np.max(channels[2]) - np.min(channels[2]))
    absolute_range = max(gray_range, value_range)
    gray_std = float(stats[1])
    value_std = float(stats[5])
    gradient_mean = float(stats[6])
    return (
        absolute_range < LOW_CONTRAST_RANGE_THRESHOLD
        and max(gray_std, value_std) < LOW_CONTRAST_STD_THRESHOLD
        and gradient_mean < LOW_CONTRAST_GRADIENT_THRESHOLD
    )


def _suppress_low_contrast_view(channels: np.ndarray, stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not _is_low_contrast_view(channels, stats):
        return channels, stats
    return np.zeros_like(channels, dtype=np.float32), np.zeros_like(stats, dtype=np.float32)


def _precompute_feature_planes(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(np.sqrt(grad_x * grad_x + grad_y * grad_y), 0.0, 1.0)
    return np.ascontiguousarray(
        np.stack(
            [
                gray,
                hsv[:, :, 1] / 255.0,
                hsv[:, :, 2] / 255.0,
                lab[:, :, 2] / 255.0,
                gradient,
            ],
            axis=2,
        ).astype(np.float32)
    )


def _resize_feature_planes(crop: np.ndarray, feature_size: int) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((feature_size, feature_size, FEATURE_CHANNELS), dtype=np.float32)
    crop = np.ascontiguousarray(crop.astype(np.float32, copy=False))
    if crop.shape[2] <= 4:
        resized = cv2.resize(crop, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return resized.astype(np.float32, copy=False)
    head = cv2.resize(crop[:, :, :4], (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    tail = cv2.resize(crop[:, :, 4], (feature_size, feature_size), interpolation=cv2.INTER_AREA)[:, :, None]
    return np.concatenate([head, tail], axis=2).astype(np.float32, copy=False)


def _visual_channels_from_planes(
    planes: np.ndarray,
    bbox: tuple[int, int, int, int],
    feature_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    crop = planes[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    resized = _resize_feature_planes(crop, feature_size)
    channels = np.moveaxis(resized, 2, 0)
    gray = channels[0]
    saturation = channels[1]
    value = channels[2]
    gradient = channels[4]
    stats = np.asarray(
        [
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.mean(saturation)),
            float(np.std(saturation)),
            float(np.mean(value)),
            float(np.std(value)),
            float(np.mean(gradient)),
            float(np.std(gradient)),
        ],
        dtype=np.float32,
    )
    channels, stats = _suppress_low_contrast_view(channels, stats)
    return channels, stats


def _visual_channels(crop: np.ndarray, feature_size: int) -> tuple[np.ndarray, np.ndarray]:
    if crop.size == 0:
        crop = np.zeros((feature_size, feature_size, 3), dtype=np.uint8)
    resized = cv2.resize(crop, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB).astype(np.float32)
    blue_yellow = lab[:, :, 2] / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(np.sqrt(grad_x * grad_x + grad_y * grad_y), 0.0, 1.0)
    channels = np.stack([gray, saturation, value, blue_yellow, gradient], axis=0)
    stats = np.asarray(
        [
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.mean(saturation)),
            float(np.std(saturation)),
            float(np.mean(value)),
            float(np.std(value)),
            float(np.mean(gradient)),
            float(np.std(gradient)),
        ],
        dtype=np.float32,
    )
    channels, stats = _suppress_low_contrast_view(channels, stats)
    return channels, stats


def crop_feature_vector(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    context_padding: float = 0.30,
    feature_size: int = FEATURE_SIZE,
) -> np.ndarray:
    return batch_crop_feature_matrix(
        image,
        [bbox],
        context_padding=context_padding,
        feature_size=feature_size,
        backend="cpu",
    )[0]


def _feature_vector_from_precomputed(
    planes: np.ndarray,
    bbox: tuple[int, int, int, int],
    context_padding: float,
    feature_size: int,
) -> np.ndarray:
    height, width = planes.shape[:2]
    padded = expand_bbox(bbox, width, height, context_padding)
    target_channels, target_stats = _visual_channels_from_planes(planes, bbox, feature_size)
    context_channels, context_stats = _visual_channels_from_planes(planes, padded, feature_size)
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    padded_w = max(1.0, float(padded[2] - padded[0]))
    padded_h = max(1.0, float(padded[3] - padded[1]))
    geometry = np.asarray(
        [
            box_w / max(1.0, float(width)),
            box_h / max(1.0, float(height)),
            min(6.0, box_w / box_h) / 6.0,
            min(1.0, (box_w * box_h) / max(1.0, float(width * height))),
            ((x1 + x2) * 0.5) / max(1.0, float(width)),
            ((y1 + y2) * 0.5) / max(1.0, float(height)),
            padded_w / max(1.0, float(width)),
            padded_h / max(1.0, float(height)),
            min(6.0, padded_w / padded_h) / 6.0,
            min(1.0, (box_w * box_h) / max(1.0, padded_w * padded_h)),
        ],
        dtype=np.float32,
    )
    stat_delta = target_stats - context_stats
    quality = _quality_features(target_stats, context_stats)
    return np.concatenate(
        [
            target_channels.reshape(-1),
            context_channels.reshape(-1),
            target_stats,
            context_stats,
            stat_delta,
            quality,
            geometry,
        ],
        axis=0,
    ).astype(np.float32)


def _batch_crop_feature_matrix_cv2(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    context_padding: float,
    feature_size: int,
) -> np.ndarray:
    planes = _precompute_feature_planes(image)
    return batch_crop_feature_matrix_from_planes(
        planes,
        bboxes,
        context_padding=context_padding,
        feature_size=feature_size,
    )


def batch_crop_feature_matrix_from_planes(
    planes: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    context_padding: float = 0.30,
    feature_size: int = FEATURE_SIZE,
) -> np.ndarray:
    if not bboxes:
        return np.zeros((0, feature_dimension(feature_size)), dtype=np.float32)
    return np.stack(
        [
            _feature_vector_from_precomputed(
                planes,
                bbox,
                context_padding=context_padding,
                feature_size=feature_size,
            )
            for bbox in bboxes
        ]
    ).astype(np.float32)


def _batch_crop_feature_matrix_taichi(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    context_padding: float,
    feature_size: int,
    backend: str,
    require_taichi: bool = False,
) -> np.ndarray | None:
    enabled = os.environ.get("SOLO_EXPERIMENTAL_TAICHI_FEATURES", "").strip().lower()
    if not require_taichi and enabled not in {"1", "true", "yes"}:
        return None
    if backend == "cpu":
        if require_taichi:
            raise RuntimeError("Taichi feature backend requires a GPU backend; got backend='cpu'")
        return None

    from solo.taichi_detector.feature_kernels import batch_crop_feature_matrix_taichi

    planes = _precompute_feature_planes(image)
    chunk_size = max(0, int(os.environ.get("SOLO_TAICHI_FEATURE_BATCH", "768") or "0"))
    if chunk_size > 0 and len(bboxes) > chunk_size:
        chunks = []
        for start in range(0, len(bboxes), chunk_size):
            matrix = batch_crop_feature_matrix_taichi(
                planes,
                bboxes[start : start + chunk_size],
                context_padding=context_padding,
                feature_size=feature_size,
                backend=backend,
            )
            if matrix is None:
                if require_taichi:
                    raise RuntimeError("Taichi feature backend requested, but no GPU Taichi runtime is available")
                return None
            chunks.append(matrix)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    matrix = batch_crop_feature_matrix_taichi(
        planes,
        bboxes,
        context_padding=context_padding,
        feature_size=feature_size,
        backend=backend,
    )
    if matrix is None and require_taichi:
        raise RuntimeError("Taichi feature backend requested, but no GPU Taichi runtime is available")
    return matrix


def batch_crop_feature_matrix(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    context_padding: float = 0.30,
    feature_size: int = FEATURE_SIZE,
    backend: str = "auto",
    feature_backend: str = "opencv",
) -> np.ndarray:
    if not bboxes:
        return np.zeros((0, feature_dimension(feature_size)), dtype=np.float32)
    resolved_feature_backend = feature_backend.lower().strip()
    if resolved_feature_backend not in {"opencv", "taichi"}:
        raise ValueError(f"unsupported feature_backend: {feature_backend}")
    try:
        matrix = _batch_crop_feature_matrix_taichi(
            image,
            bboxes,
            context_padding=context_padding,
            feature_size=feature_size,
            backend=backend,
            require_taichi=resolved_feature_backend == "taichi",
        )
    except Exception:
        if resolved_feature_backend == "taichi":
            raise
        matrix = None
    if matrix is not None:
        return matrix
    return _batch_crop_feature_matrix_cv2(
        image,
        bboxes,
        context_padding=context_padding,
        feature_size=feature_size,
    )


def bbox_payload_from_tuple(bbox: tuple[int, int, int, int], image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape[:2]
    return _bbox_payload(bbox, width, height)


__all__ = [
    "FEATURE_SIZE",
    "batch_crop_feature_matrix",
    "batch_crop_feature_matrix_from_planes",
    "bbox_payload_from_tuple",
    "crop_feature_vector",
    "expand_bbox",
    "feature_dimension",
    "image_paths",
    "_precompute_feature_planes",
    "read_rgb_image",
]
