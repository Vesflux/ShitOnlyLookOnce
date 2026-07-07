from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from solo.config import IMAGE_SUFFIXES


def neural_image_paths(path: str | Path) -> list[Path]:
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


def neural_label_path(image_path: Path, labels_dir: str | Path | None = None) -> Path | None:
    if labels_dir is not None:
        candidate = Path(labels_dir) / f"{image_path.stem}.txt"
        return candidate if candidate.exists() else None
    candidates = [image_path.with_suffix(".txt"), image_path.parent / f"{image_path.stem}.txt"]
    parts = image_path.parts
    if "images" in parts:
        index = parts.index("images")
        candidates.append(Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt"))
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return candidate
    return None


def read_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_yolo_boxes(label_path: Path | None) -> tuple[np.ndarray, list[str]]:
    if label_path is None or not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), []
    boxes: list[list[float]] = []
    labels: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO annotation at {label_path}:{line_number}")
        label = parts[0]
        values = [float(value) for value in parts[1:]]
        if len(values) == 4:
            cx, cy, width, height = values
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            raise ValueError(f"invalid YOLO annotation at {label_path}:{line_number}")
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), []
    return np.asarray(boxes, dtype=np.float32), labels


def letterbox_cv2(image: np.ndarray, image_size: int = 640) -> tuple[np.ndarray, dict[str, Any]]:
    if image_size <= 0:
        height, width = image.shape[:2]
        return image, {
            "enabled": False,
            "input_size": 0,
            "source_width": width,
            "source_height": height,
            "scale": 1.0,
            "pad_x": 0,
            "pad_y": 0,
            "resized_width": width,
            "resized_height": height,
        }
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    boxed = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    boxed[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return boxed, {
        "enabled": True,
        "input_size": image_size,
        "source_width": width,
        "source_height": height,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }


def boxes_to_letterbox(boxes: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    if boxes.size == 0:
        return boxes.astype(np.float32)
    source_width = float(info["source_width"])
    source_height = float(info["source_height"])
    input_size = float(info["input_size"] or max(source_width, source_height))
    scale = float(info["scale"])
    pad_x = float(info["pad_x"])
    pad_y = float(info["pad_y"])
    converted = boxes.astype(np.float32).copy()
    converted[:, [0, 2]] = (converted[:, [0, 2]] * source_width * scale + pad_x) / input_size
    converted[:, [1, 3]] = (converted[:, [1, 3]] * source_height * scale + pad_y) / input_size
    converted = np.clip(converted, 0.0, 1.0)
    keep = (converted[:, 2] > converted[:, 0]) & (converted[:, 3] > converted[:, 1])
    return converted[keep]


def boxes_to_letterbox_labeled(
    boxes: np.ndarray,
    labels: list[str],
    info: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    if boxes.size == 0:
        return boxes.astype(np.float32), []
    source_width = float(info["source_width"])
    source_height = float(info["source_height"])
    input_size = float(info["input_size"] or max(source_width, source_height))
    scale = float(info["scale"])
    pad_x = float(info["pad_x"])
    pad_y = float(info["pad_y"])
    converted = boxes.astype(np.float32).copy()
    converted[:, [0, 2]] = (converted[:, [0, 2]] * source_width * scale + pad_x) / input_size
    converted[:, [1, 3]] = (converted[:, [1, 3]] * source_height * scale + pad_y) / input_size
    converted = np.clip(converted, 0.0, 1.0)
    keep = (converted[:, 2] > converted[:, 0]) & (converted[:, 3] > converted[:, 1])
    kept_labels = [label for label, should_keep in zip(labels, keep.tolist()) if should_keep]
    return converted[keep], kept_labels


def map_square_box_to_source(box: tuple[float, float, float, float], info: dict[str, Any]) -> tuple[int, int, int, int] | None:
    input_size = float(info["input_size"] or max(info["source_width"], info["source_height"]))
    scale = float(info["scale"])
    if scale <= 0:
        return None
    pad_x = float(info["pad_x"])
    pad_y = float(info["pad_y"])
    source_width = int(info["source_width"])
    source_height = int(info["source_height"])
    x1, y1, x2, y2 = box
    mapped = (
        (x1 * input_size - pad_x) / scale,
        (y1 * input_size - pad_y) / scale,
        (x2 * input_size - pad_x) / scale,
        (y2 * input_size - pad_y) / scale,
    )
    left = max(0, min(source_width, round(min(mapped[0], mapped[2]))))
    top = max(0, min(source_height, round(min(mapped[1], mapped[3]))))
    right = max(0, min(source_width, round(max(mapped[0], mapped[2]))))
    bottom = max(0, min(source_height, round(max(mapped[1], mapped[3]))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _clip_boxes(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.astype(np.float32)
    clipped = np.clip(boxes.astype(np.float32), 0.0, 1.0)
    keep = (clipped[:, 2] - clipped[:, 0] > 0.004) & (clipped[:, 3] - clipped[:, 1] > 0.004)
    return clipped[keep]


def _clip_boxes_labeled(boxes: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    if boxes.size == 0:
        return boxes.astype(np.float32), []
    clipped = np.clip(boxes.astype(np.float32), 0.0, 1.0)
    keep = (clipped[:, 2] - clipped[:, 0] > 0.004) & (clipped[:, 3] - clipped[:, 1] > 0.004)
    kept_labels = [label for label, should_keep in zip(labels, keep.tolist()) if should_keep]
    return clipped[keep], kept_labels


def augment_letterboxed(
    image: np.ndarray,
    boxes: np.ndarray,
    rng: random.Random,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    output = image.copy()
    augmented_boxes = boxes.astype(np.float32).copy()
    augmented_labels = list(labels or [])
    size = output.shape[0]

    if augmented_boxes.size and rng.random() < 0.5:
        output = np.ascontiguousarray(output[:, ::-1])
        x1 = augmented_boxes[:, 0].copy()
        x2 = augmented_boxes[:, 2].copy()
        augmented_boxes[:, 0] = 1.0 - x2
        augmented_boxes[:, 2] = 1.0 - x1

    if rng.random() < 0.75:
        scale = rng.uniform(0.86, 1.14)
        tx = rng.uniform(-0.08, 0.08) * size
        ty = rng.uniform(-0.06, 0.06) * size
        matrix = np.asarray([[scale, 0.0, tx + (1.0 - scale) * size / 2], [0.0, scale, ty + (1.0 - scale) * size / 2]])
        output = cv2.warpAffine(
            output,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        if augmented_boxes.size:
            corners = augmented_boxes.copy() * size
            x1, y1, x2, y2 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
            points = np.stack(
                [
                    np.stack([x1, y1, np.ones_like(x1)], axis=1),
                    np.stack([x2, y1, np.ones_like(x1)], axis=1),
                    np.stack([x1, y2, np.ones_like(x1)], axis=1),
                    np.stack([x2, y2, np.ones_like(x1)], axis=1),
                ],
                axis=1,
            )
            transformed = points @ matrix.T
            augmented_boxes = np.stack(
                [
                    transformed[:, :, 0].min(axis=1) / size,
                    transformed[:, :, 1].min(axis=1) / size,
                    transformed[:, :, 0].max(axis=1) / size,
                    transformed[:, :, 1].max(axis=1) / size,
                ],
                axis=1,
            )
            if labels is None:
                augmented_boxes = _clip_boxes(augmented_boxes)
            else:
                augmented_boxes, augmented_labels = _clip_boxes_labeled(augmented_boxes, augmented_labels)

    if rng.random() < 0.85:
        hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
        hsv[:, :, 1] *= rng.uniform(0.75, 1.25)
        hsv[:, :, 2] *= rng.uniform(0.72, 1.28)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if rng.random() < 0.45:
        alpha = rng.uniform(0.82, 1.18)
        beta = rng.uniform(-18, 18)
        output = np.clip(output.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.12:
        output = cv2.GaussianBlur(output, (3, 3), 0)

    if labels is None:
        return output, augmented_boxes
    return output, augmented_boxes, augmented_labels


def labels_to_tensor(labels: list[str]) -> torch.Tensor:
    values: list[int] = []
    for label in labels:
        try:
            value = int(float(label))
        except ValueError:
            value = 0
        values.append(max(0, value))
    return torch.tensor(values, dtype=torch.long)


class NeuralDetectionDataset(Dataset):
    def __init__(
        self,
        images: str | Path,
        labels_dir: str | Path | None,
        image_size: int = 640,
        augment: bool = False,
        seed: int = 42,
        cache_images: bool = False,
    ) -> None:
        self.paths = neural_image_paths(images)
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.cache_images = bool(cache_images)
        self.cache: list[dict[str, Any]] | None = None
        if self.cache_images:
            self.cache = [self._load_item(path) for path in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def _load_item(self, path: Path) -> dict[str, Any]:
        image = read_rgb_image(path)
        boxes, labels = read_yolo_boxes(neural_label_path(path, self.labels_dir))
        boxed, letterbox = letterbox_cv2(image, self.image_size)
        boxes, labels = boxes_to_letterbox_labeled(boxes, labels, letterbox)
        return {
            "boxed": np.ascontiguousarray(boxed),
            "boxes": boxes.astype(np.float32),
            "labels": list(labels),
            "letterbox": letterbox,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        if self.cache is None:
            item = self._load_item(path)
        else:
            item = self.cache[index]
        boxed = item["boxed"]
        boxes = item["boxes"]
        labels = item["labels"]
        letterbox = item["letterbox"]
        if self.augment:
            rng = random.Random(self.seed + index * 1000003 + random.randint(0, 999999))
            boxed, boxes, labels = augment_letterboxed(boxed, boxes, rng, labels=labels)
        tensor = torch.from_numpy(np.ascontiguousarray(boxed)).permute(2, 0, 1).float().div(255.0)
        return {
            "image": tensor,
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": labels_to_tensor(labels),
            "path": str(path),
            "letterbox": letterbox,
        }


def collate_detection_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "path": [item["path"] for item in batch],
        "letterbox": [item["letterbox"] for item in batch],
    }


__all__ = [
    "NeuralDetectionDataset",
    "augment_letterboxed",
    "boxes_to_letterbox",
    "boxes_to_letterbox_labeled",
    "collate_detection_batch",
    "labels_to_tensor",
    "letterbox_cv2",
    "map_square_box_to_source",
    "neural_image_paths",
    "neural_label_path",
    "read_rgb_image",
    "read_yolo_boxes",
]
