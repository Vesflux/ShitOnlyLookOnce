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

def load_class_names(path: str | Path | None) -> dict[int, str]:
    if path is None:
        return {}
    names_path = Path(path)
    if not names_path.exists():
        raise FileNotFoundError(f"class names file not found: {names_path}")
    names = {}
    for index, line in enumerate(names_path.read_text(encoding="utf-8-sig").splitlines()):
        name = line.strip()
        if name:
            names[index] = name
    return names

def _annotation_candidates(image_path: Path, labels_dir: str | Path | None, suffix: str) -> list[Path]:
    if labels_dir is not None:
        return [Path(labels_dir) / f"{image_path.stem}{suffix}"]

    candidates = [image_path.with_suffix(suffix), image_path.parent / f"{image_path.stem}{suffix}"]
    parts = image_path.parts
    if "images" in parts:
        index = parts.index("images")
        replaced = Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(suffix)
        candidates.append(replaced)
    return list(dict.fromkeys(candidates))

def find_annotation_path(image_path: Path, annotations: str, labels_dir: str | Path | None = None) -> Path | None:
    suffix = ".txt" if annotations == "yolo" else ".json"
    for candidate in _annotation_candidates(image_path, labels_dir, suffix):
        if candidate.exists():
            return candidate
    return None

def load_yolo_annotations(
    image_path: Path,
    annotation_path: Path,
    class_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    class_names = class_names or {}
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    annotations = []
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"invalid YOLO annotation at {annotation_path}:{line_number}")

        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        annotation_format = "yolo"
        if len(values) == 4:
            center_x, center_y, width, height = values
            box_width = width * image_width
            box_height = height * image_height
            x1 = center_x * image_width - box_width / 2
            y1 = center_y * image_height - box_height / 2
            x2 = center_x * image_width + box_width / 2
            y2 = center_y * image_height + box_height / 2
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x1 = min(xs) * image_width
            y1 = min(ys) * image_height
            x2 = max(xs) * image_width
            y2 = max(ys) * image_height
            annotation_format = "yolo_segmentation"
        else:
            raise ValueError(f"invalid YOLO annotation at {annotation_path}:{line_number}")
        bbox = _clamp_bbox((x1, y1, x2, y2), image_width, image_height)
        if bbox is None:
            continue

        annotations.append(
            {
                "format": annotation_format,
                "class_id": class_id,
                "label": class_names.get(class_id, str(class_id)),
                "bbox": _bbox_payload(bbox, image_width, image_height),
                "line": line_number,
            }
        )
    return annotations

def load_labelme_annotations(image_path: Path, annotation_path: Path) -> list[dict[str, Any]]:
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    annotations = []
    for index, shape in enumerate(payload.get("shapes", [])):
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        bbox = _clamp_bbox((min(xs), min(ys), max(xs), max(ys)), image_width, image_height)
        if bbox is None:
            continue
        annotations.append(
            {
                "format": "labelme",
                "class_id": None,
                "label": shape.get("label", ""),
                "shape_type": shape.get("shape_type", "polygon"),
                "bbox": _bbox_payload(bbox, image_width, image_height),
                "shape_index": index,
            }
        )
    return annotations

def load_annotations_for_image(
    image_path: Path,
    annotations: str,
    labels_dir: str | Path | None = None,
    class_names: dict[int, str] | None = None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    _validate_annotation_config(annotations, DEFAULT_CROP_MODE)
    if annotations == "none":
        return None, []

    annotation_path = find_annotation_path(image_path, annotations, labels_dir)
    if annotation_path is None:
        return None, []
    if annotations == "yolo":
        return annotation_path, load_yolo_annotations(image_path, annotation_path, class_names)
    return annotation_path, load_labelme_annotations(image_path, annotation_path)

def _crop_annotation(image: Image.Image, bbox: dict[str, Any], crop_mode: str) -> Image.Image:
    return _prepare_crop_for_features(image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])), crop_mode)

def _prepare_crop_for_features(crop: Image.Image, crop_mode: str, pad_color: tuple[int, int, int] = (114, 114, 114)) -> Image.Image:
    _validate_crop_mode(crop_mode)
    crop = crop.convert("RGB")
    source_width, source_height = crop.size
    if crop_mode == LEGACY_CROP_MODE:
        crop.info["solo_source_width"] = source_width
        crop.info["solo_source_height"] = source_height
        crop.info["solo_crop_mode"] = crop_mode
        return crop
    width, height = source_width, source_height
    side = max(width, height, 1)
    boxed = Image.new("RGB", (side, side), pad_color)
    boxed.paste(crop, ((side - width) // 2, (side - height) // 2))
    boxed.info["solo_source_width"] = source_width
    boxed.info["solo_source_height"] = source_height
    boxed.info["solo_crop_mode"] = crop_mode
    return boxed

def _crop_mode_from_config(config: dict[str, Any] | None) -> str:
    crop_mode = str((config or {}).get("crop_mode") or DEFAULT_CROP_MODE)
    _validate_crop_mode(crop_mode)
    return crop_mode

def _stats_version_from_config(config: dict[str, Any] | None) -> str:
    return str((config or {}).get("stats_version") or LEGACY_FEATURE_STATS_VERSION)

def _source_key(path: str | Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).casefold()

def _image_paths(path: str | Path) -> list[Path]:
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
__all__ = [
    'load_class_names',
    '_annotation_candidates',
    'find_annotation_path',
    'load_yolo_annotations',
    'load_labelme_annotations',
    'load_annotations_for_image',
    '_crop_annotation',
    '_prepare_crop_for_features',
    '_crop_mode_from_config',
    '_stats_version_from_config',
    '_source_key',
    '_image_paths',
]
