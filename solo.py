from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_WEIGHT_PATH = "solo_weights.json"
DEFAULT_KERNEL = "weighted"
DEFAULT_FIELD = "global"
DEFAULT_NAB = 0.25
DEFAULT_MAX_RADIUS = 0
DEFAULT_ANNOTATIONS = "none"
DEFAULT_CROP_MODE = "stretch"
DEFAULT_DETECTION_RESULTS = "solo_detection_results.txt"


def _validate_config(size: int, qua: int, nab: float, pt_size: int, kernel: str, field: str, max_radius: int) -> None:
    source_size = 16 * size
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if qua < 0:
        raise ValueError("qua must be 0 or greater")
    if pt_size <= 0:
        raise ValueError("pt_size must be greater than 0")
    if pt_size > source_size:
        raise ValueError("pt_size cannot be larger than 16 * size")
    if (source_size - pt_size) % 2 != 0:
        raise ValueError("pt_size must have the same even/odd parity as 16 * size")
    if kernel not in {"weighted", "legacy"}:
        raise ValueError("kernel must be 'weighted' or 'legacy'")
    if field not in {"local", "global"}:
        raise ValueError("field must be 'local' or 'global'")
    if max_radius < 0:
        raise ValueError("max_radius must be 0 or greater")
    if _kernel_denominator(nab, kernel) == 0:
        raise ValueError("nab cannot make the kernel denominator 0")


def _validate_annotation_config(annotations: str, crop_mode: str) -> None:
    if annotations not in {"none", "yolo", "labelme"}:
        raise ValueError("annotations must be 'none', 'yolo', or 'labelme'")
    if crop_mode != "stretch":
        raise ValueError("crop_mode currently supports only 'stretch'")


def _validate_pt(pt: list[list[float]], pt_size: int, nab: float, kernel: str, field: str, max_radius: int) -> None:
    if not pt or not pt[0]:
        raise ValueError("pt cannot be empty")
    width = len(pt[0])
    if any(len(row) != width for row in pt):
        raise ValueError("pt rows must have the same length")
    if len(pt) != width:
        raise ValueError("pt must be a square matrix")
    if pt_size <= 0:
        raise ValueError("pt_size must be greater than 0")
    if pt_size > len(pt):
        raise ValueError("pt_size cannot be larger than the source pt")
    if (len(pt) - pt_size) % 2 != 0:
        raise ValueError("pt_size must have the same even/odd parity as the source pt")
    if kernel not in {"weighted", "legacy"}:
        raise ValueError("kernel must be 'weighted' or 'legacy'")
    if field not in {"local", "global"}:
        raise ValueError("field must be 'local' or 'global'")
    if max_radius < 0:
        raise ValueError("max_radius must be 0 or greater")
    if _kernel_denominator(nab, kernel) == 0:
        raise ValueError("nab cannot make the kernel denominator 0")


def _kernel_denominator(nab: float, kernel: str) -> float:
    if kernel == "legacy":
        return nab * 8 + 1
    side_weight = nab
    diagonal_weight = nab / 2
    return 1 + side_weight * 4 + diagonal_weight * 4


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


def _clamp_bbox(bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    left = max(0, min(image_width, round(min(x1, x2))))
    top = max(0, min(image_height, round(min(y1, y2))))
    right = max(0, min(image_width, round(max(x1, x2))))
    bottom = max(0, min(image_height, round(max(y1, y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _bbox_payload(bbox: tuple[int, int, int, int], image_width: int, image_height: int) -> dict[str, Any]:
    left, top, right, bottom = bbox
    return {
        "x1": left,
        "y1": top,
        "x2": right,
        "y2": bottom,
        "width": right - left,
        "height": bottom - top,
        "normalized": {
            "x1": left / image_width,
            "y1": top / image_height,
            "x2": right / image_width,
            "y2": bottom / image_height,
        },
    }


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
        center_x, center_y, width, height = map(float, parts[1:5])
        box_width = width * image_width
        box_height = height * image_height
        x1 = center_x * image_width - box_width / 2
        y1 = center_y * image_height - box_height / 2
        x2 = center_x * image_width + box_width / 2
        y2 = center_y * image_height + box_height / 2
        bbox = _clamp_bbox((x1, y1, x2, y2), image_width, image_height)
        if bbox is None:
            continue

        annotations.append(
            {
                "format": "yolo",
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
    if crop_mode != "stretch":
        raise ValueError("crop_mode currently supports only 'stretch'")
    return image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))


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


def get_image_pt(
    dirpath: str | Path,
    size: int = 2,
    qua: int = 8,
    nab: float = DEFAULT_NAB,
    pt_size: int = 8,
    kernel: str = DEFAULT_KERNEL,
    field: str = DEFAULT_FIELD,
    max_radius: int = DEFAULT_MAX_RADIUS,
    normalize: bool = True,
    normalize_each_step: bool = True,
    annotations: str = DEFAULT_ANNOTATIONS,
    labels_dir: str | Path | None = None,
    class_names_path: str | Path | None = None,
    crop_mode: str = DEFAULT_CROP_MODE,
    save_path: str | Path | None = None,
    print_pt: bool = True,
) -> list[dict[str, Any]]:
    _validate_annotation_config(annotations, crop_mode)
    image_dir = Path(dirpath)
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no image files found in: {image_dir}")

    class_names = load_class_names(class_names_path)
    weights = []
    for image_path in image_paths:
        if annotations == "none":
            pt = get_image_single_pt(
                image_path,
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
            item = {"name": image_path.name, "path": str(image_path), "pt": pt}
            weights.append(item)
            if print_pt:
                print(pt)
            continue

        annotation_path, boxes = load_annotations_for_image(image_path, annotations, labels_dir, class_names)
        if annotation_path is None:
            continue
        with Image.open(image_path) as image:
            for box_index, annotation in enumerate(boxes):
                crop = _crop_annotation(image, annotation["bbox"], crop_mode)
                pt = image_to_pt(
                    crop,
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
                item = {
                    "name": f"{image_path.stem}_{box_index}{image_path.suffix}",
                    "source_image": str(image_path),
                    "annotation_path": str(annotation_path),
                    "annotation": annotation,
                    "crop_mode": crop_mode,
                    "pt": pt,
                }
                weights.append(item)
                if print_pt:
                    print(pt)

    if not weights:
        raise FileNotFoundError(f"no weights generated from: {image_dir}")

    if save_path is not None:
        save_weights(
            weights,
            save_path,
            config={
                "size": size,
                "qua": qua,
                "nab": nab,
                "pt_size": pt_size,
                "kernel": kernel,
                "field": field,
                "max_radius": max_radius,
                "normalize": normalize,
                "normalize_each_step": normalize_each_step,
                "annotations": annotations,
                "labels_dir": str(labels_dir) if labels_dir is not None else None,
                "class_names_path": str(class_names_path) if class_names_path is not None else None,
                "crop_mode": crop_mode,
            },
        )

    return weights


def save_weights(weights: list[dict[str, Any]], path: str | Path = DEFAULT_WEIGHT_PATH, config: dict[str, Any] | None = None) -> Path:
    payload = {"version": 1, "config": config or {}, "weights": weights}
    weight_path = Path(path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return weight_path


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


def _bbox_tuple(box: dict[str, Any]) -> tuple[int, int, int, int]:
    return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_tuple(a)
    bx1, by1, bx2, by2 = _bbox_tuple(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator else 0.0


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


def color_proposals(
    image: Image.Image,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
) -> list[dict[str, Any]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    mask = bytearray(width * height)

    for y in range(height):
        row_offset = y * width
        for x in range(width):
            red, green, blue = pixels[x, y]
            max_channel = max(red, green, blue)
            min_channel = min(red, green, blue)
            saturation = (max_channel - min_channel) / max(max_channel, 1)
            yellowish = (
                red > 85
                and green > 75
                and blue < 205
                and red > blue * 1.08
                and green > blue * 1.02
                and saturation > 0.12
            )
            bright_object = max_channel > 180 and saturation > 0.10 and blue < 220
            if yellowish or bright_object:
                mask[row_offset + x] = 1

    seen = bytearray(width * height)
    proposals = []
    max_area = int(width * height * max_area_ratio)
    for start_y in range(height):
        for start_x in range(width):
            index = start_y * width + start_x
            if not mask[index] or seen[index]:
                continue
            stack = [(start_x, start_y)]
            seen[index] = 1
            xs = []
            ys = []
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    row_offset = ny * width
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        next_index = row_offset + nx
                        if mask[next_index] and not seen[next_index]:
                            seen[next_index] = 1
                            stack.append((nx, ny))

            area = len(xs)
            if area < min_area or area > max_area:
                continue
            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < min_box_size or box_height < min_box_size:
                continue
            if max_box_size and (box_width > max_box_size or box_height > max_box_size):
                continue
            aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
            if max_aspect_ratio and aspect > max_aspect_ratio:
                continue
            expanded = _expand_bbox((x1, y1, x2, y2), width, height, expand)
            if expanded is None:
                continue
            ex1, ey1, ex2, ey2 = expanded
            proposals.append(
                {
                    "bbox": _bbox_payload((ex1, ey1, ex2, ey2), width, height),
                    "proposal": "color",
                    "area": area,
                }
            )
    return proposals


def sliding_proposals(
    image: Image.Image,
    window_sizes: list[int],
    stride_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    width, height = image.size
    proposals = []
    for window_size in window_sizes:
        if window_size <= 0:
            continue
        window = min(window_size, width, height)
        stride = max(1, int(window * stride_ratio))
        y_values = list(range(0, max(1, height - window + 1), stride))
        x_values = list(range(0, max(1, width - window + 1), stride))
        if y_values[-1] != height - window:
            y_values.append(height - window)
        if x_values[-1] != width - window:
            x_values.append(width - window)
        for y in y_values:
            for x in x_values:
                bbox = (x, y, x + window, y + window)
                proposals.append(
                    {
                        "bbox": _bbox_payload(bbox, width, height),
                        "proposal": "sliding",
                        "window_size": window,
                    }
                )
    return proposals


def generate_proposals(
    image: Image.Image,
    proposal: str = "color",
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    stride_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    if proposal == "color":
        return color_proposals(
            image,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            expand=expand,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
        )
    if proposal == "sliding":
        return sliding_proposals(image, window_sizes or [32, 48, 64], stride_ratio=stride_ratio)
    if proposal == "both":
        color = color_proposals(
            image,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            expand=expand,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
        )
        sliding = sliding_proposals(image, window_sizes or [32, 48, 64], stride_ratio=stride_ratio)
        return color + sliding
    raise ValueError("proposal must be 'color', 'sliding', or 'both'")


def prepare_weight_index(weight_paths: list[str | Path]) -> dict[str, Any]:
    if not weight_paths:
        raise ValueError("at least one weight file is required for detection")

    entries = []
    configs = []
    for weight_path in weight_paths:
        payload = load_weights(weight_path)
        config = payload.get("config", {})
        configs.append({"path": str(weight_path), "config": config})
        for item_index, item in enumerate(payload.get("weights", [])):
            pt = item.get("pt")
            if not pt:
                continue
            annotation = item.get("annotation") or {}
            label = annotation.get("label") or item.get("label") or annotation.get("class_id") or "unknown"
            flat = [float(value) for row in pt for value in row]
            entries.append(
                {
                    "weight_path": str(weight_path),
                    "index": item_index,
                    "label": str(label),
                    "pt": flat,
                    "source": item.get("source_image") or item.get("path") or item.get("name"),
                }
            )
    if not entries:
        raise ValueError("weight files do not contain usable pt entries")

    base_config = configs[0]["config"]
    vector_length = len(entries[0]["pt"])
    entries = [entry for entry in entries if len(entry["pt"]) == vector_length]
    return {"entries": entries, "configs": configs, "config": base_config}


def _mse(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


def match_pt(pt: list[list[float]], weight_index: dict[str, Any]) -> dict[str, Any]:
    vector = [float(value) for row in pt for value in row]
    best = None
    for entry in weight_index["entries"]:
        if len(entry["pt"]) != len(vector):
            continue
        distance = _mse(vector, entry["pt"])
        if best is None or distance < best["distance"]:
            best = {
                "label": entry["label"],
                "distance": distance,
                "score": 1 / (1 + distance),
                "weight_path": entry["weight_path"],
                "weight_index": entry["index"],
                "weight_source": entry["source"],
            }
    if best is None:
        raise ValueError("no compatible weight entries were found")
    return best


def nms(detections: list[dict[str, Any]], iou_threshold: float = 0.35) -> list[dict[str, Any]]:
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if all(bbox_iou(detection["bbox"], item["bbox"]) <= iou_threshold for item in kept):
            kept.append(detection)
    return kept


def detect_image(
    image_path: str | Path,
    weight_index: dict[str, Any],
    proposal: str = "color",
    score_threshold: float = 0.9,
    nms_threshold: float = 0.35,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    proposal_expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
) -> dict[str, Any]:
    path = Path(image_path)
    config = weight_index["config"]
    started = time.time()
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        proposals = generate_proposals(
            image,
            proposal=proposal,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            expand=proposal_expand,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            window_sizes=window_sizes,
            stride_ratio=stride_ratio,
        )
        detections = []
        for proposal_item in proposals:
            bbox = _bbox_tuple(proposal_item["bbox"])
            scaled_bbox = _expand_bbox(bbox, width, height, bbox_scale)
            if scaled_bbox is None:
                continue
            crop = image.crop(scaled_bbox)
            pt = image_to_pt(
                crop,
                size=config.get("size", 2),
                qua=config.get("qua", 8),
                nab=config.get("nab", DEFAULT_NAB),
                pt_size=config.get("pt_size", 8),
                kernel=config.get("kernel", DEFAULT_KERNEL),
                field=config.get("field", DEFAULT_FIELD),
                max_radius=config.get("max_radius", DEFAULT_MAX_RADIUS),
                normalize=config.get("normalize", True),
                normalize_each_step=config.get("normalize_each_step", True),
            )
            match = match_pt(pt, weight_index)
            if match["score"] < score_threshold:
                continue
            detections.append(
                {
                    "label": match["label"],
                    "score": round(match["score"], 6),
                    "distance": round(match["distance"], 8),
                    "bbox": _bbox_payload(scaled_bbox, width, height),
                    "proposal": proposal_item.get("proposal", proposal),
                    "weight_path": match["weight_path"],
                    "weight_index": match["weight_index"],
                    "weight_source": match["weight_source"],
                }
            )

    return {
        "image": str(path),
        "width": width,
        "height": height,
        "proposal_count": len(proposals),
        "raw_detection_count": len(detections),
        "detections": nms(detections, nms_threshold),
        "elapsed_seconds": round(time.time() - started, 4),
    }


def _truth_boxes_for_image(
    image_path: Path,
    annotations: str,
    labels_dir: str | Path | None,
    class_names: dict[int, str] | None,
) -> list[dict[str, Any]]:
    _annotation_path, annotations_payload = load_annotations_for_image(image_path, annotations, labels_dir, class_names)
    truth = []
    for item in annotations_payload:
        label = item.get("label") or item.get("class_id") or "unknown"
        truth.append({"label": str(label), "bbox": item["bbox"]})
    return truth


def _best_truth_match(detection: dict[str, Any], truth_boxes: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    same_label = [truth for truth in truth_boxes if truth["label"] == detection["label"]]
    candidates = same_label or truth_boxes
    if not candidates:
        return 0.0, None
    best_truth = None
    best_iou = 0.0
    for truth in candidates:
        iou = bbox_iou(detection["bbox"], truth["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_truth = truth
    return best_iou, best_truth


def _area_from_bbox_payload(bbox: dict[str, Any]) -> float:
    return max(0, float(bbox["width"])) * max(0, float(bbox["height"]))


def calibrate_detection(
    train_images: str | Path | None,
    val_images: str | Path | None,
    weight_index: dict[str, Any],
    annotations: str = "yolo",
    train_labels_dir: str | Path | None = None,
    val_labels_dir: str | Path | None = None,
    class_names_path: str | Path | None = None,
    proposal: str = "color",
    score_threshold: float = 0.9,
    nms_threshold: float = 0.35,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    proposal_expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
    max_samples: int = 0,
) -> dict[str, Any]:
    if not train_images or not val_images:
        return {"enabled": False, "score_threshold": score_threshold, "bbox_scale": bbox_scale}

    train_paths = _image_paths(train_images)
    val_paths = _image_paths(val_images)
    if not val_paths:
        return {"enabled": False, "score_threshold": score_threshold, "bbox_scale": bbox_scale}

    ratio = max(1, round(len(train_paths) / len(val_paths))) if train_paths else 1
    class_names = load_class_names(class_names_path)
    val_iter = iter(val_paths)
    score_samples = []
    scale_samples = []
    processed_val = 0

    for train_index, _train_image in enumerate(train_paths, start=1):
        if train_index % ratio != 0:
            continue
        try:
            val_image = next(val_iter)
        except StopIteration:
            break
        truth_boxes = _truth_boxes_for_image(val_image, annotations, val_labels_dir, class_names)
        if not truth_boxes:
            continue
        result = detect_image(
            val_image,
            weight_index,
            proposal=proposal,
            score_threshold=max(0.0, score_threshold - 0.2),
            nms_threshold=nms_threshold,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            proposal_expand=proposal_expand,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            window_sizes=window_sizes,
            stride_ratio=stride_ratio,
            bbox_scale=bbox_scale,
        )
        processed_val += 1
        for detection in result["detections"]:
            best_iou, best_truth = _best_truth_match(detection, truth_boxes)
            if best_iou >= 0.25:
                score_samples.append(detection["score"])
                if best_truth is not None:
                    detection_area = _area_from_bbox_payload(detection["bbox"])
                    truth_area = _area_from_bbox_payload(best_truth["bbox"])
                    if detection_area > 0 and truth_area > 0:
                        scale_samples.append(math.sqrt(truth_area / detection_area))
        if max_samples and processed_val >= max_samples:
            break

    calibrated_threshold = score_threshold
    if score_samples:
        calibrated_threshold = max(0.0, min(0.999999, statistics.median(score_samples) * 0.98))
    calibrated_bbox_scale = bbox_scale
    if scale_samples:
        calibrated_bbox_scale = max(0.2, min(5.0, bbox_scale * statistics.median(scale_samples)))

    return {
        "enabled": True,
        "ratio": ratio,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "processed_val": processed_val,
        "score_samples": len(score_samples),
        "scale_samples": len(scale_samples),
        "score_threshold": calibrated_threshold,
        "bbox_scale": calibrated_bbox_scale,
    }


def draw_detections(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    output_path: str | Path,
    hide_labels: bool = False,
) -> Path:
    path = Path(image_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for detection in detections:
            x1, y1, x2, y2 = _bbox_tuple(detection["bbox"])
            draw.rectangle((x1, y1, x2, y2), outline=(40, 255, 120), width=3)
            if not hide_labels:
                label = f"{detection['label']} {detection['score']:.2f}"
                text_bbox = draw.textbbox((x1, y1), label)
                text_height = text_bbox[3] - text_bbox[1]
                text_width = text_bbox[2] - text_bbox[0]
                label_y = max(0, y1 - text_height - 4)
                draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill=(40, 255, 120))
                draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0))
        image.save(out_path)
    return out_path


def save_detection_report(
    results: list[dict[str, Any]],
    output_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def run_detection(
    image_path: str | Path,
    weight_paths: list[str | Path],
    output_path: str | Path = DEFAULT_DETECTION_RESULTS,
    draw_dir: str | Path | None = None,
    hide_labels: bool = False,
    proposal: str = "color",
    score_threshold: float = 0.9,
    nms_threshold: float = 0.35,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
    proposal_expand: float = 1.1,
    min_box_size: int = 6,
    max_box_size: int = 0,
    max_aspect_ratio: float = 6.0,
    window_sizes: list[int] | None = None,
    stride_ratio: float = 0.5,
    bbox_scale: float = 1.0,
    train_images: str | Path | None = None,
    val_images: str | Path | None = None,
    train_labels_dir: str | Path | None = None,
    val_labels_dir: str | Path | None = None,
    class_names_path: str | Path | None = None,
    annotation_format: str = "yolo",
    calibration_samples: int = 0,
) -> dict[str, Any]:
    weight_index = prepare_weight_index(weight_paths)
    calibration = calibrate_detection(
        train_images,
        val_images,
        weight_index,
        annotations=annotation_format,
        train_labels_dir=train_labels_dir,
        val_labels_dir=val_labels_dir,
        class_names_path=class_names_path,
        proposal=proposal,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        proposal_expand=proposal_expand,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        max_aspect_ratio=max_aspect_ratio,
        window_sizes=window_sizes,
        stride_ratio=stride_ratio,
        bbox_scale=bbox_scale,
        max_samples=calibration_samples,
    )
    score_threshold = calibration.get("score_threshold", score_threshold)
    bbox_scale = calibration.get("bbox_scale", bbox_scale)

    results = []
    for path in _image_paths(image_path):
        result = detect_image(
            path,
            weight_index,
            proposal=proposal,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            proposal_expand=proposal_expand,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            max_aspect_ratio=max_aspect_ratio,
            window_sizes=window_sizes,
            stride_ratio=stride_ratio,
            bbox_scale=bbox_scale,
        )
        if draw_dir is not None:
            draw_path = Path(draw_dir) / path.name
            result["draw_path"] = str(draw_detections(path, result["detections"], draw_path, hide_labels=hide_labels))
        results.append(result)

    metadata = {
        "weights": [str(path) for path in weight_paths],
        "proposal": proposal,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "bbox_scale": bbox_scale,
        "calibration": calibration,
        "total_images": len(results),
        "total_detections": sum(len(result["detections"]) for result in results),
    }
    report_path = save_detection_report(results, output_path, metadata)
    return {"metadata": metadata, "results": results, "report_path": str(report_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, save, and load SOLOv1 image pt weights.")
    parser.add_argument("dirpath", nargs="?", default="train", help="image directory used when generating weights")
    parser.add_argument(
        "--detect",
        nargs="+",
        help="detect objects with one or more weight JSON files; dirpath becomes image file or image directory",
    )
    parser.add_argument("--output", default=DEFAULT_DETECTION_RESULTS, help="detection report JSON path")
    parser.add_argument("--draw-dir", help="optional directory for images with drawn detection boxes")
    parser.add_argument("--hide-labels", action="store_true", help="hide text labels above drawn boxes")
    parser.add_argument("--proposal", choices=["color", "sliding", "both"], default="color", help="detection proposal mode")
    parser.add_argument("--score-threshold", type=float, default=0.9, help="minimum detection score")
    parser.add_argument("--nms-threshold", type=float, default=0.35, help="NMS IoU threshold")
    parser.add_argument("--min-area", type=int, default=80, help="minimum color proposal area")
    parser.add_argument("--max-area-ratio", type=float, default=0.15, help="maximum color proposal area as image ratio")
    parser.add_argument("--proposal-expand", type=float, default=1.1, help="proposal bbox expansion scale")
    parser.add_argument("--min-box-size", type=int, default=6, help="minimum proposal width and height")
    parser.add_argument("--max-box-size", type=int, default=0, help="maximum proposal width or height; 0 disables")
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0, help="maximum proposal aspect ratio; 0 disables")
    parser.add_argument("--window-sizes", help="comma-separated sliding window sizes, for example 32,48,64")
    parser.add_argument("--stride-ratio", type=float, default=0.5, help="sliding proposal stride ratio")
    parser.add_argument("--bbox-scale", type=float, default=1.0, help="extra detection bbox scaling before matching")
    parser.add_argument("--calibrate-train-images", help="train image folder used for train/val calibration ratio")
    parser.add_argument("--calibrate-val-images", help="val image folder used for periodic calibration")
    parser.add_argument("--calibrate-train-labels", help="train labels folder used during calibration")
    parser.add_argument("--calibrate-val-labels", help="val labels folder used during calibration")
    parser.add_argument("--calibration-samples", type=int, default=0, help="limit val images used for calibration; 0 means all")
    parser.add_argument("--annotation-format", choices=["yolo", "labelme"], default="yolo", help="calibration annotation format")
    parser.add_argument("--size", type=int, default=8, help="image is resized to 16 * size")
    parser.add_argument("--qua", type=int, default=8, help="rounding precision")
    parser.add_argument("--nab", type=float, default=DEFAULT_NAB, help="side-neighbor weight; diagonals use nab / 2")
    parser.add_argument("--pt-size", type=int, default=8, help="target pt matrix size")
    parser.add_argument("--kernel", choices=["weighted", "legacy"], default=DEFAULT_KERNEL, help="compression kernel")
    parser.add_argument("--field", choices=["local", "global"], default=DEFAULT_FIELD, help="visual field mode")
    parser.add_argument(
        "--max-radius",
        type=int,
        default=DEFAULT_MAX_RADIUS,
        help="global visual field radius limit; 0 means full image",
    )
    parser.add_argument("--annotations", choices=["none", "yolo", "labelme"], default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--labels-dir", help="annotation directory; defaults to same folder or images->labels mapping")
    parser.add_argument("--class-names", help="YOLO class names file, one name per line")
    parser.add_argument("--crop-mode", choices=["stretch"], default=DEFAULT_CROP_MODE)
    parser.add_argument("--save", default=DEFAULT_WEIGHT_PATH, help="output weight JSON path")
    parser.add_argument("--load", help="load weight JSON instead of generating new weights")
    parser.add_argument("--no-save", action="store_true", help="do not save generated weights")
    parser.add_argument("--no-normalize", action="store_true", help="skip per-image normalization")
    parser.add_argument("--normalize-final-only", action="store_true", help="normalize only after all compression steps")
    parser.add_argument("--no-print", action="store_true", help="do not print every pt matrix")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.load:
        payload = load_weights(args.load)
        print(f"loaded {len(payload['weights'])} weights from {Path(args.load).resolve()}")
        print(f"config: {payload.get('config', {})}")
        if not args.no_print:
            for item in payload["weights"]:
                print(item.get("pt"))
        return

    if args.detect:
        window_sizes = None
        if args.window_sizes:
            window_sizes = [int(value.strip()) for value in args.window_sizes.split(",") if value.strip()]
        result = run_detection(
            args.dirpath,
            args.detect,
            output_path=args.output,
            draw_dir=args.draw_dir,
            hide_labels=args.hide_labels,
            proposal=args.proposal,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            proposal_expand=args.proposal_expand,
            min_box_size=args.min_box_size,
            max_box_size=args.max_box_size,
            max_aspect_ratio=args.max_aspect_ratio,
            window_sizes=window_sizes,
            stride_ratio=args.stride_ratio,
            bbox_scale=args.bbox_scale,
            train_images=args.calibrate_train_images,
            val_images=args.calibrate_val_images,
            train_labels_dir=args.calibrate_train_labels,
            val_labels_dir=args.calibrate_val_labels,
            class_names_path=args.class_names,
            annotation_format=args.annotation_format,
            calibration_samples=args.calibration_samples,
        )
        print(f"detected {result['metadata']['total_detections']} objects in {result['metadata']['total_images']} images")
        print(f"saved detection report to {Path(result['report_path']).resolve()}")
        if args.draw_dir:
            print(f"saved drawn images to {Path(args.draw_dir).resolve()}")
        return

    save_path = None if args.no_save else args.save
    weights = get_image_pt(
        args.dirpath,
        size=args.size,
        qua=args.qua,
        nab=args.nab,
        pt_size=args.pt_size,
        kernel=args.kernel,
        field=args.field,
        max_radius=args.max_radius,
        normalize=not args.no_normalize,
        normalize_each_step=not args.normalize_final_only,
        annotations=args.annotations,
        labels_dir=args.labels_dir,
        class_names_path=args.class_names,
        crop_mode=args.crop_mode,
        save_path=save_path,
        print_pt=not args.no_print,
    )
    print(f"generated {len(weights)} weights")
    if save_path is not None:
        print(f"saved weights to {Path(save_path).resolve()}")


if __name__ == "__main__":
    main()
