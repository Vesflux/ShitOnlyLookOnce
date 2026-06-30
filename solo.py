from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_WEIGHT_PATH = "solo_weights.json"
DEFAULT_KERNEL = "weighted"
DEFAULT_FIELD = "global"
DEFAULT_NAB = 0.25
DEFAULT_MAX_RADIUS = 0
DEFAULT_ANNOTATIONS = "none"
DEFAULT_CROP_MODE = "stretch"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, save, and load SOLOv1 image pt weights.")
    parser.add_argument("dirpath", nargs="?", default="train", help="image directory used when generating weights")
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
