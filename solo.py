from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_WEIGHT_PATH = "solo_weights.json"
DEFAULT_KERNEL = "weighted"
DEFAULT_NAB = 0.25


def _validate_config(size: int, qua: int, nab: float, pt_size: int, kernel: str) -> None:
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
    if _kernel_denominator(nab, kernel) == 0:
        raise ValueError("nab cannot make the kernel denominator 0")


def _validate_pt(pt: list[list[float]], pt_size: int, nab: float, kernel: str) -> None:
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


def _compress_once(pt: list[list[float]], qua: int, nab: float, kernel: str = DEFAULT_KERNEL) -> list[list[float]]:
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
    normalize: bool = True,
    normalize_each_step: bool = True,
) -> list[list[float]]:
    _validate_pt(pt, weight, nab, kernel)
    compressed = [[float(value) for value in row] for row in pt]
    compressed = _add_zero_border(compressed)

    while len(compressed) > weight:
        compressed = _compress_once(compressed, qua, nab, kernel)
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
    normalize: bool = True,
    normalize_each_step: bool = True,
) -> list[list[float]]:
    _validate_config(size, qua, nab, pt_size, kernel)

    source_size = 16 * size
    image_path = Path(path)
    with Image.open(image_path) as image:
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
    normalize: bool = True,
    normalize_each_step: bool = True,
    save_path: str | Path | None = None,
    print_pt: bool = True,
) -> list[dict[str, Any]]:
    image_dir = Path(dirpath)
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no image files found in: {image_dir}")

    weights = []
    for image_path in image_paths:
        pt = get_image_single_pt(
            image_path,
            size=size,
            qua=qua,
            nab=nab,
            pt_size=pt_size,
            kernel=kernel,
            normalize=normalize,
            normalize_each_step=normalize_each_step,
        )
        item = {"name": image_path.name, "path": str(image_path), "pt": pt}
        weights.append(item)
        if print_pt:
            print(pt)

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
                "normalize": normalize,
                "normalize_each_step": normalize_each_step,
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
    payload = json.loads(weight_path.read_text(encoding="utf-8"))
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
        normalize=not args.no_normalize,
        normalize_each_step=not args.normalize_final_only,
        save_path=save_path,
        print_pt=not args.no_print,
    )
    print(f"generated {len(weights)} weights")
    if save_path is not None:
        print(f"saved weights to {Path(save_path).resolve()}")


if __name__ == "__main__":
    main()
