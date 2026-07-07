from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from solo.config import *
from solo.utils.bbox import *
from solo.core.ops import *
from solo.core.features import *
from solo.data.dataset import *
from solo.utils.cv_image import Image
from solo.data.negatives import _hard_objectness_negative_candidates, _median_box_size, _sample_negative_boxes
from solo.data.weights import build_weight_metadata, get_weight_by_name, load_weights, save_weights, _prototype_entries_from_weights

def _weight_config(
    size: int,
    qua: int,
    nab: float,
    pt_size: int,
    kernel: str,
    field: str,
    max_radius: int,
    normalize: bool,
    normalize_each_step: bool,
    annotations: str,
    labels_dir: str | Path | None,
    class_names_path: str | Path | None,
    crop_mode: str,
    negative_label: str,
    negative_samples_per_image: int,
    negative_ratio: float,
    negative_iou: float,
    negative_seed: int,
    feature_mode: str,
    selected_channels: list[str],
    prototype_count: int,
    structure_mode: str,
    structure_grid: int,
    accelerator: str,
    workers: int,
    compact_weights: bool,
    weight_precision: int,
    compact_exemplars: int,
    compact_sample_limit: int,
) -> dict[str, Any]:
    return {
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
        "negative_label": negative_label,
        "negative_samples_per_image": negative_samples_per_image,
        "negative_ratio": negative_ratio,
        "negative_iou": negative_iou,
        "negative_seed": negative_seed,
        "feature_mode": feature_mode,
        "channels": selected_channels,
        "stats_version": DEFAULT_FEATURE_STATS_VERSION,
        "prototype_count": prototype_count,
        "structure_mode": structure_mode,
        "structure_grid": structure_grid,
        "accelerator": accelerator,
        "workers": workers,
        "compact_weights": compact_weights,
        "weight_precision": weight_precision,
        "compact_exemplars": compact_exemplars,
        "compact_sample_limit": compact_sample_limit,
    }


def _features_for_crop(
    crop: Image.Image,
    size: int,
    qua: int,
    nab: float,
    pt_size: int,
    kernel: str,
    field: str,
    max_radius: int,
    normalize: bool,
    normalize_each_step: bool,
    feature_mode: str,
    selected_channels: list[str],
    structure_mode: str,
    structure_grid: int,
) -> dict[str, Any]:
    return extract_image_features(
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
        feature_mode=feature_mode,
        channels=selected_channels,
        structure_mode=structure_mode,
        structure_grid=structure_grid,
    )


def _process_image_weight_task(payload: dict[str, Any]) -> dict[str, Any]:
    image_index = int(payload["image_index"])
    image_count = int(payload["image_count"])
    image_path = Path(payload["image_path"])
    annotations = str(payload["annotations"])
    labels_dir = payload.get("labels_dir")
    class_names = payload.get("class_names") or {}
    crop_mode = str(payload["crop_mode"])
    size = int(payload["size"])
    qua = int(payload["qua"])
    nab = float(payload["nab"])
    pt_size = int(payload["pt_size"])
    kernel = str(payload["kernel"])
    field = str(payload["field"])
    max_radius = int(payload["max_radius"])
    normalize = bool(payload["normalize"])
    normalize_each_step = bool(payload["normalize_each_step"])
    negative_samples_per_image = int(payload["negative_samples_per_image"])
    negative_ratio = float(payload["negative_ratio"])
    negative_iou = float(payload["negative_iou"])
    negative_seed = int(payload["negative_seed"])
    negative_label = str(payload["negative_label"])
    feature_mode = str(payload["feature_mode"])
    selected_channels = list(payload["selected_channels"])
    structure_mode = str(payload["structure_mode"])
    structure_grid = int(payload["structure_grid"])
    print_pt = bool(payload.get("print_pt", False))
    rng = random.Random(negative_seed + image_index * 1000003)
    weights: list[dict[str, Any]] = []
    printed_pts: list[Any] = []
    skipped = None
    annotation_path: Path | None = None
    positive_count = 0
    generated_negative_count = 0
    negative_count = 0

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
        with Image.open(image_path) as image:
            features = _features_for_crop(
                image,
                size,
                qua,
                nab,
                pt_size,
                kernel,
                field,
                max_radius,
                normalize,
                normalize_each_step,
                feature_mode,
                selected_channels,
                structure_mode,
                structure_grid,
            )
        weights.append({"name": image_path.name, "path": str(image_path), "pt": pt, "features": features})
        if print_pt:
            printed_pts.append(pt)
        return {
            "image_index": image_index,
            "image_count": image_count,
            "image_name": image_path.name,
            "weights": weights,
            "printed_pts": printed_pts,
            "positive_count": 0,
            "negative_count": 0,
            "generated_negative_count": 0,
            "skipped": None,
        }

    annotation_path, boxes = load_annotations_for_image(image_path, annotations, labels_dir, class_names)
    if annotation_path is None:
        skipped = "no_annotation"
    else:
        negative_count = max(negative_samples_per_image, round(len(boxes) * negative_ratio))
        if not boxes and negative_count <= 0:
            skipped = "empty_annotation"

    if skipped:
        return {
            "image_index": image_index,
            "image_count": image_count,
            "image_name": image_path.name,
            "weights": [],
            "printed_pts": [],
            "positive_count": 0,
            "negative_count": negative_count,
            "generated_negative_count": 0,
            "skipped": skipped,
        }

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size
        for box_index, annotation in enumerate(boxes):
            crop = _crop_annotation(image, annotation["bbox"], crop_mode)
            features = _features_for_crop(
                crop,
                size,
                qua,
                nab,
                pt_size,
                kernel,
                field,
                max_radius,
                normalize,
                normalize_each_step,
                feature_mode,
                selected_channels,
                structure_mode,
                structure_grid,
            )
            weights.append(
                {
                    "name": f"{image_path.stem}_{box_index}{image_path.suffix}",
                    "source_image": str(image_path),
                    "annotation_path": str(annotation_path),
                    "annotation": annotation,
                    "crop_mode": crop_mode,
                    "pt": features["pt"],
                    "features": features,
                }
            )
            positive_count += 1
            if print_pt:
                printed_pts.append(features["pt"])
        if negative_count > 0:
            negative_boxes = _sample_negative_boxes(
                image_width,
                image_height,
                boxes,
                negative_count,
                rng,
                min_iou=negative_iou,
                image=image,
            )
            for negative_index, bbox in enumerate(negative_boxes):
                crop = _prepare_crop_for_features(image.crop(bbox), crop_mode)
                features = _features_for_crop(
                    crop,
                    size,
                    qua,
                    nab,
                    pt_size,
                    kernel,
                    field,
                    max_radius,
                    normalize,
                    normalize_each_step,
                    feature_mode,
                    selected_channels,
                    structure_mode,
                    structure_grid,
                )
                weights.append(
                    {
                        "name": f"{image_path.stem}_negative_{negative_index}{image_path.suffix}",
                        "source_image": str(image_path),
                        "annotation_path": str(annotation_path),
                        "annotation": {
                            "format": "negative",
                            "class_id": None,
                            "label": negative_label,
                            "bbox": _bbox_payload(bbox, image_width, image_height),
                        },
                        "crop_mode": crop_mode,
                        "negative": True,
                        "pt": features["pt"],
                        "features": features,
                    }
                )
                generated_negative_count += 1
                if print_pt:
                    printed_pts.append(features["pt"])

    return {
        "image_index": image_index,
        "image_count": image_count,
        "image_name": image_path.name,
        "weights": weights,
        "printed_pts": printed_pts,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "generated_negative_count": generated_negative_count,
        "skipped": None,
    }


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
    negative_samples_per_image: int = DEFAULT_NEGATIVE_SAMPLES_PER_IMAGE,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    negative_iou: float = DEFAULT_NEGATIVE_IOU,
    negative_seed: int = 42,
    negative_label: str = DEFAULT_NEGATIVE_LABEL,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    channels: list[str] | str | None = None,
    prototype_count: int = DEFAULT_PROTOTYPE_COUNT,
    structure_mode: str = DEFAULT_STRUCTURE_MODE,
    structure_grid: int = DEFAULT_STRUCTURE_GRID,
    accelerator: str = DEFAULT_ACCELERATOR,
    workers: int = 1,
    compact_weights: bool = DEFAULT_COMPACT_WEIGHTS,
    weight_precision: int = DEFAULT_WEIGHT_PRECISION,
    compact_exemplars: int = DEFAULT_COMPACT_EXEMPLARS,
    compact_sample_limit: int = DEFAULT_COMPACT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    _validate_annotation_config(annotations, crop_mode)
    _validate_feature_mode(feature_mode)
    _validate_structure_mode(structure_mode)
    _validate_accelerator(accelerator)
    selected_channels = parse_channels(channels)
    if negative_samples_per_image < 0:
        raise ValueError("negative_samples_per_image must be 0 or greater")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be 0 or greater")
    if negative_iou < 0:
        raise ValueError("negative_iou must be 0 or greater")
    if workers < 1:
        raise ValueError("workers must be 1 or greater")
    if weight_precision < 0:
        raise ValueError("weight_precision must be 0 or greater")
    if compact_exemplars < 0:
        raise ValueError("compact_exemplars must be 0 or greater")
    if compact_sample_limit < 0:
        raise ValueError("compact_sample_limit must be 0 or greater")
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
    started = time.time()
    print(
        f"[train] source={image_dir.resolve()} images={len(image_paths)} "
        f"annotations={annotations} field={field} pt_size={pt_size} "
        f"negative_per_image={negative_samples_per_image} negative_ratio={negative_ratio} "
        f"workers={workers} accelerator={accelerator}",
        flush=True,
    )
    task_payloads = [
        {
            "image_index": image_index,
            "image_count": len(image_paths),
            "image_path": str(image_path),
            "annotations": annotations,
            "labels_dir": str(labels_dir) if labels_dir is not None else None,
            "class_names": class_names,
            "crop_mode": crop_mode,
            "size": size,
            "qua": qua,
            "nab": nab,
            "pt_size": pt_size,
            "kernel": kernel,
            "field": field,
            "max_radius": max_radius,
            "normalize": normalize,
            "normalize_each_step": normalize_each_step,
            "negative_samples_per_image": negative_samples_per_image,
            "negative_ratio": negative_ratio,
            "negative_iou": negative_iou,
            "negative_seed": negative_seed,
            "negative_label": negative_label,
            "feature_mode": feature_mode,
            "selected_channels": selected_channels,
            "structure_mode": structure_mode,
            "structure_grid": structure_grid,
            "print_pt": print_pt,
        }
        for image_index, image_path in enumerate(image_paths, start=1)
    ]

    def consume_result(result: dict[str, Any]) -> None:
        before_count = len(weights)
        weights.extend(result["weights"])
        if print_pt:
            for pt in result.get("printed_pts", []):
                print(pt)
        skipped = result.get("skipped")
        if skipped:
            print(
                f"[train] image {result['image_index']}/{result['image_count']} {result['image_name']} "
                f"skipped={skipped} elapsed={_elapsed(started)}",
                flush=True,
            )
            return
        if annotations == "none":
            print(
                f"[train] image {result['image_index']}/{result['image_count']} {result['image_name']} "
                f"weights={len(weights)} elapsed={_elapsed(started)}",
                flush=True,
            )
            return
        print(
            f"[train] image {result['image_index']}/{result['image_count']} {result['image_name']} "
            f"boxes={result['positive_count']} negatives={result['generated_negative_count']}/{result['negative_count']} "
            f"new_weights={len(weights) - before_count} "
            f"total_weights={len(weights)} elapsed={_elapsed(started)}",
            flush=True,
        )

    if workers == 1:
        for payload in task_payloads:
            consume_result(_process_image_weight_task(payload))
    else:
        pending: dict[int, dict[str, Any]] = {}
        next_index = 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_image_weight_task, payload) for payload in task_payloads]
            for future in as_completed(futures):
                result = future.result()
                pending[int(result["image_index"])] = result
                while next_index in pending:
                    consume_result(pending.pop(next_index))
                    next_index += 1

    if not weights:
        raise FileNotFoundError(f"no weights generated from: {image_dir}")

    config = _weight_config(
        size,
        qua,
        nab,
        pt_size,
        kernel,
        field,
        max_radius,
        normalize,
        normalize_each_step,
        annotations,
        labels_dir,
        class_names_path,
        crop_mode,
        negative_label,
        negative_samples_per_image,
        negative_ratio,
        negative_iou,
        negative_seed,
        feature_mode,
        selected_channels,
        prototype_count,
        structure_mode,
        structure_grid,
        accelerator,
        workers,
        compact_weights,
        weight_precision,
        compact_exemplars,
        compact_sample_limit,
    )
    if save_path is not None:
        save_weights(
            weights,
            save_path,
            config=config,
            compact=compact_weights,
            precision=weight_precision,
            compact_exemplars=compact_exemplars,
            compact_sample_limit=compact_sample_limit,
        )

    return weights

__all__ = [
    '_weight_config',
    '_features_for_crop',
    '_process_image_weight_task',
    'get_image_pt',
    'save_weights',
    '_prototype_entries_from_weights',
    'build_weight_metadata',
    'load_weights',
    'get_weight_by_name',
    '_median_box_size',
    '_sample_negative_boxes',
    '_hard_objectness_negative_candidates',
]
