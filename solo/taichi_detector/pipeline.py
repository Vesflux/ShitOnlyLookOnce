from __future__ import annotations

import random
import time
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from solo.config import DEFAULT_MAX_DETECTIONS, DEFAULT_VAL_MATCH_IOU
from solo.config import DEFAULT_NMS_DIOU_THRESHOLD
from solo.data.dataset import load_annotations_for_image
from solo.engine.evaluation import evaluate_detection_results
from solo.taichi_detector.anchors import (
    DEFAULT_ANCHOR_RATIOS,
    DEFAULT_ANCHOR_SIZES,
    DEFAULT_ANCHOR_STRIDE_RATIO,
    DEFAULT_PYRAMID_SCALES,
    build_anchor_hardness_index,
    generate_pyramid_anchors,
    max_iou_with_truth,
    parse_float_list,
    parse_int_list,
    quick_anchor_hardness_from_index,
)
from solo.taichi_detector.features import (
    FEATURE_SIZE,
    batch_crop_feature_matrix,
    batch_crop_feature_matrix_from_planes,
    bbox_payload_from_tuple,
    feature_dimension,
    image_paths,
    _precompute_feature_planes,
    read_rgb_image,
)
from solo.taichi_detector.image_pyramid import build_image_pyramid
from solo.taichi_detector.model import load_model, save_model, score_mlp_taichi, standardize_features, train_mlp_taichi
from solo.taichi_detector.runtime import initialize_taichi
from solo.utils.bbox import _bbox_float_tuple, _bbox_payload, _bbox_tuple, bbox_containment, bbox_iou, distance_box_fusion, filter_detections, nms, snap_detections_to_edges, split_overmerged_fusions
from solo.utils.visualization import draw_detections, save_detection_report


_MEMORY_CACHE_MAX_BYTES = int(os.environ.get("SOLO_TAICHI_MEMORY_CACHE_MB", "0") or "0") * 1024 * 1024
_TRAIN_SAMPLE_FLUSH_IMAGES = max(1, int(os.environ.get("SOLO_TAICHI_TRAIN_FLUSH_IMAGES", "16") or "16"))
_TRAIN_SAMPLE_FLUSH_BOXES = max(1, int(os.environ.get("SOLO_TAICHI_TRAIN_FLUSH_BOXES", "8192") or "8192"))
_SMALL_TRUTH_AREA = 24.0 * 24.0
_SMALL_TRUTH_SOFT_POSITIVE = 0.24


def _jitter_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = (x1 + x2) * 0.5 + rng.uniform(-0.10, 0.10) * width
    center_y = (y1 + y2) * 0.5 + rng.uniform(-0.08, 0.08) * height
    scale_w = rng.uniform(0.88, 1.28)
    scale_h = rng.uniform(0.88, 1.22)
    from solo.utils.bbox import _clamp_bbox

    return _clamp_bbox(
        (
            center_x - width * scale_w * 0.5,
            center_y - height * scale_h * 0.5,
            center_x + width * scale_w * 0.5,
            center_y + height * scale_h * 0.5,
        ),
        image_width,
        image_height,
    )


def _truth_for_image(path: Path, labels_dir: str | Path | None) -> list[dict[str, Any]]:
    _annotation_path, annotations = load_annotations_for_image(path, "yolo", labels_dir, None)
    return [{"label": str(item.get("label", item.get("class_id", "0"))), "bbox": item["bbox"]} for item in annotations]


def _truth_tuple_list(truth: list[dict[str, Any]]) -> list[tuple[int, int, int, int]]:
    return [_bbox_tuple(item["bbox"]) for item in truth]


def _truth_geometry(truth: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not truth:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "x1": empty,
            "y1": empty,
            "x2": empty,
            "y2": empty,
            "cx": empty,
            "cy": empty,
            "area": empty,
            "width": empty,
            "height": empty,
        }
    boxes = np.asarray([_bbox_float_tuple(item["bbox"]) for item in truth], dtype=np.float32)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return {
        "x1": boxes[:, 0],
        "y1": boxes[:, 1],
        "x2": boxes[:, 2],
        "y2": boxes[:, 3],
        "cx": (boxes[:, 0] + boxes[:, 2]) * 0.5,
        "cy": (boxes[:, 1] + boxes[:, 3]) * 0.5,
        "area": np.maximum(1.0, widths * heights),
        "width": np.maximum(1.0, widths),
        "height": np.maximum(1.0, heights),
    }


def _anchor_truth_stats(
    bbox: dict[str, Any],
    geometry: dict[str, np.ndarray],
) -> tuple[float, int, float]:
    if geometry["x1"].size == 0:
        return 0.0, 0, 1.0
    x1, y1, x2, y2 = _bbox_float_tuple(bbox)
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    ix1 = np.maximum(x1, geometry["x1"])
    iy1 = np.maximum(y1, geometry["y1"])
    ix2 = np.minimum(x2, geometry["x2"])
    iy2 = np.minimum(y2, geometry["y2"])
    intersections = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    denominators = np.maximum(1e-6, box_area + geometry["area"] - intersections)
    ious = intersections / denominators
    match_scores = ious.copy()
    small_truth_mask = geometry["area"] <= _SMALL_TRUTH_AREA
    if np.any(small_truth_mask):
        anchor_width = max(1.0, x2 - x1)
        anchor_height = max(1.0, y2 - y1)
        anchor_cx = (x1 + x2) * 0.5
        anchor_cy = (y1 + y2) * 0.5
        distances = np.sqrt((geometry["cx"] - anchor_cx) ** 2 + (geometry["cy"] - anchor_cy) ** 2)
        sigma = np.maximum(10.0, np.sqrt(geometry["area"]) * 1.10)
        center_scores = np.exp(-distances / sigma)
        area_ratios = np.minimum(box_area, geometry["area"]) / np.maximum(box_area, geometry["area"])
        width_ratios = np.minimum(anchor_width, geometry["width"]) / np.maximum(anchor_width, geometry["width"])
        height_ratios = np.minimum(anchor_height, geometry["height"]) / np.maximum(anchor_height, geometry["height"])
        shape_scores = np.power(np.maximum(0.0, area_ratios), 0.25) * np.power(
            np.maximum(0.0, np.minimum(width_ratios, height_ratios)),
            0.10,
        )
        soft_scores = center_scores * shape_scores
        match_scores = np.where(small_truth_mask, np.maximum(match_scores, soft_scores), match_scores)
    best_index = int(np.argmax(match_scores))
    centers_inside = int(
        np.count_nonzero(
            (geometry["cx"] >= x1)
            & (geometry["cx"] <= x2)
            & (geometry["cy"] >= y1)
            & (geometry["cy"] <= y2)
        )
    )
    return float(match_scores[best_index]), centers_inside, float(geometry["area"][best_index])


def _estimate_training_record_bytes(image: np.ndarray, planes: np.ndarray, hardness_index: dict[str, np.ndarray]) -> int:
    return int(image.nbytes + planes.nbytes + sum(array.nbytes for array in hardness_index.values()))


def _build_training_records(
    images: str | Path,
    labels_dir: str | Path,
) -> list[dict[str, Any]] | None:
    paths = image_paths(images)
    records = []
    estimated_bytes = 0
    for index, path in enumerate(paths, start=1):
        image = read_rgb_image(path)
        planes = _precompute_feature_planes(image)
        hardness_index = build_anchor_hardness_index(planes)
        estimated_bytes += _estimate_training_record_bytes(image, planes, hardness_index)
        if _MEMORY_CACHE_MAX_BYTES > 0 and estimated_bytes > _MEMORY_CACHE_MAX_BYTES:
            print(
                "[taichi-train] memory cache skipped "
                f"(estimated={estimated_bytes / (1024 * 1024):.1f}MB limit={_MEMORY_CACHE_MAX_BYTES / (1024 * 1024):.1f}MB)",
                flush=True,
            )
            return None
        records.append(
            {
                "index": index,
                "total": len(paths),
                "path": path,
                "image": image,
                "planes": planes,
                "hardness_index": hardness_index,
                "truth": _truth_for_image(path, labels_dir),
            }
        )
    print(
        f"[taichi-train] memory cache loaded {len(records)} images "
        f"({estimated_bytes / (1024 * 1024):.1f}MB)",
        flush=True,
    )
    return records


class _TrainingFeatureBatcher:
    def __init__(
        self,
        feature_chunks: list[np.ndarray],
        labels: list[float],
        weights: list[float],
        context_padding: float,
        backend: str,
        feature_backend: str,
        flush_images: int = _TRAIN_SAMPLE_FLUSH_IMAGES,
        flush_boxes: int = _TRAIN_SAMPLE_FLUSH_BOXES,
    ) -> None:
        self.feature_chunks = feature_chunks
        self.labels = labels
        self.weights = weights
        self.context_padding = context_padding
        self.backend = backend
        self.feature_backend = feature_backend
        self.flush_images = max(1, int(flush_images))
        self.flush_boxes = max(1, int(flush_boxes))
        self.pending: list[dict[str, Any]] = []
        self.pending_boxes = 0

    def append(
        self,
        image: np.ndarray,
        planes: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        batch_labels: list[float],
        batch_weights: list[float],
    ) -> None:
        if not bboxes:
            return
        self.pending.append(
            {
                "image": image,
                "planes": planes,
                "bboxes": list(bboxes),
                "labels": list(batch_labels),
                "weights": list(batch_weights),
            }
        )
        self.pending_boxes += len(bboxes)
        if len(self.pending) >= self.flush_images or self.pending_boxes >= self.flush_boxes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        for item in self.pending:
            _append_feature_batch_from_planes(
                self.feature_chunks,
                self.labels,
                self.weights,
                item["image"],
                item["planes"],
                item["bboxes"],
                item["labels"],
                item["weights"],
                context_padding=self.context_padding,
                backend=self.backend,
                feature_backend=self.feature_backend,
            )
        self.pending.clear()
        self.pending_boxes = 0


def _iter_training_records(
    images: str | Path,
    labels_dir: str | Path,
    records: list[dict[str, Any]] | None = None,
):
    if records is not None:
        yield from records
        return
    paths = image_paths(images)
    for index, path in enumerate(paths, start=1):
        image = read_rgb_image(path)
        planes = _precompute_feature_planes(image)
        yield {
            "index": index,
            "total": len(paths),
            "path": path,
            "image": image,
            "planes": planes,
            "hardness_index": build_anchor_hardness_index(planes),
            "truth": _truth_for_image(path, labels_dir),
        }


def _truth_centers_inside_count(
    bbox: dict[str, Any],
    truth: list[dict[str, Any]],
) -> int:
    x1, y1, x2, y2 = _bbox_tuple(bbox)
    count = 0
    for item in truth:
        tx1, ty1, tx2, ty2 = _bbox_tuple(item["bbox"])
        cx = (tx1 + tx2) * 0.5
        cy = (ty1 + ty2) * 0.5
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            count += 1
    return count


def _append_feature_batch(
    feature_chunks: list[np.ndarray],
    labels: list[float],
    weights: list[float],
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    batch_labels: list[float],
    batch_weights: list[float],
    context_padding: float,
    backend: str,
    feature_backend: str,
) -> None:
    if not bboxes:
        return
    feature_chunks.append(
        batch_crop_feature_matrix(
            image,
            bboxes,
            context_padding=context_padding,
            backend=backend,
            feature_backend=feature_backend,
        ).astype(np.float32)
    )
    labels.extend(batch_labels)
    weights.extend(batch_weights)


def _append_feature_batch_from_planes(
    feature_chunks: list[np.ndarray],
    labels: list[float],
    weights: list[float],
    image: np.ndarray,
    planes: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    batch_labels: list[float],
    batch_weights: list[float],
    context_padding: float,
    backend: str,
    feature_backend: str,
) -> None:
    if not bboxes:
        return
    if feature_backend.lower().strip() == "opencv":
        feature_chunks.append(
            batch_crop_feature_matrix_from_planes(
                planes,
                bboxes,
                context_padding=context_padding,
            ).astype(np.float32)
        )
    else:
        feature_chunks.append(
            batch_crop_feature_matrix(
                image,
                bboxes,
                context_padding=context_padding,
                backend=backend,
                feature_backend=feature_backend,
            ).astype(np.float32)
        )
    labels.extend(batch_labels)
    weights.extend(batch_weights)


def _rebalance_binary_sample_weights(
    labels: np.ndarray,
    weights: np.ndarray,
    target_positive_to_negative: float = 0.92,
    max_positive_multiplier: float = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels_np = labels.astype(np.float32)
    weights_np = weights.astype(np.float32).copy()
    positive_mask = labels_np >= 0.5
    negative_mask = ~positive_mask
    positive_weight = float(np.sum(weights_np[positive_mask]))
    negative_weight = float(np.sum(weights_np[negative_mask]))
    multiplier = 1.0
    if positive_weight > 0 and negative_weight > 0:
        multiplier = min(max_positive_multiplier, max(1.0, negative_weight * target_positive_to_negative / positive_weight))
        weights_np[positive_mask] *= multiplier
    return weights_np, {
        "positive_weight_before": round(positive_weight, 4),
        "negative_weight_before": round(negative_weight, 4),
        "positive_multiplier": round(multiplier, 4),
        "positive_weight_after": round(float(np.sum(weights_np[positive_mask])), 4),
        "negative_weight_after": round(float(np.sum(weights_np[negative_mask])), 4),
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = max(1e-6, beta * beta)
    return (1.0 + beta_sq) * precision * recall / max(1e-9, beta_sq * precision + recall)


def _collect_training_samples(
    images: str | Path,
    labels_dir: str | Path,
    context_padding: float,
    scales: tuple[float, ...],
    sizes: tuple[int, ...],
    ratios: tuple[float, ...],
    stride_ratio: float,
    negatives_per_image: int,
    positive_jitter: int,
    seed: int,
    backend: str,
    feature_backend: str,
    max_positive_boxes_per_image: int = 0,
    records: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    feature_chunks: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    positive_count = 0
    negative_count = 0
    image_count = 0
    batcher = _TrainingFeatureBatcher(
        feature_chunks,
        labels,
        weights,
        context_padding=context_padding,
        backend=backend,
        feature_backend=feature_backend,
    )
    for record in _iter_training_records(images, labels_dir, records):
        index = int(record["index"])
        total = int(record["total"])
        path = Path(record["path"])
        image = record["image"]
        height, width = image.shape[:2]
        planes = record["planes"]
        hardness_index = record["hardness_index"]
        truth = record["truth"]
        truth_geometry = _truth_geometry(truth)
        truth_count = len(truth)
        image_count += 1
        image_bboxes: list[tuple[int, int, int, int]] = []
        image_labels: list[float] = []
        image_weights: list[float] = []
        sampled_truth = truth
        if max_positive_boxes_per_image > 0 and len(sampled_truth) > max_positive_boxes_per_image:
            sampled_truth = rng.sample(sampled_truth, max_positive_boxes_per_image)
        effective_truth_count = len(sampled_truth) if max_positive_boxes_per_image > 0 else truth_count
        for truth_item in sampled_truth:
            bbox = _bbox_tuple(truth_item["bbox"])
            for sample_index in range(max(1, positive_jitter)):
                candidate = bbox if sample_index == 0 else _jitter_bbox(bbox, width, height, rng)
                if candidate is None:
                    continue
                image_bboxes.append(candidate)
                image_labels.append(1.0)
                image_weights.append(2.4 if sample_index == 0 else 1.9)
                positive_count += 1

        anchors = generate_pyramid_anchors(
            width,
            height,
            scales=scales,
            sizes=sizes,
            ratios=ratios,
            stride_ratio=stride_ratio,
            max_anchors=6500,
        )
        aligned_positives = []
        partial_negatives = []
        background_negatives = []
        multi_truth_negatives = []
        oversized_negatives = []
        inside_fragment_negatives = []
        for anchor in anchors:
            best_iou, centers_inside, truth_area = _anchor_truth_stats(anchor["bbox"], truth_geometry)
            if centers_inside >= 2:
                multi_truth_negatives.append(anchor)
            elif best_iou >= 0.58 or (truth_area <= _SMALL_TRUTH_AREA and best_iou >= _SMALL_TRUTH_SOFT_POSITIVE):
                aligned_positives.append(anchor)
            elif centers_inside == 1 and best_iou < 0.42:
                inside_fragment_negatives.append(anchor)
            elif best_iou >= 0.34:
                bbox_area = max(1.0, float(anchor["bbox"]["width"]) * float(anchor["bbox"]["height"]))
                if bbox_area > truth_area * 1.85:
                    oversized_negatives.append(anchor)
            elif best_iou <= 0.04:
                background_negatives.append(anchor)
            elif best_iou < 0.34:
                partial_negatives.append(anchor)

        rng.shuffle(aligned_positives)
        for anchor in aligned_positives[: max(2, effective_truth_count * 8)]:
            image_bboxes.append(_bbox_tuple(anchor["bbox"]))
            image_labels.append(1.0)
            image_weights.append(1.55)
            positive_count += 1

        rng.shuffle(multi_truth_negatives)
        for anchor in multi_truth_negatives[: max(4, effective_truth_count * 10)]:
            image_bboxes.append(_bbox_tuple(anchor["bbox"]))
            image_labels.append(0.0)
            image_weights.append(3.4)
            negative_count += 1

        rng.shuffle(oversized_negatives)
        for anchor in oversized_negatives[: max(4, effective_truth_count * 8)]:
            image_bboxes.append(_bbox_tuple(anchor["bbox"]))
            image_labels.append(0.0)
            image_weights.append(3.0)
            negative_count += 1

        rng.shuffle(inside_fragment_negatives)
        for anchor in inside_fragment_negatives[: max(8, effective_truth_count * 14)]:
            image_bboxes.append(_bbox_tuple(anchor["bbox"]))
            image_labels.append(0.0)
            image_weights.append(3.2)
            negative_count += 1

        rng.shuffle(partial_negatives)
        for anchor in partial_negatives[: max(6, effective_truth_count * 12)]:
            image_bboxes.append(_bbox_tuple(anchor["bbox"]))
            image_labels.append(0.0)
            image_weights.append(2.7)
            negative_count += 1

        negatives = background_negatives
        hard_ranked = sorted(
            negatives,
            key=lambda anchor: quick_anchor_hardness_from_index(hardness_index, anchor["bbox"]),
            reverse=True,
        )
        selected = hard_ranked[: max(0, negatives_per_image // 2)]
        if len(hard_ranked) > negatives_per_image:
            selected.extend(rng.sample(hard_ranked[negatives_per_image // 2 :], min(negatives_per_image - len(selected), len(hard_ranked[negatives_per_image // 2 :]))))
        else:
            selected = hard_ranked
        for anchor in selected[:negatives_per_image]:
            image_bboxes.append(_bbox_tuple(anchor["bbox"]))
            image_labels.append(0.0)
            image_weights.append(1.15)
            negative_count += 1
        batcher.append(
            image,
            planes,
            image_bboxes,
            image_labels,
            image_weights,
        )
        print(
            f"[taichi-train] samples image {index}/{total} {path.name} "
            f"truth={len(truth)} sampled_truth={len(sampled_truth)} aligned={min(len(aligned_positives), max(2, effective_truth_count * 8))} "
            f"multi_neg={min(len(multi_truth_negatives), max(4, effective_truth_count * 10))} "
            f"oversize_neg={min(len(oversized_negatives), max(4, effective_truth_count * 8))} "
            f"inside_frag={min(len(inside_fragment_negatives), max(8, effective_truth_count * 14))} "
            f"partial_neg={min(len(partial_negatives), max(6, effective_truth_count * 12))} "
            f"negatives={min(len(selected), negatives_per_image)}",
            flush=True,
        )
    batcher.flush()
    if not feature_chunks:
        raise ValueError("no Taichi training samples were collected")
    return (
        np.concatenate(feature_chunks, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        {"images": image_count, "positives": positive_count, "negatives": negative_count},
    )


def _append_hard_negatives(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    model_payload: dict[str, Any],
    images: str | Path,
    labels_dir: str | Path,
    max_per_image: int,
    score_threshold: float,
    hard_negative_weight: float,
    hard_negative_max_iou: float,
    max_hard_negatives: int,
    seed: int,
    records: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = random.Random(seed)
    mined: list[tuple[float, Path, tuple[int, int, int, int]]] = []
    record_by_path: dict[Path, dict[str, Any]] = {}
    for record in _iter_training_records(images, labels_dir, records):
        path = Path(record["path"])
        record_by_path[path] = record
        result = detect_image_taichi(
            path,
            model_payload,
            score_threshold=score_threshold,
            max_detections=max_per_image,
            quiet=True,
        )
        truth = record["truth"]
        for detection in result["detections"]:
            best_iou = max_iou_with_truth(detection["bbox"], truth)
            if best_iou <= hard_negative_max_iou or _truth_centers_inside_count(detection["bbox"], truth) >= 2:
                score = float(detection.get("adjusted_score", detection.get("score", 0.0)))
                mined.append((score, path, _bbox_tuple(detection["bbox"])))
    if not mined:
        return features, labels, weights, 0

    mined.sort(key=lambda item: item[0], reverse=True)
    if max_hard_negatives > 0 and len(mined) > max_hard_negatives:
        head_count = max(1, int(max_hard_negatives * 0.65))
        head = mined[:head_count]
        tail = mined[head_count:]
        sampled_tail = rng.sample(tail, min(max_hard_negatives - len(head), len(tail))) if tail else []
        mined = head + sampled_tail

    added_chunks: list[np.ndarray] = []
    by_path: dict[Path, list[tuple[int, int, int, int]]] = {}
    for _score, path, bbox in mined:
        by_path.setdefault(path, []).append(bbox)
    for path, bboxes in by_path.items():
        record = record_by_path.get(path)
        image = record["image"] if record is not None else read_rgb_image(path)
        if bboxes:
            if record is not None and str(model_payload["config"].get("feature_backend", "opencv")).lower().strip() == "opencv":
                added_chunks.append(
                    batch_crop_feature_matrix_from_planes(
                        record["planes"],
                        bboxes,
                        context_padding=float(model_payload["config"].get("context_padding", 0.30)),
                    ).astype(np.float32)
                )
            else:
                added_chunks.append(
                    batch_crop_feature_matrix(
                        image,
                        bboxes,
                        context_padding=float(model_payload["config"].get("context_padding", 0.30)),
                        backend=str(model_payload["config"].get("backend", "auto")),
                        feature_backend=str(model_payload["config"].get("feature_backend", "opencv")),
                    ).astype(np.float32)
                )
    if not added_chunks:
        return features, labels, weights, 0
    hard_features = np.concatenate(added_chunks, axis=0).astype(np.float32)
    added_count = int(hard_features.shape[0])
    return (
        np.concatenate([features, hard_features], axis=0),
        np.concatenate([labels, np.zeros((added_count,), dtype=np.float32)], axis=0),
        np.concatenate([weights, np.full((added_count,), hard_negative_weight, dtype=np.float32)], axis=0),
        added_count,
    )


def _calibrate_score_threshold(
    model_payload: dict[str, Any],
    val_images: str | Path,
    val_labels: str | Path,
    base_threshold: float,
    nms_threshold: float,
    backend: str,
    beta: float,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    match_iou: float = DEFAULT_VAL_MATCH_IOU,
) -> dict[str, Any]:
    base_threshold = max(0.0, min(0.995, float(base_threshold)))
    candidate_thresholds = sorted(
        {
            round(value, 3)
            for value in (
                0.05,
                0.08,
                0.10,
                0.12,
                0.16,
                0.20,
                0.24,
                0.30,
                0.38,
                0.46,
                0.54,
                0.62,
                0.70,
                0.74,
                0.78,
                0.82,
                0.86,
                0.88,
                0.90,
                0.92,
                0.94,
                0.96,
                0.98,
                base_threshold,
                max(0.0, base_threshold - 0.20),
                max(0.0, base_threshold - 0.12),
                max(0.0, base_threshold - 0.04),
                min(0.995, base_threshold + 0.04),
            )
        }
    )
    sweep = []
    best: dict[str, Any] | None = None
    low_threshold = min(candidate_thresholds)
    detection_limit = max(1, int(max_detections))
    base_results = [
        detect_image_taichi(
            path,
            model_payload,
            score_threshold=low_threshold,
            nms_threshold=nms_threshold,
            max_detections=max(60, detection_limit),
            backend=backend,
            quiet=True,
        )
        for path in image_paths(val_images)
    ]
    for threshold in candidate_thresholds:
        results = []
        for result in base_results:
            filtered = dict(result)
            detections = [
                detection
                for detection in result.get("detections", [])
                if float(detection.get("adjusted_score", detection.get("score", 0.0))) >= threshold
            ]
            filtered["detections"] = detections[:detection_limit]
            results.append(filtered)
        evaluation = evaluate_detection_results(
            results,
            val_labels,
            annotations="yolo",
            match_iou=match_iou,
            evaluate_proposals=False,
        )
        summary = evaluation.get("summary", {})
        precision = float(summary.get("precision", 0.0))
        recall = float(summary.get("recall", 0.0))
        fbeta = _fbeta(precision, recall, beta)
        row = {
            "threshold": threshold,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(float(summary.get("f1", 0.0)), 6),
            "fbeta": round(fbeta, 6),
            "tp": int(summary.get("tp", 0)),
            "fp": int(summary.get("fp", 0)),
            "fn": int(summary.get("fn", 0)),
            "detections": int(summary.get("detections", 0)),
        }
        sweep.append(row)
        if best is None or _calibration_rank(row) > _calibration_rank(best):
            best = row
    assert best is not None
    exact_thresholds = _select_exact_thresholds(sweep, base_threshold=base_threshold)
    exact_sweep = []
    exact_best: dict[str, Any] | None = None
    for threshold in exact_thresholds:
        exact_results = [
            detect_image_taichi(
                path,
                model_payload,
                score_threshold=threshold,
                nms_threshold=nms_threshold,
                max_detections=detection_limit,
                backend=backend,
                quiet=True,
            )
            for path in image_paths(val_images)
        ]
        exact_evaluation = evaluate_detection_results(
            exact_results,
            val_labels,
            annotations="yolo",
            match_iou=match_iou,
            evaluate_proposals=False,
        )
        exact_summary = exact_evaluation.get("summary", {})
        exact_precision = float(exact_summary.get("precision", 0.0))
        exact_recall = float(exact_summary.get("recall", 0.0))
        exact_fbeta = _fbeta(exact_precision, exact_recall, beta)
        exact_row = {
            "threshold": threshold,
            "precision": round(exact_precision, 6),
            "recall": round(exact_recall, 6),
            "f1": round(float(exact_summary.get("f1", 0.0)), 6),
            "fbeta": round(exact_fbeta, 6),
            "tp": int(exact_summary.get("tp", 0)),
            "fp": int(exact_summary.get("fp", 0)),
            "fn": int(exact_summary.get("fn", 0)),
            "detections": int(exact_summary.get("detections", 0)),
        }
        exact_sweep.append(exact_row)
        if exact_best is None or _calibration_rank(exact_row) > _calibration_rank(exact_best):
            exact_best = exact_row
    assert exact_best is not None
    if int(exact_best.get("tp", 0)) <= 0 and int(best.get("tp", 0)) > 0:
        exact_best = best
    return {
        "beta": beta,
        "selected_threshold": float(exact_best["threshold"]),
        "selected": exact_best,
        "sweep": sweep,
        "exact_sweep": exact_sweep,
        "approx_selected": best,
    }


def _calibration_rank(row: dict[str, Any]) -> tuple[float, float, int, int, int, float]:
    tp = int(row.get("tp", 0))
    detections = int(row.get("detections", tp + int(row.get("fp", 0))))
    has_detection = 1 if detections > 0 else 0
    has_tp = 1 if tp > 0 else 0
    return (
        float(row.get("fbeta", 0.0)),
        float(row.get("recall", 0.0)),
        has_tp,
        tp,
        has_detection,
        -float(row.get("threshold", 0.0)),
    )


def _select_exact_thresholds(
    sweep: list[dict[str, Any]],
    max_thresholds: int = 6,
    base_threshold: float = 0.90,
) -> list[float]:
    ranked = sorted(
        sweep,
        key=_calibration_rank,
        reverse=True,
    )
    selected: set[float] = set()
    for row in ranked[:max_thresholds]:
        threshold = float(row["threshold"])
        selected.add(round(threshold, 3))
        selected.add(round(max(0.0, threshold - 0.02), 3))
        selected.add(round(min(0.995, threshold + 0.02), 3))
        if len(selected) >= max_thresholds:
            break
    if not selected:
        selected.add(round(max(0.0, min(0.995, base_threshold)), 3))
        selected.add(0.10)
    return sorted(selected)[:max_thresholds]


def train_taichi_detector(
    train_images: str | Path,
    train_labels: str | Path,
    output_path: str | Path,
    val_images: str | Path | None = None,
    val_labels: str | Path | None = None,
    backend: str = "auto",
    feature_backend: str = "opencv",
    context_padding: float = 0.30,
    pyramid_scales: str | None = None,
    anchor_sizes: str | None = None,
    anchor_ratios: str | None = None,
    anchor_stride_ratio: float = DEFAULT_ANCHOR_STRIDE_RATIO,
    epochs: int = 280,
    learning_rate: float = 0.08,
    hard_negative_rounds: int = 1,
    negatives_per_image: int = 140,
    positive_jitter: int = 7,
    max_positive_boxes_per_image: int = 0,
    hidden_size: int = 24,
    hard_mine_score_threshold: float = 0.86,
    hard_negative_weight: float = 1.0,
    hard_negative_max_iou: float = 0.05,
    max_hard_negatives: int = 180,
    threshold_calibration_beta: float = 1.15,
    score_threshold: float = 0.58,
    nms_threshold: float = 0.28,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    match_iou: float = DEFAULT_VAL_MATCH_IOU,
    seed: int = 42,
    draw_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    runtime = initialize_taichi(backend)
    resolved_feature_backend = feature_backend.lower().strip()
    if resolved_feature_backend not in {"opencv", "taichi"}:
        raise ValueError(f"unsupported feature_backend: {feature_backend}")
    scales = parse_float_list(pyramid_scales, DEFAULT_PYRAMID_SCALES)
    sizes = parse_int_list(anchor_sizes, DEFAULT_ANCHOR_SIZES)
    ratios = parse_float_list(anchor_ratios, DEFAULT_ANCHOR_RATIOS)
    training_records = _build_training_records(train_images, train_labels)
    features, labels, sample_weights, sample_report = _collect_training_samples(
        train_images,
        train_labels,
        context_padding=context_padding,
        scales=scales,
        sizes=sizes,
        ratios=ratios,
        stride_ratio=anchor_stride_ratio,
        negatives_per_image=negatives_per_image,
        positive_jitter=positive_jitter,
        seed=seed,
        backend=backend,
        feature_backend=resolved_feature_backend,
        max_positive_boxes_per_image=max_positive_boxes_per_image,
        records=training_records,
    )
    sample_weights, balance_report = _rebalance_binary_sample_weights(labels, sample_weights)
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0).astype(np.float32), 1e-4)
    normalized = standardize_features(features, mean, std)
    network, train_report = train_mlp_taichi(
        normalized,
        labels,
        sample_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        backend=backend,
    )
    payload = {
        "format": "solo_taichi_mlp_detector_v1",
        "network": _serialize_network(network),
        "mean": mean.round(7).tolist(),
        "std": std.round(7).tolist(),
        "config": {
            "backend": backend,
            "runtime_arch": runtime.get("arch"),
            "feature_backend": resolved_feature_backend,
            "context_padding": context_padding,
            "feature_size": FEATURE_SIZE,
            "feature_dimension": feature_dimension(),
            "hidden_size": hidden_size,
            "pyramid_scales": list(scales),
            "anchor_sizes": list(sizes),
            "anchor_ratios": list(ratios),
            "anchor_stride_ratio": anchor_stride_ratio,
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "max_detections": int(max_detections),
            "val_match_iou": float(match_iou),
            "label": "0",
            "threshold_calibration_beta": threshold_calibration_beta,
        },
        "training": {
            "initial_samples": sample_report,
            "class_balance": balance_report,
            "hard_negative_config": {
                "score_threshold": hard_mine_score_threshold,
                "sample_weight": hard_negative_weight,
                "max_iou": hard_negative_max_iou,
                "max_hard_negatives": max_hard_negatives,
            },
            "rounds": [train_report],
        },
    }
    for round_index in range(max(0, hard_negative_rounds)):
        features, labels, sample_weights, added = _append_hard_negatives(
            features,
            labels,
            sample_weights,
            payload,
            train_images,
            train_labels,
            max_per_image=50,
            score_threshold=hard_mine_score_threshold,
            hard_negative_weight=hard_negative_weight,
            hard_negative_max_iou=hard_negative_max_iou,
            max_hard_negatives=max_hard_negatives,
            seed=seed + round_index + 1000,
            records=training_records,
        )
        if not added:
            break
        sample_weights, balance_report = _rebalance_binary_sample_weights(labels, sample_weights)
        mean = features.mean(axis=0).astype(np.float32)
        std = np.maximum(features.std(axis=0).astype(np.float32), 1e-4)
        normalized = standardize_features(features, mean, std)
        network, train_report = train_mlp_taichi(
            normalized,
            labels,
            sample_weights,
            epochs=max(80, epochs // 2),
            learning_rate=learning_rate * 0.75,
            hidden_size=hidden_size,
            backend=backend,
        )
        payload.update(
            {
                "network": _serialize_network(network),
                "mean": mean.round(7).tolist(),
                "std": std.round(7).tolist(),
            }
        )
        payload["training"]["rounds"].append({"hard_negative_round": round_index + 1, "added": added, **train_report})
        payload["training"]["class_balance"] = balance_report
        print(f"[taichi-train] hard_negative_round={round_index + 1} added={added}", flush=True)

    calibration = {"enabled": False}
    if val_images and val_labels and threshold_calibration_beta > 0:
        calibration = _calibrate_score_threshold(
            payload,
            val_images,
            val_labels,
            base_threshold=score_threshold,
            nms_threshold=nms_threshold,
            backend=backend,
            beta=threshold_calibration_beta,
            max_detections=max_detections,
            match_iou=match_iou,
        )
        payload["config"]["score_threshold"] = calibration["selected_threshold"]
        payload["training"]["threshold_calibration"] = calibration
        score_threshold = calibration["selected_threshold"]
        print(
            "[taichi-train] calibrated threshold="
            f"{score_threshold:.3f} precision={calibration['selected']['precision']:.4f} "
            f"recall={calibration['selected']['recall']:.4f} fbeta={calibration['selected']['fbeta']:.4f}",
            flush=True,
        )

    save_model(output_path, payload)
    evaluation = {"enabled": False}
    results: list[dict[str, Any]] = []
    if val_images and val_labels:
        results, evaluation = detect_images_taichi(
            val_images,
            output_path,
            labels_dir=val_labels,
            output_path=report_path,
            draw_dir=draw_dir,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            match_iou=match_iou,
        )
    return {
        "weights_path": str(output_path),
        "training": payload["training"],
        "threshold_calibration": calibration,
        "evaluation": evaluation,
        "results": results,
    }


def _serialize_network(network: dict[str, Any]) -> dict[str, Any]:
    return {
        "w1": np.asarray(network.get("w1", []), dtype=np.float32).round(7).tolist(),
        "b1": np.asarray(network.get("b1", []), dtype=np.float32).round(7).tolist(),
        "w2": np.asarray(network.get("w2", []), dtype=np.float32).round(7).tolist(),
        "b2": round(float(network.get("b2", 0.0)), 8),
    }


def _network_from_payload(model_payload: dict[str, Any]) -> dict[str, Any]:
    if "network" in model_payload:
        return model_payload["network"]
    return {
        "w1": [],
        "b1": [],
        "w2": model_payload.get("weights", []),
        "b2": model_payload.get("bias", 0.0),
    }


def _merge_partial_boxes(detections: list[dict[str, Any]], image_width: int, image_height: int) -> tuple[list[dict[str, Any]], int]:
    groups: list[list[dict[str, Any]]] = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        target = None
        for group in groups:
            union = _union_group_bbox(group, image_width, image_height)
            if detection["label"] != group[0]["label"]:
                continue
            candidate_union = _union_group_bbox([*group, detection], image_width, image_height)
            shape_compatible = _merge_shape_compatible(detection["bbox"], union, candidate_union)
            distance_compatible = _distance_merge_compatible(detection["bbox"], union, candidate_union, image_width, image_height)
            if not shape_compatible and not distance_compatible:
                continue
            largest_area = max(
                max(1.0, float(item["bbox"]["width"]) * float(item["bbox"]["height"]))
                for item in [*group, detection]
            )
            union_area = max(1.0, float(candidate_union["width"]) * float(candidate_union["height"]))
            max_growth = 2.35 if distance_compatible else 1.55
            if union_area / largest_area > max_growth:
                continue
            if (
                bbox_iou(detection["bbox"], union) >= 0.34
                or bbox_containment(detection["bbox"], union) >= 0.74
                or distance_compatible
            ):
                target = group
                break
        if target is None:
            groups.append([detection])
        else:
            target.append(detection)
    merged: list[dict[str, Any]] = []
    merged_count = 0
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        best = max(group, key=lambda item: item["score"])
        fused = dict(best)
        fused["bbox"] = _union_group_bbox(group, image_width, image_height)
        if _fused_box_too_large(fused["bbox"], group, image_width, image_height):
            guarded = dict(best)
            guarded["proposal"] = "taichi_anchor_fused_guarded"
            guarded["fused_rejected_count"] = len(group)
            merged.append(guarded)
            continue
        fused["score"] = max(item["score"] for item in group)
        fused["adjusted_score"] = fused["score"]
        fused["proposal"] = "taichi_anchor_fused"
        fused["fused_count"] = len(group)
        fused["fused_components"] = [
            {
                "bbox": item["bbox"],
                "score": round(float(item.get("adjusted_score", item.get("score", 0.0))), 6),
                "proposal": item.get("proposal"),
            }
            for item in group
        ]
        merged_count += len(group) - 1
        merged.append(fused)
    return merged, merged_count


def _fused_box_too_large(
    fused_bbox: dict[str, Any],
    group: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> bool:
    image_area = max(1.0, float(image_width * image_height))
    fused_area = max(1.0, float(fused_bbox["width"]) * float(fused_bbox["height"]))
    largest_area = max(
        max(1.0, float(item["bbox"]["width"]) * float(item["bbox"]["height"]))
        for item in group
    )
    area_ratio = fused_area / image_area
    growth = fused_area / largest_area
    too_much_frame = area_ratio > 0.32
    very_wide_frame = (
        float(fused_bbox["width"]) > float(image_width) * 0.68
        and float(fused_bbox["height"]) > float(image_height) * 0.42
    )
    chain_growth = len(group) >= 8 and area_ratio > 0.22 and growth > 1.28
    aspect = float(fused_bbox["width"]) / max(1.0, float(fused_bbox["height"]))
    chain_bridge = (
        len(group) >= 8
        and aspect > 2.55
        and float(fused_bbox["width"]) > float(image_width) * 0.38
        and area_ratio > 0.08
    )
    return too_much_frame or very_wide_frame or chain_growth or chain_bridge


def _bbox_center_xy(bbox: dict[str, Any]) -> tuple[float, float]:
    return (float(bbox["x1"]) + float(bbox["x2"])) * 0.5, (float(bbox["y1"]) + float(bbox["y2"])) * 0.5


def _bbox_horizontal_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    if float(a["x2"]) < float(b["x1"]):
        return float(b["x1"]) - float(a["x2"])
    if float(b["x2"]) < float(a["x1"]):
        return float(a["x1"]) - float(b["x2"])
    return 0.0


def _bbox_vertical_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = min(float(a["y2"]), float(b["y2"])) - max(float(a["y1"]), float(b["y1"]))
    if overlap <= 0.0:
        return 0.0
    return overlap / max(1.0, min(float(a["height"]), float(b["height"])))


def _distance_merge_compatible(
    bbox: dict[str, Any],
    group_bbox: dict[str, Any],
    union_bbox: dict[str, Any],
    image_width: int,
    image_height: int,
) -> bool:
    width = max(1.0, float(bbox["width"]))
    height = max(1.0, float(bbox["height"]))
    group_width = max(1.0, float(group_bbox["width"]))
    group_height = max(1.0, float(group_bbox["height"]))
    union_width = max(1.0, float(union_bbox["width"]))
    union_height = max(1.0, float(union_bbox["height"]))
    height_ratio = min(height, group_height) / max(height, group_height)
    if height_ratio < 0.45:
        return False
    if _bbox_vertical_overlap_ratio(bbox, group_bbox) < 0.38:
        return False
    _center_x, center_y = _bbox_center_xy(bbox)
    _group_center_x, group_center_y = _bbox_center_xy(group_bbox)
    if abs(center_y - group_center_y) > max(height, group_height) * 0.42:
        return False
    gap = _bbox_horizontal_gap(bbox, group_bbox)
    if _looks_like_separate_taichi_instances(bbox, group_bbox, union_bbox, gap):
        return False
    allowed_gap = max(10.0, min(width, group_width) * 0.42, max(width, group_width) * 0.16)
    if gap > allowed_gap:
        return False
    if union_height / max(height, group_height) > 1.34:
        return False
    union_aspect = union_width / union_height
    if union_aspect < 1.15 or union_aspect > 7.2:
        return False
    image_area = max(1.0, float(image_width * image_height))
    if (union_width * union_height) / image_area > 0.30:
        return False
    if union_width > float(image_width) * 0.78 and union_height > float(image_height) * 0.34:
        return False
    largest_area = max(width * height, group_width * group_height)
    if union_width * union_height / max(1.0, largest_area) > 2.35:
        return False
    return True


def _looks_like_separate_taichi_instances(
    bbox: dict[str, Any],
    group_bbox: dict[str, Any],
    union_bbox: dict[str, Any],
    gap: float,
) -> bool:
    width = max(1.0, float(bbox["width"]))
    height = max(1.0, float(bbox["height"]))
    group_width = max(1.0, float(group_bbox["width"]))
    group_height = max(1.0, float(group_bbox["height"]))
    union_width = max(1.0, float(union_bbox["width"]))
    union_height = max(1.0, float(union_bbox["height"]))
    if min(height, group_height) < 10.0 or min(width, group_width) < 18.0:
        return False
    height_ratio = min(height, group_height) / max(height, group_height)
    if height_ratio < 0.58:
        return False
    aspect_a = width / height
    aspect_b = group_width / group_height
    if min(aspect_a, aspect_b) < 0.95:
        return False
    center_x, _center_y = _bbox_center_xy(bbox)
    group_center_x, _group_center_y = _bbox_center_xy(group_bbox)
    center_spread = abs(center_x - group_center_x)
    clear_gap = gap >= max(14.0, min(width, group_width) * 0.24)
    large_union = union_width / max(width, group_width) >= 1.42 or (union_width * union_height) / max(width * height, group_width * group_height) >= 1.58
    return clear_gap and large_union and center_spread >= max(16.0, min(width, group_width) * 0.52)


def _merge_shape_compatible(
    bbox: dict[str, Any],
    group_bbox: dict[str, Any],
    union_bbox: dict[str, Any],
) -> bool:
    width = max(1.0, float(bbox["width"]))
    height = max(1.0, float(bbox["height"]))
    group_width = max(1.0, float(group_bbox["width"]))
    group_height = max(1.0, float(group_bbox["height"]))
    union_width = max(1.0, float(union_bbox["width"]))
    union_height = max(1.0, float(union_bbox["height"]))
    height_ratio = min(height, group_height) / max(height, group_height)
    if height_ratio < 0.52:
        return False
    center_x, center_y = _bbox_center_xy(bbox)
    group_center_x, group_center_y = _bbox_center_xy(group_bbox)
    y_delta = abs(center_y - group_center_y) / max(height, group_height)
    if y_delta > 0.32:
        return False
    x_delta = abs(center_x - group_center_x) / max(width, group_width)
    if x_delta > 0.34 and bbox_iou(bbox, group_bbox) < 0.52:
        return False
    if union_width / max(width, group_width) > 1.38:
        return False
    if union_height / max(height, group_height) > 1.38:
        return False
    if union_width / max(1.0, min(width, group_width)) > 2.35 and bbox_iou(bbox, group_bbox) < 0.42:
        return False
    if union_width * union_height / max(1.0, width * height, group_width * group_height) > 1.48:
        return False
    return True


def _suppress_multi_object_covers(detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if len(detections) < 3:
        return detections, 0
    suppressed: set[int] = set()
    for index, detection in enumerate(detections):
        box = detection["bbox"]
        area = max(1.0, float(box["width"]) * float(box["height"]))
        score = float(detection.get("adjusted_score", detection.get("score", 0.0)))
        contained = []
        for other_index, other in enumerate(detections):
            if other_index == index or other.get("label") != detection.get("label"):
                continue
            other_area = max(1.0, float(other["bbox"]["width"]) * float(other["bbox"]["height"]))
            if other_area >= area * 0.72:
                continue
            if bbox_containment(other["bbox"], box) < 0.82:
                continue
            other_score = float(other.get("adjusted_score", other.get("score", 0.0)))
            if other_score >= score - 0.04:
                contained.append(other)
        if len(contained) < 2:
            continue
        centers = sorted(_bbox_center_xy(item["bbox"])[0] for item in contained)
        if centers[-1] - centers[0] < max(8.0, float(box["width"]) * 0.24):
            continue
        total_area = sum(max(1.0, float(item["bbox"]["width"]) * float(item["bbox"]["height"])) for item in contained)
        cover_center_x, _cover_center_y = _bbox_center_xy(box)
        split_sides = centers[0] < cover_center_x - float(box["width"]) * 0.08 and centers[-1] > cover_center_x + float(box["width"]) * 0.08
        strong_pair = len(contained) >= 2 and centers[-1] - centers[0] >= max(12.0, float(box["width"]) * 0.34)
        if total_area / area < 0.18 and not (split_sides and strong_pair):
            continue
        suppressed.add(index)
    if not suppressed:
        return detections, 0
    return [item for index, item in enumerate(detections) if index not in suppressed], len(suppressed)


def _suppress_attached_fragments(detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if len(detections) < 2:
        return detections, 0
    suppressed: set[int] = set()
    for index, detection in enumerate(detections):
        box = detection["bbox"]
        area = max(1.0, float(box["width"]) * float(box["height"]))
        center_x, center_y = _bbox_center_xy(box)
        for other_index, other in enumerate(detections):
            if other_index == index or other.get("label") != detection.get("label"):
                continue
            other_box = other["bbox"]
            other_area = max(1.0, float(other_box["width"]) * float(other_box["height"]))
            if area >= other_area * 0.72:
                continue
            other_center_x, other_center_y = _bbox_center_xy(other_box)
            height_ratio = float(box["height"]) / max(1.0, float(other_box["height"]))
            width_ratio = float(box["width"]) / max(1.0, float(other_box["width"]))
            containment = bbox_containment(box, other_box)
            if height_ratio < 0.46 and bbox_iou(box, other_box) > 0.12:
                suppressed.add(index)
                break
            if (
                area < other_area * 0.46
                and containment > 0.34
                and bbox_iou(box, other_box) > 0.05
                and float(other.get("bbox", {}).get("width", 0)) >= max(48.0, float(box["width"]) * 1.45)
            ):
                suppressed.add(index)
                break
            vertical_band = (
                float(other_box["y1"]) + float(other_box["height"]) * 0.42
                <= center_y
                <= float(other_box["y2"]) + float(other_box["height"]) * 0.28
            )
            horizontal_touch = (
                float(other_box["x1"]) - float(other_box["width"]) * 0.10
                <= center_x
                <= float(other_box["x2"]) + float(other_box["width"]) * 0.24
            )
            close_to_vehicle = bbox_iou(box, other_box) > 0.06 or containment > 0.16
            edge_fragment = abs(center_x - other_center_x) > float(other_box["width"]) * 0.30
            top_fragment = height_ratio < 0.62 and width_ratio < 0.64 and center_y < other_center_y + float(other_box["height"]) * 0.08
            if vertical_band and horizontal_touch and close_to_vehicle and (edge_fragment or top_fragment):
                suppressed.add(index)
                break
            if (
                height_ratio < 0.92
                and width_ratio < 0.72
                and horizontal_touch
                and abs(center_y - other_center_y) <= max(12.0, float(other_box["height"]) * 0.38)
                and bbox_iou(box, other_box) > 0.03
                and float(other_box["width"]) >= 80.0
            ):
                suppressed.add(index)
                break
    if not suppressed:
        return detections, 0
    return [item for index, item in enumerate(detections) if index not in suppressed], len(suppressed)


def _union_group_bbox(group: list[dict[str, Any]], image_width: int, image_height: int) -> dict[str, Any]:
    left = min(float(item["bbox"]["x1"]) for item in group)
    top = min(float(item["bbox"]["y1"]) for item in group)
    right = max(float(item["bbox"]["x2"]) for item in group)
    bottom = max(float(item["bbox"]["y2"]) for item in group)
    return _bbox_payload((left, top, right, bottom), image_width, image_height)


def _expand_tuple(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> tuple[int, int, int, int] | None:
    from solo.utils.bbox import _clamp_bbox

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return _clamp_bbox(
        (
            x1 - width * left_ratio,
            y1 - height * top_ratio,
            x2 + width * right_ratio,
            y2 + height * bottom_ratio,
        ),
        image_width,
        image_height,
    )


def _append_context_expansions(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    model_payload: dict[str, Any],
    threshold: float,
    backend: str,
    feature_backend: str,
    limit: int = 28,
) -> tuple[list[dict[str, Any]], int]:
    if not detections:
        return detections, 0
    height, width = image.shape[:2]
    config = model_payload["config"]
    context_padding = float(config.get("context_padding", 0.30))
    mean = np.asarray(model_payload["mean"], dtype=np.float32)
    std = np.asarray(model_payload["std"], dtype=np.float32)
    network = _network_from_payload(model_payload)
    variants: list[tuple[tuple[int, int, int, int], dict[str, Any], float, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    expansion_patterns = (
        (0.16, 0.08, 0.16, 0.10),
        (0.28, 0.12, 0.28, 0.14),
        (0.45, 0.16, 0.45, 0.18),
        (0.55, 0.12, 0.25, 0.16),
        (0.25, 0.12, 0.55, 0.16),
    )
    for base_index, detection in enumerate(sorted(detections, key=lambda item: item["score"], reverse=True)[:limit]):
        base = _bbox_tuple(detection["bbox"])
        base_area = max(1, (base[2] - base[0]) * (base[3] - base[1]))
        if base_area / max(1.0, float(width * height)) > 0.20:
            continue
        for pattern in expansion_patterns:
            expanded = _expand_tuple(base, width, height, *pattern)
            if expanded is None or expanded in seen or expanded == base:
                continue
            box_w = max(1, expanded[2] - expanded[0])
            box_h = max(1, expanded[3] - expanded[1])
            aspect = box_w / max(1.0, float(box_h))
            area_ratio = (box_w * box_h) / max(1.0, float(width * height))
            if aspect < 0.65 or aspect > 5.8 or area_ratio > 0.55:
                continue
            seen.add(expanded)
            variants.append((expanded, detection, float((box_w * box_h) / base_area), base_index))
    if not variants:
        return detections, 0

    features = batch_crop_feature_matrix(
        image,
        [bbox for bbox, _base, _area_gain, _base_index in variants],
        context_padding=context_padding,
        backend=backend,
        feature_backend=feature_backend,
    ).astype(np.float32)
    if features.shape[1] != mean.shape[0] or features.shape[1] != std.shape[0]:
        raise ValueError(
            "Taichi detector weight feature dimension does not match the current extractor: "
            f"features={features.shape[1]} mean={mean.shape[0]} std={std.shape[0]}. "
            "Retrain the detector after feature extractor changes."
        )
    scores = score_mlp_taichi(
        standardize_features(features, mean, std),
        network,
        backend=backend,
    )
    best_by_base: dict[int, dict[str, Any]] = {}
    for (bbox, base_detection, area_gain, base_index), score in zip(variants, scores):
        base_score = float(base_detection["score"])
        score_value = float(score)
        if score_value < max(threshold + 0.04, base_score + 0.01):
            continue
        expanded = dict(base_detection)
        expanded["bbox"] = _bbox_payload(bbox, width, height)
        expanded["score"] = round(score_value, 6)
        expanded["adjusted_score"] = round(score_value, 6)
        expanded["proposal"] = "taichi_context_expand"
        expanded["base_score"] = round(base_score, 6)
        expanded["area_gain"] = round(area_gain, 4)
        current = best_by_base.get(base_index)
        if current is None or float(expanded["adjusted_score"]) > float(current["adjusted_score"]):
            best_by_base[base_index] = expanded
    appended = list(best_by_base.values())
    return detections + appended, len(appended)


def _shadow_road_penalty_from_feature(vector: np.ndarray, feature_size: int = FEATURE_SIZE) -> float:
    stat_base = 2 * 5 * feature_size * feature_size
    if vector.shape[0] < stat_base + 32:
        return 1.0
    target_shadow = float(vector[stat_base + 27])
    context_shadow = float(vector[stat_base + 29])
    target_edge = float(vector[stat_base + 6])
    target_sat = float(vector[stat_base + 2])
    road_like = max(target_shadow, context_shadow * 0.85) * (1.0 - min(1.0, target_edge * 4.5)) * (1.0 - min(1.0, target_sat * 3.5))
    return max(0.55, 1.0 - max(0.0, min(1.0, road_like)) * 0.35)


def _low_contrast_penalty_from_feature(vector: np.ndarray, feature_size: int = FEATURE_SIZE) -> float:
    stat_base = 2 * 5 * feature_size * feature_size
    if vector.shape[0] < stat_base + 32:
        return 1.0
    target_gray_std = float(vector[stat_base + 1])
    target_value_std = float(vector[stat_base + 5])
    target_grad = float(vector[stat_base + 6])
    context_gray_std = float(vector[stat_base + 9])
    context_value_std = float(vector[stat_base + 13])
    context_grad = float(vector[stat_base + 14])
    target_activity = max(target_gray_std, target_value_std, target_grad)
    context_activity = max(context_gray_std, context_value_std, context_grad)
    if target_activity <= 1e-6:
        return 0.08
    if target_activity < 0.035 and context_activity < 0.045:
        return 0.35
    return 1.0


def _apply_detection_penalties(
    detections: list[dict[str, Any]],
    scores: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    kept = []
    for detection, score, vector in zip(detections, scores, matrix):
        low_contrast_penalty = _low_contrast_penalty_from_feature(vector)
        penalty = _shadow_road_penalty_from_feature(vector) * low_contrast_penalty
        adjusted = float(score) * penalty
        if adjusted < threshold:
            continue
        item = dict(detection)
        item["score"] = round(float(score), 6)
        item["adjusted_score"] = round(adjusted, 6)
        if penalty < 0.999:
            item["road_shadow_penalty"] = round(penalty, 6)
        if low_contrast_penalty < 0.999:
            item["low_contrast_penalty"] = round(low_contrast_penalty, 6)
        kept.append(item)
    return kept


def _extract_pyramid_candidate_features(
    candidates: list[dict[str, Any]],
    context_padding: float,
    backend: str,
    feature_backend: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not candidates:
        return np.zeros((0, feature_dimension()), dtype=np.float32), []
    ordered_features: list[np.ndarray] = []
    ordered_candidates: list[dict[str, Any]] = []
    by_level: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_level.setdefault(int(candidate["level_index"]), []).append(candidate)
    for _level_index, level_candidates in sorted(by_level.items()):
        level = level_candidates[0]["level"]
        bboxes = [_bbox_tuple(candidate["anchor"]["bbox"]) for candidate in level_candidates]
        if str(feature_backend).lower().strip() == "opencv":
            planes = level.get("planes")
            if planes is None:
                planes = _precompute_feature_planes(level["image"])
                level["planes"] = planes
            matrix = batch_crop_feature_matrix_from_planes(
                planes,
                bboxes,
                context_padding=context_padding,
            ).astype(np.float32)
        else:
            matrix = batch_crop_feature_matrix(
                level["image"],
                bboxes,
                context_padding=context_padding,
                backend=backend,
                feature_backend=feature_backend,
            ).astype(np.float32)
        ordered_features.append(matrix)
        ordered_candidates.extend(level_candidates)
    return np.concatenate(ordered_features, axis=0).astype(np.float32), ordered_candidates


def _map_level_bbox_to_source(
    bbox: dict[str, Any],
    source_width: int,
    source_height: int,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float, float, float] | None:
    from solo.utils.bbox import _clamp_bbox_float

    return _clamp_bbox_float(
        (
            float(bbox["x1"]) / max(1e-6, scale_x),
            float(bbox["y1"]) / max(1e-6, scale_y),
            float(bbox["x2"]) / max(1e-6, scale_x),
            float(bbox["y2"]) / max(1e-6, scale_y),
        ),
        source_width,
        source_height,
    )


def _source_anchor_rank(candidate: dict[str, Any], image_width: int, image_height: int) -> float:
    bbox = candidate["source_bbox"]
    width = max(1.0, float(bbox[2] - bbox[0]))
    height = max(1.0, float(bbox[3] - bbox[1]))
    aspect = width / height
    area_ratio = (width * height) / max(1.0, float(image_width * image_height))
    center_y = (float(bbox[1]) + float(bbox[3])) * 0.5 / max(1.0, float(image_height))
    aspect_score = max(
        math.exp(-abs(math.log(aspect / 1.25))) * 0.82,
        math.exp(-abs(math.log(aspect / 1.8))),
        math.exp(-abs(math.log(aspect / 2.45))) * 0.92,
        math.exp(-abs(math.log(aspect / 3.2))) * 0.74,
    )
    small_anchor_enabled = int(candidate["anchor"].get("anchor_size", 999)) <= 14
    area_targets = [
        (0.018, 1.00),
        (0.070, 0.92),
        (0.155, 0.88),
        (0.255, 0.72),
    ]
    if small_anchor_enabled:
        area_targets = [(0.0015, 0.86), (0.0060, 0.94), *area_targets]
    area_score = max(math.exp(-abs(math.log(max(area_ratio, 1e-5) / target))) * weight for target, weight in area_targets)
    road_score = 1.0 - abs(center_y - 0.62) / 0.62
    scale = float(candidate["level"].get("scale", 1.0))
    scale_bonus = 0.06 if scale <= 0.50 and area_ratio >= 0.055 else 0.0
    small_bonus = 0.08 if small_anchor_enabled and area_ratio <= 0.008 and center_y >= 0.28 else 0.0
    return aspect_score * 0.38 + area_score * 0.34 + max(0.0, road_score) * 0.18 + scale_bonus + small_bonus


def _generate_image_pyramid_candidates(
    image: np.ndarray,
    scales: tuple[float, ...],
    sizes: tuple[int, ...],
    ratios: tuple[float, ...],
    stride_ratio: float,
    backend: str,
    max_anchors: int = 7000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = image.shape[:2]
    levels = build_image_pyramid(image, scales=scales, backend=backend)
    candidates: list[dict[str, Any]] = []
    for level_index, level in enumerate(levels):
        if str(backend).lower().strip() != "taichi":
            level["planes"] = _precompute_feature_planes(level["image"])
        level_anchors = generate_pyramid_anchors(
            int(level["width"]),
            int(level["height"]),
            scales=(1.0,),
            sizes=sizes,
            ratios=ratios,
            stride_ratio=stride_ratio,
            max_anchors=max_anchors,
        )
        for anchor in level_anchors:
            source_bbox = _map_level_bbox_to_source(
                anchor["bbox"],
                width,
                height,
                float(level["scale_x"]),
                float(level["scale_y"]),
            )
            if source_bbox is None:
                continue
            candidates.append(
                {
                    "level_index": level_index,
                    "level": level,
                    "anchor": anchor,
                    "source_bbox": source_bbox,
                }
            )
    if max_anchors > 0 and len(candidates) > max_anchors:
        candidates = sorted(
            candidates,
            key=lambda item: _source_anchor_rank(item, width, height),
            reverse=True,
        )[:max_anchors]
    return levels, candidates


def detect_image_taichi(
    image_path: str | Path,
    model_payload: dict[str, Any],
    score_threshold: float | None = None,
    nms_threshold: float | None = None,
    max_detections: int = 20,
    backend: str | None = None,
    min_detection_width: int = 8,
    min_detection_height: int = 6,
    min_detection_area: int = 80,
    quiet: bool = False,
) -> dict[str, Any]:
    started = time.time()
    path = Path(image_path)
    image = read_rgb_image(path)
    height, width = image.shape[:2]
    config = model_payload["config"]
    threshold = float(config.get("score_threshold", 0.58) if score_threshold is None else score_threshold)
    nms_value = float(config.get("nms_threshold", 0.28) if nms_threshold is None else nms_threshold)
    scales = tuple(float(value) for value in config.get("pyramid_scales", DEFAULT_PYRAMID_SCALES))
    sizes = tuple(int(value) for value in config.get("anchor_sizes", DEFAULT_ANCHOR_SIZES))
    ratios = tuple(float(value) for value in config.get("anchor_ratios", DEFAULT_ANCHOR_RATIOS))
    stride_ratio = float(config.get("anchor_stride_ratio", DEFAULT_ANCHOR_STRIDE_RATIO))
    context_padding = float(config.get("context_padding", 0.30))
    resolved_backend = backend or str(config.get("backend", "auto"))
    resolved_feature_backend = str(config.get("feature_backend", "opencv"))
    mean = np.asarray(model_payload["mean"], dtype=np.float32)
    std = np.asarray(model_payload["std"], dtype=np.float32)
    network = _network_from_payload(model_payload)

    pyramid_levels, candidates = _generate_image_pyramid_candidates(
        image,
        scales=scales,
        sizes=sizes,
        ratios=ratios,
        stride_ratio=stride_ratio,
        backend=resolved_backend,
    )
    raw_detections = []
    raw_scores = []
    matrix, scored_candidates = _extract_pyramid_candidate_features(
        candidates,
        context_padding=context_padding,
        backend=resolved_backend,
        feature_backend=resolved_feature_backend,
    )
    if matrix.shape[1] != mean.shape[0] or matrix.shape[1] != std.shape[0]:
        raise ValueError(
            "Taichi detector weight feature dimension does not match the current extractor: "
            f"features={matrix.shape[1]} mean={mean.shape[0]} std={std.shape[0]}. "
            "Retrain the detector after feature extractor changes."
        )
    scores = score_mlp_taichi(
        standardize_features(matrix, mean, std),
        network,
        backend=resolved_backend,
    )
    kept_vectors = []
    for candidate, score, vector in zip(scored_candidates, scores, matrix):
        if float(score) < threshold:
            continue
        level = candidate["level"]
        anchor = candidate["anchor"]
        raw_detections.append(
            {
                "label": str(config.get("label", "0")),
                "score": round(float(score), 6),
                "adjusted_score": round(float(score), 6),
                "bbox": _bbox_payload(candidate["source_bbox"], width, height),
                "proposal": "taichi_image_pyramid_anchor",
                "image_pyramid_scale": round(float(level["scale"]), 6),
                "pyramid_scale": round(float(level["scale"]), 6),
                "level_bbox": anchor["bbox"],
                "anchor_size": anchor.get("anchor_size"),
                "anchor_ratio": anchor.get("anchor_ratio"),
            }
        )
        raw_scores.append(float(score))
        kept_vectors.append(vector)
    detections = _apply_detection_penalties(
        raw_detections,
        np.asarray(raw_scores, dtype=np.float32),
        np.asarray(kept_vectors, dtype=np.float32),
        threshold,
    )
    detections, expanded_count = _append_context_expansions(
        image,
        detections,
        model_payload,
        threshold,
        resolved_backend,
        resolved_feature_backend,
    )
    detections, merged_count = _merge_partial_boxes(detections, width, height)
    detections, distance_fused_count = distance_box_fusion(
        detections,
        width,
        height,
        proposal_name="taichi_distance_box_fused",
    )
    detections, overmerged_split_count = split_overmerged_fusions(
        detections,
        width,
        height,
        proposal_suffix="taichi_split",
    )
    detections, edge_snapped_count = snap_detections_to_edges(
        image,
        detections,
        width,
        height,
        search_px=int(config.get("edge_snap_search_px", 5)),
        min_gain=float(config.get("edge_snap_min_gain", 0.015)),
    )
    detections, fragment_suppressed = _suppress_attached_fragments(detections)
    detections, multi_cover_suppressed = _suppress_multi_object_covers(detections)
    final = nms(
        detections,
        iou_threshold=nms_value,
        containment_threshold=0.72,
        cluster_center_distance=0.38,
        cluster_containment_threshold=0.55,
        diou_threshold=float(config.get("diou_nms_threshold", DEFAULT_NMS_DIOU_THRESHOLD)),
    )
    final, suppressed = filter_detections(
        final,
        min_width=min_detection_width,
        min_height=min_detection_height,
        min_area=min_detection_area,
        max_detections=max_detections,
    )
    if not quiet:
        print(
            f"[taichi-detect] {path.name} levels={len(pyramid_levels)} anchors={len(candidates)} raw={len(detections)} expand={expanded_count} "
            f"merge={merged_count}+{distance_fused_count} split={overmerged_split_count} snap={edge_snapped_count} det={len(final)} elapsed={time.time() - started:.2f}s",
            flush=True,
        )
    return {
        "image": str(path),
        "width": width,
        "height": height,
        "proposal_count": len(candidates),
        "pyramid_levels": [
            {
                "scale": round(float(level["scale"]), 6),
                "width": int(level["width"]),
                "height": int(level["height"]),
            }
            for level in pyramid_levels
        ],
        "raw_detection_count": len(detections),
        "taichi_context_expanded_count": expanded_count,
        "taichi_merged_count": merged_count,
        "taichi_distance_fused_count": distance_fused_count,
        "taichi_overmerged_split_count": overmerged_split_count,
        "taichi_edge_snapped_count": edge_snapped_count,
        "taichi_fragment_suppressed": fragment_suppressed,
        "taichi_multi_cover_suppressed": multi_cover_suppressed,
        "postprocess_suppressed": suppressed,
        "detections": final,
        "elapsed_seconds": round(time.time() - started, 4),
    }


def detect_images_taichi(
    image_path: str | Path,
    weights_path: str | Path,
    labels_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    draw_dir: str | Path | None = None,
    hide_labels: bool = False,
    score_threshold: float | None = None,
    nms_threshold: float | None = None,
    max_detections: int = 20,
    backend: str | None = None,
    annotation_format: str = "yolo",
    class_names_path: str | Path | None = None,
    match_iou: float = DEFAULT_VAL_MATCH_IOU,
    duplicate_iou: float = 0.85,
    min_detection_width: int = 8,
    min_detection_height: int = 6,
    min_detection_area: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_model(weights_path)
    runtime = initialize_taichi(backend or str(payload.get("config", {}).get("backend", "auto")))
    results = []
    for index, path in enumerate(image_paths(image_path), start=1):
        print(f"[taichi-detect] image {index} {path.name}", flush=True)
        result = detect_image_taichi(
            path,
            payload,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            backend=backend,
            min_detection_width=min_detection_width,
            min_detection_height=min_detection_height,
            min_detection_area=min_detection_area,
        )
        if draw_dir is not None:
            draw_path = Path(draw_dir) / path.name
            result["draw_path"] = str(draw_detections(path, result["detections"], draw_path, hide_labels=hide_labels))
        results.append(result)
    evaluation = {"enabled": False}
    if labels_dir is not None:
        evaluation = evaluate_detection_results(
            results,
            labels_dir,
            annotations=annotation_format,
            class_names_path=class_names_path,
            match_iou=match_iou,
            duplicate_iou=duplicate_iou,
            weight_index=None,
            evaluate_proposals=False,
        )
    if output_path is not None:
        save_detection_report(
            results,
            output_path,
            {
                "backend": "taichi",
                "weights": [str(weights_path)],
                "score_threshold": score_threshold if score_threshold is not None else payload["config"].get("score_threshold"),
                "nms_threshold": nms_threshold if nms_threshold is not None else payload["config"].get("nms_threshold"),
                "runtime": {key: value for key, value in runtime.items() if key != "ti"},
                "config": payload.get("config", {}),
                "evaluation": evaluation,
                "total_images": len(results),
                "total_detections": sum(len(result["detections"]) for result in results),
            },
        )
    return results, evaluation


__all__ = ["detect_image_taichi", "detect_images_taichi", "train_taichi_detector"]
