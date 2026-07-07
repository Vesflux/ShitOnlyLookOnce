from __future__ import annotations

import math
import random
import statistics
from typing import Any

from solo.config import DEFAULT_OBJECTNESS_WORK_SIZE
from solo.core.proposal_features import _mean_in_bbox, _objectness_maps, color_proposals
from solo.utils.cv_image import Image
from solo.utils.bbox import (
    _bbox_iou_tuple,
    _bbox_payload,
    _bbox_tuple,
    _bbox_tuple_from_annotation,
    _clamp_bbox,
)


def _median_box_size(boxes: list[dict[str, Any]]) -> tuple[int, int]:
    widths = [int(box["bbox"]["width"]) for box in boxes if int(box["bbox"]["width"]) > 0]
    heights = [int(box["bbox"]["height"]) for box in boxes if int(box["bbox"]["height"]) > 0]
    if not widths or not heights:
        return 32, 32
    return max(1, round(statistics.median(widths))), max(1, round(statistics.median(heights)))


def _sample_negative_boxes(
    image_width: int,
    image_height: int,
    positive_boxes: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    min_iou: float = 0.05,
    attempts_per_box: int = 200,
    image: Image.Image | None = None,
) -> list[tuple[int, int, int, int]]:
    if count <= 0 or image_width <= 1 or image_height <= 1:
        return []

    base_width, base_height = _median_box_size(positive_boxes)
    positive_tuples = [_bbox_tuple_from_annotation(item) for item in positive_boxes]
    sampled = []
    fallback_candidates = []
    if image is not None:
        hard_candidates = _hard_objectness_negative_candidates(
            image,
            positive_tuples,
            count=max(count * 3, count + 8),
            min_iou=min_iou,
        )
        for candidate, _score in hard_candidates:
            if len(sampled) >= max(1, math.ceil(count * 0.65)):
                break
            if any(_bbox_iou_tuple(candidate, existing) > 0.5 for existing in sampled):
                continue
            sampled.append(candidate)
    max_attempts = max(1, count * attempts_per_box)
    attempts = 0

    while len(sampled) < count and attempts < max_attempts:
        attempts += 1
        scale = rng.uniform(0.65, 1.8)
        jitter = rng.uniform(0.75, 1.35)
        box_width = max(6, min(image_width, round(base_width * scale)))
        box_height = max(6, min(image_height, round(base_height * scale * jitter)))
        if box_width >= image_width or box_height >= image_height:
            box_width = max(1, min(box_width, image_width))
            box_height = max(1, min(box_height, image_height))
        x1 = rng.randint(0, max(0, image_width - box_width))
        y1 = rng.randint(0, max(0, image_height - box_height))
        candidate = (x1, y1, x1 + box_width, y1 + box_height)
        if any(_bbox_iou_tuple(candidate, existing) > 0.5 for existing in sampled):
            continue
        max_positive_iou = max((_bbox_iou_tuple(candidate, positive) for positive in positive_tuples), default=0.0)
        if max_positive_iou > min_iou:
            fallback_candidates.append((max_positive_iou, candidate))
            continue
        sampled.append(candidate)

    if len(sampled) < count:
        fallback_candidates.sort(key=lambda item: item[0])
        for _max_iou, candidate in fallback_candidates:
            if len(sampled) >= count:
                break
            if any(_bbox_iou_tuple(candidate, existing) > 0.5 for existing in sampled):
                continue
            sampled.append(candidate)

    return sampled


def _hard_objectness_negative_candidates(
    image: Image.Image,
    positive_tuples: list[tuple[int, int, int, int]],
    count: int,
    min_iou: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    width, height = image.size
    if count <= 0 or width <= 1 or height <= 1:
        return []
    base_width, base_height = _median_box_size(
        [{"bbox": _bbox_payload(box, width, height)} for box in positive_tuples]
    )
    proposals = color_proposals(
        image,
        min_area=max(16, round(base_width * base_height * 0.20)),
        max_area_ratio=0.70,
        expand=1.05,
        min_box_size=max(4, min(base_width, base_height) // 3),
        max_box_size=0,
        max_aspect_ratio=12.0,
    )
    candidates = []
    for proposal in proposals:
        bbox = _bbox_tuple(proposal["bbox"])
        max_positive_iou = max((_bbox_iou_tuple(bbox, positive) for positive in positive_tuples), default=0.0)
        if max_positive_iou > min_iou:
            continue
        score = float(proposal.get("objectness", 0.0))
        candidates.append((bbox, score))

    work_scale = min(1.0, DEFAULT_OBJECTNESS_WORK_SIZE / max(width, height, 1))
    if work_scale < 1.0:
        work_width = max(16, round(width * work_scale))
        work_height = max(16, round(height * work_scale))
        work_image = image.resize((work_width, work_height))
    else:
        work_width, work_height = width, height
        work_image = image
    objectness, objectness_width, _height = _objectness_maps(work_image)
    scales = [0.75, 1.0, 1.35, 1.75]
    for scale in scales:
        box_width = max(4, min(work_width, round(base_width * scale * work_scale)))
        box_height = max(4, min(work_height, round(base_height * scale * work_scale)))
        stride_x = max(4, box_width // 2)
        stride_y = max(4, box_height // 2)
        for y in range(0, max(1, work_height - box_height + 1), stride_y):
            for x in range(0, max(1, work_width - box_width + 1), stride_x):
                work_bbox = (x, y, x + box_width, y + box_height)
                if work_scale < 1.0:
                    bbox = _clamp_bbox(
                        (
                            work_bbox[0] / work_scale,
                            work_bbox[1] / work_scale,
                            work_bbox[2] / work_scale,
                            work_bbox[3] / work_scale,
                        ),
                        width,
                        height,
                    )
                else:
                    bbox = work_bbox
                if bbox is None:
                    continue
                max_positive_iou = max((_bbox_iou_tuple(bbox, positive) for positive in positive_tuples), default=0.0)
                if max_positive_iou > min_iou:
                    continue
                candidates.append((bbox, _mean_in_bbox(objectness, objectness_width, work_bbox)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    deduped = []
    for bbox, score in candidates:
        if any(_bbox_iou_tuple(bbox, existing) > 0.55 for existing, _score in deduped):
            continue
        deduped.append((bbox, score))
        if len(deduped) >= count:
            break
    return deduped


__all__ = [
    "_median_box_size",
    "_sample_negative_boxes",
    "_hard_objectness_negative_candidates",
]
