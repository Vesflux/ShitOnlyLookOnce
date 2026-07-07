from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _as_prediction_map(prediction: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(prediction, dict):
        return prediction
    return {"p8": prediction}


def _target_scales(predictions: dict[str, torch.Tensor], boxes_batch: list[torch.Tensor]) -> list[float]:
    image_size = 1.0
    for boxes in boxes_batch:
        if boxes.numel():
            image_size = max(1.0, float(predictions[next(iter(predictions))].shape[-1]))
            break
    return [float(grid_size) for grid_size in [item.shape[-1] for item in predictions.values()]]


def _label_tensor(labels: torch.Tensor | list[str], device: torch.device) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        return labels.to(device=device, dtype=torch.long)
    values: list[int] = []
    for label in labels:
        try:
            value = int(float(label))
        except ValueError:
            value = 0
        values.append(max(0, value))
    return torch.tensor(values, dtype=torch.long, device=device)


def _class_balanced_weights(
    labels_batch: list[torch.Tensor] | list[list[str]] | None,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if num_classes <= 1 or not labels_batch:
        return None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for labels in labels_batch:
        label_values = _label_tensor(labels, device)
        if label_values.numel() == 0:
            continue
        label_values = label_values.clamp(0, num_classes - 1)
        counts += torch.bincount(label_values, minlength=num_classes).float()
    if not bool((counts > 0).any()):
        return None
    present = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    weights[present] = counts[present].sum() / (counts[present] * float(present.sum()))
    return weights.clamp(0.25, 12.0)


def _scale_stride(scale_name: str | None, grid_size: int, max_grid_size: int | None = None) -> int:
    if scale_name and scale_name.startswith("p"):
        try:
            return max(1, int(scale_name[1:]))
        except ValueError:
            pass
    if max_grid_size and grid_size > 0:
        return max(1, int(round(float(max_grid_size) / float(grid_size) * 4.0)))
    return 8


def _scale_size_range(scale_name: str | None) -> tuple[float, float]:
    if scale_name == "p4":
        return 0.0, 96.0
    if scale_name == "p8":
        return 32.0, 192.0
    if scale_name == "p16":
        return 96.0, 384.0
    if scale_name == "p32":
        return 192.0, 1e9
    return 0.0, 1e9


def _range_affinity(size_pixels: float, lower: float, upper: float) -> float:
    if lower <= size_pixels <= upper:
        return 1.0
    if size_pixels < lower:
        return max(0.0, 1.0 - (lower - size_pixels) / max(lower, 1.0))
    if upper >= 1e8:
        return 1.0
    return max(0.0, 1.0 - (size_pixels - upper) / max(upper, 1.0))


def _centerness_from_ltrb(distances: torch.Tensor) -> torch.Tensor:
    left, top, right, bottom = distances.unbind(dim=1)
    lr = torch.minimum(left, right) / torch.maximum(left, right).clamp(min=1e-6)
    tb = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp(min=1e-6)
    return torch.sqrt((lr * tb).clamp(min=0.0, max=1.0))


def _pairwise_iou_aligned(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted.unbind(dim=1)
    target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=1)
    inter_x1 = torch.maximum(pred_x1, target_x1)
    inter_y1 = torch.maximum(pred_y1, target_y1)
    inter_x2 = torch.minimum(pred_x2, target_x2)
    inter_y2 = torch.minimum(pred_y2, target_y2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
    union = (pred_area + target_area - inter).clamp(min=1e-7)
    return (inter / union).clamp(min=0.0, max=1.0)


def normalized_wasserstein_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_size: float = 640.0,
    constant: float = 12.8,
) -> torch.Tensor:
    pred_cx = (predicted[:, 0] + predicted[:, 2]) * 0.5 * image_size
    pred_cy = (predicted[:, 1] + predicted[:, 3]) * 0.5 * image_size
    target_cx = (target[:, 0] + target[:, 2]) * 0.5 * image_size
    target_cy = (target[:, 1] + target[:, 3]) * 0.5 * image_size
    pred_w = (predicted[:, 2] - predicted[:, 0]).clamp(min=1e-6) * image_size
    pred_h = (predicted[:, 3] - predicted[:, 1]).clamp(min=1e-6) * image_size
    target_w = (target[:, 2] - target[:, 0]).clamp(min=1e-6) * image_size
    target_h = (target[:, 3] - target[:, 1]).clamp(min=1e-6) * image_size
    center_distance = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
    shape_distance = ((pred_w - target_w).pow(2) + (pred_h - target_h).pow(2)) * 0.25
    distance = torch.sqrt((center_distance + shape_distance).clamp(min=1e-9))
    similarity = torch.exp(-distance / max(1e-6, constant))
    return 1.0 - similarity.clamp(min=0.0, max=1.0)


def build_quality_fpn_targets(
    prediction_map: dict[str, torch.Tensor],
    boxes_batch: list[torch.Tensor],
    labels_batch: list[torch.Tensor] | list[list[str]] | None,
    *,
    num_classes: int,
    heat_radius: int = 1,
) -> dict[str, dict[str, torch.Tensor]]:
    first_prediction = next(iter(prediction_map.values()))
    batch_size = len(boxes_batch)
    device = first_prediction.device
    max_grid_size = max(int(item.shape[-1]) for item in prediction_map.values())
    level_meta: dict[str, tuple[int, int, float, float]] = {}
    for scale_name, prediction in prediction_map.items():
        grid_size = int(prediction.shape[-1])
        stride = _scale_stride(scale_name, grid_size, max_grid_size=max_grid_size)
        lower, upper = _scale_size_range(scale_name)
        level_meta[scale_name] = (grid_size, stride, lower, upper)

    targets: dict[str, dict[str, torch.Tensor]] = {}
    for scale_name, prediction in prediction_map.items():
        grid_size = int(prediction.shape[-1])
        targets[scale_name] = {
            "heatmap": torch.zeros(
                (batch_size, num_classes, grid_size, grid_size),
                dtype=torch.float32,
                device=device,
            ),
            "box": torch.zeros((batch_size, 4, grid_size, grid_size), dtype=torch.float32, device=device),
            "offset": torch.zeros((batch_size, 2, grid_size, grid_size), dtype=torch.float32, device=device),
            "positive": torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.bool, device=device),
            "quality": torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.float32, device=device),
            "small_object": torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.bool, device=device),
            "class_target": torch.zeros((batch_size, grid_size, grid_size), dtype=torch.long, device=device),
            "assigned_score": torch.full(
                (batch_size, grid_size, grid_size),
                -1.0,
                dtype=torch.float32,
                device=device,
            ),
        }

    for batch_index, boxes in enumerate(boxes_batch):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        if labels_batch is not None:
            labels = _label_tensor(labels_batch[batch_index], device)
            if labels.numel() < boxes.shape[0]:
                labels = F.pad(labels, (0, boxes.shape[0] - labels.numel()))
            labels = labels[: boxes.shape[0]].clamp(0, max(0, num_classes - 1))
        else:
            labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)

        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-5, max=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-5, max=1.0)
        centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5).clamp(0.0, 1.0)
        centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5).clamp(0.0, 1.0)

        for box_index in range(boxes.shape[0]):
            label_id = int(labels[box_index].item()) if labels.numel() else 0
            x1, y1, x2, y2 = [float(value) for value in boxes[box_index].tolist()]
            width = float(widths[box_index])
            height = float(heights[box_index])
            cx = float(centers_x[box_index])
            cy = float(centers_y[box_index])
            input_size_guess = max(float(grid_size * stride) for grid_size, stride, _lower, _upper in level_meta.values())
            size_pixels = max(width, height) * input_size_guess

            eligible_levels: list[str] = []
            level_affinities: dict[str, float] = {}
            for scale_name, (_grid_size, _stride, lower, upper) in level_meta.items():
                affinity = _range_affinity(size_pixels, lower, upper)
                level_affinities[scale_name] = affinity
                if affinity > 0.0:
                    eligible_levels.append(scale_name)
            if not eligible_levels:
                eligible_levels = [
                    min(
                        level_meta,
                        key=lambda name: abs(size_pixels - (level_meta[name][2] + min(level_meta[name][3], 512.0)) * 0.5),
                    )
                ]

            for scale_name in eligible_levels:
                grid_size, stride, _lower, _upper = level_meta[scale_name]
                target = targets[scale_name]
                gx = cx * grid_size - 0.5
                gy = cy * grid_size - 0.5
                center_x_index = max(0, min(grid_size - 1, int(round(gx))))
                center_y_index = max(0, min(grid_size - 1, int(round(gy))))
                base_radius = int(round(min(width, height) * grid_size * 0.55))
                radius = max(heat_radius, min(5, base_radius))
                sigma = max(0.65, radius / 2.0)
                for yy in range(max(0, center_y_index - radius), min(grid_size, center_y_index + radius + 1)):
                    for xx in range(max(0, center_x_index - radius), min(grid_size, center_x_index + radius + 1)):
                        distance = math.sqrt((xx - gx) * (xx - gx) + (yy - gy) * (yy - gy))
                        heat = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
                        target["heatmap"][batch_index, label_id, yy, xx] = torch.maximum(
                            target["heatmap"][batch_index, label_id, yy, xx],
                            torch.tensor(heat, dtype=torch.float32, device=device),
                        )

                candidate_radius = int(round(min(width, height) * grid_size * 0.35))
                if size_pixels >= 28.0:
                    candidate_radius = max(candidate_radius, 1)
                if size_pixels >= 128.0:
                    candidate_radius = max(candidate_radius, 2)
                candidate_radius = max(0, min(5, candidate_radius))
                candidates: list[tuple[float, int, int, tuple[float, float, float, float]]] = []
                for yy in range(
                    max(0, center_y_index - candidate_radius),
                    min(grid_size, center_y_index + candidate_radius + 1),
                ):
                    center_y = (yy + 0.5) / grid_size
                    for xx in range(
                        max(0, center_x_index - candidate_radius),
                        min(grid_size, center_x_index + candidate_radius + 1),
                    ):
                        center_x = (xx + 0.5) / grid_size
                        left = center_x - x1
                        top = center_y - y1
                        right = x2 - center_x
                        bottom = y2 - center_y
                        if min(left, top, right, bottom) <= 0.0:
                            continue
                        norm_dx = abs(center_x - cx) / max(width, 1e-6)
                        norm_dy = abs(center_y - cy) / max(height, 1e-6)
                        normalized_distance = math.sqrt(norm_dx * norm_dx + norm_dy * norm_dy)
                        if normalized_distance > 0.72:
                            continue
                        candidates.append(
                            (
                                normalized_distance,
                                yy,
                                xx,
                                (
                                    left * grid_size,
                                    top * grid_size,
                                    right * grid_size,
                                    bottom * grid_size,
                                ),
                            )
                        )
                if not candidates:
                    continue
                if size_pixels < 32.0:
                    top_k = 1
                elif size_pixels < 128.0:
                    top_k = 3
                elif size_pixels < 320.0:
                    top_k = 5
                else:
                    top_k = 7
                candidates = sorted(candidates, key=lambda item: item[0])[:top_k]
                affinity = level_affinities.get(scale_name, 1.0)
                for normalized_distance, yy, xx, distances in candidates:
                    distance_tensor = torch.tensor(distances, dtype=torch.float32, device=device)
                    quality = float(_centerness_from_ltrb(distance_tensor.view(1, 4))[0].item())
                    assignment_score = quality * 1.4 + affinity * 0.4 - normalized_distance * 0.25
                    if assignment_score < float(target["assigned_score"][batch_index, yy, xx]):
                        continue
                    target["assigned_score"][batch_index, yy, xx] = assignment_score
                    target["box"][batch_index, :, yy, xx] = distance_tensor
                    target["offset"][batch_index, :, yy, xx] = torch.tensor(
                        [cx * grid_size - (xx + 0.5), cy * grid_size - (yy + 0.5)],
                        dtype=torch.float32,
                        device=device,
                    ).clamp(-0.5, 0.5)
                    target["quality"][batch_index, 0, yy, xx] = max(0.05, quality)
                    target["small_object"][batch_index, 0, yy, xx] = size_pixels <= 64.0
                    target["class_target"][batch_index, yy, xx] = label_id
                    target["positive"][batch_index, 0, yy, xx] = True
                    target["heatmap"][batch_index, label_id, yy, xx] = torch.maximum(
                        target["heatmap"][batch_index, label_id, yy, xx],
                        torch.tensor(max(0.85, quality), dtype=torch.float32, device=device),
                    )

    for target in targets.values():
        target.pop("assigned_score", None)
    return targets


def build_detection_targets(
    boxes_batch: list[torch.Tensor],
    grid_size: int,
    device: torch.device,
    heat_radius: int = 1,
    labels_batch: list[torch.Tensor] | list[list[str]] | None = None,
    num_classes: int = 1,
    scale_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(boxes_batch)
    objectness = torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.float32, device=device)
    targets = torch.zeros((batch_size, 4, grid_size, grid_size), dtype=torch.float32, device=device)
    positive = torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.bool, device=device)
    centerness = torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.float32, device=device)
    class_target = torch.zeros((batch_size, grid_size, grid_size), dtype=torch.long, device=device)
    assigned_score = torch.full((batch_size, grid_size, grid_size), -1.0, dtype=torch.float32, device=device)

    for batch_index, boxes in enumerate(boxes_batch):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        labels = None
        if labels_batch is not None:
            labels = _label_tensor(labels_batch[batch_index], device)
            if labels.numel() < boxes.shape[0]:
                labels = F.pad(labels, (0, boxes.shape[0] - labels.numel()))
            labels = labels[: boxes.shape[0]].clamp(0, max(0, num_classes - 1))
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-5, max=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-5, max=1.0)
        centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5).clamp(0.0, 1.0 - 1e-6)
        centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5).clamp(0.0, 1.0 - 1e-6)
        for box_index in range(boxes.shape[0]):
            cx = float(centers_x[box_index])
            cy = float(centers_y[box_index])
            width = float(widths[box_index])
            height = float(heights[box_index])
            area = width * height
            box_grid_size = max(width, height) * grid_size
            if scale_name == "p4" and box_grid_size > 9.5:
                continue
            if scale_name == "p8" and box_grid_size < 1.25:
                continue
            grid_x_float = cx * grid_size
            grid_y_float = cy * grid_size
            grid_x = max(0, min(grid_size - 1, int(grid_x_float)))
            grid_y = max(0, min(grid_size - 1, int(grid_y_float)))
            base_radius = int(round(min(width, height) * grid_size * 0.55))
            radius = max(heat_radius, min(3, base_radius))
            sigma = max(0.55, radius / 2)
            for yy in range(max(0, grid_y - radius), min(grid_size, grid_y + radius + 1)):
                for xx in range(max(0, grid_x - radius), min(grid_size, grid_x + radius + 1)):
                    distance = math.sqrt((xx + 0.5 - grid_x_float) ** 2 + (yy + 0.5 - grid_y_float) ** 2)
                    heat = 0.28 * math.exp(-(distance * distance) / (2.0 * sigma * sigma))
                    objectness[batch_index, 0, yy, xx] = max(objectness[batch_index, 0, yy, xx], heat)

            assign_radius = 1 if max(width, height) * grid_size >= 4.0 else 0
            for yy in range(max(0, grid_y - assign_radius), min(grid_size, grid_y + assign_radius + 1)):
                for xx in range(max(0, grid_x - assign_radius), min(grid_size, grid_x + assign_radius + 1)):
                    offset_x = grid_x_float - xx
                    offset_y = grid_y_float - yy
                    if offset_x < 0.0 or offset_x >= 1.0 or offset_y < 0.0 or offset_y >= 1.0:
                        if assign_radius == 0:
                            continue
                        offset_x = max(0.0, min(0.999, offset_x))
                        offset_y = max(0.0, min(0.999, offset_y))
                    center_distance = math.sqrt((xx + 0.5 - grid_x_float) ** 2 + (yy + 0.5 - grid_y_float) ** 2)
                    center_score = float(math.exp(-(center_distance * center_distance) / 0.72))
                    if assign_radius > 0 and center_score < 0.20:
                        continue
                    assignment_score = center_score + min(area * 128.0, 0.5)
                    objectness[batch_index, 0, yy, xx] = max(objectness[batch_index, 0, yy, xx], center_score)
                    if assignment_score >= float(assigned_score[batch_index, yy, xx]):
                        assigned_score[batch_index, yy, xx] = assignment_score
                        targets[batch_index, :, yy, xx] = torch.tensor(
                            [offset_x, offset_y, width, height],
                            dtype=torch.float32,
                            device=device,
                        )
                        if labels is not None and labels.numel():
                            class_target[batch_index, yy, xx] = labels[box_index]
                        centerness[batch_index, 0, yy, xx] = max(0.15, center_score)
                        positive[batch_index, 0, yy, xx] = True

    return objectness, targets, positive, centerness, class_target


def build_detection_targets_class_heatmap(
    boxes_batch: list[torch.Tensor],
    grid_size: int,
    device: torch.device,
    labels_batch: list[torch.Tensor] | list[list[str]],
    num_classes: int,
    heat_radius: int = 1,
    scale_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(boxes_batch)
    heatmap = torch.zeros((batch_size, num_classes, grid_size, grid_size), dtype=torch.float32, device=device)
    targets = torch.zeros((batch_size, 4, grid_size, grid_size), dtype=torch.float32, device=device)
    positive = torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.bool, device=device)
    centerness = torch.zeros((batch_size, 1, grid_size, grid_size), dtype=torch.float32, device=device)
    class_target = torch.zeros((batch_size, grid_size, grid_size), dtype=torch.long, device=device)
    assigned_score = torch.full((batch_size, grid_size, grid_size), -1.0, dtype=torch.float32, device=device)

    for batch_index, boxes in enumerate(boxes_batch):
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        labels = _label_tensor(labels_batch[batch_index], device)
        if labels.numel() < boxes.shape[0]:
            labels = F.pad(labels, (0, boxes.shape[0] - labels.numel()))
        labels = labels[: boxes.shape[0]].clamp(0, max(0, num_classes - 1))
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-5, max=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-5, max=1.0)
        centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5).clamp(0.0, 1.0 - 1e-6)
        centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5).clamp(0.0, 1.0 - 1e-6)
        for box_index in range(boxes.shape[0]):
            label_id = int(labels[box_index].item())
            cx = float(centers_x[box_index])
            cy = float(centers_y[box_index])
            width = float(widths[box_index])
            height = float(heights[box_index])
            area = width * height
            box_grid_size = max(width, height) * grid_size
            if scale_name == "p4" and box_grid_size > 9.5:
                continue
            if scale_name == "p8" and box_grid_size < 1.25:
                continue
            grid_x_float = cx * grid_size
            grid_y_float = cy * grid_size
            grid_x = max(0, min(grid_size - 1, int(grid_x_float)))
            grid_y = max(0, min(grid_size - 1, int(grid_y_float)))
            base_radius = int(round(min(width, height) * grid_size * 0.65))
            radius = max(heat_radius, min(4 if label_id > 0 else 3, base_radius))
            sigma = max(0.55, radius / 2)
            for yy in range(max(0, grid_y - radius), min(grid_size, grid_y + radius + 1)):
                for xx in range(max(0, grid_x - radius), min(grid_size, grid_x + radius + 1)):
                    distance = math.sqrt((xx + 0.5 - grid_x_float) ** 2 + (yy + 0.5 - grid_y_float) ** 2)
                    heat = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
                    heatmap[batch_index, label_id, yy, xx] = max(heatmap[batch_index, label_id, yy, xx], heat)

            assign_radius = 1 if max(width, height) * grid_size >= 4.0 else 0
            for yy in range(max(0, grid_y - assign_radius), min(grid_size, grid_y + assign_radius + 1)):
                for xx in range(max(0, grid_x - assign_radius), min(grid_size, grid_x + assign_radius + 1)):
                    offset_x = max(0.0, min(0.999, grid_x_float - xx))
                    offset_y = max(0.0, min(0.999, grid_y_float - yy))
                    center_distance = math.sqrt((xx + 0.5 - grid_x_float) ** 2 + (yy + 0.5 - grid_y_float) ** 2)
                    center_score = float(math.exp(-(center_distance * center_distance) / 0.72))
                    if assign_radius > 0 and center_score < 0.20:
                        continue
                    assignment_score = center_score + min(area * 128.0, 0.5)
                    if assignment_score >= float(assigned_score[batch_index, yy, xx]):
                        assigned_score[batch_index, yy, xx] = assignment_score
                        targets[batch_index, :, yy, xx] = torch.tensor(
                            [offset_x, offset_y, width, height],
                            dtype=torch.float32,
                            device=device,
                        )
                        class_target[batch_index, yy, xx] = label_id
                        centerness[batch_index, 0, yy, xx] = max(0.15, center_score)
                        positive[batch_index, 0, yy, xx] = True

    return heatmap, targets, positive, centerness, class_target


def generalized_iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted.unbind(dim=1)
    target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=1)
    inter_x1 = torch.maximum(pred_x1, target_x1)
    inter_y1 = torch.maximum(pred_y1, target_y1)
    inter_x2 = torch.minimum(pred_x2, target_x2)
    inter_y2 = torch.minimum(pred_y2, target_y2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=1e-7)

    cover_x1 = torch.minimum(pred_x1, target_x1)
    cover_y1 = torch.minimum(pred_y1, target_y1)
    cover_x2 = torch.maximum(pred_x2, target_x2)
    cover_y2 = torch.maximum(pred_y2, target_y2)
    cover_area = (cover_x2 - cover_x1).clamp(min=0) * (cover_y2 - cover_y1).clamp(min=0)
    giou = iou - (cover_area - union) / cover_area.clamp(min=1e-7)
    return 1.0 - giou.clamp(min=-1.0, max=1.0)


def ciou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted.unbind(dim=1)
    target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=1)
    iou = _pairwise_iou_aligned(predicted, target)

    pred_cx = (pred_x1 + pred_x2) * 0.5
    pred_cy = (pred_y1 + pred_y2) * 0.5
    target_cx = (target_x1 + target_x2) * 0.5
    target_cy = (target_y1 + target_y2) * 0.5
    center_distance = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)

    cover_x1 = torch.minimum(pred_x1, target_x1)
    cover_y1 = torch.minimum(pred_y1, target_y1)
    cover_x2 = torch.maximum(pred_x2, target_x2)
    cover_y2 = torch.maximum(pred_y2, target_y2)
    diagonal = (cover_x2 - cover_x1).pow(2) + (cover_y2 - cover_y1).pow(2)

    pred_w = (pred_x2 - pred_x1).clamp(min=1e-6)
    pred_h = (pred_y2 - pred_y1).clamp(min=1e-6)
    target_w = (target_x2 - target_x1).clamp(min=1e-6)
    target_h = (target_y2 - target_y1).clamp(min=1e-6)
    aspect_penalty = (4.0 / (math.pi * math.pi)) * (
        torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)
    ).pow(2)
    with torch.no_grad():
        alpha = aspect_penalty / (1.0 - iou + aspect_penalty).clamp(min=1e-6)
    ciou = iou - center_distance / diagonal.clamp(min=1e-7) - alpha * aspect_penalty
    return 1.0 - ciou.clamp(min=-1.0, max=1.0)


def mpdiou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = predicted.unbind(dim=1)
    target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=1)
    iou = _pairwise_iou_aligned(predicted, target)
    x1_distance = (pred_x1 - target_x1).pow(2) + (pred_y1 - target_y1).pow(2)
    x2_distance = (pred_x2 - target_x2).pow(2) + (pred_y2 - target_y2).pow(2)
    enclosing_w = (torch.maximum(pred_x2, target_x2) - torch.minimum(pred_x1, target_x1)).clamp(min=1e-6)
    enclosing_h = (torch.maximum(pred_y2, target_y2) - torch.minimum(pred_y1, target_y1)).clamp(min=1e-6)
    normalizer = enclosing_w.pow(2) + enclosing_h.pow(2)
    mpdiou = iou - (x1_distance + x2_distance) / normalizer.clamp(min=1e-7)
    return 1.0 - mpdiou.clamp(min=-1.0, max=1.0)


def _decode_positive_boxes(
    raw_boxes: torch.Tensor,
    indexes: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    offset = torch.sigmoid(raw_boxes[:, :2])
    size = torch.sigmoid(raw_boxes[:, 2:]).clamp(min=1e-4, max=1.0)
    grid_y = indexes[:, 1].float()
    grid_x = indexes[:, 2].float()
    center_x = (grid_x + offset[:, 0]) / grid_size
    center_y = (grid_y + offset[:, 1]) / grid_size
    width = size[:, 0]
    height = size[:, 1]
    return torch.stack(
        [
            (center_x - width / 2).clamp(0.0, 1.0),
            (center_y - height / 2).clamp(0.0, 1.0),
            (center_x + width / 2).clamp(0.0, 1.0),
            (center_y + height / 2).clamp(0.0, 1.0),
        ],
        dim=1,
    )


def _decode_ltrb_positive_boxes(
    distances: torch.Tensor,
    indexes: torch.Tensor,
    grid_size: int,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    grid_y = indexes[:, 1].float()
    grid_x = indexes[:, 2].float()
    if offsets is None:
        offset_x = torch.zeros_like(grid_x)
        offset_y = torch.zeros_like(grid_y)
    else:
        offset_x = offsets[:, 0].clamp(-0.5, 0.5)
        offset_y = offsets[:, 1].clamp(-0.5, 0.5)
    center_x = (grid_x + 0.5 + offset_x) / grid_size
    center_y = (grid_y + 0.5 + offset_y) / grid_size
    left = distances[:, 0] / grid_size
    top = distances[:, 1] / grid_size
    right = distances[:, 2] / grid_size
    bottom = distances[:, 3] / grid_size
    return torch.stack(
        [
            (center_x - left).clamp(0.0, 1.0),
            (center_y - top).clamp(0.0, 1.0),
            (center_x + right).clamp(0.0, 1.0),
            (center_y + bottom).clamp(0.0, 1.0),
        ],
        dim=1,
    )


def _loss_for_quality_fpn_scale(
    prediction: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    box_weight: float,
    iou_weight: float,
    centerness_weight: float,
    num_classes: int,
    class_weights: torch.Tensor | None,
    task_aligned: bool = False,
    advanced_box_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    grid_size = prediction.shape[-1]
    heatmap_target = target["heatmap"]
    box_target = target["box"]
    positive = target["positive"]
    quality_target = target["quality"]
    offset_target = target.get("offset")
    small_object = target.get("small_object")
    heatmap_logits = prediction[:, :num_classes]
    has_offset = prediction.shape[1] >= num_classes + 7
    offset_start = num_classes
    box_start = num_classes + 2 if has_offset else num_classes
    raw_offsets_map = prediction[:, offset_start : offset_start + 2] if has_offset else None
    raw_distances_map = prediction[:, box_start : box_start + 4]
    quality_logits = prediction[:, box_start + 4 : box_start + 5]

    object_loss = _heatmap_focal_loss(heatmap_logits, heatmap_target, class_weights)
    positive_count = int(positive.sum().item())
    if positive_count:
        indexes = torch.nonzero(positive[:, 0], as_tuple=False)
        raw_distances = raw_distances_map.permute(0, 2, 3, 1)[positive[:, 0]]
        target_distances = box_target.permute(0, 2, 3, 1)[positive[:, 0]].clamp(min=1e-4)
        predicted_distances = F.softplus(raw_distances).clamp(min=1e-4, max=float(grid_size) * 2.0)
        if has_offset and raw_offsets_map is not None and offset_target is not None:
            raw_offsets = raw_offsets_map.permute(0, 2, 3, 1)[positive[:, 0]]
            predicted_offsets = torch.tanh(raw_offsets) * 0.5
            target_offsets = offset_target.permute(0, 2, 3, 1)[positive[:, 0]]
            offset_loss = F.smooth_l1_loss(predicted_offsets, target_offsets, reduction="mean", beta=0.05)
        else:
            predicted_offsets = None
            target_offsets = None
            offset_loss = raw_distances_map.sum() * 0.0
        smooth_loss = F.smooth_l1_loss(
            torch.log1p(predicted_distances),
            torch.log1p(target_distances),
            reduction="mean",
            beta=0.15,
        )
        predicted_boxes = _decode_ltrb_positive_boxes(predicted_distances, indexes, grid_size, predicted_offsets)
        target_boxes = _decode_ltrb_positive_boxes(target_distances, indexes, grid_size, target_offsets)
        giou_values = generalized_iou_loss(predicted_boxes, target_boxes)
        giou_loss = giou_values.mean()
        if advanced_box_loss:
            ciou_values = ciou_loss(predicted_boxes, target_boxes)
            mpdiou_values = mpdiou_loss(predicted_boxes, target_boxes)
            advanced_geometry_loss = ciou_values.mean() * 0.55 + mpdiou_values.mean() * 0.45
        else:
            advanced_geometry_loss = raw_distances_map.sum() * 0.0
        if small_object is not None and bool(small_object[positive].any()):
            small_mask = small_object[positive].view(-1)
            nwd_loss = normalized_wasserstein_loss(predicted_boxes[small_mask], target_boxes[small_mask]).mean()
        else:
            nwd_loss = raw_distances_map.sum() * 0.0
        with torch.no_grad():
            predicted_iou = _pairwise_iou_aligned(predicted_boxes, target_boxes)
            center_quality = quality_target[positive].view(-1)
            quality_values = center_quality.clamp(0.05, 1.0)
            if task_aligned:
                class_target = target.get("class_target")
                localization_quality = (0.35 + 0.65 * predicted_iou.clamp(0.0, 1.0)).clamp(0.05, 1.0)
                quality_values = (center_quality * localization_quality).clamp(0.02, 1.0)
                if class_target is not None:
                    class_values = class_target[positive[:, 0]]
                    positive_logits = heatmap_logits.permute(0, 2, 3, 1)[positive[:, 0]]
                    class_probability = torch.sigmoid(
                        positive_logits[torch.arange(positive_logits.shape[0], device=positive_logits.device), class_values]
                    )
                    alignment_target = (
                        class_probability.clamp(1e-4, 1.0).pow(0.5)
                        * predicted_iou.clamp(0.0, 1.0).pow(1.5)
                    ).clamp(0.0, 1.0)
                    quality_values = torch.maximum(quality_values, alignment_target * center_quality).clamp(0.02, 1.0)
        centerness_loss = F.binary_cross_entropy_with_logits(
            quality_logits[positive],
            quality_values,
            reduction="mean",
        )
        if task_aligned:
            class_target = target.get("class_target")
            if class_target is not None:
                positive_logits = heatmap_logits.permute(0, 2, 3, 1)[positive[:, 0]]
                class_values = class_target[positive[:, 0]]
                selected_logits = positive_logits[
                    torch.arange(positive_logits.shape[0], device=positive_logits.device),
                    class_values,
                ]
                with torch.no_grad():
                    alignment_scores = (predicted_iou.clamp(0.0, 1.0).pow(1.5) * quality_values).clamp(0.0, 1.0)
                alignment_loss = F.binary_cross_entropy_with_logits(
                    selected_logits,
                    alignment_scores,
                    reduction="mean",
                )
            else:
                alignment_loss = raw_distances_map.sum() * 0.0
        else:
            alignment_loss = raw_distances_map.sum() * 0.0
    else:
        smooth_loss = raw_distances_map.sum() * 0.0
        giou_loss = raw_distances_map.sum() * 0.0
        offset_loss = raw_distances_map.sum() * 0.0
        nwd_loss = raw_distances_map.sum() * 0.0
        alignment_loss = raw_distances_map.sum() * 0.0
        advanced_geometry_loss = raw_distances_map.sum() * 0.0
        centerness_loss = quality_logits.sum() * 0.0

    regression_loss = smooth_loss + offset_loss * 0.55
    total = (
        object_loss
        + regression_loss * box_weight
        + giou_loss * iou_weight
        + advanced_geometry_loss * max(0.4, iou_weight * 0.45)
        + nwd_loss * max(0.2, iou_weight * 0.35)
        + alignment_loss * 0.45
        + centerness_loss * centerness_weight
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "box_loss": float(regression_loss.detach().cpu()),
        "giou_loss": float(giou_loss.detach().cpu()),
        "advanced_geometry_loss": float(advanced_geometry_loss.detach().cpu()),
        "nwd_loss": float(nwd_loss.detach().cpu()),
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "centerness_loss": float(centerness_loss.detach().cpu()),
        "class_loss": 0.0,
        "positives": positive_count,
    }
    return total, metrics


def _loss_for_scale(
    prediction: torch.Tensor,
    boxes_batch: list[torch.Tensor],
    labels_batch: list[torch.Tensor] | list[list[str]] | None,
    *,
    heat_radius: int,
    box_weight: float,
    iou_weight: float,
    class_weight: float,
    centerness_weight: float,
    num_classes: int,
    use_centerness: bool,
    scale_name: str | None,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = prediction.device
    grid_size = prediction.shape[-1]
    objectness_target, box_target, positive, centerness_target, class_target = build_detection_targets(
        boxes_batch,
        grid_size=grid_size,
        device=device,
        heat_radius=heat_radius,
        labels_batch=labels_batch,
        num_classes=num_classes,
        scale_name=scale_name,
    )
    logits = prediction[:, 0:1]
    object_bce = F.binary_cross_entropy_with_logits(logits, objectness_target, reduction="none")
    probability = torch.sigmoid(logits)
    center_mask = objectness_target >= 0.95
    soft_mask = (objectness_target > 0.02) & ~center_mask
    negative_mask = objectness_target <= 0.02
    losses: list[torch.Tensor] = []
    if center_mask.any():
        losses.append(object_bce[center_mask].mean())
    if soft_mask.any():
        losses.append(object_bce[soft_mask].mean() * 0.25)
    if negative_mask.any():
        losses.append((object_bce[negative_mask] * probability[negative_mask].pow(2.0)).mean() * 1.25)
    object_loss = sum(losses) if losses else object_bce.mean()

    positive_count = int(positive.sum().item())
    centerness_channel = 5 if use_centerness else None
    class_start = 6 if use_centerness else 5
    if positive_count:
        indexes = torch.nonzero(positive[:, 0], as_tuple=False)
        raw_boxes = prediction[:, 1:5].permute(0, 2, 3, 1)[positive[:, 0]]
        target_values = box_target.permute(0, 2, 3, 1)[positive[:, 0]]
        predicted_values = torch.cat([torch.sigmoid(raw_boxes[:, :2]), torch.sigmoid(raw_boxes[:, 2:])], dim=1)
        smooth_loss = F.smooth_l1_loss(predicted_values, target_values, reduction="mean", beta=0.04)

        predicted_boxes = _decode_positive_boxes(raw_boxes, indexes, grid_size)
        grid_y = indexes[:, 1].float()
        grid_x = indexes[:, 2].float()
        target_cx = (grid_x + target_values[:, 0]) / grid_size
        target_cy = (grid_y + target_values[:, 1]) / grid_size
        target_width = target_values[:, 2]
        target_height = target_values[:, 3]
        target_boxes = torch.stack(
            [
                (target_cx - target_width / 2).clamp(0.0, 1.0),
                (target_cy - target_height / 2).clamp(0.0, 1.0),
                (target_cx + target_width / 2).clamp(0.0, 1.0),
                (target_cy + target_height / 2).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        giou_loss = generalized_iou_loss(predicted_boxes, target_boxes).mean()
        if use_centerness and centerness_channel is not None and prediction.shape[1] > centerness_channel:
            centerness_logits = prediction[:, centerness_channel : centerness_channel + 1]
            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness_logits[positive],
                centerness_target[positive],
                reduction="mean",
            )
        else:
            centerness_loss = prediction[:, 0].sum() * 0.0
        if num_classes > 1 and prediction.shape[1] >= class_start + num_classes:
            class_logits = prediction[:, class_start : class_start + num_classes].permute(0, 2, 3, 1)[positive[:, 0]]
            class_values = class_target[positive[:, 0]]
            class_ce = F.cross_entropy(class_logits, class_values, weight=class_weights, reduction="none")
            class_pt = torch.exp(-class_ce.detach())
            class_loss = (class_ce * (1.0 - class_pt).pow(0.75)).mean()
        else:
            class_loss = prediction[:, 0].sum() * 0.0
    else:
        smooth_loss = prediction[:, 1:5].sum() * 0.0
        giou_loss = prediction[:, 1:5].sum() * 0.0
        centerness_loss = prediction[:, 0].sum() * 0.0
        class_loss = prediction[:, 0].sum() * 0.0

    total = (
        object_loss
        + smooth_loss * box_weight
        + giou_loss * iou_weight
        + centerness_loss * centerness_weight
        + class_loss * class_weight
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "box_loss": float(smooth_loss.detach().cpu()),
        "giou_loss": float(giou_loss.detach().cpu()),
        "centerness_loss": float(centerness_loss.detach().cpu()),
        "class_loss": float(class_loss.detach().cpu()),
        "positives": positive_count,
    }
    return total, metrics


def _heatmap_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    positive_mask = target >= 0.95
    negative_mask = target < 0.95
    positive_loss = bce * (1.0 - probability).pow(2.0) * positive_mask.float()
    negative_weight = (1.0 - target).pow(4.0)
    negative_loss = bce * probability.pow(2.0) * negative_weight * negative_mask.float()
    if class_weights is not None:
        view_shape = (1, class_weights.numel(), 1, 1)
        positive_loss = positive_loss * class_weights.view(view_shape)
    positive_count = positive_mask.float().sum().clamp(min=1.0)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def _loss_for_class_heatmap_scale(
    prediction: torch.Tensor,
    boxes_batch: list[torch.Tensor],
    labels_batch: list[torch.Tensor] | list[list[str]],
    *,
    heat_radius: int,
    box_weight: float,
    iou_weight: float,
    centerness_weight: float,
    num_classes: int,
    use_centerness: bool,
    class_box: bool,
    scale_name: str | None,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = prediction.device
    grid_size = prediction.shape[-1]
    heatmap_target, box_target, positive, centerness_target, _class_target = build_detection_targets_class_heatmap(
        boxes_batch,
        grid_size=grid_size,
        device=device,
        labels_batch=labels_batch,
        num_classes=num_classes,
        heat_radius=heat_radius,
        scale_name=scale_name,
    )
    heatmap_logits = prediction[:, :num_classes]
    object_loss = _heatmap_focal_loss(heatmap_logits, heatmap_target, class_weights)
    box_start = num_classes
    box_channels = num_classes * 4 if class_box else 4
    positive_count = int(positive.sum().item())
    if positive_count:
        indexes = torch.nonzero(positive[:, 0], as_tuple=False)
        if class_box:
            all_raw_boxes = prediction[:, box_start : box_start + box_channels].view(
                prediction.shape[0],
                num_classes,
                4,
                prediction.shape[-2],
                prediction.shape[-1],
            )
            class_values = _class_target[positive[:, 0]]
            raw_boxes = all_raw_boxes.permute(0, 3, 4, 1, 2)[positive[:, 0]]
            raw_boxes = raw_boxes[torch.arange(raw_boxes.shape[0], device=device), class_values]
        else:
            raw_boxes = prediction[:, box_start : box_start + 4].permute(0, 2, 3, 1)[positive[:, 0]]
        target_values = box_target.permute(0, 2, 3, 1)[positive[:, 0]]
        predicted_values = torch.cat([torch.sigmoid(raw_boxes[:, :2]), torch.sigmoid(raw_boxes[:, 2:])], dim=1)
        smooth_loss = F.smooth_l1_loss(predicted_values, target_values, reduction="mean", beta=0.04)

        predicted_boxes = _decode_positive_boxes(raw_boxes, indexes, grid_size)
        grid_y = indexes[:, 1].float()
        grid_x = indexes[:, 2].float()
        target_cx = (grid_x + target_values[:, 0]) / grid_size
        target_cy = (grid_y + target_values[:, 1]) / grid_size
        target_width = target_values[:, 2]
        target_height = target_values[:, 3]
        target_boxes = torch.stack(
            [
                (target_cx - target_width / 2).clamp(0.0, 1.0),
                (target_cy - target_height / 2).clamp(0.0, 1.0),
                (target_cx + target_width / 2).clamp(0.0, 1.0),
                (target_cy + target_height / 2).clamp(0.0, 1.0),
            ],
            dim=1,
        )
        giou_loss = generalized_iou_loss(predicted_boxes, target_boxes).mean()
        if use_centerness and prediction.shape[1] > box_start + box_channels:
            centerness_logits = prediction[:, box_start + box_channels : box_start + box_channels + 1]
            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness_logits[positive],
                centerness_target[positive],
                reduction="mean",
            )
        else:
            centerness_loss = prediction[:, 0].sum() * 0.0
    else:
        smooth_loss = prediction[:, box_start : box_start + box_channels].sum() * 0.0
        giou_loss = prediction[:, box_start : box_start + box_channels].sum() * 0.0
        centerness_loss = prediction[:, 0].sum() * 0.0

    total = object_loss + smooth_loss * box_weight + giou_loss * iou_weight + centerness_loss * centerness_weight
    metrics = {
        "loss": float(total.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "box_loss": float(smooth_loss.detach().cpu()),
        "giou_loss": float(giou_loss.detach().cpu()),
        "centerness_loss": float(centerness_loss.detach().cpu()),
        "class_loss": 0.0,
        "positives": positive_count,
    }
    return total, metrics


def detection_loss(
    prediction: torch.Tensor | dict[str, torch.Tensor],
    boxes_batch: list[torch.Tensor],
    labels_batch: list[torch.Tensor] | list[list[str]] | None = None,
    heat_radius: int = 1,
    box_weight: float = 5.0,
    iou_weight: float = 2.0,
    class_weight: float = 1.2,
    centerness_weight: float = 0.7,
    num_classes: int = 1,
    use_centerness: bool = False,
    class_weights: torch.Tensor | list[float] | None = None,
    class_heatmap: bool = False,
    class_box: bool = False,
    quality_fpn: bool = False,
    task_aligned: bool = False,
    advanced_box_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prediction_map = _as_prediction_map(prediction)
    first_prediction = next(iter(prediction_map.values()))
    device = first_prediction.device
    if class_weights is None:
        class_weights_tensor = _class_balanced_weights(labels_batch, num_classes, device)
    elif isinstance(class_weights, torch.Tensor):
        class_weights_tensor = class_weights.to(device=device, dtype=torch.float32)
    else:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    if class_weights_tensor is not None and class_weights_tensor.numel() != num_classes:
        class_weights_tensor = None
    total_loss = first_prediction.sum() * 0.0
    metrics_by_scale: list[dict[str, Any]] = []
    total_positives = 0

    if quality_fpn:
        quality_targets = build_quality_fpn_targets(
            prediction_map,
            boxes_batch,
            labels_batch,
            num_classes=max(1, num_classes),
            heat_radius=heat_radius,
        )
        for scale_name, scale_prediction in prediction_map.items():
            scale_weight = 1.0
            if scale_name == "p4":
                scale_weight = 1.20
            elif scale_name == "p8":
                scale_weight = 1.05
            elif scale_name == "p16":
                scale_weight = 0.95
            elif scale_name == "p32":
                scale_weight = 0.80
            scale_loss, scale_metrics = _loss_for_quality_fpn_scale(
                scale_prediction,
                quality_targets[scale_name],
                box_weight=box_weight,
                iou_weight=iou_weight,
                centerness_weight=centerness_weight,
                num_classes=max(1, num_classes),
                class_weights=class_weights_tensor,
                task_aligned=task_aligned,
                advanced_box_loss=advanced_box_loss,
            )
            total_loss = total_loss + scale_loss * scale_weight
            total_positives += int(scale_metrics.get("positives", 0))
            metrics_by_scale.append(scale_metrics)
        averaged = {
            key: float(sum(float(item.get(key, 0.0)) for item in metrics_by_scale) / len(metrics_by_scale))
            for key in (
                "object_loss",
                "box_loss",
                "giou_loss",
                "advanced_geometry_loss",
                "nwd_loss",
                "alignment_loss",
                "centerness_loss",
                "class_loss",
            )
        }
        return total_loss, {
            "loss": float(total_loss.detach().cpu()),
            **averaged,
            "positives": total_positives,
        }

    for scale_name, scale_prediction in prediction_map.items():
        scale_weight = 1.0
        if scale_name == "p4":
            scale_weight = 1.15
        elif scale_name == "p8":
            scale_weight = 0.9 if len(prediction_map) > 1 else 1.0
        if class_heatmap and num_classes > 1 and labels_batch is not None:
            scale_loss, scale_metrics = _loss_for_class_heatmap_scale(
                scale_prediction,
                boxes_batch,
                labels_batch,
                heat_radius=heat_radius,
                box_weight=box_weight,
                iou_weight=iou_weight,
                centerness_weight=centerness_weight,
                num_classes=num_classes,
                use_centerness=use_centerness,
                class_box=class_box,
                scale_name=scale_name if len(prediction_map) > 1 else None,
                class_weights=class_weights_tensor,
            )
        else:
            scale_loss, scale_metrics = _loss_for_scale(
                scale_prediction,
                boxes_batch,
                labels_batch,
                heat_radius=heat_radius,
                box_weight=box_weight,
                iou_weight=iou_weight,
                class_weight=class_weight,
                centerness_weight=centerness_weight,
                num_classes=num_classes,
                use_centerness=use_centerness,
                scale_name=scale_name if len(prediction_map) > 1 else None,
                class_weights=class_weights_tensor,
            )
        total_loss = total_loss + scale_loss * scale_weight
        total_positives += int(scale_metrics.get("positives", 0))
        metrics_by_scale.append(scale_metrics)

    averaged = {
        key: float(sum(float(item.get(key, 0.0)) for item in metrics_by_scale) / len(metrics_by_scale))
        for key in ("object_loss", "box_loss", "giou_loss", "centerness_loss", "class_loss")
    }
    metrics = {
        "loss": float(total_loss.detach().cpu()),
        **averaged,
        "positives": total_positives,
    }
    return total_loss, metrics


__all__ = [
    "build_detection_targets",
    "build_quality_fpn_targets",
    "ciou_loss",
    "detection_loss",
    "generalized_iou_loss",
    "mpdiou_loss",
    "normalized_wasserstein_loss",
]
