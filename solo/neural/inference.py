from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from solo.neural.data import letterbox_cv2, map_square_box_to_source, neural_image_paths, read_rgb_image
from solo.neural.model import count_parameters, create_detector_model
from solo.utils.bbox import (
    _bbox_payload,
    bbox_center_distance_ratio,
    bbox_containment,
    bbox_iou,
    filter_detections,
    nms,
    soft_nms,
)


def load_neural_checkpoint(
    weights_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(weights_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = create_detector_model(
        config.get("model", "tiny_context"),
        pretrained=False,
        num_classes=int(config.get("num_classes", 1)),
        use_centerness=bool(config.get("use_centerness", False)),
        class_heatmap=bool(config.get("class_heatmap", False)),
        class_box=bool(config.get("class_box", False)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    config = {
        **config,
        "weights_path": str(checkpoint_path),
        "parameter_count": count_parameters(model),
        "best_score": checkpoint.get("best_score"),
        "epoch": checkpoint.get("epoch"),
    }
    return model, config


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _label_for_class(class_id: int, class_names: list[str] | dict[int, str] | None, fallback: str) -> str:
    if class_names is None:
        return str(class_id) if fallback == "0" or int(class_id) != 0 else str(fallback)
    if isinstance(class_names, dict):
        return str(class_names.get(int(class_id), class_id))
    if 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    return str(class_id)


def _config_class_names(config: dict[str, Any]) -> list[str] | dict[int, str] | None:
    names = config.get("class_names")
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        resolved: dict[int, str] = {}
        for key, value in names.items():
            try:
                resolved[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return resolved
    return None


def _use_soft_nms_for_detections(config: dict[str, Any], detections: list[dict[str, Any]], image_width: int, image_height: int) -> bool:
    legacy_enabled = config.get("soft_nms")
    mode = str(config.get("soft_nms_mode", "on" if legacy_enabled is True else "auto")).lower().strip()
    if mode in {"off", "false", "0", "hard"}:
        return False
    if mode in {"on", "true", "1", "soft"}:
        return True
    if not bool(config.get("quality_fpn", False)) or len(detections) < 24:
        return False

    image_area = max(1.0, float(image_width * image_height))
    compact_count = 0
    for detection in detections:
        bbox = detection.get("bbox") or {}
        area = float(bbox.get("width", 0)) * float(bbox.get("height", 0))
        if area <= image_area * 0.018:
            compact_count += 1
    return compact_count >= 18 and compact_count / max(1, len(detections)) >= 0.55


def decode_neural_prediction(
    prediction: torch.Tensor | dict[str, torch.Tensor],
    score_threshold: float,
    max_candidates: int = 200,
    num_classes: int = 1,
    use_centerness: bool = False,
    class_heatmap: bool = False,
    class_box: bool = False,
    quality_fpn: bool = False,
    center_offset: bool = False,
    class_names: list[str] | dict[int, str] | None = None,
    fallback_label: str = "0",
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    prediction_map = prediction if isinstance(prediction, dict) else {"p8": prediction}
    for scale_name, scale_prediction in prediction_map.items():
        grid_size = scale_prediction.shape[-1]
        if quality_fpn:
            class_count = max(1, int(num_classes))
            heatmaps = torch.sigmoid(scale_prediction[0, :class_count])
            has_offset = center_offset and scale_prediction.shape[1] >= class_count + 7
            if has_offset:
                offsets_map = torch.tanh(scale_prediction[0, class_count : class_count + 2]) * 0.5
                box_start = class_count + 2
            else:
                offsets_map = None
                box_start = class_count
            raw_distances = scale_prediction[0, box_start : box_start + 4]
            quality = torch.sigmoid(scale_prediction[0, box_start + 4]).clamp(0.01, 1.0)
            decoded_per_level: list[dict[str, Any]] = []
            for class_id in range(class_count):
                score_map = heatmaps[class_id] * quality
                local_max = F.max_pool2d(score_map.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
                peaks = score_map >= local_max
                mask = (score_map >= score_threshold) & peaks
                if not bool(mask.any()):
                    continue
                scores = score_map[mask]
                indexes = torch.nonzero(mask, as_tuple=False)
                if max_candidates > 0 and scores.numel() > max_candidates:
                    scores, order = torch.topk(scores, max_candidates)
                    indexes = indexes[order]
                for score, index in zip(scores.detach().cpu(), indexes.detach().cpu()):
                    grid_y = int(index[0])
                    grid_x = int(index[1])
                    distances = F.softplus(raw_distances[:, grid_y, grid_x]).detach().cpu()
                    if offsets_map is None:
                        offset_x = 0.0
                        offset_y = 0.0
                    else:
                        offset_x = float(offsets_map[0, grid_y, grid_x].detach().cpu())
                        offset_y = float(offsets_map[1, grid_y, grid_x].detach().cpu())
                    center_x = (grid_x + 0.5 + offset_x) / grid_size
                    center_y = (grid_y + 0.5 + offset_y) / grid_size
                    x1 = max(0.0, center_x - float(distances[0]) / grid_size)
                    y1 = max(0.0, center_y - float(distances[1]) / grid_size)
                    x2 = min(1.0, center_x + float(distances[2]) / grid_size)
                    y2 = min(1.0, center_y + float(distances[3]) / grid_size)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    decoded_per_level.append(
                        {
                            "score": float(score),
                            "objectness": float(heatmaps[class_id, grid_y, grid_x].detach().cpu()),
                            "centerness": float(quality[grid_y, grid_x].detach().cpu()),
                            "class_score": float(heatmaps[class_id, grid_y, grid_x].detach().cpu()),
                            "quality": float(quality[grid_y, grid_x].detach().cpu()),
                            "center_offset": (offset_x, offset_y),
                            "class_id": int(class_id),
                            "label": _label_for_class(class_id, class_names, fallback_label),
                            "box": (x1, y1, x2, y2),
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "scale": scale_name,
                        }
                    )
            detections.extend(decoded_per_level)
            continue

        objectness = torch.sigmoid(scale_prediction[0, 0])
        if class_heatmap and num_classes > 1:
            heatmaps = torch.sigmoid(scale_prediction[0, :num_classes])
            box_start = num_classes
            box_channels = num_classes * 4 if class_box else 4
            raw_boxes = scale_prediction[0, box_start : box_start + box_channels]
            if use_centerness and scale_prediction.shape[1] > box_start + box_channels:
                centerness = torch.sigmoid(scale_prediction[0, box_start + box_channels]).clamp(0.05, 1.0)
            else:
                centerness = torch.ones_like(heatmaps[0])
            for class_id in range(num_classes):
                score_map = heatmaps[class_id] * centerness
                local_max = F.max_pool2d(score_map.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
                peaks = score_map >= local_max
                mask = (score_map >= score_threshold) & peaks
                if not bool(mask.any()):
                    continue
                scores = score_map[mask]
                indexes = torch.nonzero(mask, as_tuple=False)
                if max_candidates > 0 and scores.numel() > max_candidates:
                    scores, order = torch.topk(scores, max_candidates)
                    indexes = indexes[order]
                for score, index in zip(scores.detach().cpu(), indexes.detach().cpu()):
                    grid_y = int(index[0])
                    grid_x = int(index[1])
                    if class_box:
                        class_box_start = class_id * 4
                        values = raw_boxes[class_box_start : class_box_start + 4, grid_y, grid_x].detach().cpu()
                    else:
                        values = raw_boxes[:, grid_y, grid_x].detach().cpu()
                    offsets = torch.sigmoid(values[:2])
                    size = torch.sigmoid(values[2:]).clamp(1e-4, 1.0)
                    center_x = (grid_x + float(offsets[0])) / grid_size
                    center_y = (grid_y + float(offsets[1])) / grid_size
                    width = float(size[0])
                    height = float(size[1])
                    x1 = max(0.0, center_x - width / 2)
                    y1 = max(0.0, center_y - height / 2)
                    x2 = min(1.0, center_x + width / 2)
                    y2 = min(1.0, center_y + height / 2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append(
                        {
                            "score": float(score),
                            "objectness": float(heatmaps[class_id, grid_y, grid_x].detach().cpu()),
                            "centerness": float(centerness[grid_y, grid_x].detach().cpu()),
                            "class_score": float(heatmaps[class_id, grid_y, grid_x].detach().cpu()),
                            "class_id": int(class_id),
                            "label": _label_for_class(class_id, class_names, fallback_label),
                            "box": (x1, y1, x2, y2),
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "scale": scale_name,
                        }
                    )
            continue

        raw_boxes = scale_prediction[0, 1:5]
        if use_centerness and scale_prediction.shape[1] > 5:
            centerness = torch.sigmoid(scale_prediction[0, 5]).clamp(0.05, 1.0)
            class_start = 6
        else:
            centerness = torch.ones_like(objectness)
            class_start = 5
        if num_classes > 1 and scale_prediction.shape[1] >= class_start + num_classes:
            class_probs = torch.softmax(scale_prediction[0, class_start : class_start + num_classes], dim=0)
            class_scores, class_ids = torch.max(class_probs, dim=0)
            score_map = objectness * centerness * class_scores
        else:
            class_ids = torch.zeros_like(objectness, dtype=torch.long)
            class_scores = torch.ones_like(objectness)
            score_map = objectness * centerness
        local_max = F.max_pool2d(score_map.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
        peaks = score_map >= local_max
        mask = (score_map >= score_threshold) & peaks
        if not bool(mask.any()):
            continue
        scores = score_map[mask]
        indexes = torch.nonzero(mask, as_tuple=False)
        if max_candidates > 0 and scores.numel() > max_candidates:
            scores, order = torch.topk(scores, max_candidates)
            indexes = indexes[order]
        for score, index in zip(scores.detach().cpu(), indexes.detach().cpu()):
            grid_y = int(index[0])
            grid_x = int(index[1])
            values = raw_boxes[:, grid_y, grid_x].detach().cpu()
            offsets = torch.sigmoid(values[:2])
            size = torch.sigmoid(values[2:]).clamp(1e-4, 1.0)
            center_x = (grid_x + float(offsets[0])) / grid_size
            center_y = (grid_y + float(offsets[1])) / grid_size
            width = float(size[0])
            height = float(size[1])
            x1 = max(0.0, center_x - width / 2)
            y1 = max(0.0, center_y - height / 2)
            x2 = min(1.0, center_x + width / 2)
            y2 = min(1.0, center_y + height / 2)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[grid_y, grid_x].detach().cpu())
            detections.append(
                {
                    "score": float(score),
                    "objectness": float(objectness[grid_y, grid_x].detach().cpu()),
                    "centerness": float(centerness[grid_y, grid_x].detach().cpu()),
                    "class_score": float(class_scores[grid_y, grid_x].detach().cpu()),
                    "class_id": class_id,
                    "label": _label_for_class(class_id, class_names, fallback_label),
                    "box": (x1, y1, x2, y2),
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "scale": scale_name,
                }
            )
    if max_candidates > 0 and len(detections) > max_candidates:
        detections = sorted(detections, key=lambda item: float(item["score"]), reverse=True)[:max_candidates]
    return detections


def _vertical_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    top = max(float(a["y1"]), float(b["y1"]))
    bottom = min(float(a["y2"]), float(b["y2"]))
    overlap = max(0.0, bottom - top)
    min_height = max(1.0, min(float(a["height"]), float(b["height"])))
    return overlap / min_height


def _union_bbox_payload(items: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    left = min(int(item["bbox"]["x1"]) for item in items)
    top = min(int(item["bbox"]["y1"]) for item in items)
    right = max(int(item["bbox"]["x2"]) for item in items)
    bottom = max(int(item["bbox"]["y2"]) for item in items)
    return _bbox_payload((left, top, right, bottom), width, height)


def _same_neural_object(
    detection: dict[str, Any],
    group: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    min_merge_score: float,
    max_area_growth: float = 2.35,
) -> bool:
    if detection.get("label") != group[0].get("label"):
        return False
    if float(detection.get("score", 0.0)) < min_merge_score:
        return False
    group_bbox = _union_bbox_payload(group, image_width, image_height)
    candidate_bbox = detection["bbox"]
    member_boxes = [item["bbox"] for item in group]
    best_iou = max(bbox_iou(candidate_bbox, member_bbox) for member_bbox in member_boxes)
    best_containment = max(bbox_containment(candidate_bbox, member_bbox) for member_bbox in member_boxes)
    vertical_overlap = _vertical_overlap_ratio(candidate_bbox, group_bbox)
    center_distance = bbox_center_distance_ratio(candidate_bbox, group_bbox)
    union = _union_bbox_payload([*group, detection], image_width, image_height)
    union_area = float(union["width"] * union["height"])
    max_member_area = max(float(item["bbox"]["width"] * item["bbox"]["height"]) for item in [*group, detection])
    if union_area > max_member_area * max_area_growth:
        return False
    if best_iou >= 0.18 or best_containment >= 0.48:
        return True
    return center_distance <= 0.42 and vertical_overlap >= 0.42


def merge_neural_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    min_merge_score: float = 0.82,
) -> tuple[list[dict[str, Any]], int]:
    groups: list[list[dict[str, Any]]] = []
    for detection in sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        matched_group = None
        for group in groups:
            if _same_neural_object(detection, group, image_width, image_height, min_merge_score=min_merge_score):
                matched_group = group
                break
        if matched_group is None:
            groups.append([detection])
        else:
            matched_group.append(detection)

    merged: list[dict[str, Any]] = []
    merged_count = 0
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        best = max(group, key=lambda item: float(item.get("score", 0.0)))
        fused = dict(best)
        fused["bbox"] = _union_bbox_payload(group, image_width, image_height)
        fused["score"] = round(max(float(item.get("score", 0.0)) for item in group), 6)
        fused["adjusted_score"] = fused["score"]
        fused["proposal"] = "neural_grid_fused"
        fused["fused_count"] = len(group)
        fused["fused_boxes"] = [
            {
                "score": item.get("score"),
                "bbox": item.get("bbox"),
            }
            for item in group[:8]
        ]
        merged_count += len(group) - 1
        merged.append(fused)
    return merged, merged_count


def detect_image_neural(
    image_path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    device: str | torch.device = "cpu",
    score_threshold: float = 0.35,
    nms_threshold: float = 0.45,
    image_size: int | None = None,
    label: str = "0",
    max_candidates: int = 200,
    min_detection_width: int = 8,
    min_detection_height: int = 6,
    min_detection_area: int = 80,
    max_detections: int = 20,
) -> dict[str, Any]:
    started = time.time()
    path = Path(image_path)
    image_size = int(image_size or config.get("image_size", 640))
    device = torch.device(device)
    source = read_rgb_image(path)
    height, width = source.shape[:2]
    boxed, letterbox = letterbox_cv2(source, image_size)
    tensor = torch.from_numpy(boxed).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(tensor)
    num_classes = int(config.get("num_classes", 1))
    use_centerness = bool(config.get("use_centerness", False))
    class_heatmap = bool(config.get("class_heatmap", False))
    class_box = bool(config.get("class_box", False))
    quality_fpn = bool(config.get("quality_fpn", False))
    center_offset = bool(config.get("center_offset", False)) or str(config.get("box_format", "")) == "center_ltrb"
    class_names = _config_class_names(config)
    decoded = decode_neural_prediction(
        {key: value.detach().cpu() for key, value in prediction.items()} if isinstance(prediction, dict) else prediction.detach().cpu(),
        score_threshold=score_threshold,
        max_candidates=max_candidates,
        num_classes=num_classes,
        use_centerness=use_centerness,
        class_heatmap=class_heatmap,
        class_box=class_box,
        quality_fpn=quality_fpn,
        center_offset=center_offset,
        class_names=class_names,
        fallback_label=label,
    )
    detections: list[dict[str, Any]] = []
    for item in decoded:
        source_bbox = map_square_box_to_source(item["box"], letterbox)
        if source_bbox is None:
            continue
        score = round(float(item["score"]), 6)
        detections.append(
            {
                "label": str(item.get("label", label)),
                "score": score,
                "adjusted_score": score,
                "bbox": _bbox_payload(source_bbox, width, height),
                "proposal": "neural_grid",
                "class_id": item.get("class_id", 0),
                "objectness": round(float(item.get("objectness", score)), 6),
                "centerness": round(float(item.get("centerness", 1.0)), 6),
                "class_score": round(float(item.get("class_score", 1.0)), 6),
                "localization_quality": round(float(item.get("quality", item.get("centerness", 1.0))), 6),
                "center_offset": item.get("center_offset"),
                "scale": item.get("scale", "p8"),
                "grid_cell": {"x": item["grid_x"], "y": item["grid_y"]},
            }
        )
    if quality_fpn:
        neural_merged_count = 0
    else:
        detections, neural_merged_count = merge_neural_detections(
            detections,
            width,
            height,
            min_merge_score=max(0.82, score_threshold),
        )
    use_soft_nms = _use_soft_nms_for_detections(config, detections, width, height)
    if use_soft_nms:
        final_detections = soft_nms(
            detections,
            iou_threshold=max(0.10, nms_threshold),
            sigma=float(config.get("soft_nms_sigma", 0.50)),
            score_threshold=max(score_threshold * 0.82, float(config.get("soft_nms_min_score", 0.10))),
            containment_threshold=0.82,
            diou_threshold=0.18,
            class_aware=True,
        )
    else:
        final_detections = nms(
            detections,
            iou_threshold=nms_threshold,
            containment_threshold=0.75,
            cluster_center_distance=0.0,
            cluster_containment_threshold=0.0,
            diou_threshold=0.16 if quality_fpn else 0.20,
            class_aware=True,
        )
    final_detections, postprocess_suppressed = filter_detections(
        final_detections,
        min_width=min_detection_width,
        min_height=min_detection_height,
        min_area=min_detection_area,
        max_detections=max_detections,
        min_refine_edge_gain=0.0,
    )
    return {
        "image": str(path),
        "width": width,
        "height": height,
        "letterbox": letterbox,
        "proposal_count": sum(item.shape[-1] * item.shape[-2] for item in prediction.values()) if isinstance(prediction, dict) else prediction.shape[-1] * prediction.shape[-2],
        "raw_detection_count": len(decoded),
        "negative_suppressed_count": 0,
        "quality_suppressed_count": 0,
        "context_suppressed_count": 0,
        "fragmentation_suppressed_count": 0,
        "shape_suppressed_count": 0,
        "score_suppressed_count": 0,
        "neural_merged_count": neural_merged_count,
        "second_stage": {"enabled": False},
        "postprocess_suppressed": postprocess_suppressed,
        "nms_mode": "soft" if use_soft_nms else "hard",
        "detections": final_detections,
        "elapsed_seconds": round(time.time() - started, 4),
    }


def detect_images_neural(
    image_path: str | Path,
    weights_path: str | Path,
    device: str = "auto",
    score_threshold: float = 0.35,
    nms_threshold: float = 0.45,
    image_size: int | None = None,
    label: str = "0",
    max_candidates: int = 200,
    soft_nms_mode: str | None = None,
    min_detection_width: int = 8,
    min_detection_height: int = 6,
    min_detection_area: int = 80,
    max_detections: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_device = _device_from_arg(device)
    model, config = load_neural_checkpoint(weights_path, resolved_device)
    if score_threshold < 0:
        score_threshold = float(config.get("score_threshold", 0.35))
    if nms_threshold < 0:
        nms_threshold = float(config.get("nms_threshold", 0.45))
    if soft_nms_mode is not None:
        config = {
            **config,
            "soft_nms_mode": soft_nms_mode,
            "soft_nms": soft_nms_mode == "on",
        }
    paths = neural_image_paths(image_path)
    results = []
    started = time.time()
    for index, path in enumerate(paths, start=1):
        print(f"[neural-detect] image {index}/{len(paths)} {path.name}", flush=True)
        result = detect_image_neural(
            path,
            model,
            config,
            device=resolved_device,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            image_size=image_size,
            label=label,
            max_candidates=max_candidates,
            min_detection_width=min_detection_width,
            min_detection_height=min_detection_height,
            min_detection_area=min_detection_area,
            max_detections=max_detections,
        )
        results.append(result)
        print(
            f"[neural-detect] image {index}/{len(paths)} {path.name} "
            f"detections={len(result['detections'])} elapsed={result['elapsed_seconds']}s",
            flush=True,
        )
    metadata = {
        "device": str(resolved_device),
        "config": config,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "soft_nms_mode": config.get("soft_nms_mode", "auto"),
        "elapsed_seconds": round(time.time() - started, 4),
    }
    return results, metadata


def describe_checkpoint(weights_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    return {
        "path": str(weights_path),
        "config": config,
        "epoch": checkpoint.get("epoch"),
        "best_score": checkpoint.get("best_score"),
        "metrics": checkpoint.get("metrics"),
    }


__all__ = [
    "decode_neural_prediction",
    "describe_checkpoint",
    "detect_image_neural",
    "detect_images_neural",
    "load_neural_checkpoint",
    "merge_neural_detections",
]
