from __future__ import annotations

import json
import random
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from solo.config import DEFAULT_NEURAL_SOFT_NMS_MODE
from solo.data.dataset import load_class_names
from solo.engine.evaluation import evaluate_detection_results
from solo.neural.data import NeuralDetectionDataset, collate_detection_batch, neural_label_path, read_yolo_boxes
from solo.neural.inference import detect_image_neural
from solo.neural.losses import detection_loss
from solo.neural.model import count_parameters, create_detector_model
from solo.utils.visualization import draw_detections, save_detection_report


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_neural_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _average_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item if isinstance(item.get(key), (int, float))})
    return {key: float(sum(float(item.get(key, 0.0)) for item in items) / len(items)) for key in keys}


def _class_names_list(class_names_path: str | Path | None, num_classes: int | None = None) -> list[str]:
    names = load_class_names(class_names_path)
    if names:
        size = max(max(names) + 1, int(num_classes or 0))
        return [str(names.get(index, index)) for index in range(size)]
    if num_classes is None or num_classes <= 0:
        return []
    return [str(index) for index in range(num_classes)]


def _infer_num_classes(labels_dir: str | Path, class_names_path: str | Path | None = None) -> int:
    class_names = load_class_names(class_names_path)
    max_label = max(class_names.keys(), default=-1)
    for path in Path(labels_dir).glob("*.txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                max_label = max(max_label, int(float(parts[0])))
            except ValueError:
                continue
    return max(1, max_label + 1)


def _label_counts(labels_dir: str | Path, num_classes: int) -> list[int]:
    counts = [0 for _ in range(max(1, num_classes))]
    for path in Path(labels_dir).glob("*.txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                label = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= label < len(counts):
                counts[label] += 1
    return counts


def _global_class_weights(labels_dir: str | Path, num_classes: int) -> list[float] | None:
    if num_classes <= 1:
        return None
    counts = _label_counts(labels_dir, num_classes)
    present = [count for count in counts if count > 0]
    if not present:
        return None
    total = float(sum(present))
    present_classes = float(len(present))
    weights = []
    for count in counts:
        if count <= 0:
            weights.append(1.0)
        else:
            weights.append(max(0.35, min(6.0, total / (float(count) * present_classes))))
    return weights


def _rare_class_sampler(dataset: NeuralDetectionDataset, num_classes: int, rare_boost: float = 5.0) -> WeightedRandomSampler | None:
    if num_classes <= 1:
        return None
    image_labels: list[set[int]] = []
    counts = [0 for _ in range(num_classes)]
    for path in dataset.paths:
        _boxes, labels = read_yolo_boxes(neural_label_path(path, dataset.labels_dir))
        label_set: set[int] = set()
        for label in labels:
            try:
                label_id = int(float(label))
            except ValueError:
                continue
            if 0 <= label_id < num_classes:
                label_set.add(label_id)
                counts[label_id] += 1
        image_labels.append(label_set)
    present = [count for count in counts if count > 0]
    if not present:
        return None
    max_count = max(present)
    rare_classes = {index for index, count in enumerate(counts) if 0 < count <= max_count * 0.35}
    if not rare_classes:
        return None
    weights = []
    for labels in image_labels:
        rare_hits = len(labels & rare_classes)
        weights.append(1.0 + rare_hits * rare_boost)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _should_cache_training_images(image_dir: str | Path, image_size: int, max_images: int = 512, max_gb: float = 2.5) -> bool:
    try:
        paths = list(Path(image_dir).iterdir())
    except OSError:
        return False
    image_paths = [path for path in paths if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}]
    if not image_paths or len(image_paths) > max_images:
        return False
    estimated_bytes = len(image_paths) * int(image_size) * int(image_size) * 3
    return estimated_bytes <= max_gb * (1024**3)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    box_weight: float,
    iou_weight: float,
    class_weight: float,
    centerness_weight: float,
    num_classes: int,
    use_centerness: bool,
    class_weights: list[float] | None,
    class_heatmap: bool,
    class_box: bool,
    quality_fpn: bool,
    task_aligned: bool,
    advanced_box_loss: bool,
) -> dict[str, float]:
    model.train()
    metrics = []
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        boxes = [item.to(device, non_blocking=True) for item in batch["boxes"]]
        labels = [item.to(device, non_blocking=True) for item in batch["labels"]]
        prediction = model(images)
        loss, loss_metrics = detection_loss(
            prediction,
            boxes,
            labels_batch=labels,
            box_weight=box_weight,
            iou_weight=iou_weight,
            class_weight=class_weight,
            centerness_weight=centerness_weight,
            num_classes=num_classes,
            use_centerness=use_centerness,
            class_weights=class_weights,
            class_heatmap=class_heatmap,
            class_box=class_box,
            quality_fpn=quality_fpn,
            task_aligned=task_aligned,
            advanced_box_loss=advanced_box_loss,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        metrics.append(loss_metrics)
        if step == 1 or step % 10 == 0 or step == len(loader):
            averaged = _average_metrics(metrics[-10:])
            nwd_text = f" nwd={averaged.get('nwd_loss', 0):.4f}" if "nwd_loss" in averaged else ""
            align_text = f" align={averaged.get('alignment_loss', 0):.4f}" if "alignment_loss" in averaged else ""
            adv_text = f" adv={averaged.get('advanced_geometry_loss', 0):.4f}" if "advanced_geometry_loss" in averaged else ""
            print(
                f"[neural-train] epoch={epoch}/{epochs} step={step}/{len(loader)} "
                f"loss={averaged.get('loss', 0):.4f} obj={averaged.get('object_loss', 0):.4f} "
                f"box={averaged.get('box_loss', 0):.4f} giou={averaged.get('giou_loss', 0):.4f} "
                f"cls={averaged.get('class_loss', 0):.4f} ctr={averaged.get('centerness_loss', 0):.4f}"
                f"{nwd_text}{align_text}{adv_text} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return _average_metrics(metrics)


def _evaluate_checkpoint_candidate(
    model: torch.nn.Module,
    image_dir: str | Path,
    labels_dir: str | Path,
    device: torch.device,
    image_size: int,
    score_threshold: float,
    nms_threshold: float,
    val_match_iou: float,
    max_detections: int,
    class_names_path: str | Path | None = None,
    soft_nms_mode: str = DEFAULT_NEURAL_SOFT_NMS_MODE,
    quiet: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    was_training = model.training
    model.eval()
    dataset = NeuralDetectionDataset(image_dir, labels_dir, image_size=image_size, augment=False)
    config = {
        "image_size": image_size,
        "model": "checkpoint_candidate",
        "num_classes": int(getattr(model, "num_classes", 1)),
        "use_centerness": bool(getattr(model, "use_centerness", False)),
        "class_heatmap": bool(getattr(model, "class_heatmap", False)),
        "class_box": bool(getattr(model, "class_box", False)),
        "quality_fpn": bool(getattr(model, "quality_fpn", False)),
        "center_offset": bool(getattr(model, "center_offset", False)),
        "task_aligned": bool(getattr(model, "task_aligned", False)),
        "panet": bool(getattr(model, "panet", False)),
        "advanced_box_loss": bool(getattr(model, "advanced_box_loss", False)),
        "soft_nms_mode": soft_nms_mode,
        "soft_nms": soft_nms_mode == "on",
        "box_format": str(getattr(model, "box_format", "")),
        "class_names": _class_names_list(class_names_path, int(getattr(model, "num_classes", 1))),
    }
    with torch.inference_mode():
        results = [
            detect_image_neural(
                path,
                model,
                config,
                device=device,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                image_size=image_size,
                max_detections=max_detections,
            )
            for path in dataset.paths
        ]
    if quiet:
        with redirect_stdout(StringIO()):
            evaluation = evaluate_detection_results(
                results,
                labels_dir,
                annotations="yolo",
                class_names_path=class_names_path,
                match_iou=val_match_iou,
                duplicate_iou=0.85,
                weight_index=None,
                evaluate_proposals=False,
                strict_labels=True,
            )
    else:
        evaluation = evaluate_detection_results(
            results,
            labels_dir,
            annotations="yolo",
            class_names_path=class_names_path,
            match_iou=val_match_iou,
            duplicate_iou=0.85,
            weight_index=None,
            evaluate_proposals=False,
            strict_labels=True,
        )
    if was_training:
        model.train()
    return evaluation, results


def _evaluate_thresholds(
    model: torch.nn.Module,
    image_dir: str | Path,
    labels_dir: str | Path,
    device: torch.device,
    image_size: int,
    thresholds: list[float],
    nms_threshold: float,
    val_match_iou: float,
    max_detections: int,
    class_names_path: str | Path | None = None,
    soft_nms_mode: str = DEFAULT_NEURAL_SOFT_NMS_MODE,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    best_evaluation: dict[str, Any] | None = None
    best_results: list[dict[str, Any]] = []
    best_threshold = thresholds[0]
    best_score = -1.0
    for threshold in thresholds:
        evaluation, results = _evaluate_checkpoint_candidate(
            model,
            image_dir,
            labels_dir,
            device,
            image_size=image_size,
            score_threshold=threshold,
            nms_threshold=nms_threshold,
            val_match_iou=val_match_iou,
            max_detections=max_detections,
            class_names_path=class_names_path,
            soft_nms_mode=soft_nms_mode,
            quiet=True,
        )
        summary = evaluation["summary"]
        score = float(summary.get("f1", 0.0)) + 0.15 * float(summary.get("recall", 0.0))
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_evaluation = evaluation
            best_results = results
    assert best_evaluation is not None
    return best_evaluation, best_results, best_threshold


def _save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    best_score: float,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "solo_neural_detector_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            "best_score": best_score,
        },
        checkpoint_path,
    )
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "format": "solo_neural_detector_v1",
                "epoch": epoch,
                "config": config,
                "metrics": metrics,
                "best_score": best_score,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def train_neural_detector(
    train_images: str | Path,
    train_labels: str | Path,
    output_path: str | Path,
    val_images: str | Path | None = None,
    val_labels: str | Path | None = None,
    image_size: int = 640,
    epochs: int = 80,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    workers: int = 0,
    score_threshold: float = 0.28,
    nms_threshold: float = 0.45,
    val_match_iou: float = 0.25,
    max_detections: int = 20,
    box_weight: float = 5.0,
    iou_weight: float = 2.0,
    class_weight: float = 1.2,
    centerness_weight: float = 0.7,
    draw_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    model_name: str = "mobilenet_context",
    pretrained: bool = True,
    class_names_path: str | Path | None = None,
    num_classes: int | None = None,
    soft_nms_mode: str = DEFAULT_NEURAL_SOFT_NMS_MODE,
) -> dict[str, Any]:
    set_neural_seed(seed)
    resolved_device = _device_from_arg(device)
    inferred_classes = int(num_classes or _infer_num_classes(train_labels, class_names_path))
    class_names = _class_names_list(class_names_path, inferred_classes)
    use_centerness = model_name == "multiscale_context"
    class_heatmap = inferred_classes > 1 and model_name == "multiscale_context"
    class_box = class_heatmap
    quality_fpn = model_name in {"fpn_quality", "fpn_quality_v2", "fpn_quality_v3", "fpn_quality_v4"}
    center_offset = model_name in {"fpn_quality_v2", "fpn_quality_v3", "fpn_quality_v4"}
    task_aligned = model_name in {"fpn_quality_v3", "fpn_quality_v4"}
    advanced_box_loss = model_name == "fpn_quality_v4"
    soft_nms_mode = str(soft_nms_mode or DEFAULT_NEURAL_SOFT_NMS_MODE).lower().strip()
    if soft_nms_mode not in {"auto", "on", "off"}:
        raise ValueError("soft_nms_mode must be 'auto', 'on', or 'off'")
    if quality_fpn:
        use_centerness = True
        class_heatmap = True
        class_box = False
    cache_images = _should_cache_training_images(train_images, image_size)
    train_dataset = NeuralDetectionDataset(
        train_images,
        train_labels,
        image_size=image_size,
        augment=True,
        seed=seed,
        cache_images=cache_images,
    )
    sampler = _rare_class_sampler(train_dataset, inferred_classes)
    class_weights = _global_class_weights(train_labels, inferred_classes)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=resolved_device.type == "cuda",
        collate_fn=collate_detection_batch,
    )
    model = create_detector_model(
        model_name,
        pretrained=pretrained,
        num_classes=inferred_classes,
        use_centerness=use_centerness,
        class_heatmap=class_heatmap,
        class_box=class_box,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05)
    config = {
        "model": model_name,
        "pretrained": bool(pretrained),
        "image_size": image_size,
        "grid_size": image_size // 8,
        "num_classes": inferred_classes,
        "class_names": class_names,
        "class_names_path": str(class_names_path) if class_names_path else None,
        "use_centerness": use_centerness,
        "class_heatmap": class_heatmap,
        "class_box": class_box,
        "quality_fpn": quality_fpn,
        "center_offset": center_offset,
        "nwd_loss": center_offset,
        "task_aligned": task_aligned,
        "panet": model_name == "fpn_quality_v4",
        "advanced_box_loss": advanced_box_loss,
        "box_loss_family": "smooth_l1+giou+ciou+mpdiou+nwd" if advanced_box_loss else "smooth_l1+giou+nwd",
        "soft_nms_mode": soft_nms_mode,
        "soft_nms": soft_nms_mode == "on",
        "soft_nms_sigma": 0.50,
        "soft_nms_min_score": 0.10,
        "box_format": "center_ltrb" if center_offset else ("ltrb" if quality_fpn else "xywh"),
        "train_images": str(train_images),
        "train_labels": str(train_labels),
        "val_images": str(val_images) if val_images else None,
        "val_labels": str(val_labels) if val_labels else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "val_match_iou": val_match_iou,
        "max_detections": max_detections,
        "box_weight": box_weight,
        "iou_weight": iou_weight,
        "class_weight": class_weight,
        "centerness_weight": centerness_weight,
        "class_weights": class_weights,
        "rare_class_sampling": sampler is not None,
        "cache_images": cache_images,
        "seed": seed,
        "parameter_count": count_parameters(model),
    }
    print(
        f"[neural-train] model={model_name} pretrained={bool(pretrained)} classes={inferred_classes} "
        f"centerness={use_centerness} class_heatmap={class_heatmap} class_box={class_box} "
        f"quality_fpn={quality_fpn} center_offset={center_offset} task_aligned={task_aligned} "
        f"advanced_box_loss={advanced_box_loss} panet={model_name == 'fpn_quality_v4'} "
        f"soft_nms_mode={soft_nms_mode} "
        f"params={config['parameter_count']} "
        f"device={resolved_device} train_images={len(train_dataset)} image_size={image_size} cache_images={cache_images}",
        flush=True,
    )
    if class_weights:
        print(f"[neural-train] class_weights={class_weights} rare_class_sampling={sampler is not None}", flush=True)
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    output_path = Path(output_path)
    last_train_metrics: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        last_train_metrics = _run_epoch(
            model,
            loader,
            optimizer,
            resolved_device,
            epoch,
            epochs,
            box_weight=box_weight,
            iou_weight=iou_weight,
            class_weight=class_weight,
            centerness_weight=centerness_weight,
            num_classes=inferred_classes,
            use_centerness=use_centerness,
            class_weights=class_weights,
            class_heatmap=class_heatmap,
            class_box=class_box,
            quality_fpn=quality_fpn,
            task_aligned=task_aligned,
            advanced_box_loss=advanced_box_loss,
        )
        scheduler.step()
        metrics: dict[str, Any] = {"train": last_train_metrics, "lr": scheduler.get_last_lr()[0]}
        score = -float(last_train_metrics.get("loss", 0.0))
        if val_images and val_labels and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            thresholds = sorted(
                {
                    max(0.03, min(0.95, score_threshold * scale))
                    for scale in (0.5, 0.7, 1.0, 1.35, 1.8, 2.4, 3.2, 4.1)
                }
                | {0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.93}
            )
            evaluation, _results, best_threshold = _evaluate_thresholds(
                model,
                val_images,
                val_labels,
                resolved_device,
                image_size=image_size,
                thresholds=thresholds,
                nms_threshold=nms_threshold,
                val_match_iou=val_match_iou,
                max_detections=max_detections,
                class_names_path=class_names_path,
                soft_nms_mode=soft_nms_mode,
            )
            summary = evaluation["summary"]
            metrics["validation"] = summary
            metrics["score_threshold"] = best_threshold
            score = float(summary.get("f1", 0.0)) + 0.15 * float(summary.get("recall", 0.0))
            print(
                f"[neural-train] validation epoch={epoch} threshold={best_threshold:.3f} "
                f"tp={summary['tp']} fp={summary['fp']} fn={summary['fn']} "
                f"precision={summary['precision']:.4f} recall={summary['recall']:.4f} f1={summary['f1']:.4f}",
                flush=True,
            )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            if "score_threshold" in metrics:
                config["score_threshold"] = metrics["score_threshold"]
            _save_checkpoint(output_path, model, optimizer, epoch, config, metrics, best_score)
            print(f"[neural-train] saved best checkpoint to {output_path.resolve()} score={best_score:.4f}", flush=True)

    if not output_path.exists():
        _save_checkpoint(output_path, model, optimizer, epochs, config, {"train": last_train_metrics}, best_score)

    final_results: list[dict[str, Any]] = []
    final_evaluation: dict[str, Any] = {"enabled": False}
    if val_images and val_labels:
        checkpoint = torch.load(output_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        config = checkpoint.get("config", config)
        model.eval()
        final_evaluation, final_results = _evaluate_checkpoint_candidate(
            model,
            val_images,
            val_labels,
            resolved_device,
            image_size=image_size,
            score_threshold=float(config.get("score_threshold", score_threshold)),
            nms_threshold=nms_threshold,
            val_match_iou=val_match_iou,
            max_detections=max_detections,
            class_names_path=class_names_path,
            soft_nms_mode=str(config.get("soft_nms_mode", soft_nms_mode)),
        )
        if draw_dir is not None:
            for result in final_results:
                draw_path = Path(draw_dir) / Path(result["image"]).name
                result["draw_path"] = str(draw_detections(result["image"], result["detections"], draw_path))
        if report_path is not None:
            save_detection_report(
                final_results,
                report_path,
                {
                    "backend": "neural",
                    "weights": str(output_path),
                    "config": config,
                    "evaluation": final_evaluation,
                    "total_images": len(final_results),
                    "total_detections": sum(len(result["detections"]) for result in final_results),
                },
            )

    return {
        "weights_path": str(output_path),
        "config": config,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "evaluation": final_evaluation,
    }


__all__ = ["set_neural_seed", "train_neural_detector"]
