import torch

from solo.neural.inference import _use_soft_nms_for_detections, decode_neural_prediction
from solo.neural.losses import detection_loss
from solo.neural.model import create_detector_model


def test_fpn_quality_forward_loss_and_decode():
    model = create_detector_model("fpn_quality", pretrained=False, num_classes=2)
    images = torch.rand(2, 3, 256, 256)
    prediction = model(images)

    assert {name: tuple(value.shape[-2:]) for name, value in prediction.items()} == {
        "p4": (64, 64),
        "p8": (32, 32),
        "p16": (16, 16),
        "p32": (8, 8),
    }
    assert all(value.shape[1] == 7 for value in prediction.values())

    boxes = [
        torch.tensor([[0.20, 0.20, 0.60, 0.50], [0.65, 0.55, 0.72, 0.66]], dtype=torch.float32),
        torch.zeros((0, 4), dtype=torch.float32),
    ]
    labels = [torch.tensor([1, 0], dtype=torch.long), torch.zeros((0,), dtype=torch.long)]
    loss, metrics = detection_loss(prediction, boxes, labels_batch=labels, num_classes=2, quality_fpn=True)

    assert torch.isfinite(loss)
    assert metrics["positives"] > 0
    loss.backward()

    decoded = decode_neural_prediction(
        {key: value.detach() for key, value in prediction.items()},
        score_threshold=0.0,
        max_candidates=8,
        num_classes=2,
        quality_fpn=True,
        class_names=["feed", "crab"],
    )
    assert decoded
    assert {item["label"] for item in decoded} <= {"feed", "crab"}


def test_fpn_quality_v2_uses_center_offsets_and_nwd_loss():
    model = create_detector_model("fpn_quality_v2", pretrained=False, num_classes=2)
    images = torch.rand(2, 3, 256, 256)
    prediction = model(images)

    assert all(value.shape[1] == 9 for value in prediction.values())

    boxes = [
        torch.tensor([[0.20, 0.20, 0.24, 0.24], [0.50, 0.45, 0.90, 0.70]], dtype=torch.float32),
        torch.zeros((0, 4), dtype=torch.float32),
    ]
    labels = [torch.tensor([0, 1], dtype=torch.long), torch.zeros((0,), dtype=torch.long)]
    loss, metrics = detection_loss(prediction, boxes, labels_batch=labels, num_classes=2, quality_fpn=True)

    assert torch.isfinite(loss)
    assert metrics["positives"] > 0
    assert metrics["nwd_loss"] >= 0.0
    loss.backward()

    decoded = decode_neural_prediction(
        {key: value.detach() for key, value in prediction.items()},
        score_threshold=0.0,
        max_candidates=8,
        num_classes=2,
        quality_fpn=True,
        center_offset=True,
        class_names=["feed", "crab"],
    )
    assert decoded
    assert "center_offset" in decoded[0]


def test_fpn_quality_v3_enables_task_aligned_loss():
    model = create_detector_model("fpn_quality_v3", pretrained=False, num_classes=2)
    images = torch.rand(2, 3, 256, 256)
    prediction = model(images)

    assert model.task_aligned is True
    assert all(value.shape[1] == 9 for value in prediction.values())

    boxes = [
        torch.tensor([[0.18, 0.20, 0.25, 0.27], [0.50, 0.45, 0.88, 0.72]], dtype=torch.float32),
        torch.zeros((0, 4), dtype=torch.float32),
    ]
    labels = [torch.tensor([0, 1], dtype=torch.long), torch.zeros((0,), dtype=torch.long)]
    loss, metrics = detection_loss(
        prediction,
        boxes,
        labels_batch=labels,
        num_classes=2,
        quality_fpn=True,
        task_aligned=True,
    )

    assert torch.isfinite(loss)
    assert metrics["alignment_loss"] >= 0.0


def test_fpn_quality_v4_uses_panet_and_advanced_geometry_loss():
    model = create_detector_model("fpn_quality_v4", pretrained=False, num_classes=2)
    images = torch.rand(2, 3, 256, 256)
    prediction = model(images)

    assert model.panet is True
    assert model.task_aligned is True
    assert model.advanced_box_loss is True
    assert all(value.shape[1] == 9 for value in prediction.values())

    boxes = [
        torch.tensor([[0.08, 0.10, 0.13, 0.15], [0.22, 0.30, 0.86, 0.62]], dtype=torch.float32),
        torch.zeros((0, 4), dtype=torch.float32),
    ]
    labels = [torch.tensor([0, 1], dtype=torch.long), torch.zeros((0,), dtype=torch.long)]
    loss, metrics = detection_loss(
        prediction,
        boxes,
        labels_batch=labels,
        num_classes=2,
        quality_fpn=True,
        task_aligned=True,
        advanced_box_loss=True,
    )

    assert torch.isfinite(loss)
    assert metrics["advanced_geometry_loss"] >= 0.0
    assert metrics["nwd_loss"] >= 0.0
    loss.backward()


def test_soft_nms_auto_policy_prefers_hard_nms_for_sparse_scenes():
    detections = [
        {"bbox": {"width": 120, "height": 60}, "label": "car", "score": 0.9},
        {"bbox": {"width": 100, "height": 50}, "label": "car", "score": 0.8},
    ]

    assert not _use_soft_nms_for_detections({"quality_fpn": True, "soft_nms_mode": "auto"}, detections, 640, 480)


def test_soft_nms_auto_policy_enables_dense_compact_scenes():
    detections = [
        {"bbox": {"width": 18, "height": 16}, "label": "feed", "score": 0.4}
        for _index in range(30)
    ]

    assert _use_soft_nms_for_detections({"quality_fpn": True, "soft_nms_mode": "auto"}, detections, 640, 480)
