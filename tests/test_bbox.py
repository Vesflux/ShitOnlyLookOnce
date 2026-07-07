import numpy as np

from solo.utils.bbox import bbox_containment, bbox_diou, bbox_iou, distance_box_fusion, nms, snap_bbox_to_edges, soft_nms, split_overmerged_fusions


def box(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}


def test_bbox_iou_and_containment():
    left = box(0, 0, 10, 10)
    right = box(5, 5, 15, 15)
    inner = box(2, 2, 8, 8)

    assert round(bbox_iou(left, right), 6) == round(25 / 175, 6)
    assert bbox_containment(left, inner) == 1.0


def test_nms_keeps_highest_scoring_non_overlapping_boxes():
    detections = [
        {"bbox": box(0, 0, 10, 10), "score": 0.90},
        {"bbox": box(1, 1, 11, 11), "score": 0.80},
        {"bbox": box(30, 30, 40, 40), "score": 0.70},
    ]

    kept = nms(detections, iou_threshold=0.5)

    assert len(kept) == 2
    assert kept[0]["score"] == 0.90
    assert kept[1]["score"] == 0.70


def test_diou_nms_suppresses_same_center_low_iou_duplicates():
    detections = [
        {"label": "car", "bbox": box(20, 20, 100, 60), "score": 0.92},
        {"label": "car", "bbox": box(35, 17, 115, 57), "score": 0.88},
        {"label": "car", "bbox": box(150, 20, 220, 60), "score": 0.86},
    ]

    assert bbox_diou(detections[0]["bbox"], detections[1]["bbox"]) > 0.24
    kept = nms(detections, iou_threshold=0.95, diou_threshold=0.24)

    assert len(kept) == 2
    assert kept[0]["score"] == 0.92
    assert kept[1]["score"] == 0.86


def test_soft_nms_decays_duplicate_scores_without_hard_deleting_neighbors():
    detections = [
        {"label": "car", "bbox": box(20, 20, 100, 60), "score": 0.92, "adjusted_score": 0.92},
        {"label": "car", "bbox": box(24, 21, 98, 59), "score": 0.88, "adjusted_score": 0.88},
        {"label": "car", "bbox": box(112, 20, 182, 60), "score": 0.70, "adjusted_score": 0.70},
    ]

    kept = soft_nms(detections, iou_threshold=0.35, sigma=0.5, score_threshold=0.05)

    assert len(kept) == 3
    assert kept[0]["score"] == 0.92
    assert kept[1]["score"] == 0.70
    assert kept[2]["score"] < 0.20


def test_distance_box_fusion_merges_nearby_same_label_fragments_without_iou():
    detections = [
        {"label": "car", "score": 0.92, "adjusted_score": 0.92, "bbox": box(20, 40, 66, 74)},
        {"label": "car", "score": 0.90, "adjusted_score": 0.90, "bbox": box(74, 41, 124, 75)},
        {"label": "car", "score": 0.82, "adjusted_score": 0.82, "bbox": box(20, 110, 80, 144)},
    ]

    fused, fused_count = distance_box_fusion(detections, 220, 160)

    assert fused_count == 1
    assert len(fused) == 2
    assert fused[0]["bbox"]["x1"] == 20
    assert fused[0]["bbox"]["x2"] == 124
    assert fused[0]["proposal"] == "distance_box_fused"


def test_distance_box_fusion_preserves_separate_vehicle_centers():
    detections = [
        {"label": "car", "score": 0.92, "adjusted_score": 0.92, "bbox": box(20, 40, 74, 75)},
        {"label": "car", "score": 0.90, "adjusted_score": 0.90, "bbox": box(88, 41, 144, 76)},
    ]

    fused, fused_count = distance_box_fusion(detections, 220, 160)

    assert fused_count == 0
    assert len(fused) == 2


def test_split_overmerged_fusions_restores_legacy_fused_vehicle_centers():
    detections = [
        {
            "label": "car",
            "score": 0.92,
            "adjusted_score": 0.92,
            "proposal": "distance_box_fused",
            "bbox": box(20, 40, 144, 76),
            "fused_components": [
                {"bbox": box(20, 40, 74, 75), "score": 0.92},
                {"bbox": box(88, 41, 144, 76), "score": 0.90},
            ],
        }
    ]

    split, split_count = split_overmerged_fusions(detections, 220, 160)

    assert split_count == 1
    assert len(split) == 2


def test_snap_bbox_to_edges_tightens_loose_box_on_strong_gradient():
    image = np.full((80, 120, 3), 20, dtype=np.uint8)
    image[24:54, 34:88] = 230
    loose = box(29, 19, 93, 59)

    snapped, gain = snap_bbox_to_edges(image, loose, 120, 80, search_px=6, min_gain=0.001)

    assert gain > 0.0
    assert snapped["x1"] in {33, 34}
    assert snapped["y1"] in {23, 24}
    assert snapped["x2"] in {88, 89}
    assert snapped["y2"] in {54, 55}
