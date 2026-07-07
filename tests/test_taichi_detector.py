import numpy as np
import pytest

from solo.taichi_detector.anchors import build_anchor_hardness_index, generate_pyramid_anchors, quick_anchor_hardness, quick_anchor_hardness_from_index
from solo.taichi_detector.feature_kernels import batch_crop_feature_matrix_taichi
from solo.taichi_detector.features import _batch_crop_feature_matrix_cv2, _precompute_feature_planes, batch_crop_feature_matrix_from_planes, crop_feature_vector, feature_dimension
from solo.taichi_detector.image_pyramid import build_image_pyramid, gaussian_blur_3x3
from solo.taichi_detector.model import score_mlp_taichi, standardize_features, train_mlp_taichi
from solo.taichi_detector.pipeline import _anchor_truth_stats, _calibration_rank, _extract_pyramid_candidate_features, _generate_image_pyramid_candidates, _low_contrast_penalty_from_feature, _map_level_bbox_to_source, _merge_partial_boxes, _select_exact_thresholds, _suppress_multi_object_covers, _truth_geometry


def test_taichi_feature_vector_uses_context_and_geometry():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[18:34, 20:48] = (60, 110, 180)

    vector = crop_feature_vector(image, (20, 18, 48, 34), context_padding=0.30)

    assert vector.shape == (feature_dimension(),)
    assert float(vector[-10]) > 0.0
    assert float(vector[-4]) > float(vector[-10])
    assert 0.0 < float(vector[-1]) < 1.0


def test_pyramid_anchors_include_wide_vehicle_ratios():
    anchors = generate_pyramid_anchors(
        160,
        96,
        scales=(1.0, 0.5),
        sizes=(24, 40),
        ratios=(1.0, 2.4, 3.2),
        max_anchors=500,
    )
    ratios = {round(float(anchor["anchor_ratio"]), 1) for anchor in anchors}

    assert 2.4 in ratios
    assert 3.2 in ratios
    assert any(anchor["bbox"]["width"] > anchor["bbox"]["height"] * 2 for anchor in anchors)


def test_pyramid_anchors_include_small_vehicle_cues():
    anchors = generate_pyramid_anchors(
        320,
        180,
        scales=(1.0,),
        sizes=(10, 14),
        ratios=(1.0, 1.8),
        stride_ratio=0.46,
        max_anchors=900,
    )

    assert any(anchor["bbox"]["width"] <= 16 and anchor["bbox"]["height"] <= 16 for anchor in anchors)


def test_small_truth_matching_uses_center_soft_score_when_iou_dies():
    truth = [{"bbox": {"x1": 50, "y1": 50, "x2": 60, "y2": 60, "width": 10, "height": 10}}]
    geometry = _truth_geometry(truth)
    near_miss = {"x1": 61, "y1": 50, "x2": 75, "y2": 64, "width": 14, "height": 14}

    score, centers_inside, truth_area = _anchor_truth_stats(near_miss, geometry)

    assert centers_inside == 0
    assert truth_area == pytest.approx(100.0)
    assert score > 0.24


def test_image_pyramid_builds_real_downsampled_levels_on_cpu():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(120, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(80, dtype=np.uint8)[:, None]

    levels = build_image_pyramid(image, scales=(1.0, 0.5, 0.25), backend="cpu", min_side=8)

    assert [level["image"].shape[:2] for level in levels] == [(80, 120), (40, 60), (20, 30)]
    assert levels[1]["scale_x"] == pytest.approx(0.5)
    assert levels[2]["scale_y"] == pytest.approx(0.25)
    assert not np.array_equal(levels[1]["image"], image[:40, :60])


def test_gaussian_blur_softens_high_frequency_before_downsample():
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[4, 4] = 255

    blurred = gaussian_blur_3x3(image)

    assert 0 < int(blurred[4, 4, 0]) < 255
    assert int(blurred[4, 3, 0]) > 0


def test_image_pyramid_candidates_map_level_boxes_back_to_source():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    levels, candidates = _generate_image_pyramid_candidates(
        image,
        scales=(1.0, 0.25),
        sizes=(24,),
        ratios=(2.4,),
        stride_ratio=0.46,
        backend="cpu",
        max_anchors=200,
    )

    assert [round(level["scale"], 2) for level in levels] == [1.0, 0.25]
    coarse = [candidate for candidate in candidates if candidate["level"]["scale"] == 0.25]
    assert coarse
    assert any((bbox := candidate["source_bbox"])[2] - bbox[0] > 80 for candidate in coarse)


def test_pyramid_candidate_features_are_flattened_across_levels():
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    image[18:42, 24:70] = (80, 140, 220)
    _levels, candidates = _generate_image_pyramid_candidates(
        image,
        scales=(1.0, 0.5),
        sizes=(18,),
        ratios=(1.0, 1.8),
        stride_ratio=0.60,
        backend="cpu",
        max_anchors=80,
    )

    matrix, flattened = _extract_pyramid_candidate_features(
        candidates,
        context_padding=0.30,
        backend="cpu",
        feature_backend="opencv",
    )

    assert flattened
    assert len(flattened) == matrix.shape[0]
    assert matrix.shape[1] == feature_dimension()
    assert {candidate["level_index"] for candidate in flattened} <= {0, 1}


def test_level_bbox_mapping_keeps_float_precision_until_rendering():
    mapped = _map_level_bbox_to_source(
        {"x1": 3.25, "y1": 4.5, "x2": 25.75, "y2": 14.25},
        source_width=200,
        source_height=100,
        scale_x=0.25,
        scale_y=0.5,
    )

    assert mapped == pytest.approx((13.0, 9.0, 103.0, 28.5))


def test_low_contrast_feature_window_is_suppressed():
    image = np.full((48, 64, 3), 128, dtype=np.uint8)

    vector = crop_feature_vector(image, (10, 10, 44, 34), context_padding=0.30)

    visual_length = 2 * 5 * 14 * 14
    assert np.count_nonzero(vector[:visual_length]) == 0
    assert _low_contrast_penalty_from_feature(vector) < 0.1


def test_experimental_taichi_features_keep_opencv_layout():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(96, dtype=np.uint8)[None, :] * 2
    image[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None] * 3
    image[20:42, 25:70, 2] = 220
    bboxes = [(24, 18, 72, 44), (5, 5, 40, 30), (50, 25, 90, 60)]
    planes = _precompute_feature_planes(image)

    expected = _batch_crop_feature_matrix_cv2(image, bboxes, 0.30, 14)
    actual = batch_crop_feature_matrix_taichi(planes, bboxes, 0.30, backend="auto")
    if actual is None:
        pytest.skip("Taichi GPU backend is not available")

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual[:, -10:], expected[:, -10:], rtol=1e-6, atol=1e-6)
    assert float(np.mean(np.abs(actual - expected))) < 0.01


def test_feature_matrix_from_precomputed_planes_matches_opencv_path():
    image = np.zeros((48, 72, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(72, dtype=np.uint8)[None, :] * 2
    image[12:36, 18:58] = (90, 130, 210)
    bboxes = [(18, 12, 58, 36), (4, 8, 30, 24)]
    planes = _precompute_feature_planes(image)

    expected = _batch_crop_feature_matrix_cv2(image, bboxes, 0.30, 14)
    actual = batch_crop_feature_matrix_from_planes(planes, bboxes, 0.30, 14)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_integral_anchor_hardness_tracks_legacy_score():
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    image[15:35, 20:58] = (50, 170, 220)
    bbox = {"x1": 18, "y1": 12, "x2": 60, "y2": 38, "width": 42, "height": 26}
    planes = _precompute_feature_planes(image)

    legacy = quick_anchor_hardness(image, bbox)
    fast = quick_anchor_hardness_from_index(build_anchor_hardness_index(planes), bbox)

    assert fast > 0.0
    assert abs(fast - legacy) < 0.20


def test_mlp_training_learns_simple_separable_samples_on_cpu():
    features = np.asarray(
        [
            [1.0, 1.0, 0.9],
            [0.9, 0.8, 1.0],
            [-1.0, -0.8, -1.0],
            [-0.9, -1.0, -0.7],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    weights = np.ones((4,), dtype=np.float32)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-4)

    network, report = train_mlp_taichi(
        standardize_features(features, mean, std),
        labels,
        weights,
        epochs=80,
        learning_rate=0.08,
        hidden_size=4,
        backend="cpu",
    )
    scores = score_mlp_taichi(standardize_features(features, mean, std), network, backend="cpu")

    assert report["samples"] == 4
    assert scores[:2].min() > scores[2:].max()

def test_taichi_multi_object_cover_suppression_removes_wide_cover():
    detections = [
        {
            "label": "0",
            "score": 0.96,
            "bbox": {"x1": 20, "y1": 32, "x2": 170, "y2": 76, "width": 150, "height": 44},
        },
        {
            "label": "0",
            "score": 0.94,
            "bbox": {"x1": 22, "y1": 34, "x2": 72, "y2": 74, "width": 50, "height": 40},
        },
        {
            "label": "0",
            "score": 0.93,
            "bbox": {"x1": 116, "y1": 34, "x2": 168, "y2": 74, "width": 52, "height": 40},
        },
    ]

    kept, suppressed = _suppress_multi_object_covers(detections)

    assert suppressed == 1
    assert len(kept) == 2
    assert all(item["bbox"]["width"] < 100 for item in kept)


def test_taichi_distance_merge_fuses_low_iou_vehicle_fragments():
    detections = [
        {
            "label": "0",
            "score": 0.93,
            "adjusted_score": 0.93,
            "bbox": {"x1": 30, "y1": 52, "x2": 92, "y2": 86, "width": 62, "height": 34},
        },
        {
            "label": "0",
            "score": 0.91,
            "adjusted_score": 0.91,
            "bbox": {"x1": 104, "y1": 53, "x2": 168, "y2": 88, "width": 64, "height": 35},
        },
    ]

    merged, merged_count = _merge_partial_boxes(detections, 220, 120)

    assert merged_count == 1
    assert len(merged) == 1
    assert merged[0]["bbox"]["x1"] == 30
    assert merged[0]["bbox"]["x2"] == 168
    assert merged[0]["proposal"] == "taichi_anchor_fused"


def test_taichi_distance_merge_preserves_clear_adjacent_vehicle_instances():
    detections = [
        {
            "label": "0",
            "score": 0.93,
            "adjusted_score": 0.93,
            "bbox": {"x1": 30, "y1": 52, "x2": 92, "y2": 86, "width": 62, "height": 34},
        },
        {
            "label": "0",
            "score": 0.91,
            "adjusted_score": 0.91,
            "bbox": {"x1": 110, "y1": 53, "x2": 174, "y2": 88, "width": 64, "height": 35},
        },
    ]

    merged, merged_count = _merge_partial_boxes(detections, 240, 120)

    assert merged_count == 0
    assert len(merged) == 2


def test_calibration_rank_does_not_prefer_all_zero_high_threshold():
    zero_high = {"threshold": 0.995, "fbeta": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "detections": 0}
    zero_low = {"threshold": 0.10, "fbeta": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "detections": 12}
    positive = {"threshold": 0.30, "fbeta": 0.01, "recall": 0.02, "tp": 1, "fp": 20, "detections": 21}

    assert _calibration_rank(zero_low) > _calibration_rank(zero_high)
    assert _calibration_rank(positive) > _calibration_rank(zero_low)


def test_select_exact_thresholds_keeps_low_candidates_when_sweep_is_flat():
    thresholds = _select_exact_thresholds(
        [
            {"threshold": 0.995, "fbeta": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "detections": 0},
            {"threshold": 0.10, "fbeta": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "detections": 8},
        ],
        max_thresholds=4,
        base_threshold=0.82,
    )

    assert min(thresholds) <= 0.10
    assert 0.995 not in thresholds[:1]
