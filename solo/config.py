from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_WEIGHT_PATH = "solo_weights.json"
DEFAULT_SCORE_THRESHOLD = 0.76
DEFAULT_NMS_THRESHOLD = 0.30
DEFAULT_KERNEL = "weighted"
DEFAULT_FIELD = "global"
DEFAULT_NAB = 0.25
DEFAULT_MAX_RADIUS = 0
DEFAULT_ANNOTATIONS = "none"
LETTERBOX_CROP_MODE = "letterbox"
DEFAULT_CROP_MODE = "stretch"
LEGACY_CROP_MODE = "stretch"
LEGACY_FEATURE_STATS_VERSION = "geometry_v1"
DEFAULT_FEATURE_STATS_VERSION = "geometry_v2"
DEFAULT_STATS_WEIGHT = 0.12
DEFAULT_DETECTION_RESULTS = "solo_detection_results.txt"
DEFAULT_NEGATIVE_LABEL = "__negative__"
DEFAULT_COMPACT_WEIGHTS = False
DEFAULT_WEIGHT_PRECISION = 5
DEFAULT_COMPACT_SAMPLE_LIMIT = 0
DEFAULT_VAL_MATCH_IOU = 0.25
DEFAULT_VAL_DUPLICATE_IOU = 0.85
DEFAULT_MISSING_LABEL_SCORE = 0.995
DEFAULT_NEGATIVE_PENALTY = 1.0
DEFAULT_MIN_NEGATIVE_MARGIN = 0.005
DEFAULT_MAX_CALIBRATED_SCORE_THRESHOLD = 0.9985
DEFAULT_CALIBRATION_SCORE_SLACK = 0.0005
DEFAULT_FEATURE_MODE = "multi"
DEFAULT_BBOX_PRIOR_MODE = "soft"
DEFAULT_MATCH_MODE = "nearest"
DEFAULT_CHANNEL_MODE = "hybrid"
DEFAULT_CHANNEL_TOP_K = 4
DEFAULT_ACCELERATOR = "auto"
DEFAULT_CHANNELS = ["gray", "saturation", "edge", "hue", "lab_b", "local_contrast", "texture"]
SUPPORTED_CHANNELS = DEFAULT_CHANNELS + ["yellow_mask"]
DEFAULT_STRUCTURE_MODE = "grid"
DEFAULT_STRUCTURE_GRID = 4
DEFAULT_STRUCTURE_WEIGHT = 0.08
DEFAULT_BOX_QUALITY_WEIGHT = 0.08
DEFAULT_MIN_BOX_QUALITY = 0.0
DEFAULT_CONTEXT_WEIGHT = 0.04
DEFAULT_MIN_CONTEXT_QUALITY = 0.0
DEFAULT_CONTEXT_EXPAND = True
DEFAULT_FRAGMENTATION_WEIGHT = 0.04
DEFAULT_MAX_FRAGMENTATION = 0.0
DEFAULT_PROTOTYPE_COUNT = 16
DEFAULT_COMPACT_EXEMPLARS = 0
DEFAULT_PROPOSAL_DEDUPE_IOU = 0.86
DEFAULT_MAX_PROPOSALS = 220
DEFAULT_EDGE_PROPOSALS = True
DEFAULT_BODY_PROPOSALS = True
DEFAULT_PROPOSAL_REFINE = False
DEFAULT_SECOND_STAGE_RESCORING = True
DEFAULT_SECOND_STAGE_THRESHOLD = 0.42
DEFAULT_SECOND_STAGE_MARGIN_WEIGHT = 0.28
DEFAULT_SECOND_STAGE_SUPPORT_WEIGHT = 0.20
DEFAULT_SECOND_STAGE_PROPOSAL_WEIGHT = 0.12
DEFAULT_SECOND_STAGE_QUALITY_WEIGHT = 0.12
DEFAULT_SECOND_STAGE_SKY_REGION = 0.38
DEFAULT_SECOND_STAGE_SKY_PENALTY = 0.55
DEFAULT_SECOND_STAGE_MIN_ANCHOR_MARGIN = 0.025
DEFAULT_SECOND_STAGE_MIN_ANCHOR_QUALITY = 0.42
DEFAULT_CLUSTER_NMS_CENTER_DISTANCE = 0.42
DEFAULT_CLUSTER_NMS_CONTAINMENT = 0.60
DEFAULT_NMS_DIOU_THRESHOLD = 0.30
DEFAULT_MIN_DETECTION_WIDTH = 8
DEFAULT_MIN_DETECTION_HEIGHT = 6
DEFAULT_MIN_DETECTION_AREA = 80
DEFAULT_MAX_DETECTIONS = 20
DEFAULT_MIN_REFINE_EDGE_GAIN = 0.0
DEFAULT_NMS_CONTAINMENT_THRESHOLD = 0.75
DEFAULT_REFINE_BOXES = True
DEFAULT_REFINE_TOP_K = 16
DEFAULT_REFINE_EDGE_WEIGHT = 0.025
DEFAULT_REFINE_EDGE_GAIN = 0.03
DEFAULT_REFINE_REMATCH_TOP_K = 3
DEFAULT_REFINE_REQUIRE_REMATCH = True
DEFAULT_HARD_POSITIVE_WEIGHT = 2.0
DEFAULT_HARD_NEGATIVE_WEIGHT = 1.5
DEFAULT_HARD_MINE_SCORE_THRESHOLD = 0.9
DEFAULT_HARD_NEGATIVE_MIN_IOU = 0.05
DEFAULT_NEGATIVE_SAMPLES_PER_IMAGE = 3
DEFAULT_NEGATIVE_RATIO = 0.20
DEFAULT_NEGATIVE_IOU = 0.03
DEFAULT_MINING_ROUNDS = 0
DEFAULT_INPUT_SIZE = 640
DEFAULT_OBJECTNESS_WORK_SIZE = 320
DEFAULT_DETECTOR_BACKEND = "solo"
DEFAULT_TAICHI_BACKEND = "auto"
DEFAULT_TAICHI_FEATURE_BACKEND = "opencv"
DEFAULT_TAICHI_CONTEXT_PADDING = 0.30
DEFAULT_TAICHI_PYRAMID_SCALES = "1.0,0.75,0.5,0.35"
DEFAULT_TAICHI_ANCHOR_SIZES = "18,24,32,44,60,82,112,152"
DEFAULT_TAICHI_ANCHOR_RATIOS = "0.65,0.85,1.0,1.2,1.35,1.8,2.4,3.2"
DEFAULT_TAICHI_ANCHOR_STRIDE_RATIO = 0.58
DEFAULT_TAICHI_EPOCHS = 280
DEFAULT_TAICHI_LEARNING_RATE = 0.08
DEFAULT_TAICHI_HARD_NEGATIVE_ROUNDS = 1
DEFAULT_TAICHI_NEGATIVES_PER_IMAGE = 140
DEFAULT_TAICHI_POSITIVE_JITTER = 7
DEFAULT_TAICHI_MAX_POSITIVE_BOXES_PER_IMAGE = 0
DEFAULT_TAICHI_SCORE_THRESHOLD = 0.90
DEFAULT_TAICHI_NMS_THRESHOLD = 0.28
DEFAULT_TAICHI_HIDDEN_SIZE = 18
DEFAULT_TAICHI_HARD_MINE_SCORE_THRESHOLD = 0.86
DEFAULT_TAICHI_HARD_NEGATIVE_WEIGHT = 1.0
DEFAULT_TAICHI_HARD_NEGATIVE_MAX_IOU = 0.05
DEFAULT_TAICHI_MAX_HARD_NEGATIVES = 180
DEFAULT_TAICHI_THRESHOLD_CALIBRATION_BETA = 1.15
DEFAULT_NEURAL_IMAGE_SIZE = 640
DEFAULT_NEURAL_EPOCHS = 80
DEFAULT_NEURAL_BATCH_SIZE = 8
DEFAULT_NEURAL_LEARNING_RATE = 0.002
DEFAULT_NEURAL_WEIGHT_DECAY = 0.0001
DEFAULT_NEURAL_SCORE_THRESHOLD = 0.28
DEFAULT_NEURAL_NMS_THRESHOLD = 0.45
DEFAULT_NEURAL_MAX_CANDIDATES = 200
DEFAULT_NEURAL_DEVICE = "auto"
DEFAULT_NEURAL_MODEL = "mobilenet_context"
DEFAULT_NEURAL_SOFT_NMS_MODE = "auto"
MAX_CORRECTION_REPORTS = 50

def _validate_config(size: int, qua: int, nab: float, pt_size: int, kernel: str, field: str, max_radius: int) -> None:
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
    if field not in {"local", "global"}:
        raise ValueError("field must be 'local' or 'global'")
    if max_radius < 0:
        raise ValueError("max_radius must be 0 or greater")
    if _kernel_denominator(nab, kernel) == 0:
        raise ValueError("nab cannot make the kernel denominator 0")

def _validate_annotation_config(annotations: str, crop_mode: str) -> None:
    if annotations not in {"none", "yolo", "labelme"}:
        raise ValueError("annotations must be 'none', 'yolo', or 'labelme'")
    _validate_crop_mode(crop_mode)

def _validate_crop_mode(crop_mode: str) -> None:
    if crop_mode not in {LETTERBOX_CROP_MODE, LEGACY_CROP_MODE}:
        raise ValueError("crop_mode must be 'letterbox' or 'stretch'")

def _validate_feature_mode(feature_mode: str) -> None:
    if feature_mode not in {"gray", "multi"}:
        raise ValueError("feature_mode must be 'gray' or 'multi'")

def _validate_bbox_prior_mode(bbox_prior_mode: str) -> None:
    if bbox_prior_mode not in {"none", "soft", "hard"}:
        raise ValueError("bbox_prior_mode must be 'none', 'soft', or 'hard'")

def _validate_match_mode(match_mode: str) -> None:
    if match_mode not in {"prototype", "nearest"}:
        raise ValueError("match_mode must be 'prototype' or 'nearest'")

def _validate_channel_mode(channel_mode: str) -> None:
    if channel_mode not in {"fixed", "adaptive", "hybrid"}:
        raise ValueError("channel_mode must be 'fixed', 'adaptive', or 'hybrid'")

def _validate_accelerator(accelerator: str) -> None:
    if accelerator not in {"auto", "cpu", "taichi"}:
        raise ValueError("accelerator must be 'auto', 'cpu', or 'taichi'")

def _validate_detector_backend(backend: str) -> None:
    if backend not in {"solo", "taichi", "neural"}:
        raise ValueError("backend must be 'solo', 'taichi', or 'neural'")

def _validate_neural_model(model_name: str) -> None:
    if model_name not in {
        "tiny_context",
        "mobilenet_context",
        "multiscale_context",
        "fpn_quality",
        "fpn_quality_v2",
        "fpn_quality_v3",
        "fpn_quality_v4",
    }:
        raise ValueError(
            "neural model must be 'tiny_context', 'mobilenet_context', 'multiscale_context', 'fpn_quality', "
            "'fpn_quality_v2', 'fpn_quality_v3', or 'fpn_quality_v4'"
        )

def _validate_neural_soft_nms_mode(mode: str) -> None:
    if mode not in {"auto", "on", "off"}:
        raise ValueError("neural soft NMS mode must be 'auto', 'on', or 'off'")

def _validate_structure_mode(structure_mode: str) -> None:
    if structure_mode not in {"none", "grid"}:
        raise ValueError("structure_mode must be 'none' or 'grid'")

def parse_channels(channels: str | list[str] | None = None) -> list[str]:
    if channels is None:
        return list(DEFAULT_CHANNELS)
    if isinstance(channels, str):
        parsed = [item.strip() for item in channels.split(",") if item.strip()]
    else:
        parsed = [str(item).strip() for item in channels if str(item).strip()]
    if not parsed:
        raise ValueError("at least one channel is required")
    invalid = [item for item in parsed if item not in SUPPORTED_CHANNELS]
    if invalid:
        raise ValueError(f"unsupported channels: {', '.join(invalid)}")
    return list(dict.fromkeys(parsed))

def _validate_pt(pt: list[list[float]], pt_size: int, nab: float, kernel: str, field: str, max_radius: int) -> None:
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
    if field not in {"local", "global"}:
        raise ValueError("field must be 'local' or 'global'")
    if max_radius < 0:
        raise ValueError("max_radius must be 0 or greater")
    if _kernel_denominator(nab, kernel) == 0:
        raise ValueError("nab cannot make the kernel denominator 0")

def _kernel_denominator(nab: float, kernel: str) -> float:
    if kernel == "legacy":
        return nab * 8 + 1
    side_weight = nab
    diagonal_weight = nab / 2
    return 1 + side_weight * 4 + diagonal_weight * 4

def _elapsed(started: float) -> str:
    return f"{time.time() - started:.1f}s"
__all__ = [
    'IMAGE_SUFFIXES',
    'DEFAULT_WEIGHT_PATH',
    'DEFAULT_SCORE_THRESHOLD',
    'DEFAULT_NMS_THRESHOLD',
    'DEFAULT_KERNEL',
    'DEFAULT_FIELD',
    'DEFAULT_NAB',
    'DEFAULT_MAX_RADIUS',
    'DEFAULT_ANNOTATIONS',
    'DEFAULT_CROP_MODE',
    'LETTERBOX_CROP_MODE',
    'LEGACY_CROP_MODE',
    'LEGACY_FEATURE_STATS_VERSION',
    'DEFAULT_FEATURE_STATS_VERSION',
    'DEFAULT_STATS_WEIGHT',
    'DEFAULT_DETECTION_RESULTS',
    'DEFAULT_NEGATIVE_LABEL',
    'DEFAULT_COMPACT_WEIGHTS',
    'DEFAULT_WEIGHT_PRECISION',
    'DEFAULT_COMPACT_SAMPLE_LIMIT',
    'DEFAULT_VAL_MATCH_IOU',
    'DEFAULT_VAL_DUPLICATE_IOU',
    'DEFAULT_MISSING_LABEL_SCORE',
    'DEFAULT_NEGATIVE_PENALTY',
    'DEFAULT_MIN_NEGATIVE_MARGIN',
    'DEFAULT_MAX_CALIBRATED_SCORE_THRESHOLD',
    'DEFAULT_CALIBRATION_SCORE_SLACK',
    'DEFAULT_FEATURE_MODE',
    'DEFAULT_BBOX_PRIOR_MODE',
    'DEFAULT_MATCH_MODE',
    'DEFAULT_CHANNEL_MODE',
    'DEFAULT_CHANNEL_TOP_K',
    'DEFAULT_ACCELERATOR',
    'DEFAULT_CHANNELS',
    'SUPPORTED_CHANNELS',
    'DEFAULT_STRUCTURE_MODE',
    'DEFAULT_STRUCTURE_GRID',
    'DEFAULT_STRUCTURE_WEIGHT',
    'DEFAULT_BOX_QUALITY_WEIGHT',
    'DEFAULT_MIN_BOX_QUALITY',
    'DEFAULT_CONTEXT_WEIGHT',
    'DEFAULT_MIN_CONTEXT_QUALITY',
    'DEFAULT_CONTEXT_EXPAND',
    'DEFAULT_FRAGMENTATION_WEIGHT',
    'DEFAULT_MAX_FRAGMENTATION',
    'DEFAULT_PROTOTYPE_COUNT',
    'DEFAULT_COMPACT_EXEMPLARS',
    'DEFAULT_PROPOSAL_DEDUPE_IOU',
    'DEFAULT_MAX_PROPOSALS',
    'DEFAULT_EDGE_PROPOSALS',
    'DEFAULT_BODY_PROPOSALS',
    'DEFAULT_PROPOSAL_REFINE',
    'DEFAULT_SECOND_STAGE_RESCORING',
    'DEFAULT_SECOND_STAGE_THRESHOLD',
    'DEFAULT_SECOND_STAGE_MARGIN_WEIGHT',
    'DEFAULT_SECOND_STAGE_SUPPORT_WEIGHT',
    'DEFAULT_SECOND_STAGE_PROPOSAL_WEIGHT',
    'DEFAULT_SECOND_STAGE_QUALITY_WEIGHT',
    'DEFAULT_SECOND_STAGE_SKY_REGION',
    'DEFAULT_SECOND_STAGE_SKY_PENALTY',
    'DEFAULT_SECOND_STAGE_MIN_ANCHOR_MARGIN',
    'DEFAULT_SECOND_STAGE_MIN_ANCHOR_QUALITY',
    'DEFAULT_CLUSTER_NMS_CENTER_DISTANCE',
    'DEFAULT_CLUSTER_NMS_CONTAINMENT',
    'DEFAULT_NMS_DIOU_THRESHOLD',
    'DEFAULT_MIN_DETECTION_WIDTH',
    'DEFAULT_MIN_DETECTION_HEIGHT',
    'DEFAULT_MIN_DETECTION_AREA',
    'DEFAULT_MAX_DETECTIONS',
    'DEFAULT_MIN_REFINE_EDGE_GAIN',
    'DEFAULT_NMS_CONTAINMENT_THRESHOLD',
    'DEFAULT_REFINE_BOXES',
    'DEFAULT_REFINE_TOP_K',
    'DEFAULT_REFINE_EDGE_WEIGHT',
    'DEFAULT_REFINE_EDGE_GAIN',
    'DEFAULT_REFINE_REMATCH_TOP_K',
    'DEFAULT_REFINE_REQUIRE_REMATCH',
    'DEFAULT_HARD_POSITIVE_WEIGHT',
    'DEFAULT_HARD_NEGATIVE_WEIGHT',
    'DEFAULT_HARD_MINE_SCORE_THRESHOLD',
    'DEFAULT_HARD_NEGATIVE_MIN_IOU',
    'DEFAULT_NEGATIVE_SAMPLES_PER_IMAGE',
    'DEFAULT_NEGATIVE_RATIO',
    'DEFAULT_NEGATIVE_IOU',
    'DEFAULT_MINING_ROUNDS',
    'DEFAULT_INPUT_SIZE',
    'DEFAULT_OBJECTNESS_WORK_SIZE',
    'DEFAULT_DETECTOR_BACKEND',
    'DEFAULT_TAICHI_BACKEND',
    'DEFAULT_TAICHI_FEATURE_BACKEND',
    'DEFAULT_TAICHI_CONTEXT_PADDING',
    'DEFAULT_TAICHI_PYRAMID_SCALES',
    'DEFAULT_TAICHI_ANCHOR_SIZES',
    'DEFAULT_TAICHI_ANCHOR_RATIOS',
    'DEFAULT_TAICHI_ANCHOR_STRIDE_RATIO',
    'DEFAULT_TAICHI_EPOCHS',
    'DEFAULT_TAICHI_LEARNING_RATE',
    'DEFAULT_TAICHI_HARD_NEGATIVE_ROUNDS',
    'DEFAULT_TAICHI_NEGATIVES_PER_IMAGE',
    'DEFAULT_TAICHI_POSITIVE_JITTER',
    'DEFAULT_TAICHI_MAX_POSITIVE_BOXES_PER_IMAGE',
    'DEFAULT_TAICHI_SCORE_THRESHOLD',
    'DEFAULT_TAICHI_NMS_THRESHOLD',
    'DEFAULT_TAICHI_HIDDEN_SIZE',
    'DEFAULT_TAICHI_HARD_MINE_SCORE_THRESHOLD',
    'DEFAULT_TAICHI_HARD_NEGATIVE_WEIGHT',
    'DEFAULT_TAICHI_HARD_NEGATIVE_MAX_IOU',
    'DEFAULT_TAICHI_MAX_HARD_NEGATIVES',
    'DEFAULT_TAICHI_THRESHOLD_CALIBRATION_BETA',
    'DEFAULT_NEURAL_IMAGE_SIZE',
    'DEFAULT_NEURAL_EPOCHS',
    'DEFAULT_NEURAL_BATCH_SIZE',
    'DEFAULT_NEURAL_LEARNING_RATE',
    'DEFAULT_NEURAL_WEIGHT_DECAY',
    'DEFAULT_NEURAL_SCORE_THRESHOLD',
    'DEFAULT_NEURAL_NMS_THRESHOLD',
    'DEFAULT_NEURAL_MAX_CANDIDATES',
    'DEFAULT_NEURAL_DEVICE',
    'DEFAULT_NEURAL_MODEL',
    'DEFAULT_NEURAL_SOFT_NMS_MODE',
    'MAX_CORRECTION_REPORTS',
    '_validate_config',
    '_validate_annotation_config',
    '_validate_crop_mode',
    '_validate_feature_mode',
    '_validate_bbox_prior_mode',
    '_validate_match_mode',
    '_validate_channel_mode',
    '_validate_accelerator',
    '_validate_detector_backend',
    '_validate_neural_model',
    '_validate_neural_soft_nms_mode',
    '_validate_structure_mode',
    'parse_channels',
    '_validate_pt',
    '_kernel_denominator',
    '_elapsed',
]
