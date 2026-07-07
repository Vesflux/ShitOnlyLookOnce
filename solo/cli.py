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

from solo.config import *
from solo.data.dataloader import *
from solo.core.ops import *
from solo.engine.detector import *
from solo.engine.mining import *

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate, save, and load SOLOv1 image pt weights.")
    parser.add_argument("dirpath", nargs="?", default="train", help="image directory used when generating weights")
    parser.add_argument(
        "--detect",
        nargs="+",
        help="detect objects with one or more weight JSON files; dirpath becomes image file or image directory",
    )
    parser.add_argument(
        "--backend",
        choices=["solo", "taichi", "neural"],
        default=DEFAULT_DETECTOR_BACKEND,
        help="detection backend: solo uses prototype weights; taichi uses the handcrafted Taichi detector",
    )
    parser.add_argument(
        "--train-taichi",
        action="store_true",
        help="train the OpenCV + Taichi detector from YOLO labels",
    )
    parser.add_argument(
        "--train-neural",
        action="store_true",
        help="deprecated: train the optional PyTorch detector when torch is installed",
    )
    parser.add_argument("--output", default=DEFAULT_DETECTION_RESULTS, help="detection report JSON path")
    parser.add_argument("--draw-dir", help="optional directory for images with drawn detection boxes")
    parser.add_argument("--hide-labels", action="store_true", help="hide text labels above drawn boxes")
    parser.add_argument("--eval-labels", help="optional labels directory used to score detection output")
    parser.add_argument(
        "--no-proposal-eval",
        action="store_true",
        help="skip proposal recall calculation when --eval-labels is used",
    )
    parser.add_argument("--proposal", choices=["color", "sliding", "both"], default="color", help="detection proposal mode")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD, help="minimum detection score")
    parser.add_argument("--nms-threshold", type=float, default=DEFAULT_NMS_THRESHOLD, help="NMS IoU threshold")
    parser.add_argument(
        "--nms-containment-threshold",
        type=float,
        default=DEFAULT_NMS_CONTAINMENT_THRESHOLD,
        help="suppress boxes that contain this fraction of a smaller kept box; 0 disables",
    )
    parser.add_argument(
        "--cluster-nms-center-distance",
        type=float,
        default=DEFAULT_CLUSTER_NMS_CENTER_DISTANCE,
        help="same-label boxes with nearby centers are treated as one object; 0 disables",
    )
    parser.add_argument(
        "--cluster-nms-containment",
        type=float,
        default=DEFAULT_CLUSTER_NMS_CONTAINMENT,
        help="same-label boxes with this containment are merged during cluster NMS; 0 disables",
    )
    parser.add_argument("--min-area", type=int, default=80, help="minimum objectness proposal area")
    parser.add_argument("--max-area-ratio", type=float, default=0.15, help="maximum objectness proposal area as image ratio")
    parser.add_argument("--proposal-expand", type=float, default=1.1, help="proposal bbox expansion scale")
    parser.add_argument("--min-box-size", type=int, default=6, help="minimum proposal width and height")
    parser.add_argument("--max-box-size", type=int, default=0, help="maximum proposal width or height; 0 disables")
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0, help="maximum proposal aspect ratio; 0 disables")
    parser.add_argument("--window-sizes", help="comma-separated sliding window sizes, for example 32,48,64")
    parser.add_argument(
        "--window-ratios",
        help="comma-separated sliding window ratios relative to the active input size, for example 0.025,0.04,0.06",
    )
    parser.add_argument("--stride-ratio", type=float, default=0.5, help="sliding proposal stride ratio")
    parser.add_argument("--bbox-scale", type=float, default=1.0, help="extra detection bbox scaling before matching")
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="letterbox image size used before proposal generation; 0 uses original image size",
    )
    parser.add_argument(
        "--taichi-backend",
        choices=["auto", "cpu", "cuda", "vulkan", "opengl"],
        default=DEFAULT_TAICHI_BACKEND,
        help="Taichi arch for training/detection; auto tries CUDA, Vulkan, OpenGL, then CPU",
    )
    parser.add_argument(
        "--taichi-feature-backend",
        choices=["opencv", "taichi"],
        default=DEFAULT_TAICHI_FEATURE_BACKEND,
        help="feature extractor for Taichi detector crops; use taichi only with weights trained the same way",
    )
    parser.add_argument(
        "--taichi-context-padding",
        type=float,
        default=DEFAULT_TAICHI_CONTEXT_PADDING,
        help="candidate crop padding ratio used before feature extraction",
    )
    parser.add_argument(
        "--taichi-pyramid-scales",
        default=DEFAULT_TAICHI_PYRAMID_SCALES,
        help="comma-separated image pyramid scales for Taichi anchors",
    )
    parser.add_argument(
        "--taichi-anchor-sizes",
        default=DEFAULT_TAICHI_ANCHOR_SIZES,
        help="comma-separated base anchor sizes for the Taichi detector",
    )
    parser.add_argument(
        "--taichi-anchor-ratios",
        default=DEFAULT_TAICHI_ANCHOR_RATIOS,
        help="comma-separated dynamic anchor aspect ratios, for example 1.0,1.8,2.4,3.2",
    )
    parser.add_argument(
        "--taichi-anchor-stride-ratio",
        type=float,
        default=DEFAULT_TAICHI_ANCHOR_STRIDE_RATIO,
        help="stride as a ratio of the smaller Taichi anchor side; lower values improve small-object recall",
    )
    parser.add_argument(
        "--taichi-epochs",
        type=int,
        default=DEFAULT_TAICHI_EPOCHS,
        help="training epochs for the Taichi MLP detector",
    )
    parser.add_argument(
        "--taichi-lr",
        type=float,
        default=DEFAULT_TAICHI_LEARNING_RATE,
        help="learning rate for the Taichi MLP detector",
    )
    parser.add_argument(
        "--taichi-hard-negative-rounds",
        type=int,
        default=DEFAULT_TAICHI_HARD_NEGATIVE_ROUNDS,
        help="automatic false-positive mining rounds after initial Taichi training",
    )
    parser.add_argument(
        "--taichi-negatives-per-image",
        type=int,
        default=DEFAULT_TAICHI_NEGATIVES_PER_IMAGE,
        help="background anchors sampled per training image for Taichi training",
    )
    parser.add_argument(
        "--taichi-positive-jitter",
        type=int,
        default=DEFAULT_TAICHI_POSITIVE_JITTER,
        help="augmented positive crops per labeled box for Taichi training",
    )
    parser.add_argument(
        "--taichi-max-positive-boxes-per-image",
        type=int,
        default=DEFAULT_TAICHI_MAX_POSITIVE_BOXES_PER_IMAGE,
        help="cap labeled boxes sampled per image during Taichi training; 0 uses all labels",
    )
    parser.add_argument(
        "--taichi-hidden-size",
        type=int,
        default=DEFAULT_TAICHI_HIDDEN_SIZE,
        help="hidden layer width for the Taichi MLP detector; 0 uses a linear model",
    )
    parser.add_argument(
        "--taichi-hard-mine-score-threshold",
        type=float,
        default=DEFAULT_TAICHI_HARD_MINE_SCORE_THRESHOLD,
        help="Taichi training hard-negative mining keeps unmatched detections at or above this score",
    )
    parser.add_argument(
        "--taichi-hard-negative-weight",
        type=float,
        default=DEFAULT_TAICHI_HARD_NEGATIVE_WEIGHT,
        help="sample weight for Taichi mined hard negatives",
    )
    parser.add_argument(
        "--taichi-hard-negative-max-iou",
        type=float,
        default=DEFAULT_TAICHI_HARD_NEGATIVE_MAX_IOU,
        help="Taichi mined detections with IoU at or below this value become hard negatives",
    )
    parser.add_argument(
        "--taichi-max-hard-negatives",
        type=int,
        default=DEFAULT_TAICHI_MAX_HARD_NEGATIVES,
        help="maximum Taichi hard negatives appended per mining round; 0 disables the cap",
    )
    parser.add_argument(
        "--taichi-threshold-calibration-beta",
        type=float,
        default=DEFAULT_TAICHI_THRESHOLD_CALIBRATION_BETA,
        help="F-beta used to pick the saved Taichi score threshold from the validation set",
    )
    parser.add_argument(
        "--neural-image-size",
        type=int,
        default=DEFAULT_NEURAL_IMAGE_SIZE,
        help="square OpenCV letterbox size for the built-in neural detector",
    )
    parser.add_argument(
        "--neural-model",
        choices=[
            "tiny_context",
            "mobilenet_context",
            "multiscale_context",
            "fpn_quality",
            "fpn_quality_v2",
            "fpn_quality_v3",
            "fpn_quality_v4",
        ],
        default=DEFAULT_NEURAL_MODEL,
        help="built-in neural detector architecture",
    )
    parser.add_argument(
        "--no-neural-pretrained",
        action="store_true",
        help="initialize mobilenet_context from scratch instead of ImageNet features",
    )
    parser.add_argument(
        "--neural-device",
        default=DEFAULT_NEURAL_DEVICE,
        help="torch device for neural training/detection: auto, cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--neural-epochs",
        type=int,
        default=DEFAULT_NEURAL_EPOCHS,
        help="epochs used by --train-neural",
    )
    parser.add_argument(
        "--neural-batch-size",
        type=int,
        default=DEFAULT_NEURAL_BATCH_SIZE,
        help="batch size used by --train-neural",
    )
    parser.add_argument(
        "--neural-lr",
        type=float,
        default=DEFAULT_NEURAL_LEARNING_RATE,
        help="learning rate used by --train-neural",
    )
    parser.add_argument(
        "--neural-weight-decay",
        type=float,
        default=DEFAULT_NEURAL_WEIGHT_DECAY,
        help="AdamW weight decay used by --train-neural",
    )
    parser.add_argument(
        "--neural-label",
        default="0",
        help="label written to neural detector outputs; keep 0 for one-class YOLO car datasets",
    )
    parser.add_argument(
        "--neural-max-candidates",
        type=int,
        default=DEFAULT_NEURAL_MAX_CANDIDATES,
        help="maximum heatmap cells decoded before NMS in neural detection",
    )
    parser.add_argument(
        "--neural-soft-nms-mode",
        choices=["auto", "on", "off"],
        default=DEFAULT_NEURAL_SOFT_NMS_MODE,
        help="Soft-NMS policy for neural FPN heads: auto enables it only for dense compact scenes",
    )
    parser.add_argument("--neural-seed", type=int, default=42, help="random seed used by --train-neural")
    parser.add_argument(
        "--bbox-prior-mode",
        choices=["none", "soft", "hard"],
        default=DEFAULT_BBOX_PRIOR_MODE,
        help="use trained annotation box statistics to score proposals",
    )
    parser.add_argument(
        "--match-mode",
        choices=["prototype", "nearest"],
        default=DEFAULT_MATCH_MODE,
        help="match candidates against class prototypes or every saved weight",
    )
    parser.add_argument(
        "--channel-mode",
        choices=["fixed", "adaptive", "hybrid"],
        default=DEFAULT_CHANNEL_MODE,
        help="channel weighting mode used during detection",
    )
    parser.add_argument(
        "--channel-top-k",
        type=int,
        default=DEFAULT_CHANNEL_TOP_K,
        help="keep only the best K feature channels per candidate; 0 uses all",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "taichi"],
        default=DEFAULT_ACCELERATOR,
        help="distance-matching backend; auto uses Taichi when installed and falls back to CPU/NumPy",
    )
    parser.add_argument(
        "--second-stage-rescoring",
        action="store_true",
        default=DEFAULT_SECOND_STAGE_RESCORING,
        help="enable a second-stage filter that re-scores boxes with margin/support/proposal/quality signals",
    )
    parser.add_argument("--no-second-stage-rescoring", action="store_true", help="disable second-stage rescoring")
    parser.add_argument(
        "--second-stage-threshold",
        type=float,
        default=DEFAULT_SECOND_STAGE_THRESHOLD,
        help="discard boxes below this second-stage score; 0 keeps all re-scored boxes",
    )
    parser.add_argument(
        "--second-stage-margin-weight",
        type=float,
        default=DEFAULT_SECOND_STAGE_MARGIN_WEIGHT,
        help="second-stage weight for positive-vs-negative margin",
    )
    parser.add_argument(
        "--second-stage-support-weight",
        type=float,
        default=DEFAULT_SECOND_STAGE_SUPPORT_WEIGHT,
        help="second-stage weight for positive prototype support",
    )
    parser.add_argument(
        "--second-stage-proposal-weight",
        type=float,
        default=DEFAULT_SECOND_STAGE_PROPOSAL_WEIGHT,
        help="second-stage weight for proposal-source reliability",
    )
    parser.add_argument(
        "--second-stage-quality-weight",
        type=float,
        default=DEFAULT_SECOND_STAGE_QUALITY_WEIGHT,
        help="second-stage weight for box/context/fragmentation quality",
    )
    parser.add_argument(
        "--second-stage-sky-region",
        type=float,
        default=DEFAULT_SECOND_STAGE_SKY_REGION,
        help="top normalized image region where weak anchor boxes are penalized; 0 disables",
    )
    parser.add_argument(
        "--second-stage-sky-penalty",
        type=float,
        default=DEFAULT_SECOND_STAGE_SKY_PENALTY,
        help="penalty strength for weak anchor boxes in the sky/top region",
    )
    parser.add_argument(
        "--structure-weight",
        type=float,
        default=DEFAULT_STRUCTURE_WEIGHT,
        help="blend structure score into matching; 0 disables structure scoring during detection",
    )
    parser.add_argument(
        "--box-quality-weight",
        type=float,
        default=DEFAULT_BOX_QUALITY_WEIGHT,
        help="penalize boxes with weak centered object support or border-heavy texture; 0 disables",
    )
    parser.add_argument(
        "--min-box-quality",
        type=float,
        default=DEFAULT_MIN_BOX_QUALITY,
        help="discard candidate boxes below this learned box-quality score; 0 disables",
    )
    parser.add_argument(
        "--context-weight",
        type=float,
        default=DEFAULT_CONTEXT_WEIGHT,
        help="penalize isolated partial-object crops whose surrounding context still looks object-like; 0 disables",
    )
    parser.add_argument(
        "--min-context-quality",
        type=float,
        default=DEFAULT_MIN_CONTEXT_QUALITY,
        help="discard candidate boxes below this context-completeness score; 0 disables",
    )
    parser.add_argument(
        "--context-expand",
        action="store_true",
        default=DEFAULT_CONTEXT_EXPAND,
        help="try to expand partial candidate crops when surrounding context still looks object-like",
    )
    parser.add_argument("--no-context-expand", action="store_true", help="disable context-based partial crop expansion")
    parser.add_argument(
        "--fragmentation-weight",
        type=float,
        default=DEFAULT_FRAGMENTATION_WEIGHT,
        help="penalize high-frequency texture fragments that collide with object statistics; 0 disables",
    )
    parser.add_argument(
        "--max-fragmentation",
        type=float,
        default=DEFAULT_MAX_FRAGMENTATION,
        help="discard candidate boxes above this fragmentation score; 0 disables",
    )
    parser.add_argument(
        "--proposal-dedupe-iou",
        type=float,
        default=DEFAULT_PROPOSAL_DEDUPE_IOU,
        help="IoU threshold used to remove near-duplicate proposals; 0 disables",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=DEFAULT_MAX_PROPOSALS,
        help="maximum proposals evaluated per image; 0 disables",
    )
    parser.add_argument(
        "--edge-proposals",
        action="store_true",
        default=DEFAULT_EDGE_PROPOSALS,
        help="add edge-connected-component proposals",
    )
    parser.add_argument("--no-edge-proposals", action="store_true", help="disable edge-connected-component proposals")
    parser.add_argument(
        "--body-proposals",
        action="store_true",
        default=DEFAULT_BODY_PROPOSALS,
        help="add horizontal car-body structure proposals",
    )
    parser.add_argument("--no-body-proposals", action="store_true", help="disable horizontal body proposals")
    parser.add_argument(
        "--proposal-refine",
        action="store_true",
        default=DEFAULT_PROPOSAL_REFINE,
        help="locally regress learned anchor proposals toward edge/texture boundaries before matching",
    )
    parser.add_argument("--no-proposal-refine", action="store_true", help="disable learned anchor proposal refinement")
    parser.add_argument(
        "--min-detection-width",
        type=int,
        default=DEFAULT_MIN_DETECTION_WIDTH,
        help="discard final detections narrower than this many source-image pixels; 0 disables",
    )
    parser.add_argument(
        "--min-detection-height",
        type=int,
        default=DEFAULT_MIN_DETECTION_HEIGHT,
        help="discard final detections shorter than this many source-image pixels; 0 disables",
    )
    parser.add_argument(
        "--min-detection-area",
        type=int,
        default=DEFAULT_MIN_DETECTION_AREA,
        help="discard final detections smaller than this source-image pixel area; 0 disables",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=DEFAULT_MAX_DETECTIONS,
        help="maximum final detections kept per image after NMS and size filters; 0 disables",
    )
    parser.add_argument(
        "--min-refine-edge-gain",
        type=float,
        default=DEFAULT_MIN_REFINE_EDGE_GAIN,
        help="discard final detections whose edge refinement gain is below this value; 0 disables",
    )
    parser.add_argument(
        "--refine-boxes",
        action="store_true",
        default=DEFAULT_REFINE_BOXES,
        help="locally adjust high-score boxes before final NMS",
    )
    parser.add_argument("--no-refine-boxes", action="store_true", help="disable final local box refinement")
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=DEFAULT_REFINE_TOP_K,
        help="number of high-score boxes refined per image when --refine-boxes is enabled",
    )
    parser.add_argument(
        "--refine-edge-weight",
        type=float,
        default=DEFAULT_REFINE_EDGE_WEIGHT,
        help="edge-alignment bonus used during box refinement",
    )
    parser.add_argument(
        "--refine-edge-gain",
        type=float,
        default=DEFAULT_REFINE_EDGE_GAIN,
        help="minimum edge-alignment gain needed for direct edge snapping",
    )
    parser.add_argument(
        "--refine-rematch-top-k",
        type=int,
        default=DEFAULT_REFINE_REMATCH_TOP_K,
        help="number of edge-ranked local boxes to re-match per refined detection; 0 disables re-match",
    )
    parser.add_argument(
        "--refine-require-rematch",
        action="store_true",
        default=DEFAULT_REFINE_REQUIRE_REMATCH,
        help="accept refined boxes only after a fresh crop match confirms the label and margin",
    )
    parser.add_argument("--no-refine-require-rematch", action="store_true", help="allow edge snapping without rematch confirmation")
    parser.add_argument("--calibrate-train-images", help="train image folder used for train/val calibration ratio")
    parser.add_argument("--calibrate-val-images", help="val image folder used for periodic calibration")
    parser.add_argument("--calibrate-train-labels", help="train labels folder used during calibration")
    parser.add_argument("--calibrate-val-labels", help="val labels folder used during calibration")
    parser.add_argument("--calibration-samples", type=int, default=0, help="limit val images used for calibration; 0 means all")
    parser.add_argument("--hard-mine-output", help="save a new weight file with val hard positives/negatives appended")
    parser.add_argument("--hard-mine-val-images", help="val image folder used for hard mining")
    parser.add_argument("--hard-mine-val-labels", help="val label folder used for hard mining")
    parser.add_argument(
        "--hard-mine-score-threshold",
        type=float,
        default=DEFAULT_HARD_MINE_SCORE_THRESHOLD,
        help="unmatched detections at or above this score become hard negatives",
    )
    parser.add_argument("--hard-positive-weight", type=float, default=DEFAULT_HARD_POSITIVE_WEIGHT)
    parser.add_argument("--hard-negative-weight", type=float, default=DEFAULT_HARD_NEGATIVE_WEIGHT)
    parser.add_argument(
        "--hard-negative-max-iou",
        type=float,
        default=DEFAULT_HARD_NEGATIVE_MIN_IOU,
        help="only unmatched detections at or below this IoU become hard negatives",
    )
    parser.add_argument("--max-hard-positives", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-hard-negatives", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--mined-prototype-count",
        type=int,
        default=0,
        help="raise prototype_count in mined weight files; 0 keeps the source config",
    )
    parser.add_argument(
        "--mine-rounds",
        type=int,
        default=DEFAULT_MINING_ROUNDS,
        help="run automatic multi-round hard mining after initial training; 0 disables",
    )
    parser.add_argument(
        "--mine-output-dir",
        default="outputs/solo_mining_rounds",
        help="directory for round weights, val reports, and mining summary",
    )
    parser.add_argument("--annotation-format", choices=["yolo", "labelme"], default="yolo", help="calibration annotation format")
    parser.add_argument("--self-calibrate", action="store_true", help="learn score threshold from train self-prediction")
    parser.add_argument(
        "--self-calibration-samples",
        type=int,
        default=0,
        help="limit train images used for self-calibration; 0 means all",
    )
    parser.add_argument(
        "--self-calibration-beta",
        type=float,
        default=2.0,
        help="F-beta used when learning threshold; higher favors recall",
    )
    parser.add_argument(
        "--self-calibration-min-threshold",
        type=float,
        default=0.98,
        help="lower bound for self-calibration threshold search",
    )
    parser.add_argument(
        "--val-updates-threshold",
        action="store_true",
        help="allow val calibration to override the self-learned score threshold",
    )
    parser.add_argument("--val-match-iou", type=float, default=DEFAULT_VAL_MATCH_IOU, help="IoU used to match val detections")
    parser.add_argument(
        "--val-duplicate-iou",
        type=float,
        default=DEFAULT_VAL_DUPLICATE_IOU,
        help="IoU used to flag duplicate val labels or duplicate detections",
    )
    parser.add_argument(
        "--missing-label-score",
        type=float,
        default=DEFAULT_MISSING_LABEL_SCORE,
        help="high unmatched val score treated as possible missing label",
    )
    parser.add_argument(
        "--max-calibrated-score-threshold",
        type=float,
        default=DEFAULT_MAX_CALIBRATED_SCORE_THRESHOLD,
        help="upper bound for automatic val score-threshold calibration",
    )
    parser.add_argument(
        "--calibration-score-slack",
        type=float,
        default=DEFAULT_CALIBRATION_SCORE_SLACK,
        help="amount subtracted from matched val median score during threshold calibration",
    )
    parser.add_argument(
        "--negative-penalty",
        type=float,
        default=DEFAULT_NEGATIVE_PENALTY,
        help="penalty multiplier when a candidate is closer to negative samples than positive samples",
    )
    parser.add_argument(
        "--min-negative-margin",
        type=float,
        default=DEFAULT_MIN_NEGATIVE_MARGIN,
        help="suppress only when positive_score - negative_score is below this margin",
    )
    parser.add_argument("--size", type=int, default=8, help="image is resized to 16 * size")
    parser.add_argument("--qua", type=int, default=8, help="rounding precision")
    parser.add_argument("--nab", type=float, default=DEFAULT_NAB, help="side-neighbor weight; diagonals use nab / 2")
    parser.add_argument("--pt-size", type=int, default=8, help="target pt matrix size")
    parser.add_argument("--kernel", choices=["weighted", "legacy"], default=DEFAULT_KERNEL, help="compression kernel")
    parser.add_argument("--field", choices=["local", "global"], default=DEFAULT_FIELD, help="visual field mode")
    parser.add_argument(
        "--feature-mode",
        choices=["gray", "multi"],
        default=DEFAULT_FEATURE_MODE,
        help="feature vector mode: gray pt only or multi-channel pt/statistics",
    )
    parser.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        help="comma-separated feature channels: gray,saturation,edge,hue,lab_b,local_contrast,texture,yellow_mask",
    )
    parser.add_argument(
        "--prototype-count",
        type=int,
        default=DEFAULT_PROTOTYPE_COUNT,
        help="maximum prototypes per label/negative group",
    )
    parser.add_argument(
        "--structure-mode",
        choices=["none", "grid"],
        default=DEFAULT_STRUCTURE_MODE,
        help="save additional spatial structure features in generated weights",
    )
    parser.add_argument(
        "--structure-grid",
        type=int,
        default=DEFAULT_STRUCTURE_GRID,
        help="grid size used by --structure-mode grid",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=DEFAULT_MAX_RADIUS,
        help="global visual field radius limit; 0 means full image",
    )
    parser.add_argument("--annotations", choices=["none", "yolo", "labelme"], default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--labels-dir", help="annotation directory; defaults to same folder or images->labels mapping")
    parser.add_argument("--class-names", help="YOLO class names file, one name per line")
    parser.add_argument(
        "--crop-mode",
        choices=[LETTERBOX_CROP_MODE, LEGACY_CROP_MODE],
        default=DEFAULT_CROP_MODE,
        help="object crop normalization: stretch is the quality default; letterbox preserves aspect ratio with padding",
    )
    parser.add_argument(
        "--negative-samples-per-image",
        type=int,
        default=DEFAULT_NEGATIVE_SAMPLES_PER_IMAGE,
        help="minimum background/other-object crops per annotated image; high-objectness hard negatives are sampled first",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=DEFAULT_NEGATIVE_RATIO,
        help="negative samples per image as a ratio of positive annotation boxes",
    )
    parser.add_argument(
        "--negative-iou",
        type=float,
        default=DEFAULT_NEGATIVE_IOU,
        help="maximum IoU allowed between a negative crop and any positive annotation",
    )
    parser.add_argument("--negative-seed", type=int, default=42, help="random seed for negative sample crops")
    parser.add_argument("--workers", type=int, default=1, help="parallel image workers used while generating weights")
    parser.add_argument(
        "--compact-weights",
        action="store_true",
        default=DEFAULT_COMPACT_WEIGHTS,
        help="save compact detection weights instead of every training crop",
    )
    parser.add_argument("--full-weights", action="store_true", help="save every training crop in the weight JSON")
    parser.add_argument(
        "--weight-precision",
        type=int,
        default=DEFAULT_WEIGHT_PRECISION,
        help="decimal precision used when compacting prototype vectors",
    )
    parser.add_argument(
        "--compact-exemplars",
        type=int,
        default=DEFAULT_COMPACT_EXEMPLARS,
        help="representative samples kept per compact prototype; 0 saves prototypes only",
    )
    parser.add_argument(
        "--compact-sample-limit",
        type=int,
        default=DEFAULT_COMPACT_SAMPLE_LIMIT,
        help="training samples kept in compact weights for nearest matching; 0 disables compact entries",
    )
    parser.add_argument("--save", default=DEFAULT_WEIGHT_PATH, help="output weight JSON path")
    parser.add_argument("--load", help="load weight JSON instead of generating new weights")
    parser.add_argument("--no-save", action="store_true", help="do not save generated weights")
    parser.add_argument("--no-normalize", action="store_true", help="skip per-image normalization")
    parser.add_argument("--normalize-final-only", action="store_true", help="normalize only after all compression steps")
    parser.add_argument("--no-print", action="store_true", help="do not print every pt matrix")
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.no_second_stage_rescoring:
        args.second_stage_rescoring = False
    if args.no_context_expand:
        args.context_expand = False
    if args.no_proposal_refine:
        args.proposal_refine = False
    if args.no_edge_proposals:
        args.edge_proposals = False
    if args.no_body_proposals:
        args.body_proposals = False
    if args.no_refine_boxes:
        args.refine_boxes = False
    if args.no_refine_require_rematch:
        args.refine_require_rematch = False
    if args.full_weights:
        args.compact_weights = False

    if args.load:
        payload = load_weights(args.load)
        print(f"loaded {len(payload['weights'])} weights from {Path(args.load).resolve()}")
        print(f"config: {payload.get('config', {})}")
        if not args.no_print:
            for item in payload["weights"]:
                print(item.get("pt"))
        return

    if args.train_taichi:
        from solo.taichi_detector.pipeline import train_taichi_detector

        labels_dir = args.labels_dir or args.calibrate_train_labels
        if not labels_dir:
            raise ValueError("--train-taichi requires --labels-dir or --calibrate-train-labels")
        val_images = args.calibrate_val_images
        val_labels = args.calibrate_val_labels
        result = train_taichi_detector(
            args.dirpath,
            labels_dir,
            args.save,
            val_images=val_images,
            val_labels=val_labels,
            backend=args.taichi_backend,
            feature_backend=args.taichi_feature_backend,
            context_padding=args.taichi_context_padding,
            pyramid_scales=args.taichi_pyramid_scales,
            anchor_sizes=args.taichi_anchor_sizes,
            anchor_ratios=args.taichi_anchor_ratios,
            anchor_stride_ratio=args.taichi_anchor_stride_ratio,
            epochs=args.taichi_epochs,
            learning_rate=args.taichi_lr,
            hard_negative_rounds=args.taichi_hard_negative_rounds,
            negatives_per_image=args.taichi_negatives_per_image,
            positive_jitter=args.taichi_positive_jitter,
            max_positive_boxes_per_image=args.taichi_max_positive_boxes_per_image,
            hidden_size=args.taichi_hidden_size,
            hard_mine_score_threshold=args.taichi_hard_mine_score_threshold,
            hard_negative_weight=args.taichi_hard_negative_weight,
            hard_negative_max_iou=args.taichi_hard_negative_max_iou,
            max_hard_negatives=args.taichi_max_hard_negatives,
            threshold_calibration_beta=args.taichi_threshold_calibration_beta,
            score_threshold=DEFAULT_TAICHI_SCORE_THRESHOLD if args.score_threshold == DEFAULT_SCORE_THRESHOLD else args.score_threshold,
            nms_threshold=DEFAULT_TAICHI_NMS_THRESHOLD if args.nms_threshold == DEFAULT_NMS_THRESHOLD else args.nms_threshold,
            max_detections=args.max_detections,
            match_iou=args.val_match_iou,
            seed=args.negative_seed,
            draw_dir=args.draw_dir,
            report_path=args.output if val_images and val_labels else None,
        )
        print(f"saved Taichi detector weights to {Path(result['weights_path']).resolve()}")
        if result.get("evaluation", {}).get("enabled"):
            summary = result["evaluation"]["summary"]
            print(
                f"validation: precision={summary['precision']:.4f} recall={summary['recall']:.4f} "
                f"f1={summary['f1']:.4f} tp={summary['tp']} fp={summary['fp']} fn={summary['fn']}"
            )
        return

    if args.train_neural:
        from solo.neural.train import train_neural_detector

        labels_dir = args.labels_dir or args.calibrate_train_labels
        if not labels_dir:
            raise ValueError("--train-neural requires --labels-dir or --calibrate-train-labels")
        val_images = args.calibrate_val_images
        val_labels = args.calibrate_val_labels
        result = train_neural_detector(
            args.dirpath,
            labels_dir,
            args.save,
            val_images=val_images,
            val_labels=val_labels,
            image_size=args.neural_image_size,
            epochs=args.neural_epochs,
            batch_size=args.neural_batch_size,
            learning_rate=args.neural_lr,
            weight_decay=args.neural_weight_decay,
            device=args.neural_device,
            seed=args.neural_seed,
            workers=args.workers,
            score_threshold=DEFAULT_NEURAL_SCORE_THRESHOLD if args.score_threshold == DEFAULT_SCORE_THRESHOLD else args.score_threshold,
            nms_threshold=DEFAULT_NEURAL_NMS_THRESHOLD if args.nms_threshold == DEFAULT_NMS_THRESHOLD else args.nms_threshold,
            val_match_iou=args.val_match_iou,
            max_detections=args.max_detections,
            draw_dir=args.draw_dir,
            report_path=args.output if val_images and val_labels else None,
            model_name=args.neural_model,
            pretrained=not args.no_neural_pretrained,
            class_names_path=args.class_names,
            soft_nms_mode=args.neural_soft_nms_mode,
        )
        print(f"saved neural weights to {Path(result['weights_path']).resolve()}")
        if result.get("evaluation", {}).get("enabled"):
            summary = result["evaluation"]["summary"]
            print(
                f"validation: precision={summary['precision']:.4f} recall={summary['recall']:.4f} "
                f"f1={summary['f1']:.4f} tp={summary['tp']} fp={summary['fp']} fn={summary['fn']}"
            )
        return

    if args.detect:
        window_sizes = _parse_window_sizes(args.window_sizes)
        window_ratios = _parse_window_ratios(args.window_ratios)
        if args.backend != "solo" and args.hard_mine_output:
            raise ValueError("--hard-mine-output is only supported with --backend solo")
        if args.hard_mine_output:
            val_images = args.hard_mine_val_images or args.calibrate_val_images
            val_labels = args.hard_mine_val_labels or args.calibrate_val_labels
            if not val_images or not val_labels:
                raise ValueError("--hard-mine-output requires val images and val labels")
            result = mine_val_hard_examples(
                args.detect,
                args.hard_mine_output,
                val_images,
                val_labels,
                annotations=args.annotation_format,
                class_names_path=args.class_names,
                proposal=args.proposal,
                score_threshold=max(0.0, args.score_threshold - 0.2),
                false_positive_score=args.hard_mine_score_threshold,
                nms_threshold=args.nms_threshold,
                min_area=args.min_area,
                max_area_ratio=args.max_area_ratio,
                proposal_expand=args.proposal_expand,
                min_box_size=args.min_box_size,
                max_box_size=args.max_box_size,
                max_aspect_ratio=args.max_aspect_ratio,
                window_sizes=window_sizes,
                window_ratios=window_ratios,
                stride_ratio=args.stride_ratio,
                bbox_scale=args.bbox_scale,
                val_match_iou=args.val_match_iou,
                bbox_prior_mode=args.bbox_prior_mode,
                match_mode=args.match_mode,
                channel_mode=args.channel_mode,
                channel_top_k=args.channel_top_k,
                accelerator=args.accelerator,
                second_stage_rescoring=args.second_stage_rescoring,
                second_stage_threshold=args.second_stage_threshold,
                second_stage_margin_weight=args.second_stage_margin_weight,
                second_stage_support_weight=args.second_stage_support_weight,
                second_stage_proposal_weight=args.second_stage_proposal_weight,
                second_stage_quality_weight=args.second_stage_quality_weight,
                proposal_dedupe_iou=args.proposal_dedupe_iou,
                max_proposals=args.max_proposals,
                edge_proposals=args.edge_proposals,
                body_proposals=args.body_proposals,
                proposal_refine=args.proposal_refine,
                min_detection_width=args.min_detection_width,
                min_detection_height=args.min_detection_height,
                min_detection_area=args.min_detection_area,
                max_detections=args.max_detections,
                min_refine_edge_gain=args.min_refine_edge_gain,
                refine_boxes=args.refine_boxes,
                refine_top_k=args.refine_top_k,
                refine_edge_weight=args.refine_edge_weight,
                refine_edge_gain=args.refine_edge_gain,
                refine_rematch_top_k=args.refine_rematch_top_k,
                refine_require_rematch=args.refine_require_rematch,
                nms_containment_threshold=args.nms_containment_threshold,
                cluster_nms_center_distance=args.cluster_nms_center_distance,
                cluster_nms_containment=args.cluster_nms_containment,
                structure_weight=args.structure_weight,
                box_quality_weight=args.box_quality_weight,
                min_box_quality=args.min_box_quality,
                context_weight=args.context_weight,
                min_context_quality=args.min_context_quality,
                context_expand=args.context_expand,
                fragmentation_weight=args.fragmentation_weight,
                max_fragmentation=args.max_fragmentation,
                input_size=args.input_size,
                second_stage_sky_region=args.second_stage_sky_region,
                second_stage_sky_penalty=args.second_stage_sky_penalty,
                hard_positive_weight=args.hard_positive_weight,
                hard_negative_weight=args.hard_negative_weight,
                hard_negative_max_iou=args.hard_negative_max_iou,
                mined_prototype_count=args.mined_prototype_count,
                max_hard_positives=args.max_hard_positives,
                max_hard_negatives=args.max_hard_negatives,
                compact_weights=args.compact_weights,
                weight_precision=args.weight_precision,
            )
            print(f"saved mined weights to {Path(result['output_path']).resolve()}")
            print(
                f"added {result['hard_positive_count']} hard positives and "
                f"{result['hard_negative_count']} hard negatives"
            )
            return
        result = run_detection(
            args.dirpath,
            args.detect,
            output_path=args.output,
            draw_dir=args.draw_dir,
            hide_labels=args.hide_labels,
            proposal=args.proposal,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            proposal_expand=args.proposal_expand,
            min_box_size=args.min_box_size,
            max_box_size=args.max_box_size,
            max_aspect_ratio=args.max_aspect_ratio,
            window_sizes=window_sizes,
            window_ratios=window_ratios,
            stride_ratio=args.stride_ratio,
            bbox_scale=args.bbox_scale,
            input_size=args.input_size,
            bbox_prior_mode=args.bbox_prior_mode,
            match_mode=args.match_mode,
            channel_mode=args.channel_mode,
            channel_top_k=args.channel_top_k,
            accelerator=args.accelerator,
            second_stage_rescoring=args.second_stage_rescoring,
            second_stage_threshold=args.second_stage_threshold,
            second_stage_margin_weight=args.second_stage_margin_weight,
            second_stage_support_weight=args.second_stage_support_weight,
            second_stage_proposal_weight=args.second_stage_proposal_weight,
            second_stage_quality_weight=args.second_stage_quality_weight,
            proposal_dedupe_iou=args.proposal_dedupe_iou,
            max_proposals=args.max_proposals,
            edge_proposals=args.edge_proposals,
            body_proposals=args.body_proposals,
            proposal_refine=args.proposal_refine,
            min_detection_width=args.min_detection_width,
            min_detection_height=args.min_detection_height,
            min_detection_area=args.min_detection_area,
            max_detections=args.max_detections,
            min_refine_edge_gain=args.min_refine_edge_gain,
            refine_boxes=args.refine_boxes,
            refine_top_k=args.refine_top_k,
            refine_edge_weight=args.refine_edge_weight,
            refine_edge_gain=args.refine_edge_gain,
            refine_rematch_top_k=args.refine_rematch_top_k,
            refine_require_rematch=args.refine_require_rematch,
            nms_containment_threshold=args.nms_containment_threshold,
            cluster_nms_center_distance=args.cluster_nms_center_distance,
            cluster_nms_containment=args.cluster_nms_containment,
            structure_weight=args.structure_weight,
            box_quality_weight=args.box_quality_weight,
            min_box_quality=args.min_box_quality,
            context_weight=args.context_weight,
            min_context_quality=args.min_context_quality,
            context_expand=args.context_expand,
            fragmentation_weight=args.fragmentation_weight,
            max_fragmentation=args.max_fragmentation,
            train_images=args.calibrate_train_images,
            val_images=args.calibrate_val_images,
            train_labels_dir=args.calibrate_train_labels,
            val_labels_dir=args.calibrate_val_labels,
            eval_labels_dir=args.eval_labels,
            class_names_path=args.class_names,
            annotation_format=args.annotation_format,
            calibration_samples=args.calibration_samples,
            val_match_iou=args.val_match_iou,
            val_duplicate_iou=args.val_duplicate_iou,
            missing_label_score=args.missing_label_score,
            max_calibrated_score_threshold=args.max_calibrated_score_threshold,
            calibration_score_slack=args.calibration_score_slack,
            negative_penalty=args.negative_penalty,
            min_negative_margin=args.min_negative_margin,
            second_stage_sky_region=args.second_stage_sky_region,
            second_stage_sky_penalty=args.second_stage_sky_penalty,
            self_calibrate=args.self_calibrate,
            self_calibration_samples=args.self_calibration_samples,
            self_calibration_beta=args.self_calibration_beta,
            self_calibration_min_threshold=args.self_calibration_min_threshold,
            val_updates_threshold=args.val_updates_threshold,
            evaluate_proposals=not args.no_proposal_eval,
            backend=args.backend,
            taichi_backend=args.taichi_backend,
            neural_device=args.neural_device,
            neural_image_size=args.neural_image_size,
            neural_label=args.neural_label,
            neural_max_candidates=args.neural_max_candidates,
            neural_soft_nms_mode=args.neural_soft_nms_mode,
        )
        print(f"detected {result['metadata']['total_detections']} objects in {result['metadata']['total_images']} images")
        print(f"saved detection report to {Path(result['report_path']).resolve()}")
        if args.draw_dir:
            print(f"saved drawn images to {Path(args.draw_dir).resolve()}")
        return

    if args.mine_rounds:
        val_images = args.hard_mine_val_images or args.calibrate_val_images
        val_labels = args.hard_mine_val_labels or args.calibrate_val_labels
        train_labels = args.labels_dir or args.calibrate_train_labels
        if args.annotations == "none":
            raise ValueError("--mine-rounds requires --annotations yolo or labelme")
        if not train_labels:
            raise ValueError("--mine-rounds requires --labels-dir or --calibrate-train-labels")
        if not val_images or not val_labels:
            raise ValueError("--mine-rounds requires val images/labels via hard-mine or calibrate arguments")
        result = run_mining_rounds(
            args.dirpath,
            train_labels,
            val_images,
            val_labels,
            args.mine_output_dir,
            rounds=args.mine_rounds,
            annotations=args.annotations,
            class_names_path=args.class_names,
            size=args.size,
            qua=args.qua,
            nab=args.nab,
            pt_size=args.pt_size,
            kernel=args.kernel,
            field=args.field,
            max_radius=args.max_radius,
            normalize=not args.no_normalize,
            normalize_each_step=not args.normalize_final_only,
            crop_mode=args.crop_mode,
            feature_mode=args.feature_mode,
        channels=parse_channels(args.channels),
        prototype_count=args.prototype_count,
        negative_samples_per_image=args.negative_samples_per_image,
            negative_ratio=args.negative_ratio,
            negative_iou=args.negative_iou,
            negative_seed=args.negative_seed,
            proposal=args.proposal,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            min_area=args.min_area,
            max_area_ratio=args.max_area_ratio,
            proposal_expand=args.proposal_expand,
            min_box_size=args.min_box_size,
            max_box_size=args.max_box_size,
            max_aspect_ratio=args.max_aspect_ratio,
            window_sizes=_parse_window_sizes(args.window_sizes),
            window_ratios=_parse_window_ratios(args.window_ratios),
            stride_ratio=args.stride_ratio,
            bbox_scale=args.bbox_scale,
            bbox_prior_mode=args.bbox_prior_mode,
            match_mode=args.match_mode,
            channel_mode=args.channel_mode,
            channel_top_k=args.channel_top_k,
            accelerator=args.accelerator,
            second_stage_rescoring=args.second_stage_rescoring,
            second_stage_threshold=args.second_stage_threshold,
            second_stage_margin_weight=args.second_stage_margin_weight,
            second_stage_support_weight=args.second_stage_support_weight,
            second_stage_proposal_weight=args.second_stage_proposal_weight,
            second_stage_quality_weight=args.second_stage_quality_weight,
            second_stage_sky_region=args.second_stage_sky_region,
            second_stage_sky_penalty=args.second_stage_sky_penalty,
            proposal_dedupe_iou=args.proposal_dedupe_iou,
            max_proposals=args.max_proposals,
            edge_proposals=args.edge_proposals,
            body_proposals=args.body_proposals,
            proposal_refine=args.proposal_refine,
            nms_containment_threshold=args.nms_containment_threshold,
            cluster_nms_center_distance=args.cluster_nms_center_distance,
            cluster_nms_containment=args.cluster_nms_containment,
            input_size=args.input_size,
            val_match_iou=args.val_match_iou,
            val_duplicate_iou=args.val_duplicate_iou,
            hard_mine_score_threshold=args.hard_mine_score_threshold,
            hard_positive_weight=args.hard_positive_weight,
            hard_negative_weight=args.hard_negative_weight,
            hard_negative_max_iou=args.hard_negative_max_iou,
            mined_prototype_count=args.mined_prototype_count,
            max_hard_positives=args.max_hard_positives or 300,
            max_hard_negatives=args.max_hard_negatives or 120,
            structure_mode=args.structure_mode,
            structure_grid=args.structure_grid,
            structure_weight=args.structure_weight,
            box_quality_weight=args.box_quality_weight,
            min_box_quality=args.min_box_quality,
            context_weight=args.context_weight,
            min_context_quality=args.min_context_quality,
            context_expand=args.context_expand,
            fragmentation_weight=args.fragmentation_weight,
            max_fragmentation=args.max_fragmentation,
            compact_weights=args.compact_weights,
            weight_precision=args.weight_precision,
            workers=args.workers,
        )
        print(f"saved mining summary to {Path(result['summary_path']).resolve()}")
        if result.get("best_round"):
            print(f"best weights: {Path(result['best_round']['weights']).resolve()}")
        return

    save_path = None if args.no_save else args.save
    weights = get_image_pt(
        args.dirpath,
        size=args.size,
        qua=args.qua,
        nab=args.nab,
        pt_size=args.pt_size,
        kernel=args.kernel,
        field=args.field,
        max_radius=args.max_radius,
        normalize=not args.no_normalize,
        normalize_each_step=not args.normalize_final_only,
        annotations=args.annotations,
        labels_dir=args.labels_dir,
        class_names_path=args.class_names,
        crop_mode=args.crop_mode,
        save_path=save_path,
        print_pt=not args.no_print,
        negative_samples_per_image=args.negative_samples_per_image,
        negative_ratio=args.negative_ratio,
        negative_iou=args.negative_iou,
        negative_seed=args.negative_seed,
        feature_mode=args.feature_mode,
        channels=parse_channels(args.channels),
        prototype_count=args.prototype_count,
        structure_mode=args.structure_mode,
        structure_grid=args.structure_grid,
        accelerator=args.accelerator,
        workers=args.workers,
        compact_weights=args.compact_weights,
        weight_precision=args.weight_precision,
        compact_exemplars=args.compact_exemplars,
        compact_sample_limit=args.compact_sample_limit,
    )
    print(f"generated {len(weights)} weights")
    if save_path is not None:
        print(f"saved weights to {Path(save_path).resolve()}")
__all__ = [
    '_build_parser',
    'main',
]
