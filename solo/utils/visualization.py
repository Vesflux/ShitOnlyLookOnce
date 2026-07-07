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

import cv2

from solo.config import *
from solo.utils.bbox import *

def draw_detections(
    image_path: str | Path,
    detections: list[dict[str, Any]],
    output_path: str | Path,
    hide_labels: bool = False,
) -> Path:
    path = Path(image_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    for detection in detections:
        x1, y1, x2, y2 = _bbox_tuple(detection["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 255, 120), 2, lineType=cv2.LINE_AA)
        if hide_labels:
            continue
        label = f"{detection['label']} {float(detection['score']):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = max(text_height + baseline + 4, y1)
        cv2.rectangle(
            image,
            (x1, label_y - text_height - baseline - 4),
            (x1 + text_width + 7, label_y + 2),
            (40, 255, 120),
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), image)
    return out_path

def save_detection_report(
    results: list[dict[str, Any]],
    output_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
__all__ = [
    'draw_detections',
    'save_detection_report',
]
