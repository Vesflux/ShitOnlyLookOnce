# ShitOnlyLookOnce (SOLO)

SOLO is a tiny image-to-weight experiment. It reads images, converts them into grayscale point matrices, repeatedly compresses them with a decaying visual field, normalizes the result, and saves the generated pt weights as JSON.

The current default kernel follows the later `letptsmall` idea:

```text
0.125  0.25  0.125
0.25   1.00  0.25
0.125  0.25  0.125
```

It is not a neural-network framework. It is a compact, hackable SOLO-style feature/weight generator for quick image experiments.

## Features

- Convert image folders into pt weight matrices.
- Use local 3x3 or global visual-field compression.
- Train from YOLO `.txt` boxes or LabelMe `.json` shapes.
- Crop annotated objects and stretch them into a normalized SOLO input.
- Detect objects with one or more SOLO weight files.
- Save detection result JSON/TXT and optionally draw boxes on images.
- Calibrate detection thresholds and box scaling from train/val annotation ratios.
- Save and load generated weights as JSON.
- Use the newer weighted compression kernel by default.
- Keep the old equal-neighbor behavior with `--kernel legacy`.
- Normalize after each compression step to preserve contrast in the generated matrix.
- Use as a script or import as a small Python module.

## Installation

```bash
git clone https://github.com/Vesflux/ShitOnlyLookOnce.git
cd ShitOnlyLookOnce
python -m pip install -r requirements.txt
```

## Quick Start

Put training images in a folder:

```text
train/
  apple.png
  sample.jpg
  icon.webp
```

Generate and save SOLO weights:

```bash
python solo.py train --size 8 --pt-size 8 --field global --save solo_weights.json
```

Load saved weights:

```bash
python solo.py --load solo_weights.json
```

Run without printing every matrix:

```bash
python solo.py train --size 8 --field global --save solo_weights.json --no-print
```

## Operation Examples

Generate 8x8 pt weights from images resized to 128x128:

```bash
python solo.py train --size 8 --pt-size 8 --field global --save weights/solo_8x8.json
```

Generate 4x4 pt weights with lower precision:

```bash
python solo.py train --size 4 --pt-size 4 --qua 4 --field global --save weights/solo_4x4.json
```

Use the original local 3x3 visual field:

```bash
python solo.py train --size 8 --pt-size 8 --field local --save weights/local_3x3.json
```

Use global visual field but cap the radius for speed:

```bash
python solo.py train --size 8 --pt-size 8 --field global --max-radius 12 --save weights/global_r12.json
```

Use the legacy kernel from early SOLO experiments:

```bash
python solo.py train --size 8 --pt-size 8 --kernel legacy --field local --nab 0.5 --save weights/legacy.json
```

Normalize only after all compression steps:

```bash
python solo.py train --size 8 --pt-size 8 --normalize-final-only --save weights/final_norm.json
```

Generate weights without saving:

```bash
python solo.py train --size 8 --no-save
```

## Annotation Training

SOLO can generate pt weights from annotated objects instead of whole images. Each annotation box is cropped from the source image, stretched to the configured SOLO input size, compressed, and saved as one training item.

### YOLO TXT

Expected structure:

```text
dataset/
  images/
    crab001.jpg
  labels/
    crab001.txt
  classes.txt
```

YOLO boxes use the standard normalized format:

```text
class_id x_center y_center width height
```

Generate object-level weights:

```bash
python solo.py dataset/images \
  --annotations yolo \
  --labels-dir dataset/labels \
  --class-names dataset/classes.txt \
  --size 8 \
  --pt-size 8 \
  --field global \
  --save weights/yolo_objects.json
```

If your folder uses the common `images/...` and `labels/...` layout, `--labels-dir` can be omitted:

```bash
python solo.py dataset/images --annotations yolo --class-names dataset/classes.txt --save weights/yolo_objects.json
```

### LabelMe JSON

Expected structure:

```text
dataset/
  image001.png
  image001.json
```

SOLO reads `rectangle` shapes directly. For polygons, it uses the polygon's bounding rectangle.

```bash
python solo.py dataset \
  --annotations labelme \
  --size 8 \
  --pt-size 8 \
  --field global \
  --save weights/labelme_objects.json
```

The output item includes `source_image`, `annotation_path`, `label`, `bbox`, `crop_mode`, and `pt`.

## Detection

SOLO detection is a proposal-and-match pipeline:

```text
image -> proposals -> crop each box -> generate pt -> match against one or more weight files -> NMS -> report
```

Detect one image:

```bash
python solo.py image.png \
  --detect weights/yolo_objects.json \
  --proposal color \
  --score-threshold 0.999 \
  --output results/image_result.json
```

Detect a folder:

```bash
python solo.py dataset/images \
  --detect weights/yolo_objects.json \
  --proposal color \
  --output results/dataset_result.json
```

Use multiple weight files at the same time:

```bash
python solo.py image.png \
  --detect weights/feed.json weights/crab.json weights/other.json \
  --proposal both \
  --output results/multi_weight_result.json
```

Draw boxes on images:

```bash
python solo.py image.png \
  --detect weights/feed.json \
  --draw-dir results/drawn \
  --output results/feed_result.json
```

Hide labels above boxes:

```bash
python solo.py image.png \
  --detect weights/feed.json \
  --draw-dir results/drawn \
  --hide-labels \
  --output results/feed_result.json
```

Useful detection options:

```bash
python solo.py image.png \
  --detect weights/feed.json \
  --proposal color \
  --score-threshold 0.999 \
  --nms-threshold 0.35 \
  --min-box-size 8 \
  --max-aspect-ratio 4 \
  --proposal-expand 1.1 \
  --output results/feed_result.json
```

For more generic detection, use sliding windows:

```bash
python solo.py image.png \
  --detect weights/feed.json \
  --proposal sliding \
  --window-sizes 32,48,64 \
  --stride-ratio 0.5 \
  --output results/sliding_result.json
```

## Train/Val Calibration

SOLO can use train/val dataset proportions to periodically correct detection behavior. For example, if train has 1000 images and val has 200 images, SOLO processes one val image for every five train images during calibration.

The calibration currently adjusts:

- score threshold, using matched val detections
- bbox scale, using predicted-vs-ground-truth area ratio

Example:

```bash
python solo.py dataset/test/image.png \
  --detect weights/feed.json \
  --proposal color \
  --calibrate-train-images dataset/train/images \
  --calibrate-val-images dataset/val/images \
  --calibrate-train-labels dataset/train/labels \
  --calibrate-val-labels dataset/val/labels \
  --class-names dataset/classes.txt \
  --calibration-samples 20 \
  --output results/calibrated_result.json
```

Set `--calibration-samples 0` to use all scheduled val images.

## Python API

```python
from solo import get_image_single_pt, get_image_pt, load_weights, get_weight_by_name

pt = get_image_single_pt("train/apple.png", size=8, pt_size=8, field="global")

weights = get_image_pt(
    "train",
    size=8,
    pt_size=8,
    field="global",
    save_path="solo_weights.json",
    print_pt=False,
)

yolo_weights = get_image_pt(
    "dataset/images",
    annotations="yolo",
    labels_dir="dataset/labels",
    class_names_path="dataset/classes.txt",
    size=8,
    pt_size=8,
    field="global",
    save_path="weights/yolo_objects.json",
    print_pt=False,
)

payload = load_weights("solo_weights.json")
apple = get_weight_by_name(payload, "apple.png")
```

## Output Format

Saved weight files are JSON:

```json
{
  "version": 1,
  "config": {
    "size": 8,
    "qua": 8,
    "nab": 0.25,
    "pt_size": 8,
    "kernel": "weighted",
    "field": "global",
    "max_radius": 0,
    "normalize": true,
    "normalize_each_step": true,
    "annotations": "none",
    "crop_mode": "stretch"
  },
  "weights": [
    {
      "name": "apple.png",
      "path": "train/apple.png",
      "pt": [[0.0, 0.12, 0.42]]
    }
  ]
}
```

## CLI Options

```bash
python solo.py --help
```

Important options:

- `dirpath`: image folder, default `train`
- `--size`: source image is resized to `16 * size`
- `--pt-size`: final square matrix size
- `--qua`: decimal precision
- `--nab`: side-neighbor weight, diagonals use `nab / 2` in weighted mode
- `--kernel weighted|legacy`: choose the compression algorithm
- `--field global|local`: choose full-image decaying field or local 3x3 compression
- `--max-radius`: cap global field radius; `0` means full image
- `--annotations none|yolo|labelme`: train from whole images or annotation boxes
- `--labels-dir`: label folder for YOLO or LabelMe files
- `--class-names`: YOLO class names file
- `--crop-mode stretch`: normalize annotation crops by stretching them to the SOLO input
- `--detect`: one or more weight files for object detection
- `--output`: detection result JSON path
- `--draw-dir`: optional directory for images with drawn boxes
- `--hide-labels`: draw boxes without label text
- `--proposal color|sliding|both`: proposal generation mode
- `--score-threshold`: minimum score for detections
- `--nms-threshold`: overlap threshold for NMS
- `--min-box-size`, `--max-box-size`, `--max-aspect-ratio`: proposal shape filters
- `--calibrate-train-images`, `--calibrate-val-images`: train/val folders for periodic correction
- `--calibrate-train-labels`, `--calibrate-val-labels`: annotation folders for calibration
- `--save`: output JSON path
- `--load`: load an existing JSON weight file
- `--no-save`: generate without writing a file
- `--no-print`: suppress matrix printing

## Notes

- Supported image suffixes: `.bmp`, `.jpeg`, `.jpg`, `.png`, `.webp`.
- `pt_size` must be compatible with `16 * size`; for example, `size=8` gives a 128x128 source matrix and works with even targets like 8 or 4.
- Pure-color images are handled safely and normalize to all zeros.
- `global` mode is more expressive but heavier than `local`; use `--max-radius` for faster experiments.
