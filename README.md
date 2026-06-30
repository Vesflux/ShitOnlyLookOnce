# ShitOnlyLookOnce (SOLO)

SOLO is a tiny image-to-weight experiment. It reads images, converts them into grayscale point matrices, repeatedly compresses them with a small 3x3 neighborhood kernel, normalizes the result, and saves the generated pt weights as JSON.

The current default kernel follows the later `letptsmall` idea:

```text
0.125  0.25  0.125
0.25   1.00  0.25
0.125  0.25  0.125
```

It is not a neural-network framework. It is a compact, hackable SOLO-style feature/weight generator for quick image experiments.

## Features

- Convert image folders into pt weight matrices.
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
python solo.py train --size 8 --pt-size 8 --save solo_weights.json
```

Load saved weights:

```bash
python solo.py --load solo_weights.json
```

Run without printing every matrix:

```bash
python solo.py train --size 8 --save solo_weights.json --no-print
```

## Operation Examples

Generate 8x8 pt weights from images resized to 128x128:

```bash
python solo.py train --size 8 --pt-size 8 --save weights/solo_8x8.json
```

Generate 4x4 pt weights with lower precision:

```bash
python solo.py train --size 4 --pt-size 4 --qua 4 --save weights/solo_4x4.json
```

Use the legacy kernel from early SOLO experiments:

```bash
python solo.py train --size 8 --pt-size 8 --kernel legacy --nab 0.5 --save weights/legacy.json
```

Normalize only after all compression steps:

```bash
python solo.py train --size 8 --pt-size 8 --normalize-final-only --save weights/final_norm.json
```

Generate weights without saving:

```bash
python solo.py train --size 8 --no-save
```

## Python API

```python
from solo import get_image_single_pt, get_image_pt, load_weights, get_weight_by_name

pt = get_image_single_pt("train/apple.png", size=8, pt_size=8)

weights = get_image_pt(
    "train",
    size=8,
    pt_size=8,
    save_path="solo_weights.json",
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
    "normalize": true,
    "normalize_each_step": true
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
- `--save`: output JSON path
- `--load`: load an existing JSON weight file
- `--no-save`: generate without writing a file
- `--no-print`: suppress matrix printing

## Notes

- Supported image suffixes: `.bmp`, `.jpeg`, `.jpg`, `.png`, `.webp`.
- `pt_size` must be compatible with `16 * size`; for example, `size=8` gives a 128x128 source matrix and works with even targets like 8 or 4.
- Pure-color images are handled safely and normalize to all zeros.
