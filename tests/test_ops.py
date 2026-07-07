from pathlib import Path

import numpy as np

from solo.core.ops import _integral_image, _rect_sum, letptsmall
from solo.utils.cv_image import Image


def test_integral_rect_sum_matches_manual_region():
    pt = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    integral = _integral_image(pt)

    assert _rect_sum(integral, 0, 0, 2, 2) == 45.0
    assert _rect_sum(integral, 1, 1, 2, 2) == 28.0
    assert _rect_sum(integral, 0, 1, 1, 2) == 24.0


def test_letptsmall_preserves_shape_and_normalized_range():
    pt = [[float(x + y * 4) for x in range(4)] for y in range(4)]
    compressed = letptsmall(pt, weight=2, qua=6, field="global")

    assert len(compressed) == 2
    assert all(len(row) == 2 for row in compressed)
    assert all(0.0 <= value <= 1.0 for row in compressed for value in row)


def test_cv_image_replaces_pillow_for_core_image_ops():
    image = Image.new("RGB", (20, 10), (10, 20, 30))
    resized = image.resize((10, 5))
    flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
    resized.thumbnail((4, 4))

    assert resized.size == (4, 2)
    assert flipped.size == (10, 5)
    assert np.asarray(flipped.array).shape == (5, 10, 3)


def test_project_source_does_not_import_pillow():
    root = Path(__file__).resolve().parents[1] / "solo"
    forbidden = ("from " + "PIL", "import " + "PIL")

    offenders = [
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if any(pattern in path.read_text(encoding="utf-8") for pattern in forbidden)
    ]

    assert offenders == []
