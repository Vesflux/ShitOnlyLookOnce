from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
ROTATE_90 = 2
ROTATE_180 = 3
ROTATE_270 = 4


class _PixelAccess:
    def __init__(self, image: "CvImage") -> None:
        self._image = image

    def __getitem__(self, xy: tuple[int, int]) -> int | tuple[int, int, int]:
        x, y = xy
        array = self._image.array
        if array.ndim == 2:
            return int(array[y, x])
        return tuple(int(value) for value in array[y, x])


class CvImage:
    def __init__(self, array: np.ndarray, mode: str = "RGB", info: dict[str, Any] | None = None) -> None:
        if mode not in {"RGB", "L", "HSV"}:
            raise ValueError(f"unsupported image mode: {mode}")
        self.array = np.ascontiguousarray(array)
        self.mode = mode
        self.info: dict[str, Any] = dict(info or {})

    @classmethod
    def open(cls, path: str | Path) -> "CvImage":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cls(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "RGB")

    @classmethod
    def new(
        cls,
        mode: str,
        size: tuple[int, int],
        color: int | tuple[int, int, int] = 0,
    ) -> "CvImage":
        width, height = int(size[0]), int(size[1])
        if mode == "L":
            value = int(color if isinstance(color, int) else color[0])
            return cls(np.full((height, width), value, dtype=np.uint8), "L")
        if mode != "RGB":
            raise ValueError(f"unsupported image mode for new(): {mode}")
        if isinstance(color, int):
            fill = (color, color, color)
        else:
            fill = tuple(int(value) for value in color[:3])
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:, :] = fill
        return cls(array, "RGB")

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.array.shape[:2]
        return width, height

    def __enter__(self) -> "CvImage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    def copy(self) -> "CvImage":
        return CvImage(self.array.copy(), self.mode, self.info)

    def _as_rgb(self) -> np.ndarray:
        if self.mode == "RGB":
            return self.array
        if self.mode == "L":
            return cv2.cvtColor(self.array, cv2.COLOR_GRAY2RGB)
        hsv = self.array.astype(np.float32).copy()
        hsv[:, :, 0] = hsv[:, :, 0] * (179.0 / 255.0)
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    def convert(self, mode: str) -> "CvImage":
        if mode == self.mode:
            return self.copy()
        rgb = self._as_rgb()
        if mode == "RGB":
            return CvImage(rgb.copy(), "RGB", self.info)
        if mode == "L":
            return CvImage(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), "L", self.info)
        if mode == "HSV":
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = hsv[:, :, 0] * (255.0 / 179.0)
            return CvImage(np.clip(hsv, 0, 255).astype(np.uint8), "HSV", self.info)
        raise ValueError(f"unsupported conversion mode: {mode}")

    def resize(self, size: tuple[int, int]) -> "CvImage":
        width, height = int(size[0]), int(size[1])
        interpolation = cv2.INTER_AREA if width <= self.size[0] and height <= self.size[1] else cv2.INTER_LINEAR
        return CvImage(cv2.resize(self.array, (width, height), interpolation=interpolation), self.mode, self.info)

    def thumbnail(self, size: tuple[int, int]) -> None:
        max_width, max_height = max(1, int(size[0])), max(1, int(size[1]))
        width, height = self.size
        scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = self.resize(new_size)
        self.array = resized.array
        self.mode = resized.mode
        self.info.update(resized.info)

    def transpose(self, method: int) -> "CvImage":
        if method == FLIP_LEFT_RIGHT:
            return CvImage(cv2.flip(self.array, 1), self.mode, self.info)
        if method == FLIP_TOP_BOTTOM:
            return CvImage(cv2.flip(self.array, 0), self.mode, self.info)
        if method == ROTATE_90:
            return CvImage(cv2.rotate(self.array, cv2.ROTATE_90_COUNTERCLOCKWISE), self.mode, self.info)
        if method == ROTATE_180:
            return CvImage(cv2.rotate(self.array, cv2.ROTATE_180), self.mode, self.info)
        if method == ROTATE_270:
            return CvImage(cv2.rotate(self.array, cv2.ROTATE_90_CLOCKWISE), self.mode, self.info)
        raise ValueError(f"unsupported transpose method: {method}")

    def crop(self, box: tuple[int, int, int, int]) -> "CvImage":
        width, height = self.size
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        left = max(0, min(width, min(x1, x2)))
        right = max(0, min(width, max(x1, x2)))
        top = max(0, min(height, min(y1, y2)))
        bottom = max(0, min(height, max(y1, y2)))
        if right <= left or bottom <= top:
            shape = (1, 1) if self.array.ndim == 2 else (1, 1, self.array.shape[2])
            return CvImage(np.zeros(shape, dtype=np.uint8), self.mode, self.info)
        return CvImage(self.array[top:bottom, left:right].copy(), self.mode, self.info)

    def paste(self, image: "CvImage", xy: tuple[int, int]) -> None:
        source = image.convert(self.mode).array if image.mode != self.mode else image.array
        x, y = int(xy[0]), int(xy[1])
        target_h, target_w = self.array.shape[:2]
        source_h, source_w = source.shape[:2]
        left = max(0, x)
        top = max(0, y)
        right = min(target_w, x + source_w)
        bottom = min(target_h, y + source_h)
        if right <= left or bottom <= top:
            return
        source_left = left - x
        source_top = top - y
        self.array[top:bottom, left:right] = source[
            source_top : source_top + (bottom - top),
            source_left : source_left + (right - left),
        ]

    def getdata(self) -> list[int] | list[tuple[int, int, int]]:
        if self.array.ndim == 2:
            return [int(value) for value in self.array.reshape(-1)]
        return [tuple(int(channel) for channel in pixel) for pixel in self.array.reshape(-1, self.array.shape[2])]

    def load(self) -> _PixelAccess:
        return _PixelAccess(self)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "L":
            cv2.imwrite(str(output), self.array)
            return
        rgb = self._as_rgb()
        cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.array)


class Image:
    Image = CvImage
    open = staticmethod(CvImage.open)
    new = staticmethod(CvImage.new)
    FLIP_LEFT_RIGHT = FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = FLIP_TOP_BOTTOM
    ROTATE_90 = ROTATE_90
    ROTATE_180 = ROTATE_180
    ROTATE_270 = ROTATE_270


def read_image_size(path: str | Path) -> tuple[int, int]:
    return CvImage.open(path).size


__all__ = [
    "CvImage",
    "FLIP_LEFT_RIGHT",
    "FLIP_TOP_BOTTOM",
    "Image",
    "ROTATE_90",
    "ROTATE_180",
    "ROTATE_270",
    "read_image_size",
]
