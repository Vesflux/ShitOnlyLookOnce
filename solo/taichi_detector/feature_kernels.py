import numpy as np

from solo.taichi_detector.feature_layout import FEATURE_CHANNELS, FEATURE_SIZE, feature_dimension
from solo.taichi_detector.runtime import initialize_taichi

_KERNEL_CACHE: dict[tuple[int, int], tuple[object, object, object, object, object]] = {}


def _integral_feature_planes(planes: np.ndarray) -> np.ndarray:
    height, width, channels = planes.shape
    integral = np.zeros((height + 1, width + 1, channels), dtype=np.float32)
    integral[1:, 1:, :] = np.cumsum(np.cumsum(planes.astype(np.float32), axis=0), axis=1)
    return np.ascontiguousarray(integral)


def _get_kernels(ti, feature_size: int):
    cache_key = (id(ti), int(feature_size))
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    @ti.func
    def clamp01(value):
        return ti.max(0.0, ti.min(1.0, value))

    @ti.func
    def smooth(edge0, edge1, value):
        t = clamp01((value - edge0) / ti.max(1e-6, edge1 - edge0))
        return t * t * (3.0 - 2.0 * t)

    @ti.func
    def round_half_to_even(value):
        base_f = ti.floor(value)
        base = ti.cast(base_f, ti.i32)
        fraction = value - base_f
        rounded = base
        if fraction > 0.500001:
            rounded = base + 1
        elif fraction >= 0.499999 and ti.abs(base) % 2 == 1:
            rounded = base + 1
        return rounded

    @ti.func
    def sample_bilinear(planes: ti.template(), height, width, yf, xf, channel):
        x0 = ti.cast(ti.floor(xf), ti.i32)
        y0 = ti.cast(ti.floor(yf), ti.i32)
        x0 = ti.max(0, ti.min(width - 1, x0))
        y0 = ti.max(0, ti.min(height - 1, y0))
        x1 = ti.max(0, ti.min(width - 1, x0 + 1))
        y1 = ti.max(0, ti.min(height - 1, y0 + 1))
        wx = clamp01(xf - ti.cast(x0, ti.f32))
        wy = clamp01(yf - ti.cast(y0, ti.f32))
        top = planes[y0, x0, channel] * (1.0 - wx) + planes[y0, x1, channel] * wx
        bottom = planes[y1, x0, channel] * (1.0 - wx) + planes[y1, x1, channel] * wx
        return top * (1.0 - wy) + bottom * wy

    @ti.func
    def sample_area(integral: ti.template(), height, width, y0f, x0f, y1f, x1f, channel):
        x0 = ti.cast(ti.floor(x0f), ti.i32)
        y0 = ti.cast(ti.floor(y0f), ti.i32)
        x1 = ti.cast(ti.ceil(x1f), ti.i32)
        y1 = ti.cast(ti.ceil(y1f), ti.i32)
        x0 = ti.max(0, ti.min(width - 1, x0))
        y0 = ti.max(0, ti.min(height - 1, y0))
        x1 = ti.max(x0 + 1, ti.min(width, x1))
        y1 = ti.max(y0 + 1, ti.min(height, y1))
        area = ti.cast((x1 - x0) * (y1 - y0), ti.f32)
        total = (
            integral[y1, x1, channel]
            - integral[y0, x1, channel]
            - integral[y1, x0, channel]
            + integral[y0, x0, channel]
        )
        return total / ti.max(1.0, area)

    @ti.func
    def sample_cell(planes: ti.template(), integral: ti.template(), height, width, view_x1, view_y1, view_x2, view_y2, yy, xx, channel):
        view_w = ti.max(1.0, ti.cast(view_x2 - view_x1, ti.f32))
        view_h = ti.max(1.0, ti.cast(view_y2 - view_y1, ti.f32))
        cell_x0 = ti.cast(view_x1, ti.f32) + ti.cast(xx, ti.f32) * view_w / ti.cast(feature_size, ti.f32)
        cell_y0 = ti.cast(view_y1, ti.f32) + ti.cast(yy, ti.f32) * view_h / ti.cast(feature_size, ti.f32)
        cell_x1 = ti.cast(view_x1, ti.f32) + ti.cast(xx + 1, ti.f32) * view_w / ti.cast(feature_size, ti.f32)
        cell_y1 = ti.cast(view_y1, ti.f32) + ti.cast(yy + 1, ti.f32) * view_h / ti.cast(feature_size, ti.f32)
        value = 0.0
        if cell_x1 - cell_x0 >= 1.0 or cell_y1 - cell_y0 >= 1.0:
            value = sample_area(integral, height, width, cell_y0, cell_x0, cell_y1, cell_x1, channel)
        else:
            value = sample_bilinear(planes, height, width, (cell_y0 + cell_y1) * 0.5, (cell_x0 + cell_x1) * 0.5, channel)
        return value

    @ti.func
    def view_low_contrast(
        output: ti.template(),
        stat_sums: ti.template(),
        stat_squares: ti.template(),
        sample,
        view,
        inv_count,
    ):
        gray_min = 1e9
        gray_max = -1e9
        value_min = 1e9
        value_max = -1e9
        for yy in ti.static(range(feature_size)):
            for xx in ti.static(range(feature_size)):
                gray = output[sample, (view * FEATURE_CHANNELS + 0) * feature_size * feature_size + yy * feature_size + xx]
                value = output[sample, (view * FEATURE_CHANNELS + 2) * feature_size * feature_size + yy * feature_size + xx]
                gray_min = ti.min(gray_min, gray)
                gray_max = ti.max(gray_max, gray)
                value_min = ti.min(value_min, value)
                value_max = ti.max(value_max, value)
        gray_mean = stat_sums[sample, view, 0] * inv_count
        gray_std = ti.sqrt(ti.max(0.0, stat_squares[sample, view, 0] * inv_count - gray_mean * gray_mean))
        value_mean = stat_sums[sample, view, 2] * inv_count
        value_std = ti.sqrt(ti.max(0.0, stat_squares[sample, view, 2] * inv_count - value_mean * value_mean))
        grad_mean = stat_sums[sample, view, 3] * inv_count
        absolute_range = ti.max(gray_max - gray_min, value_max - value_min)
        return absolute_range < 30.0 / 255.0 and gray_std < 0.055 and value_std < 0.055 and grad_mean < 0.035

    @ti.kernel
    def prepare_view_boxes(
        boxes: ti.types.ndarray(dtype=ti.i32, ndim=2),
        view_boxes: ti.types.ndarray(dtype=ti.i32, ndim=3),
        sample_count: ti.i32,
        width: ti.i32,
        height: ti.i32,
        padding: ti.f32,
    ):
        for sample, view in ti.ndrange(sample_count, 2):
            bx1 = boxes[sample, 0]
            by1 = boxes[sample, 1]
            bx2 = boxes[sample, 2]
            by2 = boxes[sample, 3]
            vx1 = bx1
            vy1 = by1
            vx2 = bx2
            vy2 = by2
            if view == 1:
                box_w = ti.max(1, bx2 - bx1)
                box_h = ti.max(1, by2 - by1)
                pad_x = ti.cast(box_w, ti.f32) * padding
                pad_y = ti.cast(box_h, ti.f32) * padding
                vx1 = ti.max(0, round_half_to_even(ti.cast(bx1, ti.f32) - pad_x))
                vy1 = ti.max(0, round_half_to_even(ti.cast(by1, ti.f32) - pad_y))
                vx2 = ti.min(width, round_half_to_even(ti.cast(bx2, ti.f32) + pad_x))
                vy2 = ti.min(height, round_half_to_even(ti.cast(by2, ti.f32) + pad_y))
            view_boxes[sample, view, 0] = vx1
            view_boxes[sample, view, 1] = vy1
            view_boxes[sample, view, 2] = ti.max(vx1 + 1, vx2)
            view_boxes[sample, view, 3] = ti.max(vy1 + 1, vy2)

    @ti.kernel
    def clear_stats(
        stat_sums: ti.types.ndarray(dtype=ti.f32, ndim=3),
        stat_squares: ti.types.ndarray(dtype=ti.f32, ndim=3),
        sample_count: ti.i32,
    ):
        for sample, view, stat in ti.ndrange(sample_count, 2, 4):
            stat_sums[sample, view, stat] = 0.0
            stat_squares[sample, view, stat] = 0.0

    @ti.kernel
    def sample_visual_features(
        planes: ti.types.ndarray(dtype=ti.f32, ndim=3),
        integral: ti.types.ndarray(dtype=ti.f32, ndim=3),
        view_boxes: ti.types.ndarray(dtype=ti.i32, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
        stat_sums: ti.types.ndarray(dtype=ti.f32, ndim=3),
        stat_squares: ti.types.ndarray(dtype=ti.f32, ndim=3),
        sample_count: ti.i32,
        width: ti.i32,
        height: ti.i32,
    ):
        for sample, view, channel, yy, xx in ti.ndrange(
            sample_count,
            2,
            FEATURE_CHANNELS,
            feature_size,
            feature_size,
        ):
            vx1 = view_boxes[sample, view, 0]
            vy1 = view_boxes[sample, view, 1]
            vx2 = view_boxes[sample, view, 2]
            vy2 = view_boxes[sample, view, 3]
            value = sample_cell(planes, integral, height, width, vx1, vy1, vx2, vy2, yy, xx, channel)
            visual_index = (view * FEATURE_CHANNELS + channel) * feature_size * feature_size + yy * feature_size + xx
            output[sample, visual_index] = value
            if channel == 0:
                ti.atomic_add(stat_sums[sample, view, 0], value)
                ti.atomic_add(stat_squares[sample, view, 0], value * value)
            elif channel == 1:
                ti.atomic_add(stat_sums[sample, view, 1], value)
                ti.atomic_add(stat_squares[sample, view, 1], value * value)
            elif channel == 2:
                ti.atomic_add(stat_sums[sample, view, 2], value)
                ti.atomic_add(stat_squares[sample, view, 2], value * value)
            elif channel == 4:
                ti.atomic_add(stat_sums[sample, view, 3], value)
                ti.atomic_add(stat_squares[sample, view, 3], value * value)

    @ti.kernel
    def suppress_low_contrast_visuals(
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
        stat_sums: ti.types.ndarray(dtype=ti.f32, ndim=3),
        stat_squares: ti.types.ndarray(dtype=ti.f32, ndim=3),
        sample_count: ti.i32,
    ):
        inv_count = 1.0 / ti.cast(feature_size * feature_size, ti.f32)
        for sample, view in ti.ndrange(sample_count, 2):
            if view_low_contrast(output, stat_sums, stat_squares, sample, view, inv_count):
                for channel, yy, xx in ti.ndrange(FEATURE_CHANNELS, feature_size, feature_size):
                    visual_index = (view * FEATURE_CHANNELS + channel) * feature_size * feature_size + yy * feature_size + xx
                    output[sample, visual_index] = 0.0
                for stat in range(4):
                    stat_sums[sample, view, stat] = 0.0
                    stat_squares[sample, view, stat] = 0.0

    @ti.kernel
    def assemble_tail_features(
        boxes: ti.types.ndarray(dtype=ti.i32, ndim=2),
        view_boxes: ti.types.ndarray(dtype=ti.i32, ndim=3),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
        stat_sums: ti.types.ndarray(dtype=ti.f32, ndim=3),
        stat_squares: ti.types.ndarray(dtype=ti.f32, ndim=3),
        sample_count: ti.i32,
        width: ti.i32,
        height: ti.i32,
    ):
        stat_base = 2 * FEATURE_CHANNELS * feature_size * feature_size
        geometry_base = stat_base + 32
        inv_count = 1.0 / ti.cast(feature_size * feature_size, ti.f32)
        image_area = ti.max(1.0, ti.cast(width * height, ti.f32))
        for sample in ti.ndrange(sample_count):
            target_gray_mean = stat_sums[sample, 0, 0] * inv_count
            target_gray_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 0, 0] * inv_count - target_gray_mean * target_gray_mean))
            target_sat_mean = stat_sums[sample, 0, 1] * inv_count
            target_sat_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 0, 1] * inv_count - target_sat_mean * target_sat_mean))
            target_value_mean = stat_sums[sample, 0, 2] * inv_count
            target_value_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 0, 2] * inv_count - target_value_mean * target_value_mean))
            target_grad_mean = stat_sums[sample, 0, 3] * inv_count
            target_grad_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 0, 3] * inv_count - target_grad_mean * target_grad_mean))

            context_gray_mean = stat_sums[sample, 1, 0] * inv_count
            context_gray_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 1, 0] * inv_count - context_gray_mean * context_gray_mean))
            context_sat_mean = stat_sums[sample, 1, 1] * inv_count
            context_sat_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 1, 1] * inv_count - context_sat_mean * context_sat_mean))
            context_value_mean = stat_sums[sample, 1, 2] * inv_count
            context_value_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 1, 2] * inv_count - context_value_mean * context_value_mean))
            context_grad_mean = stat_sums[sample, 1, 3] * inv_count
            context_grad_std = ti.sqrt(ti.max(0.0, stat_squares[sample, 1, 3] * inv_count - context_grad_mean * context_grad_mean))

            output[sample, stat_base + 0] = target_gray_mean
            output[sample, stat_base + 1] = target_gray_std
            output[sample, stat_base + 2] = target_sat_mean
            output[sample, stat_base + 3] = target_sat_std
            output[sample, stat_base + 4] = target_value_mean
            output[sample, stat_base + 5] = target_value_std
            output[sample, stat_base + 6] = target_grad_mean
            output[sample, stat_base + 7] = target_grad_std
            output[sample, stat_base + 8] = context_gray_mean
            output[sample, stat_base + 9] = context_gray_std
            output[sample, stat_base + 10] = context_sat_mean
            output[sample, stat_base + 11] = context_sat_std
            output[sample, stat_base + 12] = context_value_mean
            output[sample, stat_base + 13] = context_value_std
            output[sample, stat_base + 14] = context_grad_mean
            output[sample, stat_base + 15] = context_grad_std

            output[sample, stat_base + 16] = target_gray_mean - context_gray_mean
            output[sample, stat_base + 17] = target_gray_std - context_gray_std
            output[sample, stat_base + 18] = target_sat_mean - context_sat_mean
            output[sample, stat_base + 19] = target_sat_std - context_sat_std
            output[sample, stat_base + 20] = target_value_mean - context_value_mean
            output[sample, stat_base + 21] = target_value_std - context_value_std
            output[sample, stat_base + 22] = target_grad_mean - context_grad_mean
            output[sample, stat_base + 23] = target_grad_std - context_grad_std

            target_grayness = 1.0 - smooth(0.08, 0.24, target_sat_mean)
            target_flatness = 1.0 - smooth(0.035, 0.16, target_grad_mean + target_gray_std * 0.35)
            target_darkness = 1.0 - smooth(0.32, 0.62, target_value_mean)
            target_shadow = clamp01(target_grayness * (target_darkness * 0.55 + target_flatness * 0.45))
            context_grayness = 1.0 - smooth(0.08, 0.24, context_sat_mean)
            context_flatness = 1.0 - smooth(0.035, 0.16, context_grad_mean + context_gray_std * 0.35)
            context_darkness = 1.0 - smooth(0.32, 0.62, context_value_mean)
            context_shadow = clamp01(context_grayness * (context_darkness * 0.55 + context_flatness * 0.45))

            output[sample, stat_base + 24] = target_grayness
            output[sample, stat_base + 25] = target_flatness
            output[sample, stat_base + 26] = target_darkness
            output[sample, stat_base + 27] = target_shadow
            output[sample, stat_base + 28] = context_grayness
            output[sample, stat_base + 29] = context_shadow
            output[sample, stat_base + 30] = ti.min(3.0, (target_grad_mean + 1e-4) / (context_grad_mean + 1e-4)) / 3.0
            output[sample, stat_base + 31] = clamp01((target_sat_mean - context_sat_mean + 1.0) * 0.5)

            bx1 = boxes[sample, 0]
            by1 = boxes[sample, 1]
            bx2 = boxes[sample, 2]
            by2 = boxes[sample, 3]
            px1 = view_boxes[sample, 1, 0]
            py1 = view_boxes[sample, 1, 1]
            px2 = view_boxes[sample, 1, 2]
            py2 = view_boxes[sample, 1, 3]
            box_w = ti.max(1.0, ti.cast(bx2 - bx1, ti.f32))
            box_h = ti.max(1.0, ti.cast(by2 - by1, ti.f32))
            padded_w = ti.max(1.0, ti.cast(px2 - px1, ti.f32))
            padded_h = ti.max(1.0, ti.cast(py2 - py1, ti.f32))

            output[sample, geometry_base + 0] = box_w / ti.max(1.0, ti.cast(width, ti.f32))
            output[sample, geometry_base + 1] = box_h / ti.max(1.0, ti.cast(height, ti.f32))
            output[sample, geometry_base + 2] = ti.min(6.0, box_w / box_h) / 6.0
            output[sample, geometry_base + 3] = ti.min(1.0, (box_w * box_h) / image_area)
            output[sample, geometry_base + 4] = ti.cast(bx1 + bx2, ti.f32) * 0.5 / ti.max(1.0, ti.cast(width, ti.f32))
            output[sample, geometry_base + 5] = ti.cast(by1 + by2, ti.f32) * 0.5 / ti.max(1.0, ti.cast(height, ti.f32))
            output[sample, geometry_base + 6] = padded_w / ti.max(1.0, ti.cast(width, ti.f32))
            output[sample, geometry_base + 7] = padded_h / ti.max(1.0, ti.cast(height, ti.f32))
            output[sample, geometry_base + 8] = ti.min(6.0, padded_w / padded_h) / 6.0
            output[sample, geometry_base + 9] = ti.min(1.0, (box_w * box_h) / ti.max(1.0, padded_w * padded_h))

    kernels = (prepare_view_boxes, clear_stats, sample_visual_features, suppress_low_contrast_visuals, assemble_tail_features)
    _KERNEL_CACHE[cache_key] = kernels
    return kernels


def batch_crop_feature_matrix_taichi(
    planes: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    context_padding: float,
    feature_size: int = FEATURE_SIZE,
    backend: str = "auto",
) -> np.ndarray | None:
    if feature_size <= 0:
        raise ValueError("feature_size must be positive")
    if backend == "cpu":
        return None

    runtime = initialize_taichi(backend)
    if not runtime.get("available") or runtime.get("arch") == "cpu":
        return None

    ti = runtime["ti"]
    planes_np = np.ascontiguousarray(planes.astype(np.float32))
    integral_np = _integral_feature_planes(planes_np)
    boxes_np = np.ascontiguousarray(np.asarray(bboxes, dtype=np.int32))
    height, width = planes_np.shape[:2]
    sample_count = int(len(bboxes))
    dimension = feature_dimension(feature_size)
    view_boxes_np = np.zeros((sample_count, 2, 4), dtype=np.int32)
    output_np = np.zeros((sample_count, dimension), dtype=np.float32)
    stat_sums_np = np.zeros((sample_count, 2, 4), dtype=np.float32)
    stat_squares_np = np.zeros((sample_count, 2, 4), dtype=np.float32)
    prepare_view_boxes, clear_stats, sample_visual_features, suppress_low_contrast_visuals, assemble_tail_features = _get_kernels(ti, feature_size)

    prepare_view_boxes(boxes_np, view_boxes_np, sample_count, width, height, float(context_padding))
    clear_stats(stat_sums_np, stat_squares_np, sample_count)
    sample_visual_features(
        planes_np,
        integral_np,
        view_boxes_np,
        output_np,
        stat_sums_np,
        stat_squares_np,
        sample_count,
        width,
        height,
    )
    suppress_low_contrast_visuals(output_np, stat_sums_np, stat_squares_np, sample_count)
    assemble_tail_features(
        boxes_np,
        view_boxes_np,
        output_np,
        stat_sums_np,
        stat_squares_np,
        sample_count,
        width,
        height,
    )
    return output_np.astype(np.float32, copy=False)


__all__ = ["batch_crop_feature_matrix_taichi"]
