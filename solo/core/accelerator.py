from __future__ import annotations

from typing import Any

from solo.config import *

_TAICHI_STATE: dict[str, Any] = {
    "attempted": False,
    "available": False,
    "ti": None,
    "np": None,
    "cache": {},
}
TAICHI_AUTO_MIN_WORK_ITEMS = 1_000_000

def accelerator_status(initialize: bool = False) -> dict[str, Any]:
    if initialize:
        _ensure_taichi()
    attempted = bool(_TAICHI_STATE.get("attempted"))
    available = bool(_TAICHI_STATE.get("available"))
    return {
        "taichi_attempted": attempted,
        "taichi_available": available,
        "backend": "taichi" if available else "cpu",
        "initialized": attempted and available,
        "auto_min_work_items": TAICHI_AUTO_MIN_WORK_ITEMS,
        "error": _TAICHI_STATE.get("error"),
    }

def _ensure_taichi() -> bool:
    if _TAICHI_STATE["attempted"]:
        return bool(_TAICHI_STATE["available"])
    _TAICHI_STATE["attempted"] = True
    try:
        import numpy as np  # type: ignore
        import taichi as ti  # type: ignore

        try:
            ti.init(arch=ti.gpu, offline_cache=True, log_level=ti.ERROR)
        except Exception:
            ti.init(arch=ti.cpu, offline_cache=True, log_level=ti.ERROR)

        _TAICHI_STATE.update({"available": True, "ti": ti, "np": np, "cache": {}, "error": None})
        return True
    except Exception as exc:
        _TAICHI_STATE.update({"available": False, "error": f"{type(exc).__name__}: {exc}"})
        return False

def _taichi_shape_cache(candidate_count: int, feature_count: int) -> dict[str, Any]:
    ti = _TAICHI_STATE["ti"]
    cache = _TAICHI_STATE.setdefault("cache", {})
    key = (candidate_count, feature_count)
    if key in cache:
        return cache[key]
    matrix_field = ti.field(dtype=ti.f32, shape=(candidate_count, feature_count))
    vector_field = ti.field(dtype=ti.f32, shape=feature_count)
    weight_field = ti.field(dtype=ti.f32, shape=feature_count)
    output_field = ti.field(dtype=ti.f32, shape=candidate_count)

    @ti.kernel
    def compute(denominator: ti.f32) -> None:
        for candidate_index in range(candidate_count):
            total = 0.0
            for feature_index in range(feature_count):
                delta = matrix_field[candidate_index, feature_index] - vector_field[feature_index]
                total += delta * delta * weight_field[feature_index]
            output_field[candidate_index] = total / denominator

    payload = {
        "matrix": matrix_field,
        "vector": vector_field,
        "weights": weight_field,
        "output": output_field,
        "compute": compute,
    }
    cache[key] = payload
    return payload

def _numpy_array(values: list[float]):
    try:
        import numpy as np  # type: ignore

        return np.asarray(values, dtype=np.float32)
    except Exception:
        return None

def _expanded_feature_weights(length: int, layout: dict[str, Any], channel_weights: dict[str, float] | None):
    weights = [1.0] * length
    if not channel_weights:
        return weights
    used = False
    for channel, weight in channel_weights.items():
        item = layout.get(channel)
        if not item:
            continue
        start = max(0, int(item.get("start", 0)))
        end = min(length, int(item.get("end", start)))
        if end <= start:
            continue
        used = True
        for index in range(start, end):
            weights[index] = float(weight)
    return weights if used else [1.0] * length

def _batch_weighted_mse_numpy(
    vector: list[float],
    candidates: list[list[float]],
    feature_weights: list[float],
) -> list[float] | None:
    try:
        import numpy as np  # type: ignore

        matrix = np.asarray(candidates, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(vector):
            return None
        left = np.asarray(vector, dtype=np.float32)
        weights = np.asarray(feature_weights, dtype=np.float32)
        denominator = float(np.sum(weights)) or float(len(vector))
        distances = np.sum(((matrix - left) ** 2) * weights, axis=1) / denominator
        return [float(value) for value in distances.tolist()]
    except Exception:
        return None

def _batch_weighted_mse_taichi(
    vector: list[float],
    candidates: list[list[float]],
    feature_weights: list[float],
) -> list[float] | None:
    if not _ensure_taichi():
        return None
    np = _TAICHI_STATE["np"]
    try:
        matrix = np.asarray(candidates, dtype=np.float32)
        left = np.asarray(vector, dtype=np.float32)
        weights = np.asarray(feature_weights, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(vector):
            return None
        candidate_count, feature_count = matrix.shape
        denominator = float(np.sum(weights)) or float(feature_count)
        payload = _taichi_shape_cache(int(candidate_count), int(feature_count))
        payload["matrix"].from_numpy(matrix)
        payload["vector"].from_numpy(left)
        payload["weights"].from_numpy(weights)
        payload["compute"](float(denominator))
        return [float(value) for value in payload["output"].to_numpy().tolist()]
    except Exception as exc:
        _TAICHI_STATE["error"] = f"{type(exc).__name__}: {exc}"
        return None

def batch_weighted_mse(
    vector: list[float],
    candidates: list[list[float]],
    weight_index: dict[str, Any],
    channel_weights: dict[str, float] | None = None,
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[float] | None:
    _validate_accelerator(accelerator)
    if not candidates:
        return []
    length = len(vector)
    if any(len(candidate) != length for candidate in candidates):
        return None
    feature_weights = _expanded_feature_weights(length, weight_index.get("channel_layout") or {}, channel_weights)
    work_items = len(candidates) * length
    should_try_taichi = accelerator == "taichi" or (
        accelerator == "auto" and work_items >= TAICHI_AUTO_MIN_WORK_ITEMS
    )
    if should_try_taichi:
        distances = _batch_weighted_mse_taichi(vector, candidates, feature_weights)
        if distances is not None:
            return distances
        if accelerator == "taichi":
            return None
    return _batch_weighted_mse_numpy(vector, candidates, feature_weights)

def batch_mse(
    vector: list[float],
    candidates: list[list[float]],
    accelerator: str = DEFAULT_ACCELERATOR,
) -> list[float] | None:
    _validate_accelerator(accelerator)
    if not candidates:
        return []
    length = len(vector)
    if any(len(candidate) != length for candidate in candidates):
        return None
    feature_weights = [1.0] * length
    work_items = len(candidates) * length
    should_try_taichi = accelerator == "taichi" or (
        accelerator == "auto" and work_items >= TAICHI_AUTO_MIN_WORK_ITEMS
    )
    if should_try_taichi:
        distances = _batch_weighted_mse_taichi(vector, candidates, feature_weights)
        if distances is not None:
            return distances
        if accelerator == "taichi":
            return None
    return _batch_weighted_mse_numpy(vector, candidates, feature_weights)

__all__ = [
    'accelerator_status',
    'batch_mse',
    'batch_weighted_mse',
    'TAICHI_AUTO_MIN_WORK_ITEMS',
]
