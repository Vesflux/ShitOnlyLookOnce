import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from solo.taichi_detector.runtime import initialize_taichi


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def standardize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features.astype(np.float32) - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1e-4)


def train_linear_taichi(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    epochs: int = 260,
    learning_rate: float = 0.08,
    l2: float = 0.0005,
    backend: str = "auto",
) -> tuple[np.ndarray, float, dict[str, Any]]:
    runtime = initialize_taichi(backend)
    if not runtime.get("available"):
        return train_linear_numpy(features, labels, sample_weights, epochs, learning_rate, l2)

    ti = runtime["ti"]
    x_np = np.ascontiguousarray(features.astype(np.float32))
    y_np = np.ascontiguousarray(labels.astype(np.float32))
    sw_np = np.ascontiguousarray(sample_weights.astype(np.float32))
    sample_count, feature_count = x_np.shape
    x = ti.field(dtype=ti.f32, shape=(sample_count, feature_count))
    y = ti.field(dtype=ti.f32, shape=sample_count)
    sw = ti.field(dtype=ti.f32, shape=sample_count)
    weights = ti.field(dtype=ti.f32, shape=feature_count)
    grad = ti.field(dtype=ti.f32, shape=feature_count)
    error = ti.field(dtype=ti.f32, shape=sample_count)
    bias = ti.field(dtype=ti.f32, shape=())
    bias_grad = ti.field(dtype=ti.f32, shape=())
    loss_value = ti.field(dtype=ti.f32, shape=())
    x.from_numpy(x_np)
    y.from_numpy(y_np)
    sw.from_numpy(sw_np)
    weights.from_numpy(np.zeros((feature_count,), dtype=np.float32))
    bias[None] = 0.0
    weight_sum = float(np.sum(sw_np)) or float(sample_count)

    @ti.kernel
    def compute_error():
        loss_value[None] = 0.0
        bias_grad[None] = 0.0
        for i in range(sample_count):
            logit = bias[None]
            for j in range(feature_count):
                logit += x[i, j] * weights[j]
            prob = 1.0 / (1.0 + ti.exp(-ti.max(-50.0, ti.min(50.0, logit))))
            err = (prob - y[i]) * sw[i]
            error[i] = err
            bias_grad[None] += err
            target = y[i]
            loss_value[None] += -sw[i] * (
                target * ti.log(ti.max(prob, 1e-6)) + (1.0 - target) * ti.log(ti.max(1.0 - prob, 1e-6))
            )

    @ti.kernel
    def compute_grad(l2_value: ti.f32):
        for j in range(feature_count):
            total = 0.0
            for i in range(sample_count):
                total += error[i] * x[i, j]
            grad[j] = total / weight_sum + l2_value * weights[j]

    @ti.kernel
    def apply_grad(lr: ti.f32):
        for j in range(feature_count):
            weights[j] -= lr * grad[j]
        bias[None] -= lr * bias_grad[None] / weight_sum

    last_loss = 0.0
    for epoch in range(max(1, epochs)):
        lr = learning_rate * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs))))
        compute_error()
        compute_grad(float(l2))
        apply_grad(float(lr))
        last_loss = float(loss_value[None]) / weight_sum
    return weights.to_numpy().astype(np.float32), float(bias[None]), {
        "backend": runtime.get("arch"),
        "loss": last_loss,
        "epochs": epochs,
        "samples": sample_count,
        "features": feature_count,
    }


def train_mlp_taichi(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    epochs: int = 260,
    learning_rate: float = 0.04,
    hidden_size: int = 24,
    l2: float = 0.0008,
    backend: str = "auto",
) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    if hidden_size <= 0:
        weights, bias, report = train_linear_taichi(
            features,
            labels,
            sample_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            backend=backend,
        )
        return {"w1": np.zeros((0, features.shape[1]), dtype=np.float32), "b1": np.zeros((0,), dtype=np.float32), "w2": weights, "b2": float(bias)}, report

    runtime = initialize_taichi(backend)
    if not runtime.get("available"):
        return train_mlp_numpy(features, labels, sample_weights, epochs, learning_rate, hidden_size, l2)

    ti = runtime["ti"]
    x_np = np.ascontiguousarray(features.astype(np.float32))
    y_np = np.ascontiguousarray(labels.astype(np.float32))
    sw_np = np.ascontiguousarray(sample_weights.astype(np.float32))
    sample_count, feature_count = x_np.shape
    rng = np.random.default_rng(12345)
    w1_np = rng.normal(0.0, 0.035, size=(hidden_size, feature_count)).astype(np.float32)
    b1_np = np.zeros((hidden_size,), dtype=np.float32)
    w2_np = rng.normal(0.0, 0.035, size=(hidden_size,)).astype(np.float32)

    x = ti.field(dtype=ti.f32, shape=(sample_count, feature_count))
    y = ti.field(dtype=ti.f32, shape=sample_count)
    sw = ti.field(dtype=ti.f32, shape=sample_count)
    w1 = ti.field(dtype=ti.f32, shape=(hidden_size, feature_count))
    b1 = ti.field(dtype=ti.f32, shape=hidden_size)
    w2 = ti.field(dtype=ti.f32, shape=hidden_size)
    b2 = ti.field(dtype=ti.f32, shape=())
    grad_w1 = ti.field(dtype=ti.f32, shape=(hidden_size, feature_count))
    grad_b1 = ti.field(dtype=ti.f32, shape=hidden_size)
    grad_w2 = ti.field(dtype=ti.f32, shape=hidden_size)
    grad_b2 = ti.field(dtype=ti.f32, shape=())
    loss_value = ti.field(dtype=ti.f32, shape=())
    x.from_numpy(x_np)
    y.from_numpy(y_np)
    sw.from_numpy(sw_np)
    w1.from_numpy(w1_np)
    b1.from_numpy(b1_np)
    w2.from_numpy(w2_np)
    b2[None] = 0.0
    weight_sum = float(np.sum(sw_np)) or float(sample_count)

    @ti.kernel
    def clear_grads():
        loss_value[None] = 0.0
        grad_b2[None] = 0.0
        for h in range(hidden_size):
            grad_b1[h] = 0.0
            grad_w2[h] = 0.0
            for j in range(feature_count):
                grad_w1[h, j] = 0.0

    @ti.kernel
    def compute_grads(l2_value: ti.f32):
        for i in range(sample_count):
            logit = b2[None]
            for h in range(hidden_size):
                hidden_raw = b1[h]
                for j in range(feature_count):
                    hidden_raw += w1[h, j] * x[i, j]
                hidden = ti.tanh(hidden_raw)
                logit += w2[h] * hidden
            prob = 1.0 / (1.0 + ti.exp(-ti.max(-50.0, ti.min(50.0, logit))))
            err = (prob - y[i]) * sw[i]
            grad_b2[None] += err
            target = y[i]
            loss_value[None] += -sw[i] * (
                target * ti.log(ti.max(prob, 1e-6)) + (1.0 - target) * ti.log(ti.max(1.0 - prob, 1e-6))
            )
            for h in range(hidden_size):
                hidden_raw = b1[h]
                for j in range(feature_count):
                    hidden_raw += w1[h, j] * x[i, j]
                hidden = ti.tanh(hidden_raw)
                grad_w2[h] += err * hidden
                hidden_grad = err * w2[h] * (1.0 - hidden * hidden)
                grad_b1[h] += hidden_grad
                for j in range(feature_count):
                    grad_w1[h, j] += hidden_grad * x[i, j]

        for h in range(hidden_size):
            grad_w2[h] = grad_w2[h] / weight_sum + l2_value * w2[h]
            grad_b1[h] = grad_b1[h] / weight_sum
            for j in range(feature_count):
                grad_w1[h, j] = grad_w1[h, j] / weight_sum + l2_value * w1[h, j]
        grad_b2[None] = grad_b2[None] / weight_sum

    @ti.kernel
    def apply_grads(lr: ti.f32):
        for h in range(hidden_size):
            w2[h] -= lr * grad_w2[h]
            b1[h] -= lr * grad_b1[h]
            for j in range(feature_count):
                w1[h, j] -= lr * grad_w1[h, j]
        b2[None] -= lr * grad_b2[None]

    last_loss = 0.0
    for epoch in range(max(1, epochs)):
        lr = learning_rate * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs))))
        clear_grads()
        compute_grads(float(l2))
        apply_grads(float(lr))
        last_loss = float(loss_value[None]) / weight_sum
    return {
        "w1": w1.to_numpy().astype(np.float32),
        "b1": b1.to_numpy().astype(np.float32),
        "w2": w2.to_numpy().astype(np.float32),
        "b2": float(b2[None]),
    }, {
        "backend": runtime.get("arch"),
        "loss": last_loss,
        "epochs": epochs,
        "samples": sample_count,
        "features": feature_count,
        "hidden_size": hidden_size,
    }


def train_mlp_numpy(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    l2: float,
) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    x = features.astype(np.float32)
    y = labels.astype(np.float32)
    sw = sample_weights.astype(np.float32)
    sample_count, feature_count = x.shape
    weight_sum = float(np.sum(sw)) or float(sample_count)
    rng = np.random.default_rng(12345)
    w1 = rng.normal(0.0, 0.035, size=(hidden_size, feature_count)).astype(np.float32)
    b1 = np.zeros((hidden_size,), dtype=np.float32)
    w2 = rng.normal(0.0, 0.035, size=(hidden_size,)).astype(np.float32)
    b2 = 0.0
    last_loss = 0.0
    for epoch in range(max(1, epochs)):
        lr = learning_rate * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs))))
        hidden = np.tanh(x @ w1.T + b1)
        logits = np.clip(hidden @ w2 + b2, -50.0, 50.0)
        prob = sigmoid_np(logits)
        err = (prob - y) * sw
        grad_w2 = (hidden.T @ err) / weight_sum + l2 * w2
        grad_b2 = float(np.sum(err)) / weight_sum
        hidden_grad = (err[:, None] * w2[None, :]) * (1.0 - hidden * hidden)
        grad_w1 = (hidden_grad.T @ x) / weight_sum + l2 * w1
        grad_b1 = np.sum(hidden_grad, axis=0) / weight_sum
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        last_loss = float(
            np.sum(-sw * (y * np.log(np.maximum(prob, 1e-6)) + (1.0 - y) * np.log(np.maximum(1.0 - prob, 1e-6))))
            / weight_sum
        )
    return {"w1": w1, "b1": b1, "w2": w2, "b2": float(b2)}, {
        "backend": "numpy",
        "loss": last_loss,
        "epochs": epochs,
        "samples": sample_count,
        "features": feature_count,
        "hidden_size": hidden_size,
    }


def train_linear_numpy(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    x = features.astype(np.float32)
    y = labels.astype(np.float32)
    sw = sample_weights.astype(np.float32)
    weight_sum = float(np.sum(sw)) or float(len(y))
    weights = np.zeros((x.shape[1],), dtype=np.float32)
    bias = 0.0
    last_loss = 0.0
    for epoch in range(max(1, epochs)):
        lr = learning_rate * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs))))
        logits = np.clip(x @ weights + bias, -50.0, 50.0)
        prob = sigmoid_np(logits)
        err = (prob - y) * sw
        grad = (x.T @ err) / weight_sum + l2 * weights
        weights -= lr * grad
        bias -= lr * float(np.sum(err)) / weight_sum
        last_loss = float(
            np.sum(-sw * (y * np.log(np.maximum(prob, 1e-6)) + (1.0 - y) * np.log(np.maximum(1.0 - prob, 1e-6))))
            / weight_sum
        )
    return weights, float(bias), {
        "backend": "numpy",
        "loss": last_loss,
        "epochs": epochs,
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
    }


def score_linear_taichi(features: np.ndarray, weights: np.ndarray, bias: float, backend: str = "auto") -> np.ndarray:
    runtime = initialize_taichi(backend)
    if not runtime.get("available"):
        return sigmoid_np(features.astype(np.float32) @ weights.astype(np.float32) + float(bias)).astype(np.float32)
    ti = runtime["ti"]
    x_np = np.ascontiguousarray(features.astype(np.float32))
    w_np = np.ascontiguousarray(weights.astype(np.float32))
    sample_count, feature_count = x_np.shape
    x = ti.field(dtype=ti.f32, shape=(sample_count, feature_count))
    w = ti.field(dtype=ti.f32, shape=feature_count)
    out = ti.field(dtype=ti.f32, shape=sample_count)
    x.from_numpy(x_np)
    w.from_numpy(w_np)

    @ti.kernel
    def compute(bias_value: ti.f32):
        for i in range(sample_count):
            logit = bias_value
            for j in range(feature_count):
                logit += x[i, j] * w[j]
            out[i] = 1.0 / (1.0 + ti.exp(-ti.max(-50.0, ti.min(50.0, logit))))

    compute(float(bias))
    return out.to_numpy().astype(np.float32)


def score_mlp_taichi(features: np.ndarray, network: dict[str, Any], backend: str = "auto") -> np.ndarray:
    w1_np = np.asarray(network.get("w1", []), dtype=np.float32)
    b1_np = np.asarray(network.get("b1", []), dtype=np.float32)
    w2_np = np.asarray(network.get("w2", []), dtype=np.float32)
    b2 = float(network.get("b2", 0.0))
    if w1_np.size == 0:
        return score_linear_taichi(features, w2_np, b2, backend=backend)

    runtime = initialize_taichi(backend)
    if not runtime.get("available"):
        hidden = np.tanh(features.astype(np.float32) @ w1_np.T + b1_np)
        return sigmoid_np(hidden @ w2_np + b2).astype(np.float32)

    ti = runtime["ti"]
    x_np = np.ascontiguousarray(features.astype(np.float32))
    sample_count, feature_count = x_np.shape
    hidden_size = int(w1_np.shape[0])
    x = ti.field(dtype=ti.f32, shape=(sample_count, feature_count))
    w1 = ti.field(dtype=ti.f32, shape=(hidden_size, feature_count))
    b1 = ti.field(dtype=ti.f32, shape=hidden_size)
    w2 = ti.field(dtype=ti.f32, shape=hidden_size)
    out = ti.field(dtype=ti.f32, shape=sample_count)
    x.from_numpy(x_np)
    w1.from_numpy(np.ascontiguousarray(w1_np))
    b1.from_numpy(np.ascontiguousarray(b1_np))
    w2.from_numpy(np.ascontiguousarray(w2_np))

    @ti.kernel
    def compute(bias_value: ti.f32):
        for i in range(sample_count):
            logit = bias_value
            for h in range(hidden_size):
                hidden_raw = b1[h]
                for j in range(feature_count):
                    hidden_raw += x[i, j] * w1[h, j]
                logit += ti.tanh(hidden_raw) * w2[h]
            out[i] = 1.0 / (1.0 + ti.exp(-ti.max(-50.0, ti.min(50.0, logit))))

    compute(b2)
    return out.to_numpy().astype(np.float32)


def save_model(path: str | Path, payload: dict[str, Any]) -> Path:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path


def load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "load_model",
    "save_model",
    "score_mlp_taichi",
    "score_linear_taichi",
    "standardize_features",
    "train_mlp_taichi",
    "train_linear_taichi",
]
