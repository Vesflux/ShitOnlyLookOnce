from __future__ import annotations

from typing import Any

_TAICHI_RUNTIME: dict[str, Any] = {
    "initialized": False,
    "available": False,
    "arch": None,
    "error": None,
    "ti": None,
}


def initialize_taichi(backend: str = "auto") -> dict[str, Any]:
    if _TAICHI_RUNTIME["initialized"]:
        return dict(_TAICHI_RUNTIME)
    _TAICHI_RUNTIME["initialized"] = True
    try:
        import taichi as ti  # type: ignore
    except Exception as exc:
        _TAICHI_RUNTIME.update({"available": False, "error": f"{type(exc).__name__}: {exc}"})
        return dict(_TAICHI_RUNTIME)

    requested = backend.lower().strip()
    if requested == "auto":
        candidates = ["cuda", "vulkan", "opengl", "cpu"]
    else:
        candidates = [requested]
    errors: list[str] = []
    for name in candidates:
        arch = getattr(ti, name, None)
        if arch is None:
            errors.append(f"{name}: unavailable in this Taichi build")
            continue
        try:
            ti.init(arch=arch, offline_cache=True, log_level=ti.ERROR)
            _TAICHI_RUNTIME.update({"available": True, "arch": name, "error": None, "ti": ti})
            return dict(_TAICHI_RUNTIME)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    _TAICHI_RUNTIME.update({"available": False, "error": "; ".join(errors), "ti": ti})
    return dict(_TAICHI_RUNTIME)


def taichi_status() -> dict[str, Any]:
    return {key: value for key, value in _TAICHI_RUNTIME.items() if key != "ti"}


__all__ = ["initialize_taichi", "taichi_status"]
