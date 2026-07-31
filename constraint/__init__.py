"""Lightweight helpers for constraint resources used by the multiscale model."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    from .water_level_converter import WaterLevelConverter as _WaterLevelConverter
except Exception:  # pragma: no cover - keep stub fallback
    _WaterLevelConverter = None  # type: ignore

_PACKAGE_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _PACKAGE_DIR / "constraints_config.json"


def constraints_path() -> Path:
    """Return the absolute path to the bundled constraints configuration."""
    return _CONFIG_PATH


def load_constraints_config() -> Dict[str, Any]:
    """Load the JSON constraint configuration packaged with the project."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_water_level_converter(*args: Any, **kwargs: Any) -> "WaterLevelConverter":
    """Return the configured WaterLevelConverter (falls back to a safe stub)."""
    try:
        return WaterLevelConverter(*args, **kwargs)
    except Exception as exc:
        warnings.warn(
            f"failed to initialize WaterLevelConverter, using identity fallback: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _FallbackWaterLevelConverter(*args, **kwargs)


class _FallbackWaterLevelConverter:
    """Safe fallback that treats storage and level as identical."""

    is_stub: bool = True

    def __init__(self, *_, **__):
        pass

    def to(self, *_):
        return self

    def to_level(self, volume):
        return volume

    def to_volume(self, level):
        return level

    # Legacy aliases used by older scheduling utilities.
    def s2h(self, storage, *_):
        return storage

    def h2s(self, levels, *_):
        return levels


if _WaterLevelConverter is not None:
    WaterLevelConverter = _WaterLevelConverter
else:
    WaterLevelConverter = _FallbackWaterLevelConverter


class ConstraintProjector:
    """No-op projector kept for backwards compatibility with legacy imports."""

    def __init__(self, *_, **__):
        self.stats: Dict[str, Any] = {}

    def reset(self) -> None:
        self.stats.clear()

    def project(self, flows, **_) -> Tuple[Any, Dict[str, Any]]:
        return flows, dict(self.stats)

    def project_step(self, predictions, **_) -> Tuple[Any, Dict[str, Any]]:
        return predictions, dict(self.stats)


__all__ = [
    "constraints_path",
    "load_constraints_config",
    "get_water_level_converter",
    "WaterLevelConverter",
    "ConstraintProjector",
]
