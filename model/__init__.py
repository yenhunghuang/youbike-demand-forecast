"""Model package exposing baseline and evaluation utilities."""

from __future__ import annotations

from . import evaluate

try:
    from . import baseline_lstm
except ModuleNotFoundError:  # pragma: no cover - fallback when torch isn't available
    baseline_lstm = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True

__all__ = ["evaluate", "baseline_lstm", "TORCH_AVAILABLE"]
