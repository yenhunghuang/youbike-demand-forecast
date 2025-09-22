from __future__ import annotations

import json
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import evaluate

torch = pytest.importorskip("torch")

from model.baseline_lstm import LSTMConfig, generate_synthetic_dataset


def test_evaluation_report_generation(tmp_path: pathlib.Path) -> None:
    report = evaluate.create_placeholder_report()
    metrics_path = tmp_path / "metrics.json"
    evaluate.write_report(report, metrics_path)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"t15_peak", "t30_offpeak", "t60_all"}

    acceptance = evaluate.validate_against_acceptance(report)
    assert acceptance["t15_peak"] is True
    assert acceptance["t30_offpeak"] is True
    assert acceptance["t60_all"] is True


def test_generate_synthetic_dataset_shapes() -> None:
    config = LSTMConfig()
    inputs, targets = generate_synthetic_dataset(
        num_samples=16,
        history=config.history,
        horizon=config.horizon,
        input_size=config.input_size,
    )
    assert inputs.shape == (16, config.history, config.input_size)
    assert targets.shape == (16, config.horizon)
    assert torch.isfinite(inputs).all()
    assert torch.isfinite(targets).all()
