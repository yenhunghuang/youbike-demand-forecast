"""Placeholder training script for Graph WaveNet."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingConfig:
    epochs: int = 80
    learning_rate: float = 1e-3
    batch_size: int = 64
    horizon: int = 12
    graph_path: Path = Path("data/processed/youbike_graph.json")
    adjacency_path: Path = Path("data/processed/youbike_adjacency.npy")


def load_graph_artifacts(config: TrainingConfig) -> Dict[str, Any]:
    if not config.graph_path.exists():
        raise FileNotFoundError(f"Graph file missing: {config.graph_path}")
    payload = config.graph_path.read_text(encoding="utf-8")
    return {"graph_json": payload, "adjacency_path": config.adjacency_path}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = TrainingConfig()
    logging.info("Bootstrapping Graph WaveNet training with config: %s", config)
    try:
        artifacts = load_graph_artifacts(config)
    except FileNotFoundError as err:
        logging.warning("Graph artifacts missing: %s", err)
        logging.warning("Run `python etl/build_graph.py` before launching training.")
        return

    logging.info("Loaded graph payload (truncated): %s...", artifacts["graph_json"][:120])
    logging.info("Adjacency stored at: %s", artifacts["adjacency_path"])
    logging.info("Next steps: wire up GWNet layers, training schedule, and evaluation hooks.")


if __name__ == "__main__":
    main()
