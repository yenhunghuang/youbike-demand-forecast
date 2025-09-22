"""Construct a graph for YouBike stations using geographic kNN."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

RAW_DIR = pathlib.Path("data/raw")
GRAPH_DIR = pathlib.Path("data/processed")
DEFAULT_K = 8


@dataclass
class Station:
    sno: str
    lat: float
    lng: float

    @staticmethod
    def from_record(record: dict) -> "Station":
        try:
            return Station(
                sno=str(record["sno"]),
                lat=float(record["lat"]),
                lng=float(record["lng"]),
            )
        except KeyError as exc:
            raise KeyError(f"Missing required station field: {exc}") from exc


def load_snapshot(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot {path} not found. Run etl/fetch_youbike.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def latest_snapshot_path(raw_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    files = sorted(raw_dir.glob("youbike_*.json"))
    if not files:
        raise FileNotFoundError("No raw snapshots found. Run etl/fetch_youbike.py first.")
    return files[-1]


def build_station_frame(records: Iterable[dict]) -> pd.DataFrame:
    stations = [Station.from_record(item).__dict__ for item in records]
    df = pd.DataFrame(stations)
    if df.empty:
        raise ValueError("No station records available to build graph.")
    return df


def compute_knn_edges(df: pd.DataFrame, k: int = DEFAULT_K) -> List[Tuple[int, int, float]]:
    coords = df[["lat", "lng"]].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric="euclidean").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    edges: List[Tuple[int, int, float]] = []
    for src, neighbors in enumerate(indices):
        for dist, dst in zip(distances[src][1:], neighbors[1:]):
            if src == dst:
                continue
            weight = math.exp(-dist)
            edges.append((src, int(dst), float(weight)))
    return edges


def build_adjacency_matrix(num_nodes: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=float)
    for src, dst, weight in edges:
        adj[src, dst] = weight
        adj[dst, src] = max(adj[dst, src], weight)
    return adj


def save_graph(
    df: pd.DataFrame,
    edges: List[Tuple[int, int, float]],
    output_dir: pathlib.Path = GRAPH_DIR,
    include_matrix: bool = True,
) -> Tuple[pathlib.Path, pathlib.Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": df.to_dict(orient="records"),
        "edges": [
            {"source": int(src), "target": int(dst), "weight": weight}
            for src, dst, weight in edges
        ],
    }
    graph_path = output_dir / "youbike_graph.json"
    graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    matrix_path = None
    if include_matrix:
        adj = build_adjacency_matrix(len(df), edges)
        matrix_path = output_dir / "youbike_adjacency.npy"
        np.save(matrix_path, adj)

    return graph_path, matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a geographic kNN graph for YouBike stations.")
    parser.add_argument("--snapshot", type=pathlib.Path, default=None, help="Path to the snapshot JSON (defaults to latest)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of nearest neighbors to connect")
    parser.add_argument("--output", type=pathlib.Path, default=GRAPH_DIR, help="Directory to save the graph artifacts")
    parser.add_argument("--no-matrix", action="store_true", help="Skip writing adjacency matrix npy file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_path = args.snapshot or latest_snapshot_path()
    snapshot = load_snapshot(snapshot_path)
    records = snapshot.get("retVal", {}).values()
    df = build_station_frame(records)
    edges = compute_knn_edges(df, k=args.k)
    graph_path, matrix_path = save_graph(df, edges, output_dir=args.output, include_matrix=not args.no_matrix)
    print(f"Graph saved to {graph_path}")
    if matrix_path:
        print(f"Adjacency matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
