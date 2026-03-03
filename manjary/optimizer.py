# conveyor_io.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


SHAPE_COLS = ["circle", "pentagon", "trapezoid", "triangle", "star", "moon", "heart", "cross"]


def read_ragged_numeric_csv(path: Path) -> pd.DataFrame:
    """
    Reads a ragged CSV (rows can have different lengths) and returns a DataFrame
    padded with NaN. Cells are converted to float when possible.
    """
    rows: List[List[str]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return pd.DataFrame()

    maxlen = max(len(r) for r in rows)
    padded = [r + [""] * (maxlen - len(r)) for r in rows]
    df = pd.DataFrame(padded)

    def conv(x):
        if x is None or x == "":
            return np.nan
        try:
            return float(x)
        except Exception:
            return x

    # apply elementwise conversion
    return df.applymap(conv)


def build_demands(order_itemtypes: pd.DataFrame, order_quantities: pd.DataFrame, n_types: int = 8) -> List[Dict[int, int]]:
    """
    For each order (row), return dict {item_type: qty}. Item types outside [0, n_types-1]
    are ignored (keeps you aligned to 8 conveyor shapes).
    """
    if len(order_itemtypes) != len(order_quantities):
        raise ValueError("order_itemtypes and order_quantities must have the same number of rows (orders).")

    demands: List[Dict[int, int]] = []
    for i in range(len(order_itemtypes)):
        d: Dict[int, int] = {t: 0 for t in range(n_types)}
        for j in range(order_itemtypes.shape[1]):
            t = order_itemtypes.iloc[i, j]
            q = order_quantities.iloc[i, j]
            if pd.isna(t) or pd.isna(q):
                continue
            t_int = int(t)
            q_int = int(q)
            if 0 <= t_int < n_types:
                d[t_int] += q_int
        demands.append(d)
    return demands


def order_vector(demand: Dict[int, int], n_types: int = 8) -> List[int]:
    """Convert {type: qty} into length-n_types vector aligned with SHAPE_COLS order."""
    vec = [0] * n_types
    for t, q in demand.items():
        if 0 <= t < n_types:
            vec[t] = int(q)
    return vec


def write_conveyor_input_csv(
    belt_queues: List[List[int]],
    demands: List[Dict[int, int]],
    out_csv: Path,
    conv_num_base: int = 1,
) -> None:
    """
    Writes conveyor input CSV in the common "round-based interleave" format:
    for r=0..max_queue_len-1, write (belt 0's rth order, belt 1's rth order, ...)

    conv_num_base:
      - set to 1 if your IDEAS template uses lanes 1..4
      - set to 0 if your IDEAS template uses lanes 0..3
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    max_len = max((len(q) for q in belt_queues), default=0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_num"] + SHAPE_COLS)

        for r in range(max_len):
            for b in range(len(belt_queues)):
                if r < len(belt_queues[b]):
                    oi = belt_queues[b][r]
                    vec = order_vector(demands[oi], n_types=8)
                    w.writerow([b + conv_num_base] + vec)


def load_inputs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads generator outputs from a directory.
    """
    order_itemtypes = read_ragged_numeric_csv(data_dir / "order_itemtypes.csv")
    order_quantities = read_ragged_numeric_csv(data_dir / "order_quantities.csv")
    orders_totes = read_ragged_numeric_csv(data_dir / "orders_totes.csv")
    return order_itemtypes, order_quantities, orders_totes


def csv_equal(a: Path, b: Path) -> bool:
    """Exact textual comparison after stripping trailing whitespace per line."""
    if not a.exists() or not b.exists():
        return False
    a_lines = [ln.rstrip() for ln in a.read_text().splitlines() if ln.strip() != ""]
    b_lines = [ln.rstrip() for ln in b.read_text().splitlines() if ln.strip() != ""]
    return a_lines == b_lines