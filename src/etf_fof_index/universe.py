from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


REQUIRED_COLUMNS = {
    "bucket",
    "symbol",
    "name",
    "instrument_type",
    "enabled",
    "priority",
    "liquidity_score",
    "size_score",
    "fee_bps",
    "tracking_error_bps",
}


@dataclass
class UniverseSelection:
    universe: pd.DataFrame
    selected: pd.DataFrame

    @property
    def bucket_to_symbol(self) -> Dict[str, str]:
        return dict(zip(self.selected["bucket"], self.selected["symbol"]))


def load_universe(path: Path) -> pd.DataFrame:
    universe = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(universe.columns)
    if missing:
        raise ValueError(f"Universe file is missing columns: {sorted(missing)}")
    universe["enabled"] = universe["enabled"].astype(int)
    return universe


def select_representatives(
    universe: pd.DataFrame,
    buckets: Iterable[str],
    available_symbols: Optional[Iterable[str]] = None,
) -> UniverseSelection:
    candidates = universe.copy()
    candidates = candidates[candidates["enabled"] == 1]

    if available_symbols is not None:
        available = set(available_symbols)
        candidates = candidates[candidates["symbol"].isin(available)]

    candidates["selection_score"] = (
        candidates["priority"] * 1000
        + candidates["liquidity_score"] * 10
        + candidates["size_score"] * 5
        - candidates["fee_bps"] * 0.5
        - candidates["tracking_error_bps"] * 1.0
    )

    selected_rows = []
    for bucket in buckets:
        bucket_candidates = candidates[candidates["bucket"] == bucket].sort_values(
            ["selection_score", "priority", "liquidity_score", "size_score"],
            ascending=False,
        )
        if bucket_candidates.empty:
            raise ValueError(f"No eligible instrument found for bucket '{bucket}'.")
        selected_rows.append(bucket_candidates.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    return UniverseSelection(universe=universe, selected=selected)

