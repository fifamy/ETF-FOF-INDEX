from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_inputs(
    config: Dict,
    output_dir: Path,
    start: str,
    end: str,
    seed: int = 7,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(seed)

    bucket_params = {
        "equity_core": {"drift": 0.09 / 252, "vol": 0.20 / np.sqrt(252)},
        "equity_defensive": {"drift": 0.07 / 252, "vol": 0.15 / np.sqrt(252)},
        "rate_bond": {"drift": 0.03 / 252, "vol": 0.05 / np.sqrt(252)},
        "gold": {"drift": 0.05 / 252, "vol": 0.14 / np.sqrt(252)},
        "money_market": {"drift": 0.015 / 252, "vol": 0.01 / np.sqrt(252)},
    }

    symbol_map = {
        "equity_core": "510300.SH",
        "equity_defensive": "512890.SH",
        "rate_bond": "511010.SH",
        "gold": "518880.SH",
        "money_market": "511990.SH",
    }

    regime = np.where((dates >= "2022-04-01") & (dates <= "2022-10-31"), -1.0, 1.0)
    regime += np.where((dates >= "2024-09-01") & (dates <= "2025-02-28"), 0.6, 0.0)

    prices = pd.DataFrame(index=dates)
    for bucket, params in bucket_params.items():
        shocks = rng.normal(loc=0.0, scale=params["vol"], size=len(dates))
        drift = np.full(len(dates), params["drift"])
        if bucket == "equity_core":
            drift += regime * 0.00015
        elif bucket == "equity_defensive":
            drift += regime * 0.00010
        elif bucket == "rate_bond":
            drift -= regime * 0.00005
        elif bucket == "gold":
            drift += np.where(regime < 0, 0.00012, -0.00003)
        path = 100.0 * np.cumprod(1.0 + drift + shocks)
        prices[symbol_map[bucket]] = path

    prices.index.name = "date"
    prices_path = output_dir / "demo_prices.csv"
    prices.reset_index().to_csv(prices_path, index=False)

    valuation_dates = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).last()
    valuation = pd.DataFrame(index=pd.DatetimeIndex(valuation_dates.values))
    valuation["equity_core"] = np.clip(np.sin(np.linspace(0, 8, len(valuation))) * 0.8, -1.0, 1.0)
    valuation["equity_defensive"] = np.clip(np.cos(np.linspace(0, 8, len(valuation))) * 0.4, -1.0, 1.0)
    valuation.index.name = "date"
    valuation_path = output_dir / "demo_valuation.csv"
    valuation.reset_index().to_csv(valuation_path, index=False)

    return prices_path, valuation_path

