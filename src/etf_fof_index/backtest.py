from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd


@dataclass
class BacktestResult:
    levels: pd.DataFrame
    holdings: pd.DataFrame


def _execution_schedule(index: pd.DatetimeIndex, signal_dates: Iterable[pd.Timestamp], weights: pd.DataFrame) -> Dict[pd.Timestamp, pd.Series]:
    positions = {date: idx for idx, date in enumerate(index)}
    schedule: Dict[pd.Timestamp, pd.Series] = {}
    for signal_date in signal_dates:
        if signal_date not in positions:
            continue
        loc = positions[signal_date]
        if loc + 1 >= len(index):
            continue
        execution_date = index[loc + 1]
        schedule[execution_date] = weights.loc[signal_date]
    return schedule


def run_backtest(
    bucket_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: Dict,
    label: str,
) -> BacktestResult:
    bucket_order = list(config["bucket_order"])
    returns = bucket_prices.pct_change().fillna(0.0)
    cost_rate = float(config["costs"]["transaction_cost_bps"]) / 10000.0
    strategic = pd.Series(config["strategic_weights"], dtype=float).reindex(bucket_order)

    signal_dates = list(target_weights.index)
    if not signal_dates:
        raise ValueError("Target weights are empty.")

    start_date = signal_dates[0]
    run_index = returns.loc[start_date:].index
    returns = returns.loc[run_index, bucket_order]
    schedule = _execution_schedule(run_index, signal_dates, target_weights.reindex(columns=bucket_order))

    current_weights = strategic.copy()
    levels = []
    holdings = []
    level = float(config["start_index_level"])

    for date in run_index:
        daily_ret = returns.loc[date]
        portfolio_return = float((current_weights * daily_ret).sum())
        gross_weights = current_weights * (1.0 + daily_ret)
        gross_weights = gross_weights / gross_weights.sum()
        turnover = 0.0

        if date in schedule:
            target = schedule[date].astype(float).reindex(bucket_order)
            turnover = float((gross_weights - target).abs().sum())
            portfolio_return -= turnover * cost_rate
            gross_weights = target

        level *= 1.0 + portfolio_return
        current_weights = gross_weights

        level_row = {
            "date": date,
            f"{label}_return": portfolio_return,
            f"{label}_turnover": turnover,
            f"{label}_index": level,
        }
        levels.append(level_row)

        holding_row = current_weights.to_dict()
        holding_row["date"] = date
        holdings.append(holding_row)

    levels_frame = pd.DataFrame(levels).set_index("date").sort_index()
    holdings_frame = pd.DataFrame(holdings).set_index("date").sort_index()
    return BacktestResult(levels=levels_frame, holdings=holdings_frame)

