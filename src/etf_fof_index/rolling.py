from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd


@dataclass
class RollingSelectionResult:
    decision_table: pd.DataFrame
    target_weights: pd.DataFrame


@dataclass
class RollingWindow:
    rebalance_signal_date: pd.Timestamp
    lookback_start: pd.Timestamp
    lookback_end: pd.Timestamp
    hold_signal_dates: pd.DatetimeIndex
    metrics: pd.DataFrame


SELECTION_RULES: Dict[str, Dict[str, object]] = {
    "min_drawdown": {
        "label": "回撤优先",
        "description": "先选观察窗最大回撤最浅的组合，并用收益、夏普和波动做并列决胜。",
    },
    "sharpe_guard": {
        "label": "回撤约束后夏普优先",
        "description": "先保留回撤接近最优的一组候选，再优先选择夏普更高的组合。",
    },
    "calmar_guard": {
        "label": "回撤约束后Calmar优先",
        "description": "先保留回撤接近最优的一组候选，再优先选择Calmar更高的组合。",
    },
}


def quarterly_rebalance_dates(signal_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(signal_dates).sort_values().unique()
    return dates[dates.month.isin([3, 6, 9, 12])]


def build_metric_frame(index: pd.DatetimeIndex, ret_store: np.ndarray, turnover_store: np.ndarray, levels: np.ndarray) -> pd.DataFrame:
    periods = max(len(index), 1)
    total_return = levels[-1] / levels[0] - 1.0
    annual_return = np.power(1.0 + total_return, 252.0 / periods) - 1.0
    annual_vol = ret_store.std(axis=0, ddof=0) * np.sqrt(252.0)
    sharpe = np.divide(annual_return, annual_vol, out=np.zeros_like(annual_return), where=annual_vol > 0)
    running_max = np.maximum.accumulate(levels, axis=0)
    drawdowns = levels / running_max - 1.0
    max_drawdown = drawdowns.min(axis=0)
    calmar = np.divide(annual_return, np.abs(max_drawdown), out=np.zeros_like(annual_return), where=max_drawdown < 0)
    avg_turnover = turnover_store.mean(axis=0)
    annualized_turnover = turnover_store.sum(axis=0) / periods * 252.0

    months = pd.Series(index).dt.to_period("M")
    monthly_groups = []
    for _, grp in pd.Series(range(len(index)), index=months).groupby(level=0):
        monthly_groups.append(grp.to_numpy())

    monthly_win = []
    for locs in monthly_groups:
        monthly_ret = np.prod(1.0 + ret_store[locs], axis=0) - 1.0
        monthly_win.append(monthly_ret > 0)
    monthly_win_rate = np.mean(np.vstack(monthly_win), axis=0) if monthly_win else np.zeros(ret_store.shape[1])

    return pd.DataFrame(
        {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "calmar": calmar,
            "monthly_win_rate": monthly_win_rate,
            "avg_daily_turnover": avg_turnover,
            "annualized_turnover_proxy": annualized_turnover,
        }
    )


def evaluate_window_metrics(
    index: pd.DatetimeIndex,
    ret_store: np.ndarray,
    turnover_store: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    mask = (index > start_date) & (index <= end_date)
    if not bool(mask.any()):
        raise ValueError("Lookback window does not contain any observations.")
    idx = index[mask]
    ret = ret_store[mask]
    turnover = turnover_store[mask]
    levels = 1000.0 * np.cumprod(1.0 + ret, axis=0)
    return build_metric_frame(idx, ret, turnover, levels)


def prepare_rebalance_windows(
    signal_dates: pd.DatetimeIndex,
    run_index: pd.DatetimeIndex,
    ret_store: np.ndarray,
    turnover_store: np.ndarray,
    lookback_months: int,
) -> list[RollingWindow]:
    rebalance_dates = quarterly_rebalance_dates(signal_dates)
    windows: list[RollingWindow] = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        lookback_start = rebalance_date - pd.DateOffset(months=lookback_months)
        if run_index.min() > lookback_start:
            continue
        if not bool(((run_index > lookback_start) & (run_index <= rebalance_date)).any()):
            continue

        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else signal_dates.max() + pd.Timedelta(days=1)
        hold_signal_dates = signal_dates[(signal_dates >= rebalance_date) & (signal_dates < next_rebalance)]
        if hold_signal_dates.empty:
            continue

        window_metrics = evaluate_window_metrics(
            index=run_index,
            ret_store=ret_store,
            turnover_store=turnover_store,
            start_date=lookback_start,
            end_date=rebalance_date,
        )
        windows.append(
            RollingWindow(
                rebalance_signal_date=rebalance_date,
                lookback_start=lookback_start,
                lookback_end=rebalance_date,
                hold_signal_dates=pd.DatetimeIndex(hold_signal_dates),
                metrics=window_metrics,
            )
        )

    return windows


def selection_rule_label(selection_rule: str) -> str:
    if selection_rule not in SELECTION_RULES:
        raise ValueError(f"Unsupported selection_rule: {selection_rule}")
    return str(SELECTION_RULES[selection_rule]["label"])


def selection_rule_description(selection_rule: str, drawdown_band: float) -> str:
    if selection_rule not in SELECTION_RULES:
        raise ValueError(f"Unsupported selection_rule: {selection_rule}")
    base = str(SELECTION_RULES[selection_rule]["description"])
    if selection_rule == "min_drawdown":
        return base
    return f"{base} 当前回撤保护带宽设为 `{drawdown_band:.2%}`。"


def _filter_drawdown_guard(metrics: pd.DataFrame, drawdown_band: float) -> pd.DataFrame:
    if drawdown_band < 0:
        raise ValueError("drawdown_band must be non-negative.")
    best_drawdown = float(metrics["max_drawdown"].max())
    cutoff = best_drawdown - drawdown_band
    guarded = metrics.loc[metrics["max_drawdown"] >= cutoff - 1e-12].copy()
    if guarded.empty:
        return metrics.copy()
    return guarded


def rank_candidates(metrics: pd.DataFrame, selection_rule: str = "min_drawdown", drawdown_band: float = 0.02) -> pd.DataFrame:
    if selection_rule not in SELECTION_RULES:
        raise ValueError(f"Unsupported selection_rule: {selection_rule}")

    ranked = metrics.copy()
    ranked["candidate_index"] = ranked.index

    if selection_rule == "min_drawdown":
        return ranked.sort_values(
            ["max_drawdown", "annual_return", "sharpe", "annual_volatility"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    guarded = _filter_drawdown_guard(ranked, drawdown_band)
    if selection_rule == "sharpe_guard":
        sort_cols = ["sharpe", "annual_return", "calmar", "annual_volatility", "max_drawdown"]
        ascending = [False, False, False, True, False]
    elif selection_rule == "calmar_guard":
        sort_cols = ["calmar", "annual_return", "sharpe", "annual_volatility", "max_drawdown"]
        ascending = [False, False, False, True, False]
    else:
        raise ValueError(f"Unsupported selection_rule: {selection_rule}")

    return guarded.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def build_quarterly_rolling_target_weights(
    signal_dates: pd.DatetimeIndex,
    run_index: pd.DatetimeIndex,
    ret_store: np.ndarray,
    turnover_store: np.ndarray,
    target_tensor: np.ndarray,
    base_weight_frame: pd.DataFrame,
    buckets: Sequence[str],
    lookback_months: int,
    selection_rule: str = "min_drawdown",
    drawdown_band: float = 0.02,
) -> RollingSelectionResult:
    windows = prepare_rebalance_windows(
        signal_dates=signal_dates,
        run_index=run_index,
        ret_store=ret_store,
        turnover_store=turnover_store,
        lookback_months=lookback_months,
    )
    return build_quarterly_rolling_target_weights_from_windows(
        windows=windows,
        signal_dates=signal_dates,
        target_tensor=target_tensor,
        base_weight_frame=base_weight_frame,
        buckets=buckets,
        selection_rule=selection_rule,
        drawdown_band=drawdown_band,
    )


def build_quarterly_rolling_target_weights_from_windows(
    windows: Sequence[RollingWindow],
    signal_dates: pd.DatetimeIndex,
    target_tensor: np.ndarray,
    base_weight_frame: pd.DataFrame,
    buckets: Sequence[str],
    selection_rule: str = "min_drawdown",
    drawdown_band: float = 0.02,
) -> RollingSelectionResult:
    signal_positions = {date: pos for pos, date in enumerate(signal_dates)}
    decisions = []
    target_rows = []

    for window in windows:
        ranked = rank_candidates(window.metrics, selection_rule=selection_rule, drawdown_band=drawdown_band)
        winner = ranked.iloc[0]
        winner_idx = int(winner["candidate_index"])

        base_row = base_weight_frame.iloc[winner_idx]
        decision_row = {
            "rebalance_signal_date": window.rebalance_signal_date,
            "lookback_start": window.lookback_start,
            "lookback_end": window.lookback_end,
            "hold_start_signal_date": window.hold_signal_dates[0],
            "hold_end_signal_date": window.hold_signal_dates[-1],
            "selected_candidate_index": winner_idx,
            "selection_rule": selection_rule,
            "drawdown_band": float(drawdown_band),
            "lookback_total_return": float(winner["total_return"]),
            "lookback_annual_return": float(winner["annual_return"]),
            "lookback_annual_volatility": float(winner["annual_volatility"]),
            "lookback_max_drawdown": float(winner["max_drawdown"]),
            "lookback_sharpe": float(winner["sharpe"]),
            "lookback_calmar": float(winner["calmar"]),
            "lookback_monthly_win_rate": float(winner["monthly_win_rate"]),
            **{bucket: float(base_row[bucket]) for bucket in buckets},
        }
        decisions.append(decision_row)

        for signal_date in window.hold_signal_dates:
            pos = signal_positions[signal_date]
            target_rows.append(
                {
                    "date": signal_date,
                    "selected_candidate_index": winner_idx,
                    **{bucket: float(target_tensor[pos, winner_idx, col_idx]) for col_idx, bucket in enumerate(buckets)},
                }
            )

    if not target_rows:
        raise ValueError("No quarterly target weights were generated. Check lookback window and signal dates.")

    decision_table = pd.DataFrame(decisions).sort_values("rebalance_signal_date").reset_index(drop=True)
    target_weights = pd.DataFrame(target_rows).drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    return RollingSelectionResult(decision_table=decision_table, target_weights=target_weights)
