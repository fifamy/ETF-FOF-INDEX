from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _score(raw: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    return float(np.tanh(raw / scale))


def _project_with_bounds(weights: pd.Series, bounds: Dict[str, Dict[str, float]]) -> pd.Series:
    mins = pd.Series({bucket: float(values["min"]) for bucket, values in bounds.items()})
    maxs = pd.Series({bucket: float(values["max"]) for bucket, values in bounds.items()})
    result = weights.clip(lower=mins, upper=maxs)

    for _ in range(20):
        total = float(result.sum())
        if abs(total - 1.0) <= 1e-10:
            break
        if total < 1.0:
            room = (maxs - result).clip(lower=0.0)
            free = room[room > 1e-12]
            if free.empty:
                break
            result.loc[free.index] += (1.0 - total) * free / free.sum()
        else:
            room = (result - mins).clip(lower=0.0)
            free = room[room > 1e-12]
            if free.empty:
                break
            result.loc[free.index] -= (total - 1.0) * free / free.sum()
        result = result.clip(lower=mins, upper=maxs)

    return result / result.sum()


def _normalize_mapping(mapping: Dict[str, float], index: pd.Index) -> pd.Series:
    series = pd.Series(mapping, dtype=float).reindex(index).fillna(0.0)
    total = float(series.sum())
    if total <= 0:
        raise ValueError("Allocation mapping must contain positive weights.")
    return series / total


def _enforce_equity_group_bounds(weights: pd.Series, config: Dict, bucket_order: pd.Index) -> pd.Series:
    group = config["group_bounds"]["equity_total"]
    equities = [bucket for bucket in list(group["buckets"]) if bucket in weights.index]
    current = float(weights[equities].sum())
    lower = float(group["min"])
    upper = float(group["max"])
    allocation_cfg = config["allocation"]

    if current > upper:
        excess = current - upper
        equity_split = weights[equities] / weights[equities].sum()
        weights.loc[equities] -= excess * equity_split
        over_sink_cfg = allocation_cfg.get(
            "equity_bound_over_sink",
            allocation_cfg.get("positive_equity_funding", {"rate_bond": 0.70, "money_market": 0.30}),
        )
        over_sink = _normalize_mapping(over_sink_cfg, weights.index)
        weights.loc[over_sink.index] += excess * over_sink
    elif current < lower:
        deficit = lower - current
        under_source_cfg = allocation_cfg.get(
            "equity_bound_under_source",
            allocation_cfg.get("negative_equity_sink", {"rate_bond": 0.70, "money_market": 0.30}),
        )
        under_target_cfg = allocation_cfg.get(
            "equity_bound_under_target",
            {name: config["strategic_weights"][name] for name in equities},
        )
        sink = _normalize_mapping(under_source_cfg, weights.index)
        target = _normalize_mapping(under_target_cfg, pd.Index(equities))
        available = weights[sink.index] - pd.Series(
            {name: config["bucket_bounds"][name]["min"] for name in sink.index}
        )
        take = deficit * sink
        if (available < take - 1e-12).any():
            take = available.clip(lower=0.0)
            deficit = float(take.sum())
        weights[sink.index] -= take
        weights.loc[target.index] += deficit * target

    return weights


def _apply_shift(weights: pd.Series, deltas: Dict[str, float]) -> pd.Series:
    updated = weights.copy()
    for bucket, delta in deltas.items():
        updated[bucket] += delta
    return updated


def compute_strategy_weights(signals: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bucket_order = list(config["bucket_order"])
    strategic = pd.Series(config["strategic_weights"], dtype=float).reindex(bucket_order)
    bounds = config["bucket_bounds"]
    allocation_cfg = config["allocation"]
    signal_cfg = config["signals"]
    equity_buckets = list(config["group_bounds"]["equity_total"]["buckets"])
    stress_reference_bucket = allocation_cfg.get("stress_reference_bucket", equity_buckets[0])
    equity_tilt_distribution = _normalize_mapping(
        allocation_cfg.get(
            "equity_tilt_distribution",
            {"equity_core": 2.0 / 3.0, "equity_defensive": 1.0 / 3.0},
        ),
        pd.Index(equity_buckets),
    )
    positive_equity_funding = _normalize_mapping(
        allocation_cfg.get("positive_equity_funding", {"rate_bond": 0.70, "money_market": 0.30}),
        strategic.index,
    )
    negative_equity_sink = _normalize_mapping(
        allocation_cfg.get("negative_equity_sink", {"rate_bond": 0.70, "money_market": 0.30}),
        strategic.index,
    )
    gold_bucket = allocation_cfg.get("gold_bucket", "gold")
    gold_funding = _normalize_mapping(
        allocation_cfg.get("gold_funding", {"rate_bond": 0.70, "money_market": 0.30}),
        strategic.index,
    )
    stress_sink = _normalize_mapping(
        allocation_cfg.get("stress_sink", {"gold": 0.20, "rate_bond": 0.56, "money_market": 0.24}),
        strategic.index,
    )

    weights_rows = []
    diagnostics_rows = []

    for date in signals.index.get_level_values("date").unique():
        snapshot = signals.xs(date).reindex(bucket_order)
        weights = strategic.copy()

        equity_momentum = float(snapshot.loc[equity_buckets, "composite_momentum"].mean())
        equity_valuation = float(snapshot.loc[equity_buckets, "valuation_score"].mean())
        equity_signal = (
            (1.0 - float(signal_cfg["valuation_weight"])) * _score(equity_momentum, float(signal_cfg["equity_scale"]))
            + float(signal_cfg["valuation_weight"]) * float(np.clip(equity_valuation, -1.0, 1.0))
        )

        equity_shift = float(allocation_cfg["max_equity_tilt"]) * equity_signal
        equity_funding = positive_equity_funding if equity_shift >= 0 else negative_equity_sink
        deltas: Dict[str, float] = {}
        for bucket, weight in equity_tilt_distribution.items():
            deltas[bucket] = deltas.get(bucket, 0.0) + equity_shift * float(weight)
        for bucket, weight in equity_funding.items():
            deltas[bucket] = deltas.get(bucket, 0.0) - equity_shift * float(weight)
        weights = _apply_shift(weights, deltas)

        gold_momentum = float(snapshot.loc[gold_bucket, "composite_momentum"])
        core_vol = float(snapshot.loc[stress_reference_bucket, "vol_20d"])
        core_drawdown = float(snapshot.loc[stress_reference_bucket, "drawdown_60d"])
        vol_flag = 1.0 if core_vol > float(signal_cfg["stress_vol_threshold"]) else 0.0
        drawdown_flag = 1.0 if core_drawdown < float(signal_cfg["stress_drawdown_threshold"]) else 0.0
        stress_level = 0.5 * (vol_flag + drawdown_flag)

        gold_signal = 0.6 * _score(gold_momentum, float(signal_cfg["gold_scale"])) + 0.4 * stress_level
        gold_shift = float(allocation_cfg["max_gold_tilt"]) * float(np.clip(gold_signal, -0.5, 1.0))
        gold_deltas = {gold_bucket: gold_shift}
        for bucket, weight in gold_funding.items():
            gold_deltas[bucket] = gold_deltas.get(bucket, 0.0) - gold_shift * float(weight)
        weights = _apply_shift(weights, gold_deltas)

        equity_cut = float(allocation_cfg["stress_equity_cut"]) * stress_level
        if equity_cut > 0:
            stress_deltas: Dict[str, float] = {}
            for bucket, weight in equity_tilt_distribution.items():
                stress_deltas[bucket] = stress_deltas.get(bucket, 0.0) - equity_cut * float(weight)
            for bucket, weight in stress_sink.items():
                stress_deltas[bucket] = stress_deltas.get(bucket, 0.0) + equity_cut * float(weight)
            weights = _apply_shift(weights, stress_deltas)

        weights = _enforce_equity_group_bounds(weights, config, strategic.index)
        weights = _project_with_bounds(weights, bounds)

        weight_row = weights.to_dict()
        weight_row["date"] = date
        weights_rows.append(weight_row)

        diagnostics_rows.append(
            {
                "date": date,
                "equity_signal": equity_signal,
                "equity_shift": equity_shift,
                "gold_signal": gold_signal,
                "gold_shift": gold_shift,
                "stress_level": stress_level,
                "core_vol_20d": core_vol,
                "core_drawdown_60d": core_drawdown,
            }
        )

    weight_frame = pd.DataFrame(weights_rows).set_index("date").sort_index()
    diagnostics_frame = pd.DataFrame(diagnostics_rows).set_index("date").sort_index()
    return weight_frame, diagnostics_frame


def compute_baseline_weights(signals: pd.DataFrame, config: Dict) -> pd.DataFrame:
    strategic = pd.Series(config["strategic_weights"], dtype=float)
    dates = signals.index.get_level_values("date").unique()
    baseline = pd.DataFrame(index=dates, columns=strategic.index, data=[strategic.values] * len(dates))
    baseline.index.name = "date"
    return baseline
