#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etf_fof_index.config import load_config, resolve_path, validate_config  # noqa: E402
from etf_fof_index.data import load_price_data, load_valuation_data  # noqa: E402
from etf_fof_index.signals import build_bucket_price_frame, compute_signals  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402


BUCKETS = [
    "equity_core_csi300",
    "equity_core_csia500",
    "equity_defensive_lowvol",
    "equity_defensive_dividend",
    "rate_bond_5y",
    "rate_bond_10y",
    "gold",
    "money_market",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix-based exhaustive 5% weight-grid research for V2.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "weight_grid_v2"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    return parser.parse_args()


def enumerate_weight_grid() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for a in range(5, 26, 5):
        for b in range(5, 21, 5):
            for c in range(5, 21, 5):
                for d in range(5, 21, 5):
                    eq = a + b + c + d
                    if eq < 30 or eq > 60:
                        continue
                    for e in range(10, 31, 5):
                        for f in range(10, 31, 5):
                            for g in range(5, 21, 5):
                                h = 100 - (a + b + c + d + e + f + g)
                                if h < 5 or h > 20 or h % 5 != 0:
                                    continue
                                rows.append(
                                    {
                                        "equity_core_csi300": a / 100.0,
                                        "equity_core_csia500": b / 100.0,
                                        "equity_defensive_lowvol": c / 100.0,
                                        "equity_defensive_dividend": d / 100.0,
                                        "rate_bond_5y": e / 100.0,
                                        "rate_bond_10y": f / 100.0,
                                        "gold": g / 100.0,
                                        "money_market": h / 100.0,
                                        "is_grid_candidate": True,
                                    }
                                )
    return pd.DataFrame(rows)


def add_current_v2(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    current = pd.DataFrame([{**{bucket: float(config["strategic_weights"][bucket]) for bucket in BUCKETS}, "is_grid_candidate": False}])
    return pd.concat([current, df], ignore_index=True)


def compute_signal_deltas(config: Dict, signals: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    bucket_order = list(config["bucket_order"])
    signal_dates = pd.DatetimeIndex(signals.index.get_level_values("date").unique())
    allocation_cfg = config["allocation"]
    signal_cfg = config["signals"]

    equity_buckets = list(config["group_bounds"]["equity_total"]["buckets"])
    stress_reference_bucket = allocation_cfg.get("stress_reference_bucket", equity_buckets[0])
    gold_bucket = allocation_cfg.get("gold_bucket", "gold")

    def mapping_vector(mapping: Dict[str, float], keys: List[str]) -> np.ndarray:
        series = pd.Series(mapping, dtype=float).reindex(keys).fillna(0.0)
        total = float(series.sum())
        if total <= 0:
            raise ValueError("Mapping must contain positive weights.")
        return (series / total).to_numpy(dtype=float)

    equity_dist = mapping_vector(
        allocation_cfg.get(
            "equity_tilt_distribution",
            {"equity_core_csi300": 0.35, "equity_core_csia500": 0.25, "equity_defensive_lowvol": 0.20, "equity_defensive_dividend": 0.20},
        ),
        equity_buckets,
    )
    pos_funding = mapping_vector(allocation_cfg["positive_equity_funding"], bucket_order)
    neg_funding = mapping_vector(allocation_cfg["negative_equity_sink"], bucket_order)
    gold_funding = mapping_vector(allocation_cfg["gold_funding"], bucket_order)
    stress_sink = mapping_vector(allocation_cfg["stress_sink"], bucket_order)

    deltas = []
    for date in signal_dates:
        snapshot = signals.xs(date).reindex(bucket_order)
        equity_momentum = float(snapshot.loc[equity_buckets, "composite_momentum"].mean())
        equity_valuation = float(snapshot.loc[equity_buckets, "valuation_score"].mean())
        equity_signal = (
            (1.0 - float(signal_cfg["valuation_weight"])) * math.tanh(equity_momentum / float(signal_cfg["equity_scale"]))
            + float(signal_cfg["valuation_weight"]) * float(np.clip(equity_valuation, -1.0, 1.0))
        )
        equity_shift = float(allocation_cfg["max_equity_tilt"]) * equity_signal
        equity_funding = pos_funding if equity_shift >= 0 else neg_funding

        delta = np.zeros(len(bucket_order), dtype=float)
        for i, bucket in enumerate(equity_buckets):
            delta[bucket_order.index(bucket)] += equity_shift * equity_dist[i]
        delta -= equity_shift * equity_funding

        gold_momentum = float(snapshot.loc[gold_bucket, "composite_momentum"])
        core_vol = float(snapshot.loc[stress_reference_bucket, "vol_20d"])
        core_drawdown = float(snapshot.loc[stress_reference_bucket, "drawdown_60d"])
        stress_level = 0.5 * (
            (1.0 if core_vol > float(signal_cfg["stress_vol_threshold"]) else 0.0)
            + (1.0 if core_drawdown < float(signal_cfg["stress_drawdown_threshold"]) else 0.0)
        )
        gold_signal = 0.6 * math.tanh(gold_momentum / float(signal_cfg["gold_scale"])) + 0.4 * stress_level
        gold_shift = float(allocation_cfg["max_gold_tilt"]) * float(np.clip(gold_signal, -0.5, 1.0))
        delta[bucket_order.index(gold_bucket)] += gold_shift
        delta -= gold_shift * gold_funding

        equity_cut = float(allocation_cfg["stress_equity_cut"]) * stress_level
        if equity_cut > 0:
            for i, bucket in enumerate(equity_buckets):
                delta[bucket_order.index(bucket)] -= equity_cut * equity_dist[i]
            delta += equity_cut * stress_sink

        deltas.append(delta)

    return signal_dates, np.vstack(deltas)


def enforce_equity_group_bounds(weights: np.ndarray, config: Dict) -> np.ndarray:
    out = weights.copy()
    allocation_cfg = config["allocation"]
    equity_buckets = list(config["group_bounds"]["equity_total"]["buckets"])
    eq_idx = [BUCKETS.index(b) for b in equity_buckets]
    lower = float(config["group_bounds"]["equity_total"]["min"])
    upper = float(config["group_bounds"]["equity_total"]["max"])

    def mapping(mapping: Dict[str, float], keys: List[str]) -> np.ndarray:
        s = pd.Series(mapping, dtype=float).reindex(keys).fillna(0.0)
        return (s / s.sum()).to_numpy(dtype=float)

    over_sink = mapping(allocation_cfg["equity_bound_over_sink"], BUCKETS)
    under_source = mapping(allocation_cfg["equity_bound_under_source"], BUCKETS)
    under_target = mapping(allocation_cfg["equity_bound_under_target"], equity_buckets)
    mins = np.array([float(config["bucket_bounds"][b]["min"]) for b in BUCKETS], dtype=float)

    eq_sum = out[:, eq_idx].sum(axis=1)
    over_mask = eq_sum > upper + 1e-12
    if over_mask.any():
        over_rows = np.where(over_mask)[0]
        eq_slice = out[over_rows][:, eq_idx]
        eq_share = eq_slice / eq_slice.sum(axis=1, keepdims=True)
        excess = (eq_sum[over_rows] - upper).reshape(-1, 1)
        out[np.ix_(over_rows, eq_idx)] -= excess * eq_share
        out[over_rows] += excess * over_sink

    eq_sum = out[:, eq_idx].sum(axis=1)
    under_mask = eq_sum < lower - 1e-12
    if under_mask.any():
        under_rows = np.where(under_mask)[0]
        deficit = (lower - eq_sum[under_rows]).reshape(-1, 1)
        take = deficit * under_source
        available = out[under_rows] - mins
        limited = np.minimum(take, np.clip(available, 0.0, None))
        actual = limited.sum(axis=1, keepdims=True)
        out[under_rows] -= limited
        out[np.ix_(under_rows, eq_idx)] += actual * under_target

    return out


def project_with_bounds(weights: np.ndarray, config: Dict) -> np.ndarray:
    mins = np.array([float(config["bucket_bounds"][b]["min"]) for b in BUCKETS], dtype=float)
    maxs = np.array([float(config["bucket_bounds"][b]["max"]) for b in BUCKETS], dtype=float)
    out = np.clip(weights, mins, maxs)

    for _ in range(20):
        totals = out.sum(axis=1)
        if np.allclose(totals, 1.0, atol=1e-10):
            break
        below = totals < 1.0 - 1e-12
        if below.any():
            room = np.clip(maxs - out[below], 0.0, None)
            room_sum = room.sum(axis=1, keepdims=True)
            add = np.divide((1.0 - totals[below]).reshape(-1, 1) * room, room_sum, out=np.zeros_like(room), where=room_sum > 0)
            out[below] += add
        above = totals > 1.0 + 1e-12
        if above.any():
            room = np.clip(out[above] - mins, 0.0, None)
            room_sum = room.sum(axis=1, keepdims=True)
            sub = np.divide((totals[above] - 1.0).reshape(-1, 1) * room, room_sum, out=np.zeros_like(room), where=room_sum > 0)
            out[above] -= sub
        out = np.clip(out, mins, maxs)

    totals = out.sum(axis=1, keepdims=True)
    out = np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)
    return out


def compute_target_weight_tensor(base_weights: np.ndarray, signal_deltas: np.ndarray, config: Dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    targets = []
    for delta in signal_deltas:
        w = base_weights + delta.reshape(1, -1)
        w = enforce_equity_group_bounds(w, config)
        w = project_with_bounds(w, config)
        targets.append(w)
    return np.stack(targets, axis=0)


def run_matrix_backtest(
    returns: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    target_tensor: np.ndarray,
    base_weights: np.ndarray,
    config: Dict,
) -> pd.DataFrame:
    run_returns = returns.loc[signal_dates[0] :, BUCKETS].copy()
    run_index = pd.DatetimeIndex(run_returns.index)
    ret = run_returns.to_numpy(dtype=float)
    n_cfg = base_weights.shape[0]
    n_days = ret.shape[0]
    cost_rate = float(config["costs"]["transaction_cost_bps"]) / 10000.0

    schedule = {}
    date_to_pos = {date: i for i, date in enumerate(run_index)}
    for k, signal_date in enumerate(signal_dates):
        pos = date_to_pos.get(signal_date)
        if pos is None or pos + 1 >= n_days:
            continue
        schedule[pos + 1] = target_tensor[k]

    current = base_weights.copy()
    ret_store = np.zeros((n_days, n_cfg), dtype=float)
    turnover_store = np.zeros((n_days, n_cfg), dtype=float)

    for i in range(n_days):
        daily = ret[i]
        portfolio_ret = (current * daily.reshape(1, -1)).sum(axis=1)
        gross = current * (1.0 + daily.reshape(1, -1))
        gross_sum = gross.sum(axis=1, keepdims=True)
        gross = np.divide(gross, gross_sum, out=current.copy(), where=gross_sum > 0)
        if i in schedule:
            target = schedule[i]
            turnover = np.abs(gross - target).sum(axis=1)
            portfolio_ret = portfolio_ret - turnover * cost_rate
            gross = target
            turnover_store[i] = turnover
        ret_store[i] = portfolio_ret
        current = gross

    levels = float(config["start_index_level"]) * np.cumprod(1.0 + ret_store, axis=0)
    return build_metric_frame(run_index, ret_store, turnover_store, levels)


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
    start = 0
    for _, grp in pd.Series(range(len(index)), index=months).groupby(level=0):
        locs = grp.to_numpy()
        monthly_groups.append(locs)
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


def add_rank_scores(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["rank_return"] = out["annual_return"].rank(ascending=False, method="average")
    out["rank_vol"] = out["annual_volatility"].rank(ascending=True, method="average")
    out["rank_drawdown"] = out["max_drawdown"].rank(ascending=False, method="average")
    out["rank_sharpe"] = out["sharpe"].rank(ascending=False, method="average")
    out["rank_calmar"] = out["calmar"].rank(ascending=False, method="average")
    out["rank_turnover"] = out["annualized_turnover_proxy"].rank(ascending=True, method="average")
    n = float(len(out))
    for col in ["rank_return", "rank_vol", "rank_drawdown", "rank_sharpe", "rank_calmar", "rank_turnover"]:
        out[f"score_{col[5:]}"] = (n - out[col]) / max(n - 1.0, 1.0) * 100.0
    out["composite_score"] = (
        out["score_return"] * 0.30
        + out["score_sharpe"] * 0.25
        + out["score_calmar"] * 0.20
        + out["score_drawdown"] * 0.15
        + out["score_turnover"] * 0.10
    )
    return out.sort_values(["composite_score", "annual_return"], ascending=[False, False]).reset_index(drop=True)


def mark_pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    vals = out[["annual_return", "annual_volatility", "max_drawdown", "annualized_turnover_proxy"]].copy()
    vals["risk_drawdown"] = -vals["max_drawdown"]
    out["is_pareto_efficient"] = False
    for i, row in vals.iterrows():
        dominated = (
            (vals["annual_return"] >= row["annual_return"])
            & (vals["annual_volatility"] <= row["annual_volatility"])
            & (vals["risk_drawdown"] <= row["risk_drawdown"])
            & (vals["annualized_turnover_proxy"] <= row["annualized_turnover_proxy"])
            & (
                (vals["annual_return"] > row["annual_return"])
                | (vals["annual_volatility"] < row["annual_volatility"])
                | (vals["risk_drawdown"] < row["risk_drawdown"])
                | (vals["annualized_turnover_proxy"] < row["annualized_turnover_proxy"])
            )
        )
        out.loc[i, "is_pareto_efficient"] = not bool(dominated.any())
    return out


def _circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none", stroke_width: float = 0.0, opacity: float = 1.0) -> str:
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.2f}" />'


def _text(x: float, y: float, value: str, font_size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{font_size}" text-anchor="{anchor}" font-weight="{weight}" font-family="Arial, sans-serif">{escaped}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#999", width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr} />'


def write_scatter_svg(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str, path: Path, highlight_row: int) -> None:
    width, height = 900, 620
    margin = {"left": 80, "right": 40, "top": 60, "bottom": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    data = df.copy()
    data["_x_value"] = -data[x_col] if x_col == "max_drawdown" else data[x_col]
    data["_y_value"] = data[y_col]
    xmin, xmax = float(data["_x_value"].min()), float(data["_x_value"].max())
    ymin, ymax = float(data["_y_value"].min()), float(data["_y_value"].max())
    if math.isclose(xmin, xmax):
        xmax += 1e-6
    if math.isclose(ymin, ymax):
        ymax += 1e-6

    def sx(v: float) -> float:
        return margin["left"] + (v - xmin) / (xmax - xmin) * plot_w

    def sy(v: float) -> float:
        return margin["top"] + plot_h - (v - ymin) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white" />',
        _text(width / 2, 28, title, font_size=20, anchor="middle", weight="bold"),
        _line(margin["left"], margin["top"] + plot_h, margin["left"] + plot_w, margin["top"] + plot_h, stroke="#333", width=1.5),
        _line(margin["left"], margin["top"], margin["left"], margin["top"] + plot_h, stroke="#333", width=1.5),
        _text(width / 2, height - 18, x_label, font_size=14, anchor="middle"),
        _text(18, height / 2, y_label, font_size=14, anchor="middle"),
    ]
    for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xv = xmin + (xmax - xmin) * pct
        yv = ymin + (ymax - ymin) * pct
        x = sx(xv)
        y = sy(yv)
        parts.append(_line(x, margin["top"], x, margin["top"] + plot_h, stroke="#e5e5e5", width=1.0))
        parts.append(_line(margin["left"], y, margin["left"] + plot_w, y, stroke="#e5e5e5", width=1.0))
        parts.append(_text(x, margin["top"] + plot_h + 20, f"{xv:.2%}" if xmax <= 1 else f"{xv:.2f}", font_size=11, anchor="middle"))
        parts.append(_text(margin["left"] - 8, y + 4, f"{yv:.2%}" if ymax <= 1 else f"{yv:.2f}", font_size=11, anchor="end"))

    smin, smax = float(data["composite_score"].min()), float(data["composite_score"].max())
    def color(score: float) -> str:
        t = 0.0 if math.isclose(smin, smax) else (score - smin) / (smax - smin)
        return f"rgb({int(30+180*t)},{int(120-60*t)},{int(220-120*t)})"

    for _, row in data.iterrows():
        parts.append(
            _circle(
                sx(float(row["_x_value"])),
                sy(float(row["_y_value"])),
                5.0 if bool(row.get("is_pareto_efficient", False)) else 3.0,
                fill=color(float(row["composite_score"])),
                stroke="#111" if bool(row.get("is_pareto_efficient", False)) else "none",
                stroke_width=0.8 if bool(row.get("is_pareto_efficient", False)) else 0.0,
                opacity=0.8 if bool(row.get("is_pareto_efficient", False)) else 0.45,
            )
        )

    row = data.iloc[highlight_row]
    hx, hy = sx(float(row["_x_value"])), sy(float(row["_y_value"]))
    parts.append(_circle(hx, hy, 7.5, fill="none", stroke="#d62728", stroke_width=2.0))
    parts.append(_text(hx + 10, hy - 10, "Current V2", font_size=12, weight="bold"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_top_bar_svg(df: pd.DataFrame, path: Path) -> None:
    top = df.head(12)
    width, height = 980, 560
    margin = {"left": 280, "right": 60, "top": 50, "bottom": 40}
    plot_w = width - margin["left"] - margin["right"]
    bar_h = 24
    gap = 12
    max_score = float(top["composite_score"].max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white" />',
        _text(width / 2, 26, "Top 12 Weight Configurations by Composite Score", font_size=20, anchor="middle", weight="bold"),
    ]
    for i, (_, row) in enumerate(top.iterrows()):
        y = margin["top"] + i * (bar_h + gap)
        score = float(row["composite_score"])
        bw = 0 if max_score <= 0 else score / max_score * plot_w
        label = f"#{int(row['rank_order'])} Eq={row['equity_total']:.0%} Bond={row['bond_total']:.0%} Gold={row['gold']:.0%} Cash={row['money_market']:.0%}"
        parts.append(_text(margin["left"] - 10, y + 16, label, font_size=11, anchor="end"))
        parts.append(f'<rect x="{margin["left"]:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{bar_h:.2f}" fill="#2f6db3" rx="4" ry="4" />')
        parts.append(_text(margin["left"] + bw + 8, y + 16, f"{score:.2f}", font_size=11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report_md(results: pd.DataFrame, output_dir: Path) -> str:
    top = results.head(10)
    lines = [
        "# Weight Grid Research V2",
        "",
        f"- total_feasible_configs: `{len(results)}`",
        f"- pareto_count: `{int(results['is_pareto_efficient'].sum())}`",
        "",
        "## Best Composite Configs",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(
            "- "
            f"#{int(row['rank_order'])} score={row['composite_score']:.2f} "
            f"ret={row['annual_return']:.2%} vol={row['annual_volatility']:.2%} "
            f"mdd={row['max_drawdown']:.2%} sharpe={row['sharpe']:.2f} "
            f"weights=[300 {row['equity_core_csi300']:.0%}, A500 {row['equity_core_csia500']:.0%}, "
            f"LV {row['equity_defensive_lowvol']:.0%}, DIV {row['equity_defensive_dividend']:.0%}, "
            f"5Y {row['rate_bond_5y']:.0%}, 10Y {row['rate_bond_10y']:.0%}, G {row['gold']:.0%}, M {row['money_market']:.0%}]"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- [grid_results.csv]({(output_dir / 'grid_results.csv').resolve()})",
            f"- [pareto_frontier.csv]({(output_dir / 'pareto_frontier.csv').resolve()})",
            f"- [return_drawdown.svg]({(output_dir / 'return_drawdown.svg').resolve()})",
            f"- [return_volatility.svg]({(output_dir / 'return_volatility.svg').resolve()})",
            f"- [top_composite_configs.svg]({(output_dir / 'top_composite_configs.svg').resolve()})",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    validate_config(config)
    universe = load_universe(resolve_path(config, config["paths"]["universe"]))
    prices = load_price_data(Path(args.prices))
    selection = select_representatives(universe, config["bucket_order"], available_symbols=prices.columns)
    bucket_prices = build_bucket_price_frame(prices, selection.bucket_to_symbol, config["bucket_order"])
    valuation_data = load_valuation_data(Path(args.valuation)) if args.valuation else pd.DataFrame()
    signals = compute_signals(bucket_prices=bucket_prices, valuation_data=valuation_data, config=config)

    combos = add_current_v2(enumerate_weight_grid(), config)
    base_weights = combos[BUCKETS].to_numpy(dtype=float)
    signal_dates, signal_deltas = compute_signal_deltas(config, signals)
    target_tensor = compute_target_weight_tensor(base_weights, signal_deltas, config)
    metrics = run_matrix_backtest(bucket_prices.pct_change().fillna(0.0), signal_dates, target_tensor, base_weights, config)

    results = pd.concat([combos.reset_index(drop=True), metrics], axis=1)
    results["config_id"] = range(1, len(results) + 1)
    results["equity_total"] = results[["equity_core_csi300", "equity_core_csia500", "equity_defensive_lowvol", "equity_defensive_dividend"]].sum(axis=1)
    results["bond_total"] = results[["rate_bond_5y", "rate_bond_10y"]].sum(axis=1)
    results["is_current_v2"] = ~results["is_grid_candidate"]
    results = add_rank_scores(results)
    results = mark_pareto_frontier(results)
    results["rank_order"] = range(1, len(results) + 1)

    current_idx = int(results.index[results["is_current_v2"]].tolist()[0])
    pareto = results[results["is_pareto_efficient"]].copy()

    results.to_csv(output_dir / "grid_results.csv", index=False)
    pareto.to_csv(output_dir / "pareto_frontier.csv", index=False)
    write_scatter_svg(results, "max_drawdown", "annual_return", "Max Drawdown", "Annual Return", "Weight Grid: Return vs Drawdown", output_dir / "return_drawdown.svg", current_idx)
    write_scatter_svg(results, "annual_volatility", "annual_return", "Annual Volatility", "Annual Return", "Weight Grid: Return vs Volatility", output_dir / "return_volatility.svg", current_idx)
    write_top_bar_svg(results, output_dir / "top_composite_configs.svg")
    (output_dir / "report.md").write_text(build_report_md(results, output_dir), encoding="utf-8")

    current = results.loc[current_idx]
    print(f"tested_configs={len(results)}")
    print(f"pareto_configs={len(pareto)}")
    print(
        f"current_v2_rank={int(current['rank_order'])} composite={current['composite_score']:.2f} "
        f"ret={current['annual_return']:.2%} vol={current['annual_volatility']:.2%} mdd={current['max_drawdown']:.2%}"
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
