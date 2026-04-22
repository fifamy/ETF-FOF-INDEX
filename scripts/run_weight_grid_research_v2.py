#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.config import load_config, resolve_path, validate_config  # noqa: E402
from etf_fof_index.data import load_price_data, load_valuation_data  # noqa: E402
from etf_fof_index.report import _summary_metrics  # noqa: E402
from etf_fof_index.signals import build_bucket_price_frame, compute_signals  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402
from etf_fof_index.weights import compute_strategy_weights  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Exhaustive 5% weight-grid research for V2.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "weight_grid_v2"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    return parser.parse_args()


def enumerate_weight_grid() -> List[Dict[str, float]]:
    combos: List[Dict[str, float]] = []
    for a in range(5, 26, 5):
        for b in range(5, 21, 5):
            for c in range(5, 21, 5):
                for d in range(5, 21, 5):
                    equity_total = a + b + c + d
                    if equity_total < 30 or equity_total > 60:
                        continue
                    for e in range(10, 31, 5):
                        for f in range(10, 31, 5):
                            for g in range(5, 21, 5):
                                h = 100 - (a + b + c + d + e + f + g)
                                if h < 5 or h > 20 or h % 5 != 0:
                                    continue
                                combos.append(
                                    {
                                        "equity_core_csi300": a / 100.0,
                                        "equity_core_csia500": b / 100.0,
                                        "equity_defensive_lowvol": c / 100.0,
                                        "equity_defensive_dividend": d / 100.0,
                                        "rate_bond_5y": e / 100.0,
                                        "rate_bond_10y": f / 100.0,
                                        "gold": g / 100.0,
                                        "money_market": h / 100.0,
                                    }
                                )
    return combos


def summarize_strategy(levels: pd.DataFrame) -> Dict[str, float]:
    metrics = _summary_metrics(levels["strategy_index"], levels["strategy_return"])
    metrics["avg_daily_turnover"] = float(levels["strategy_turnover"].mean())
    metrics["annualized_turnover_proxy"] = float(levels["strategy_turnover"].sum() / max(len(levels), 1) * 252.0)
    return metrics


def fast_backtest_metrics(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: Dict,
) -> Dict[str, float]:
    bucket_order = list(config["bucket_order"])
    strategic = pd.Series(config["strategic_weights"], dtype=float).reindex(bucket_order).to_numpy(dtype=float)
    returns_frame = returns.loc[target_weights.index[0] :, bucket_order].copy()
    run_index = returns_frame.index
    returns_array = returns_frame.to_numpy(dtype=float)
    cost_rate = float(config["costs"]["transaction_cost_bps"]) / 10000.0

    date_to_pos = {date: i for i, date in enumerate(run_index)}
    exec_schedule: Dict[int, pd.Series] = {}
    for signal_date in target_weights.index:
        pos = date_to_pos.get(signal_date)
        if pos is None or pos + 1 >= len(run_index):
            continue
        exec_schedule[pos + 1] = target_weights.loc[signal_date].reindex(bucket_order).astype(float)

    current = strategic.copy()
    levels: List[float] = []
    strategy_returns: List[float] = []
    turnovers: List[float] = []
    level = float(config["start_index_level"])

    for i in range(len(run_index)):
        daily_ret = returns_array[i]
        portfolio_return = float((current * daily_ret).sum())
        gross = current * (1.0 + daily_ret)
        gross_sum = float(gross.sum())
        gross = gross / gross_sum if gross_sum > 0 else current.copy()
        turnover = 0.0

        if i in exec_schedule:
            target = exec_schedule[i].to_numpy(dtype=float)
            turnover = float(abs(gross - target).sum())
            portfolio_return -= turnover * cost_rate
            gross = target

        level *= 1.0 + portfolio_return
        current = gross
        levels.append(level)
        strategy_returns.append(portfolio_return)
        turnovers.append(turnover)

    level_frame = pd.DataFrame(
        {
            "strategy_index": levels,
            "strategy_return": strategy_returns,
            "strategy_turnover": turnovers,
        },
        index=run_index,
    )
    return summarize_strategy(level_frame)


def add_rank_scores(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["rank_return"] = out["annual_return"].rank(ascending=False, method="average")
    out["rank_vol"] = out["annual_volatility"].rank(ascending=True, method="average")
    out["rank_drawdown"] = out["max_drawdown"].rank(ascending=False, method="average")
    out["rank_sharpe"] = out["sharpe"].rank(ascending=False, method="average")
    out["rank_calmar"] = out["calmar"].rank(ascending=False, method="average")
    out["rank_turnover"] = out["annualized_turnover_proxy"].rank(ascending=True, method="average")

    n = float(len(out))
    for col in [
        "rank_return",
        "rank_vol",
        "rank_drawdown",
        "rank_sharpe",
        "rank_calmar",
        "rank_turnover",
    ]:
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
    metrics = out[["annual_return", "annual_volatility", "max_drawdown", "annualized_turnover_proxy"]].copy()
    metrics["risk_drawdown"] = -metrics["max_drawdown"]
    out["is_pareto_efficient"] = False

    for i, row in metrics.iterrows():
        dominated = (
            (metrics["annual_return"] >= row["annual_return"])
            & (metrics["annual_volatility"] <= row["annual_volatility"])
            & (metrics["risk_drawdown"] <= row["risk_drawdown"])
            & (metrics["annualized_turnover_proxy"] <= row["annualized_turnover_proxy"])
            & (
                (metrics["annual_return"] > row["annual_return"])
                | (metrics["annual_volatility"] < row["annual_volatility"])
                | (metrics["risk_drawdown"] < row["risk_drawdown"])
                | (metrics["annualized_turnover_proxy"] < row["annualized_turnover_proxy"])
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


def write_scatter_svg(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
    highlight_index: int = 0,
) -> None:
    width, height = 900, 620
    margin = {"left": 80, "right": 40, "top": 60, "bottom": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    data = df.copy()
    if x_col == "max_drawdown":
        data["_x_value"] = -data[x_col]
    else:
        data["_x_value"] = data[x_col]
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

    score_min = float(data["composite_score"].min())
    score_max = float(data["composite_score"].max())
    def color(score: float) -> str:
        t = 0.0 if math.isclose(score_min, score_max) else (score - score_min) / (score_max - score_min)
        r = int(40 + 180 * t)
        g = int(80 + 100 * (1 - t))
        b = int(200 - 120 * t)
        return f"rgb({r},{g},{b})"

    for i, row in data.iterrows():
        x = sx(float(row["_x_value"]))
        y = sy(float(row["_y_value"]))
        if bool(row.get("is_pareto_efficient", False)):
            parts.append(_circle(x, y, 5.5, fill=color(float(row["composite_score"])), stroke="#111", stroke_width=1.0, opacity=0.9))
        else:
            parts.append(_circle(x, y, 3.2, fill=color(float(row["composite_score"])), opacity=0.55))

    if highlight_index in data.index:
        row = data.loc[highlight_index]
        x = sx(float(row["_x_value"]))
        y = sy(float(row["_y_value"]))
        parts.append(_circle(x, y, 7.5, fill="none", stroke="#d62728", stroke_width=2.2))
        parts.append(_text(x + 10, y - 10, "Current V2", font_size=12, anchor="start", weight="bold"))

    top = data.head(8)
    for _, row in top.iterrows():
        x = sx(float(row["_x_value"]))
        y = sy(float(row["_y_value"]))
        parts.append(_text(x + 8, y - 6, f"#{int(row['rank_order'])}", font_size=10, anchor="start"))

    legend_x = width - 170
    legend_y = 70
    parts.append(_text(legend_x, legend_y, "Color = composite score", font_size=12, weight="bold"))
    for i, label in enumerate(["low", "mid", "high"]):
        score = score_min + (score_max - score_min) * (i / 2 if score_max > score_min else 0)
        parts.append(_circle(legend_x + i * 50, legend_y + 20, 5, fill=color(score), stroke="#111", stroke_width=0.5))
        parts.append(_text(legend_x + i * 50 + 10, legend_y + 24, label, font_size=11))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_top_bar_svg(df: pd.DataFrame, path: Path) -> None:
    top = df.head(12).copy()
    width, height = 960, 560
    margin = {"left": 260, "right": 50, "top": 50, "bottom": 40}
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
        bar_w = 0 if max_score <= 0 else score / max_score * plot_w
        label = (
            f"#{int(row['rank_order'])} "
            f"E={row['equity_total']:.0%} B={row['bond_total']:.0%} G={row['gold']:.0%} M={row['money_market']:.0%}"
        )
        parts.append(_text(margin["left"] - 10, y + 16, label, font_size=11, anchor="end"))
        parts.append(
            f'<rect x="{margin["left"]:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2f6db3" rx="4" ry="4" />'
        )
        parts.append(_text(margin["left"] + bar_w + 8, y + 16, f"{score:.2f}", font_size=11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report_md(results: pd.DataFrame, output_dir: Path) -> str:
    top = results.head(10).copy()
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
            f"#{int(row['rank_order'])} "
            f"score={row['composite_score']:.2f} "
            f"ret={row['annual_return']:.2%} "
            f"vol={row['annual_volatility']:.2%} "
            f"mdd={row['max_drawdown']:.2%} "
            f"sharpe={row['sharpe']:.2f} "
            f"weights=[300 {row['equity_core_csi300']:.0%}, A500 {row['equity_core_csia500']:.0%}, "
            f"LV {row['equity_defensive_lowvol']:.0%}, DIV {row['equity_defensive_dividend']:.0%}, "
            f"5Y {row['rate_bond_5y']:.0%}, 10Y {row['rate_bond_10y']:.0%}, "
            f"G {row['gold']:.0%}, M {row['money_market']:.0%}]"
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
    returns = bucket_prices.pct_change().fillna(0.0)

    combos = enumerate_weight_grid()
    current_weights = pd.Series(config["strategic_weights"], dtype=float).reindex(BUCKETS)

    rows: List[Dict[str, float]] = []
    for i, weights in enumerate(combos, start=1):
        cfg = dict(config)
        cfg["strategic_weights"] = weights
        strategy_weights, _ = compute_strategy_weights(signals, cfg)
        metrics = fast_backtest_metrics(returns, strategy_weights, cfg)
        row = {
            "config_id": i,
            **weights,
            **metrics,
        }
        row["equity_total"] = (
            row["equity_core_csi300"]
            + row["equity_core_csia500"]
            + row["equity_defensive_lowvol"]
            + row["equity_defensive_dividend"]
        )
        row["bond_total"] = row["rate_bond_5y"] + row["rate_bond_10y"]
        row["is_current_v2"] = all(abs(row[b] - float(current_weights[b])) < 1e-9 for b in BUCKETS)
        rows.append(row)
        if i % 500 == 0:
            pd.DataFrame(rows).to_csv(output_dir / "grid_results_partial.csv", index=False)
            print(f"progress={i}/{len(combos)}")

    results = pd.DataFrame(rows)
    results = add_rank_scores(results)
    results = mark_pareto_frontier(results)
    results["rank_order"] = range(1, len(results) + 1)

    current_idx = int(results.index[results["is_current_v2"]].tolist()[0]) if results["is_current_v2"].any() else 0
    pareto = results[results["is_pareto_efficient"]].copy()

    results.to_csv(output_dir / "grid_results.csv", index=False)
    pareto.to_csv(output_dir / "pareto_frontier.csv", index=False)
    write_scatter_svg(
        results,
        x_col="max_drawdown",
        y_col="annual_return",
        x_label="Max Drawdown",
        y_label="Annual Return",
        title="Weight Grid: Annual Return vs Max Drawdown",
        path=output_dir / "return_drawdown.svg",
        highlight_index=current_idx,
    )
    write_scatter_svg(
        results,
        x_col="annual_volatility",
        y_col="annual_return",
        x_label="Annual Volatility",
        y_label="Annual Return",
        title="Weight Grid: Annual Return vs Annual Volatility",
        path=output_dir / "return_volatility.svg",
        highlight_index=current_idx,
    )
    write_top_bar_svg(results, output_dir / "top_composite_configs.svg")
    (output_dir / "report.md").write_text(build_report_md(results, output_dir), encoding="utf-8")

    print(f"tested_configs={len(results)}")
    print(f"pareto_configs={len(pareto)}")
    if results["is_current_v2"].any():
        current = results.loc[current_idx]
        print(
            "current_v2_rank="
            f"{int(current['rank_order'])} composite={current['composite_score']:.2f} "
            f"ret={current['annual_return']:.2%} vol={current['annual_volatility']:.2%} "
            f"mdd={current['max_drawdown']:.2%}"
        )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
