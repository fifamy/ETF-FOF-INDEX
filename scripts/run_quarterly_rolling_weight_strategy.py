#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import html
from pathlib import Path
from typing import Dict, Iterable, Optional

from _bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from etf_fof_index.backtest import run_backtest  # noqa: E402
from etf_fof_index.config import load_config, resolve_path, validate_config  # noqa: E402
from etf_fof_index.data import load_price_data, load_valuation_data  # noqa: E402
from etf_fof_index.report import _summary_metrics  # noqa: E402
from etf_fof_index.rolling import (  # noqa: E402
    SELECTION_RULES,
    build_quarterly_rolling_target_weights,
    selection_rule_description,
    selection_rule_label,
)
from etf_fof_index.signals import build_bucket_price_frame, compute_signals  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402
from run_weight_grid_research_v2_matrix import (  # noqa: E402
    BUCKETS,
    add_current_v2,
    compute_signal_deltas,
    compute_target_weight_tensor,
    enumerate_weight_grid,
)


PLOT_COLORS = {
    "动态季度滚动": "#d1495b",
    "当前V2": "#2e4057",
}

REPORT_LABELS = {
    "strategy": "策略",
    "total_return": "累计收益",
    "annual_return": "年化收益",
    "annual_volatility": "年化波动",
    "max_drawdown": "最大回撤",
    "sharpe": "夏普",
    "calmar": "Calmar",
    "monthly_win_rate": "月胜率",
    "avg_daily_turnover": "平均日换手",
    "annualized_turnover_proxy": "年化换手代理",
    "rebalance_signal_date": "调仓信号日",
    "lookback_start": "观察窗起点",
    "lookback_end": "观察窗终点",
    "hold_start_signal_date": "持有起点",
    "hold_end_signal_date": "持有终点",
    "selected_candidate_index": "候选编号",
    "lookback_total_return": "观察窗累计收益",
    "lookback_annual_return": "观察窗年化收益",
    "lookback_annual_volatility": "观察窗年化波动",
    "lookback_max_drawdown": "观察窗最大回撤",
    "lookback_sharpe": "观察窗夏普",
    "lookback_calmar": "观察窗Calmar",
    "lookback_monthly_win_rate": "观察窗月胜率",
    "equity_core_csi300": "沪深300",
    "equity_core_csia500": "中证A500",
    "equity_defensive_lowvol": "低波",
    "equity_defensive_dividend": "红利",
    "rate_bond_5y": "5Y债",
    "rate_bond_10y": "10Y债",
    "gold": "黄金",
    "money_market": "货币",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarterly rolling strategic-weight selection based on trailing drawdown.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2_index_proxy.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "rolling_quarterly_v2_index_proxy"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    parser.add_argument("--lookback-months", type=int, default=12, help="Trailing months used to pick each quarter's strategic weights.")
    parser.add_argument("--selection-rule", default="min_drawdown", choices=sorted(SELECTION_RULES))
    parser.add_argument("--drawdown-band", type=float, default=0.02, help="Allowed drawdown slack for guarded rules.")
    return parser.parse_args()


def _to_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.astype(object).itertuples(index=False, name=None)]
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(values: Iterable[str]) -> str:
        padded = [str(value).ljust(width) for value, width in zip(values, widths)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator] + [render_row(row) for row in rows])


def _format_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "num":
        return f"{value:.2f}"
    return str(value)


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [REPORT_LABELS.get(str(col), str(col)) for col in out.columns]
    return out


def _format_percent_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].map(lambda value: _format_value(float(value), "pct"))
    return out


def _summarize_backtest(levels: pd.DataFrame, label: str) -> Dict[str, float]:
    metrics = _summary_metrics(levels[f"{label}_index"], levels[f"{label}_return"])
    metrics["avg_daily_turnover"] = float(levels[f"{label}_turnover"].mean())
    metrics["annualized_turnover_proxy"] = float(levels[f"{label}_turnover"].sum() / max(len(levels), 1) * 252.0)
    return metrics


def _execution_date_map(index: pd.DatetimeIndex) -> Dict[pd.Timestamp, pd.Timestamp]:
    mapping = {}
    for i in range(len(index) - 1):
        mapping[index[i]] = index[i + 1]
    return mapping


def _run_matrix_backtest_detail(
    returns: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    target_tensor: np.ndarray,
    base_weights: np.ndarray,
    config: Dict,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
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

    return run_index, ret_store, turnover_store


def _format_axis_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.0%}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def build_line_chart_svg(levels: pd.DataFrame, title: str, output_path: Path, value_kind: str = "level") -> None:
    width = 1200
    height = 720
    pad_left = 90
    pad_right = 260
    pad_top = 60
    pad_bottom = 70
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    y_min = float(levels.min().min())
    y_max = float(levels.max().max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.06
    y_min -= y_pad
    y_max += y_pad

    n = max(len(levels) - 1, 1)

    def x_pos(i: int) -> float:
        return pad_left + plot_width * i / n

    def y_pos(v: float) -> float:
        return pad_top + plot_height * (1.0 - (v - y_min) / (y_max - y_min))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7" />',
        f'<text x="{pad_left}" y="32" font-size="22" font-family="Arial, sans-serif" font-weight="700" fill="#14213d">{html.escape(title)}</text>',
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_val = y_min + (y_max - y_min) * frac
        y = y_pos(y_val)
        parts.append(f'<line x1="{pad_left}" y1="{y:.2f}" x2="{pad_left + plot_width}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1" />')
        parts.append(
            f'<text x="{pad_left - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end" font-family="Arial, sans-serif" fill="#4a4a4a">{_format_axis_value(y_val, value_kind)}</text>'
        )

    tick_positions = sorted(set([0, len(levels) // 4, len(levels) // 2, (len(levels) * 3) // 4, len(levels) - 1]))
    for pos in tick_positions:
        x = x_pos(pos)
        label = levels.index[pos].strftime("%Y-%m")
        parts.append(f'<line x1="{x:.2f}" y1="{pad_top}" x2="{x:.2f}" y2="{pad_top + plot_height}" stroke="#efefef" stroke-width="1" />')
        parts.append(
            f'<text x="{x:.2f}" y="{pad_top + plot_height + 24}" font-size="12" text-anchor="middle" font-family="Arial, sans-serif" fill="#4a4a4a">{label}</text>'
        )

    for idx, column in enumerate(levels.columns):
        color = PLOT_COLORS.get(column, list(PLOT_COLORS.values())[idx % len(PLOT_COLORS)])
        stroke_width = 4 if column == "动态季度滚动" else 3
        pts = [f"{x_pos(i):.2f},{y_pos(float(v)):.2f}" for i, v in enumerate(levels[column])]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" points="{" ".join(pts)}" />')
        parts.append(
            f'<text x="{pad_left + plot_width + 24}" y="{pad_top + 24 + idx * 26}" font-size="13" font-family="Arial, sans-serif" fill="#1f2933">{html.escape(column)}</text>'
        )
        parts.append(
            f'<line x1="{pad_left + plot_width + 2}" y1="{pad_top + 18 + idx * 26}" x2="{pad_left + plot_width + 18}" y2="{pad_top + 18 + idx * 26}" stroke="{color}" stroke-width="4" />'
        )

    if y_min < 0 < y_max:
        y_zero = y_pos(0.0)
        parts.append(f'<line x1="{pad_left}" y1="{y_zero:.2f}" x2="{pad_left + plot_width}" y2="{y_zero:.2f}" stroke="#9c6644" stroke-width="1.4" />')

    parts.append(f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_height}" stroke="#555" stroke-width="1.2" />')
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_height}" x2="{pad_left + plot_width}" y2="{pad_top + plot_height}" stroke="#555" stroke-width="1.2" />'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def run_study(
    config_path: Path,
    price_path: Path,
    output_dir: Optional[Path],
    valuation_path: Optional[Path],
    lookback_months: int,
    selection_rule: str,
    drawdown_band: float,
    write_outputs: bool = True,
) -> Dict[str, object]:
    if write_outputs and output_dir is None:
        raise ValueError("output_dir is required when write_outputs=True.")
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    validate_config(config)
    universe = load_universe(resolve_path(config, config["paths"]["universe"]))
    prices = load_price_data(price_path)
    valuation_data = load_valuation_data(valuation_path) if valuation_path else pd.DataFrame()
    selection = select_representatives(universe, config["bucket_order"], available_symbols=prices.columns)
    bucket_prices = build_bucket_price_frame(prices, selection.bucket_to_symbol, config["bucket_order"]).ffill()
    signals = compute_signals(bucket_prices=bucket_prices, valuation_data=valuation_data, config=config)
    returns = bucket_prices.pct_change(fill_method=None).fillna(0.0)

    candidate_frame = add_current_v2(enumerate_weight_grid(), config)
    base_weights = candidate_frame[BUCKETS].to_numpy(dtype=float)
    signal_dates, signal_deltas = compute_signal_deltas(config, signals)
    target_tensor = compute_target_weight_tensor(base_weights, signal_deltas, config)
    run_index, ret_store, turnover_store = _run_matrix_backtest_detail(returns, signal_dates, target_tensor, base_weights, config)

    rolling_result = build_quarterly_rolling_target_weights(
        signal_dates=signal_dates,
        run_index=run_index,
        ret_store=ret_store,
        turnover_store=turnover_store,
        target_tensor=target_tensor,
        base_weight_frame=candidate_frame[BUCKETS],
        buckets=BUCKETS,
        lookback_months=int(lookback_months),
        selection_rule=selection_rule,
        drawdown_band=float(drawdown_band),
    )

    current_start = rolling_result.target_weights.index.min()
    current_target_weights = pd.DataFrame(target_tensor[:, 0, :], index=signal_dates, columns=BUCKETS).loc[current_start:]
    current_target_weights = current_target_weights.loc[rolling_result.target_weights.index]

    dynamic_config = copy.deepcopy(config)
    first_base_weights = rolling_result.decision_table.iloc[0][BUCKETS].astype(float).to_dict()
    dynamic_config["strategic_weights"] = first_base_weights
    dynamic_backtest = run_backtest(bucket_prices, rolling_result.target_weights[BUCKETS], dynamic_config, label="dynamic")
    current_backtest = run_backtest(bucket_prices, current_target_weights[BUCKETS], config, label="current")

    comparison_levels = dynamic_backtest.levels.join(current_backtest.levels, how="inner")
    comparison_index = pd.DataFrame(
        {
            "动态季度滚动": comparison_levels["dynamic_index"],
            "当前V2": comparison_levels["current_index"],
        }
    )
    comparison_drawdown = comparison_index.div(comparison_index.cummax()) - 1.0

    metrics_rows = [
        {
            "strategy": "动态季度滚动",
            "selection_rule": selection_rule,
            "selection_rule_label": selection_rule_label(selection_rule),
            "drawdown_band": float(drawdown_band),
            "lookback_months": int(lookback_months),
            **_summarize_backtest(comparison_levels[["dynamic_index", "dynamic_return", "dynamic_turnover"]], "dynamic"),
        },
        {
            "strategy": "当前V2",
            "selection_rule": selection_rule,
            "selection_rule_label": selection_rule_label(selection_rule),
            "drawdown_band": float(drawdown_band),
            "lookback_months": int(lookback_months),
            **_summarize_backtest(comparison_levels[["current_index", "current_return", "current_turnover"]], "current"),
        },
    ]
    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table["drawdown_improvement"] = metrics_table["max_drawdown"] - float(
        metrics_table.loc[metrics_table["strategy"] == "当前V2", "max_drawdown"].iloc[0]
    )

    exec_map = _execution_date_map(bucket_prices.index)
    decisions = rolling_result.decision_table.copy()
    decisions["execution_date"] = decisions["rebalance_signal_date"].map(exec_map)
    decision_cols = [
        "rebalance_signal_date",
        "execution_date",
        "lookback_start",
        "lookback_end",
        "hold_start_signal_date",
        "hold_end_signal_date",
        "selected_candidate_index",
        "selection_rule",
        "drawdown_band",
        "lookback_total_return",
        "lookback_annual_return",
        "lookback_annual_volatility",
        "lookback_max_drawdown",
        "lookback_sharpe",
        "lookback_calmar",
        "lookback_monthly_win_rate",
        *BUCKETS,
    ]
    decisions = decisions[decision_cols]

    selection_counts = decisions["selected_candidate_index"].value_counts().rename_axis("selected_candidate_index").reset_index(name="count")

    sample_start = comparison_index.index.min().strftime("%Y-%m-%d")
    sample_end = comparison_index.index.max().strftime("%Y-%m-%d")
    dynamic_row = metrics_table.loc[metrics_table["strategy"] == "动态季度滚动"].iloc[0]
    current_row = metrics_table.loc[metrics_table["strategy"] == "当前V2"].iloc[0]
    drawdown_improvement = float(dynamic_row["max_drawdown"] - current_row["max_drawdown"])
    annual_return_delta = float(dynamic_row["annual_return"] - current_row["annual_return"])
    rule_label = selection_rule_label(selection_rule)
    rule_description = selection_rule_description(selection_rule, float(drawdown_band))

    if write_outputs:
        comparison_index.index.name = "date"
        comparison_drawdown.index.name = "date"
        rolling_result.target_weights.to_csv(output_dir / "dynamic_target_weights.csv", index=True)
        decisions.to_csv(output_dir / "rolling_decisions.csv", index=False)
        selection_counts.to_csv(output_dir / "selection_counts.csv", index=False)
        comparison_index.to_csv(output_dir / "comparison_index_levels.csv", index=True)
        comparison_drawdown.to_csv(output_dir / "comparison_drawdowns.csv", index=True)
        metrics_table.to_csv(output_dir / "comparison_metrics.csv", index=False)

        build_line_chart_svg(comparison_index, "动态季度滚动 vs 当前V2：净值对比", output_dir / "nav_comparison.svg", value_kind="level")
        build_line_chart_svg(comparison_drawdown, "动态季度滚动 vs 当前V2：回撤对比", output_dir / "drawdown_comparison.svg", value_kind="pct")

        metrics_md = metrics_table[["strategy", "total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar", "monthly_win_rate", "avg_daily_turnover", "annualized_turnover_proxy"]].copy()
        metrics_md = _format_percent_columns(
            metrics_md,
            ["total_return", "annual_return", "annual_volatility", "max_drawdown", "monthly_win_rate", "avg_daily_turnover", "annualized_turnover_proxy"],
        )
        for column in ["sharpe", "calmar"]:
            metrics_md[column] = metrics_md[column].map(lambda value: _format_value(float(value), "num"))
        metrics_md = _rename_columns(metrics_md)

        decisions_md = decisions.tail(8).copy()
        decisions_md = _format_percent_columns(
            decisions_md,
            ["drawdown_band", "lookback_total_return", "lookback_annual_return", "lookback_annual_volatility", "lookback_max_drawdown", "lookback_monthly_win_rate", *BUCKETS],
        )
        for column in ["lookback_sharpe", "lookback_calmar"]:
            decisions_md[column] = decisions_md[column].map(lambda value: _format_value(float(value), "num"))
        decisions_md = _rename_columns(decisions_md)

        report_lines = [
            "# 动态季度滚动权重研究",
            "",
            "## 方法",
            "",
            f"- 样本区间：`{sample_start}` 至 `{sample_end}`",
            f"- 观察窗：每次调仓前回看 `{int(lookback_months)}` 个月",
            "- 调仓频率：每 3 个月一次，在季末信号日选新的战略权重",
            f"- 选优规则：`{rule_label}`",
            f"- 规则说明：{rule_description}",
            "- 季度内执行：保留现有月度信号调权与交易成本口径，只替换季度层面的战略配比",
            "",
            "## 结果摘要",
            "",
            f"- 动态季度滚动：年化 `{dynamic_row['annual_return']:.2%}`，最大回撤 `{dynamic_row['max_drawdown']:.2%}`，夏普 `{dynamic_row['sharpe']:.2f}`",
            f"- 当前V2：年化 `{current_row['annual_return']:.2%}`，最大回撤 `{current_row['max_drawdown']:.2%}`，夏普 `{current_row['sharpe']:.2f}`",
            f"- 回撤改善：`{drawdown_improvement:.2%}`（正值代表动态方案回撤更浅）",
            f"- 年化变化：`{annual_return_delta:.2%}`",
            "",
            "## 图表",
            "",
            f"![净值对比]({(output_dir / 'nav_comparison.svg').resolve()})",
            "",
            f"![回撤对比]({(output_dir / 'drawdown_comparison.svg').resolve()})",
            "",
            "## 指标对比",
            "",
            _to_markdown_table(metrics_md),
            "",
            "## 最近 8 次季度选权重",
            "",
            _to_markdown_table(decisions_md),
            "",
            "## 文件",
            "",
            f"- [comparison_metrics.csv]({(output_dir / 'comparison_metrics.csv').resolve()})",
            f"- [comparison_index_levels.csv]({(output_dir / 'comparison_index_levels.csv').resolve()})",
            f"- [comparison_drawdowns.csv]({(output_dir / 'comparison_drawdowns.csv').resolve()})",
            f"- [rolling_decisions.csv]({(output_dir / 'rolling_decisions.csv').resolve()})",
            f"- [dynamic_target_weights.csv]({(output_dir / 'dynamic_target_weights.csv').resolve()})",
            f"- [selection_counts.csv]({(output_dir / 'selection_counts.csv').resolve()})",
            f"- [nav_comparison.svg]({(output_dir / 'nav_comparison.svg').resolve()})",
            f"- [drawdown_comparison.svg]({(output_dir / 'drawdown_comparison.svg').resolve()})",
        ]
        (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "lookback_months": int(lookback_months),
        "selection_rule": selection_rule,
        "selection_rule_label": rule_label,
        "drawdown_band": float(drawdown_band),
        "comparison_index": comparison_index,
        "comparison_drawdown": comparison_drawdown,
        "metrics_table": metrics_table,
        "decisions": decisions,
        "selection_counts": selection_counts,
        "drawdown_improvement": drawdown_improvement,
        "annual_return_delta": annual_return_delta,
    }


def main() -> None:
    args = parse_args()
    result = run_study(
        config_path=Path(args.config),
        price_path=Path(args.prices),
        output_dir=Path(args.output_dir),
        valuation_path=Path(args.valuation) if args.valuation else None,
        lookback_months=int(args.lookback_months),
        selection_rule=str(args.selection_rule),
        drawdown_band=float(args.drawdown_band),
        write_outputs=True,
    )

    print(f"output_dir={Path(args.output_dir).resolve()}")
    print(f"start={result['sample_start']}")
    print(f"end={result['sample_end']}")
    print(f"lookback_months={int(args.lookback_months)}")
    metrics_table = result["metrics_table"]
    dynamic_row = metrics_table.loc[metrics_table["strategy"] == "动态季度滚动"].iloc[0]
    current_row = metrics_table.loc[metrics_table["strategy"] == "当前V2"].iloc[0]
    print(f"selection_rule={args.selection_rule}")
    print(f"dynamic_max_drawdown={dynamic_row['max_drawdown']:.6f}")
    print(f"current_max_drawdown={current_row['max_drawdown']:.6f}")


if __name__ == "__main__":
    main()
