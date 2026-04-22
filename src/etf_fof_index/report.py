from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _to_markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    table = frame.copy()
    if not index:
        table = table.reset_index(drop=True)

    headers = list(table.columns)
    rows = [[str(value) for value in row] for row in table.astype(object).itertuples(index=False, name=None)]

    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(values) -> str:
        padded = [str(value).ljust(width) for value, width in zip(values, widths)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    output = [render_row(headers), separator]
    output.extend(render_row(row) for row in rows)
    return "\n".join(output)


def _summary_metrics(index_series: pd.Series, return_series: pd.Series) -> Dict[str, float]:
    ann_factor = 252.0
    total_return = float(index_series.iloc[-1] / index_series.iloc[0] - 1.0)
    periods = max(len(return_series), 1)
    annual_return = float((1.0 + total_return) ** (ann_factor / periods) - 1.0)
    annual_vol = float(return_series.std(ddof=0) * np.sqrt(ann_factor))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    running_max = index_series.cummax()
    drawdown = index_series / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    monthly = return_series.resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)
    monthly_win_rate = float((monthly > 0).mean()) if not monthly.empty else 0.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": calmar,
        "monthly_win_rate": monthly_win_rate,
    }


def _format_metric_table(levels: pd.DataFrame) -> pd.DataFrame:
    strategy = _summary_metrics(levels["strategy_index"], levels["strategy_return"])
    baseline = _summary_metrics(levels["baseline_index"], levels["baseline_return"])
    table = pd.DataFrame({"strategy": strategy, "baseline": baseline})
    table["excess"] = table["strategy"] - table["baseline"]
    return table


def build_report(
    levels: pd.DataFrame,
    selected_universe: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    metric_table = _format_metric_table(levels)
    avg_weights = strategy_weights.mean().rename("average_weight").to_frame()
    latest_diag = diagnostics.tail(1).T.rename(columns={diagnostics.index[-1]: "latest"})

    lines = [
        "# ETF-FOF Index Research Report",
        "",
        "## Selected Instruments",
        "",
        _to_markdown_table(selected_universe, index=False),
        "",
        "## Performance Summary",
        "",
        _to_markdown_table(metric_table.reset_index().rename(columns={"index": "metric"}), index=False),
        "",
        "## Average Strategy Weights",
        "",
        _to_markdown_table(avg_weights.reset_index().rename(columns={"index": "bucket"}), index=False),
        "",
        "## Latest Diagnostics",
        "",
        _to_markdown_table(latest_diag.reset_index().rename(columns={"index": "metric"}), index=False),
        "",
        "## Turnover",
        "",
        f"- Average daily turnover: {levels['strategy_turnover'].mean():.4f}",
        f"- Total annualized turnover proxy: {levels['strategy_turnover'].sum() / max(len(levels), 1) * 252:.2f}",
    ]
    return "\n".join(lines)
