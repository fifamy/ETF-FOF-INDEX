#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, Iterable, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.report import _summary_metrics  # noqa: E402
from etf_fof_index.rolling import selection_rule_label  # noqa: E402
from run_quarterly_rolling_weight_strategy import run_study  # noqa: E402


STRATEGIES = [
    {
        "name": "12M回撤优先(正式版)",
        "selection_rule": "min_drawdown",
        "lookback_months": 12,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_official",
        "source": ROOT / "output" / "rolling_quarterly_v2_official" / "comparison_index_levels.csv",
        "note": "正式版",
    },
    {
        "name": "6M回撤优先(极致降回撤)",
        "selection_rule": "min_drawdown",
        "lookback_months": 6,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_mindd6",
        "source": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_mindd6" / "comparison_index_levels.csv",
        "note": "纯压回撤",
    },
    {
        "name": "12M收益优先(进攻版)",
        "selection_rule": "max_return",
        "lookback_months": 12,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_return_candidate_maxret12",
        "source": ROOT / "output" / "rolling_quarterly_v2_return_candidate_maxret12" / "comparison_index_levels.csv",
        "note": "冲收益",
    },
    {
        "name": "36M夏普护栏(均衡版)",
        "selection_rule": "sharpe_guard",
        "lookback_months": 36,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_sharpe36",
        "source": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_sharpe36" / "comparison_index_levels.csv",
        "note": "更均衡",
    },
    {
        "name": "36MCalmar护栏(防守版)",
        "selection_rule": "calmar_guard",
        "lookback_months": 36,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_calmar36",
        "source": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_calmar36" / "comparison_index_levels.csv",
        "note": "更防守",
    },
]

COLOR_MAP = {
    "当前V2": "#243b53",
    "12M回撤优先(正式版)": "#d1495b",
    "6M回撤优先(极致降回撤)": "#edae49",
    "12M收益优先(进攻版)": "#c1121f",
    "36M夏普护栏(均衡版)": "#00798c",
    "36MCalmar护栏(防守版)": "#4f772d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual comparison board for rolling quarterly strategies.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2_index_proxy.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "rolling_quarterly_strategy_comparison"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    parser.add_argument("--drawdown-band", type=float, default=0.02)
    parser.add_argument(
        "--common-summary",
        default=str(ROOT / "output" / "rolling_quarterly_v2_rule_sweep" / "common_window_summary.csv"),
    )
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


def _format_axis_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.0%}" if abs(value) >= 0.1 else f"{value:.1%}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def build_line_chart_svg(
    levels: pd.DataFrame,
    title: str,
    output_path: Path,
    value_kind: str = "level",
    color_map: Dict[str, str] | None = None,
) -> None:
    width = 1280
    height = 760
    pad_left = 90
    pad_right = 320
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
        parts.append(f'<line x1="{pad_left}" y1="{y:.2f}" x2="{pad_left + plot_width}" y2="{y:.2f}" stroke="#ddd7ca" stroke-width="1" />')
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
        color = (color_map or {}).get(column, list(COLOR_MAP.values())[idx % len(COLOR_MAP)])
        stroke_width = 4 if column == "当前V2" else 3
        pts = [f"{x_pos(i):.2f},{y_pos(float(v)):.2f}" for i, v in enumerate(levels[column])]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" points="{" ".join(pts)}" />')
        parts.append(
            f'<text x="{pad_left + plot_width + 28}" y="{pad_top + 28 + idx * 28}" font-size="13" font-family="Arial, sans-serif" fill="#1f2933">{html.escape(column)}</text>'
        )
        parts.append(
            f'<line x1="{pad_left + plot_width + 4}" y1="{pad_top + 22 + idx * 28}" x2="{pad_left + plot_width + 22}" y2="{pad_top + 22 + idx * 28}" stroke="{color}" stroke-width="4" />'
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


def build_scatter_svg(metrics: pd.DataFrame, output_path: Path, color_map: Dict[str, str]) -> None:
    width = 860
    height = 620
    pad_left = 90
    pad_right = 50
    pad_top = 60
    pad_bottom = 70
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    x_values = metrics["drawdown_depth"]
    y_values = metrics["annual_return"]
    x_min = 0.0
    x_max = float(x_values.max()) * 1.15
    y_min = min(0.0, float(y_values.min()) * 0.85)
    y_max = float(y_values.max()) * 1.15
    if y_max <= y_min:
        y_max = y_min + 0.01

    def x_pos(v: float) -> float:
        return pad_left + plot_width * (v - x_min) / (x_max - x_min)

    def y_pos(v: float) -> float:
        return pad_top + plot_height * (1.0 - (v - y_min) / (y_max - y_min))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7" />',
        '<text x="40" y="30" font-size="22" font-family="Arial, sans-serif" font-weight="700" fill="#14213d">风险收益散点图</text>',
        '<text x="40" y="52" font-size="12" font-family="Arial, sans-serif" fill="#5c5c5c">横轴越靠左越好，纵轴越高越好；圆点标签显示策略名与夏普。</text>',
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_val = x_min + (x_max - x_min) * frac
        x = x_pos(x_val)
        parts.append(f'<line x1="{x:.2f}" y1="{pad_top}" x2="{x:.2f}" y2="{pad_top + plot_height}" stroke="#efefef" stroke-width="1" />')
        parts.append(
            f'<text x="{x:.2f}" y="{pad_top + plot_height + 24:.2f}" font-size="12" text-anchor="middle" font-family="Arial, sans-serif" fill="#4a4a4a">{x_val:.1%}</text>'
        )

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_val = y_min + (y_max - y_min) * frac
        y = y_pos(y_val)
        parts.append(f'<line x1="{pad_left}" y1="{y:.2f}" x2="{pad_left + plot_width}" y2="{y:.2f}" stroke="#efefef" stroke-width="1" />')
        parts.append(
            f'<text x="{pad_left - 10:.2f}" y="{y + 4:.2f}" font-size="12" text-anchor="end" font-family="Arial, sans-serif" fill="#4a4a4a">{y_val:.1%}</text>'
        )

    for idx, row in metrics.iterrows():
        name = str(row["strategy"])
        x = x_pos(float(row["drawdown_depth"]))
        y = y_pos(float(row["annual_return"]))
        color = color_map[name]
        radius = 9 if name == "当前V2" else 11
        label_dx = 12 if idx % 2 == 0 else -12
        anchor = "start" if label_dx > 0 else "end"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" fill-opacity="0.92" stroke="#ffffff" stroke-width="2" />')
        parts.append(
            f'<text x="{x + label_dx:.2f}" y="{y - 10:.2f}" font-size="12" text-anchor="{anchor}" font-family="Arial, sans-serif" fill="#243b53">{html.escape(name)}</text>'
        )
        parts.append(
            f'<text x="{x + label_dx:.2f}" y="{y + 6:.2f}" font-size="11" text-anchor="{anchor}" font-family="Arial, sans-serif" fill="#5c5c5c">Sharpe {float(row["sharpe"]):.2f}</text>'
        )

    parts.append(f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_height}" stroke="#555" stroke-width="1.2" />')
    parts.append(
        f'<line x1="{pad_left}" y1="{pad_top + plot_height}" x2="{pad_left + plot_width}" y2="{pad_top + plot_height}" stroke="#555" stroke-width="1.2" />'
    )
    parts.append(
        f'<text x="{pad_left + plot_width / 2:.2f}" y="{height - 18:.2f}" font-size="12" text-anchor="middle" font-family="Arial, sans-serif" fill="#5c5c5c">最大回撤深度</text>'
    )
    parts.append(
        f'<text transform="translate(22 {pad_top + plot_height / 2:.2f}) rotate(-90)" font-size="12" text-anchor="middle" font-family="Arial, sans-serif" fill="#5c5c5c">年化收益</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def build_delta_svg(metrics: pd.DataFrame, output_path: Path, color_map: Dict[str, str]) -> None:
    chart_specs = [
        ("annual_return_delta", "相对当前V2年化变化", "pct"),
        ("drawdown_improvement", "相对当前V2回撤改善", "pct"),
        ("sharpe_delta", "相对当前V2夏普变化", "num"),
    ]
    rows = metrics.loc[metrics["strategy"] != "当前V2"].reset_index(drop=True)
    width = 980
    height = 760
    outer_left = 50
    outer_top = 40
    panel_height = 210
    panel_gap = 24
    label_width = 210
    right_pad = 36
    axis_width = width - outer_left - label_width - right_pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7" />',
        '<text x="40" y="28" font-size="22" font-family="Arial, sans-serif" font-weight="700" fill="#14213d">相对当前V2的改进</text>',
    ]

    for idx, (column, title, value_kind) in enumerate(chart_specs):
        y0 = outer_top + idx * (panel_height + panel_gap)
        bar_top = y0 + 48
        bottom = y0 + panel_height - 28
        axis_left = outer_left + label_width
        zero_x = axis_left + axis_width / 2
        max_abs = max(abs(float(rows[column].min())), abs(float(rows[column].max())))
        if max_abs <= 0:
            max_abs = 1.0

        def x_pos(v: float) -> float:
            return zero_x + (v / max_abs) * (axis_width / 2)

        parts.append(
            f'<rect x="{outer_left:.2f}" y="{y0:.2f}" width="{width - outer_left * 2:.2f}" height="{panel_height:.2f}" rx="10" ry="10" fill="#fffaf0" stroke="#eadfcb" stroke-width="1.2" />'
        )
        parts.append(
            f'<text x="{outer_left + 16:.2f}" y="{y0 + 26:.2f}" font-size="16" font-family="Arial, sans-serif" font-weight="700" fill="#14213d">{html.escape(title)}</text>'
        )

        for frac in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            value = max_abs * frac
            x = x_pos(value)
            parts.append(f'<line x1="{x:.2f}" y1="{bar_top - 8:.2f}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="#ece5d8" stroke-width="1" />')
            parts.append(
                f'<text x="{x:.2f}" y="{bottom + 18:.2f}" font-size="11" text-anchor="middle" font-family="Arial, sans-serif" fill="#5c5c5c">{_format_axis_value(value, value_kind)}</text>'
            )

        parts.append(f'<line x1="{zero_x:.2f}" y1="{bar_top - 12:.2f}" x2="{zero_x:.2f}" y2="{bottom:.2f}" stroke="#6b7280" stroke-width="1.6" />')

        bar_height = 22
        gap = 18
        for row_idx, row in rows.iterrows():
            y = bar_top + row_idx * (bar_height + gap)
            value = float(row[column])
            x0 = min(zero_x, x_pos(value))
            bar_width = abs(x_pos(value) - zero_x)
            strategy = str(row["strategy"])
            color = color_map[strategy]
            parts.append(
                f'<text x="{outer_left + 8:.2f}" y="{y + 16:.2f}" font-size="12" font-family="Arial, sans-serif" fill="#243b53">{html.escape(strategy)}</text>'
            )
            parts.append(
                f'<rect x="{x0:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="5" ry="5" fill="{color}" opacity="0.92" />'
            )
            text_anchor = "start" if value >= 0 else "end"
            text_x = x_pos(value) + 6 if value >= 0 else x_pos(value) - 6
            parts.append(
                f'<text x="{text_x:.2f}" y="{y + 16:.2f}" font-size="11" text-anchor="{text_anchor}" font-family="Arial, sans-serif" fill="#5c5c5c">{_format_axis_value(value, value_kind)}</text>'
            )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _load_levels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def _ensure_strategy_levels(args: argparse.Namespace, spec: Dict[str, object]) -> pd.DataFrame:
    source = Path(spec["source"])
    if source.exists():
        return _load_levels(source)

    result = run_study(
        config_path=Path(args.config),
        price_path=Path(args.prices),
        output_dir=Path(spec["detail_dir"]),
        valuation_path=Path(args.valuation) if args.valuation else None,
        lookback_months=int(spec["lookback_months"]),
        selection_rule=str(spec["selection_rule"]),
        drawdown_band=float(args.drawdown_band),
        write_outputs=True,
    )
    comparison_index = result["comparison_index"].copy()
    comparison_index.index.name = "date"
    return comparison_index


def _build_common_metrics(levels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    sample_start = levels.index.min().strftime("%Y-%m-%d")
    sample_end = levels.index.max().strftime("%Y-%m-%d")
    current = _summary_metrics(levels["当前V2"], levels["当前V2"].pct_change(fill_method=None).fillna(0.0))
    rows.append(
        {
            "strategy": "当前V2",
            "selection_rule_label": "基准",
            "lookback_months": 0,
            "note": "当前配置",
            "sample_start": sample_start,
            "sample_end": sample_end,
            "total_return": float(current["total_return"]),
            "annual_return": float(current["annual_return"]),
            "annual_volatility": float(current["annual_volatility"]),
            "max_drawdown": float(current["max_drawdown"]),
            "drawdown_depth": abs(float(current["max_drawdown"])),
            "sharpe": float(current["sharpe"]),
            "annual_return_delta": 0.0,
            "drawdown_improvement": 0.0,
            "sharpe_delta": 0.0,
        }
    )

    for spec in STRATEGIES:
        dynamic = _summary_metrics(levels[spec["name"]], levels[spec["name"]].pct_change(fill_method=None).fillna(0.0))
        rows.append(
            {
                "strategy": spec["name"],
                "selection_rule_label": selection_rule_label(str(spec["selection_rule"])),
                "lookback_months": int(spec["lookback_months"]),
                "note": spec["note"],
                "sample_start": sample_start,
                "sample_end": sample_end,
                "total_return": float(dynamic["total_return"]),
                "annual_return": float(dynamic["annual_return"]),
                "annual_volatility": float(dynamic["annual_volatility"]),
                "max_drawdown": float(dynamic["max_drawdown"]),
                "drawdown_depth": abs(float(dynamic["max_drawdown"])),
                "sharpe": float(dynamic["sharpe"]),
                "annual_return_delta": float(dynamic["annual_return"] - current["annual_return"]),
                "drawdown_improvement": float(dynamic["max_drawdown"] - current["max_drawdown"]),
                "sharpe_delta": float(dynamic["sharpe"] - current["sharpe"]),
            }
        )

    return pd.DataFrame(rows)


def _build_levels_table(args: argparse.Namespace) -> pd.DataFrame:
    levels_map: Dict[str, pd.Series] = {}
    current_levels = None

    for spec in STRATEGIES:
        comparison = _ensure_strategy_levels(args, spec)
        levels_map[str(spec["name"])] = comparison["动态季度滚动"]
        if current_levels is None:
            current_levels = comparison["当前V2"]

    if current_levels is None:
        raise ValueError("No strategy levels available.")
    levels_map["当前V2"] = current_levels

    common_start = max(series.index.min() for series in levels_map.values())
    common_end = min(series.index.max() for series in levels_map.values())

    aligned = []
    for name in ["当前V2", *[spec["name"] for spec in STRATEGIES]]:
        series = levels_map[name].loc[(levels_map[name].index >= common_start) & (levels_map[name].index <= common_end)].copy()
        aligned.append(series.rename(name))

    levels = pd.concat(aligned, axis=1, join="inner").sort_index()
    levels = levels.div(levels.iloc[0])
    levels.index.name = "date"
    return levels


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = _build_levels_table(args)
    metrics = _build_common_metrics(levels)
    drawdowns = levels.div(levels.cummax()) - 1.0

    metrics.to_csv(output_dir / "comparison_table.csv", index=False)
    levels.to_csv(output_dir / "common_window_index_levels.csv", index=True)
    drawdowns.to_csv(output_dir / "common_window_drawdowns.csv", index=True)

    build_line_chart_svg(levels, "共同样本净值对比", output_dir / "common_window_nav.svg", value_kind="level", color_map=COLOR_MAP)
    build_line_chart_svg(drawdowns, "共同样本回撤对比", output_dir / "common_window_drawdown.svg", value_kind="pct", color_map=COLOR_MAP)
    build_scatter_svg(metrics, output_dir / "risk_return_scatter.svg", COLOR_MAP)
    build_delta_svg(metrics, output_dir / "delta_vs_current.svg", COLOR_MAP)

    pretty = metrics.copy()
    pretty["观察窗"] = pretty["lookback_months"].map(lambda value: "-" if int(value) == 0 else f"{int(value)}M")
    pretty["年化收益"] = pretty["annual_return"].map(_format_percent)
    pretty["年化波动"] = pretty["annual_volatility"].map(_format_percent)
    pretty["最大回撤"] = pretty["max_drawdown"].map(_format_percent)
    pretty["夏普"] = pretty["sharpe"].map(_format_number)
    pretty["相对当前年化变化"] = pretty["annual_return_delta"].map(_format_percent)
    pretty["相对当前回撤改善"] = pretty["drawdown_improvement"].map(_format_percent)
    pretty["相对当前夏普变化"] = pretty["sharpe_delta"].map(_format_number)
    pretty = pretty[
        [
            "strategy",
            "selection_rule_label",
            "观察窗",
            "note",
            "sample_start",
            "sample_end",
            "年化收益",
            "年化波动",
            "最大回撤",
            "夏普",
            "相对当前年化变化",
            "相对当前回撤改善",
            "相对当前夏普变化",
        ]
    ].rename(
        columns={
            "strategy": "策略",
            "selection_rule_label": "规则",
            "note": "定位",
            "sample_start": "样本起点",
            "sample_end": "样本终点",
        }
    )

    report_lines = [
        "# 动态季度滚动方案对比",
        "",
        "## 口径",
        "",
        f"- 共同样本区间：`{metrics['sample_start'].iloc[0]}` 至 `{metrics['sample_end'].iloc[0]}`",
        "- 基准：当前V2",
        "- 对比对象：正式版12M、极致降回撤6M、均衡版36M夏普护栏、防守版36MCalmar护栏",
        "",
        "## 图",
        "",
        f"![共同样本净值对比]({(output_dir / 'common_window_nav.svg').resolve()})",
        "",
        f"![共同样本回撤对比]({(output_dir / 'common_window_drawdown.svg').resolve()})",
        "",
        f"![风险收益散点图]({(output_dir / 'risk_return_scatter.svg').resolve()})",
        "",
        f"![相对当前V2的改进]({(output_dir / 'delta_vs_current.svg').resolve()})",
        "",
        "## 表",
        "",
        _to_markdown_table(pretty),
        "",
        "## 文件",
        "",
        f"- [comparison_table.csv]({(output_dir / 'comparison_table.csv').resolve()})",
        f"- [common_window_index_levels.csv]({(output_dir / 'common_window_index_levels.csv').resolve()})",
        f"- [common_window_drawdowns.csv]({(output_dir / 'common_window_drawdowns.csv').resolve()})",
        f"- [common_window_nav.svg]({(output_dir / 'common_window_nav.svg').resolve()})",
        f"- [common_window_drawdown.svg]({(output_dir / 'common_window_drawdown.svg').resolve()})",
        f"- [risk_return_scatter.svg]({(output_dir / 'risk_return_scatter.svg').resolve()})",
        f"- [delta_vs_current.svg]({(output_dir / 'delta_vs_current.svg').resolve()})",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"output_dir={output_dir.resolve()}")
    print(f"sample_start={metrics['sample_start'].iloc[0]}")
    print(f"sample_end={metrics['sample_end'].iloc[0]}")


if __name__ == "__main__":
    main()
