#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import html
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.backtest import run_backtest  # noqa: E402
from etf_fof_index.config import load_config, resolve_path, validate_config  # noqa: E402
from etf_fof_index.data import load_price_data, load_valuation_data  # noqa: E402
from etf_fof_index.signals import build_bucket_price_frame, compute_signals  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402
from etf_fof_index.weights import compute_strategy_weights  # noqa: E402
from run_weight_grid_research_v2 import fast_backtest_metrics  # noqa: E402


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

CRITERIA = [
    {"id": "return_max", "label": "收益最高", "column": "annual_return", "ascending": False},
    {"id": "drawdown_min", "label": "回撤最低", "column": "max_drawdown", "ascending": False},
    {"id": "vol_min", "label": "波动最低", "column": "annual_volatility", "ascending": True},
    {"id": "sharpe_max", "label": "夏普最高", "column": "sharpe", "ascending": False},
    {"id": "calmar_max", "label": "Calmar最高", "column": "calmar", "ascending": False},
    {"id": "win_rate_max", "label": "月胜率最高", "column": "monthly_win_rate", "ascending": False},
]

DISPLAY_COLS = [
    "profile_name",
    "selected_by",
    "total_return",
    "annual_return",
    "annual_volatility",
    "max_drawdown",
    "sharpe",
    "calmar",
    "monthly_win_rate",
] + BUCKETS

PLOT_COLORS = [
    "#d1495b",
    "#2e4057",
    "#3c7a89",
    "#edae49",
    "#00798c",
    "#6b705c",
    "#8d99ae",
]

BUCKET_LABELS = {
    "equity_core_csi300": "沪深300",
    "equity_core_csia500": "中证A500",
    "equity_defensive_lowvol": "低波",
    "equity_defensive_dividend": "红利",
    "rate_bond_5y": "5Y债",
    "rate_bond_10y": "10Y债",
    "gold": "黄金",
    "money_market": "货币",
}

REPORT_LABELS = {
    "criterion": "选优维度",
    "profile_name": "方案",
    "selected_by": "入选原因",
    "rank_order": "网格名次",
    "total_return": "累计收益",
    "annual_return": "年化收益",
    "annual_volatility": "年化波动",
    "max_drawdown": "最大回撤",
    "sharpe": "夏普",
    "calmar": "Calmar",
    "monthly_win_rate": "月胜率",
    **BUCKET_LABELS,
}


@dataclass
class Profile:
    profile_key: Tuple[float, ...]
    profile_name: str
    selected_by: List[str]
    weights: Dict[str, float]
    metrics: Dict[str, float]
    is_current: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-metric-optimal weight profiles from grid search results.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2.csv"))
    parser.add_argument("--grid-results", default=str(ROOT / "output" / "weight_grid_v2_matrix" / "grid_results.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "weight_profile_choices"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=0,
        help="If >0, shortlist top N candidates per criterion from grid_results, then reselect winners using standard replay metrics.",
    )
    return parser.parse_args()


def _format_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "num":
        return f"{value:.2f}"
    if kind == "rank":
        return f"{int(value)}"
    return str(value)


def _format_axis_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.0%}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _profile_color_map(names: Iterable[str]) -> Dict[str, str]:
    return {name: PLOT_COLORS[idx % len(PLOT_COLORS)] for idx, name in enumerate(names)}


def _rename_columns(frame: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [mapping.get(str(column), str(column)) for column in renamed.columns]
    return renamed


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


def _profile_key(row: pd.Series) -> Tuple[float, ...]:
    return tuple(round(float(row[bucket]), 6) for bucket in BUCKETS)


def _weights_from_row(row: pd.Series) -> Dict[str, float]:
    return {bucket: float(row[bucket]) for bucket in BUCKETS}


def _sort_for_criterion(results: pd.DataFrame, column: str, ascending: bool) -> pd.DataFrame:
    if ascending:
        sort_cols = [column, "annual_return", "sharpe", "max_drawdown"]
        sort_asc = [True, False, False, False]
    else:
        sort_cols = [column, "annual_return", "sharpe", "annual_volatility"]
        sort_asc = [False, False, False, True]
    return results.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)


def _pick_best_row(results: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    return _sort_for_criterion(results, column, ascending).iloc[0]


def summarize_levels(index_series: pd.Series, return_series: pd.Series) -> Dict[str, float]:
    ann_factor = 252.0
    periods = max(len(return_series), 1)
    total_return = float(index_series.iloc[-1] / index_series.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (ann_factor / periods) - 1.0)
    annual_vol = float(return_series.std(ddof=0) * (ann_factor ** 0.5))
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


def build_profiles(results: pd.DataFrame, current_weights: Dict[str, float]) -> Tuple[pd.DataFrame, List[Profile]]:
    winners = []
    merged: Dict[Tuple[float, ...], Profile] = {}

    for criterion in CRITERIA:
        row = _pick_best_row(results, criterion["column"], criterion["ascending"])
        weights = _weights_from_row(row)
        winner_row = {
            "criterion": criterion["label"],
            "profile_name": "",
            "rank_order": int(row["rank_order"]),
            "annual_return": float(row["annual_return"]),
            "annual_volatility": float(row["annual_volatility"]),
            "max_drawdown": float(row["max_drawdown"]),
            "sharpe": float(row["sharpe"]),
            "calmar": float(row["calmar"]),
            "monthly_win_rate": float(row["monthly_win_rate"]),
            **weights,
        }
        winners.append((criterion, row, winner_row))

        key = _profile_key(row)
        if key not in merged:
            merged[key] = Profile(
                profile_key=key,
                profile_name="",
                selected_by=[criterion["label"]],
                weights=weights,
                metrics={},
                is_current=bool(row.get("is_current_v2", False)),
            )
        else:
            merged[key].selected_by.append(criterion["label"])

    ordered_profiles = list(merged.values())
    ordered_profiles.sort(key=lambda item: len(CRITERIA) - len(item.selected_by), reverse=False)
    ordered_profiles.sort(key=lambda item: CRITERIA.index(next(c for c in CRITERIA if c["label"] == item.selected_by[0])))

    for idx, profile in enumerate(ordered_profiles, start=1):
        profile.profile_name = f"方案{idx}"

    winner_rows = []
    for _, row, winner_row in winners:
        key = _profile_key(row)
        winner_row["profile_name"] = merged[key].profile_name
        winner_rows.append(winner_row)

    current_rows = results.loc[results["is_current_v2"] == True]
    if not current_rows.empty:
        current_row = current_rows.iloc[0]
        current_key = _profile_key(current_row)
        if current_key not in merged:
            ordered_profiles.append(
                Profile(
                    profile_key=current_key,
                    profile_name="当前V2",
                    selected_by=["当前配置"],
                    weights=_weights_from_row(current_row),
                    metrics={},
                    is_current=True,
                )
            )
        else:
            merged[current_key].is_current = True
            if "当前配置" not in merged[current_key].selected_by:
                merged[current_key].selected_by.append("当前配置")
            if merged[current_key].profile_name.startswith("方案"):
                merged[current_key].profile_name = f"{merged[current_key].profile_name}(当前V2)"
    else:
        current_key = tuple(round(float(current_weights[bucket]), 6) for bucket in BUCKETS)
        if current_key not in merged:
            ordered_profiles.append(
                Profile(
                    profile_key=current_key,
                    profile_name="当前V2",
                    selected_by=["当前配置"],
                    weights={bucket: float(current_weights[bucket]) for bucket in BUCKETS},
                    metrics={},
                    is_current=True,
                )
            )

    return pd.DataFrame(winner_rows), ordered_profiles


def build_candidate_profiles(results: pd.DataFrame, current_weights: Dict[str, float], top_n: int) -> List[Profile]:
    merged: Dict[Tuple[float, ...], Profile] = {}
    for criterion in CRITERIA:
        shortlisted = _sort_for_criterion(results, criterion["column"], criterion["ascending"]).head(top_n)
        for _, row in shortlisted.iterrows():
            key = _profile_key(row)
            if key not in merged:
                merged[key] = Profile(
                    profile_key=key,
                    profile_name="",
                    selected_by=[],
                    weights=_weights_from_row(row),
                    metrics={},
                    is_current=bool(row.get("is_current_v2", False)),
                )

    current_key = tuple(round(float(current_weights[bucket]), 6) for bucket in BUCKETS)
    if current_key not in merged:
        merged[current_key] = Profile(
            profile_key=current_key,
            profile_name="当前V2",
            selected_by=["当前配置"],
            weights={bucket: float(current_weights[bucket]) for bucket in BUCKETS},
            metrics={},
            is_current=True,
        )
    else:
        merged[current_key].is_current = True
        if "当前配置" not in merged[current_key].selected_by:
            merged[current_key].selected_by.append("当前配置")
    return list(merged.values())


def build_profiles_from_metrics(metric_frame: pd.DataFrame, current_weights: Dict[str, float]) -> Tuple[pd.DataFrame, List[Profile]]:
    winners = []
    merged: Dict[Tuple[float, ...], Profile] = {}

    for criterion in CRITERIA:
        row = _pick_best_row(metric_frame, criterion["column"], criterion["ascending"])
        weights = _weights_from_row(row)
        winner_row = {
            "criterion": criterion["label"],
            "profile_name": "",
            "annual_return": float(row["annual_return"]),
            "annual_volatility": float(row["annual_volatility"]),
            "max_drawdown": float(row["max_drawdown"]),
            "sharpe": float(row["sharpe"]),
            "calmar": float(row["calmar"]),
            "monthly_win_rate": float(row["monthly_win_rate"]),
            **weights,
        }
        winners.append((criterion, row, winner_row))

        key = _profile_key(row)
        if key not in merged:
            merged[key] = Profile(
                profile_key=key,
                profile_name="",
                selected_by=[criterion["label"]],
                weights=weights,
                metrics={k: float(row[k]) for k in ["total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar", "monthly_win_rate"]},
                is_current=bool(row.get("is_current", False)),
            )
        else:
            merged[key].selected_by.append(criterion["label"])

    ordered_profiles = list(merged.values())
    ordered_profiles.sort(key=lambda item: CRITERIA.index(next(c for c in CRITERIA if c["label"] == item.selected_by[0])))

    for idx, profile in enumerate(ordered_profiles, start=1):
        profile.profile_name = f"方案{idx}"

    winner_rows = []
    for _, row, winner_row in winners:
        key = _profile_key(row)
        winner_row["profile_name"] = merged[key].profile_name
        winner_rows.append(winner_row)

    current_key = tuple(round(float(current_weights[bucket]), 6) for bucket in BUCKETS)
    if current_key in merged:
        merged[current_key].is_current = True
        if "当前配置" not in merged[current_key].selected_by:
            merged[current_key].selected_by.append("当前配置")
        if merged[current_key].profile_name.startswith("方案"):
            merged[current_key].profile_name = f"{merged[current_key].profile_name}(当前V2)"
    else:
        current_rows = metric_frame.loc[metric_frame["is_current"] == True]
        if not current_rows.empty:
            current_row = current_rows.iloc[0]
            ordered_profiles.append(
                Profile(
                    profile_key=current_key,
                    profile_name="当前V2",
                    selected_by=["当前配置"],
                    weights=_weights_from_row(current_row),
                    metrics={k: float(current_row[k]) for k in ["total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar", "monthly_win_rate"]},
                    is_current=True,
                )
            )

    return pd.DataFrame(winner_rows), ordered_profiles


def build_line_chart_svg(
    levels: pd.DataFrame,
    title: str,
    output_path: Path,
    value_kind: str = "level",
    color_map: Dict[str, str] | None = None,
) -> None:
    width = 1200
    height = 720
    pad_left = 90
    pad_right = 280
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
        color = (color_map or {}).get(column, PLOT_COLORS[idx % len(PLOT_COLORS)])
        stroke_width = 4 if "当前V2" in column else 3
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


def build_metric_panel_svg(
    metrics_table: pd.DataFrame,
    profile_order: List[str],
    output_path: Path,
    color_map: Dict[str, str] | None = None,
) -> None:
    chart_specs = [
        ("annual_return", "年化收益", "pct", False),
        ("annual_volatility", "年化波动", "pct", True),
        ("max_drawdown", "回撤深度", "pct", True),
        ("sharpe", "夏普", "num", False),
        ("calmar", "Calmar", "num", False),
        ("monthly_win_rate", "月胜率", "pct", False),
    ]
    panel_cols = 2
    panel_rows = 3
    width = 1360
    height = 980
    outer_left = 40
    outer_top = 40
    gutter_x = 28
    gutter_y = 24
    panel_width = (width - outer_left * 2 - gutter_x) / panel_cols
    panel_height = (height - outer_top * 2 - gutter_y * (panel_rows - 1)) / panel_rows
    series_colors = color_map or _profile_color_map(profile_order)
    base_order = {name: idx for idx, name in enumerate(profile_order)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7" />',
        '<text x="40" y="28" font-size="22" font-family="Arial, sans-serif" font-weight="700" fill="#14213d">主要指标对比</text>',
    ]

    for idx, (column, title, value_kind, lower_is_better) in enumerate(chart_specs):
        panel_col = idx % panel_cols
        panel_row = idx // panel_cols
        x0 = outer_left + panel_col * (panel_width + gutter_x)
        y0 = outer_top + panel_row * (panel_height + gutter_y)
        left = x0 + 128
        right = x0 + panel_width - 36
        top = y0 + 44
        bottom = y0 + panel_height - 24
        plot_width = right - left
        bar_height = 18
        bar_gap = 10

        panel_data = metrics_table[["profile_name", column]].copy()
        if column == "max_drawdown":
            panel_data["_metric_value"] = panel_data[column].abs()
        else:
            panel_data["_metric_value"] = panel_data[column]
        panel_data = panel_data.sort_values(
            ["_metric_value", "profile_name"],
            ascending=[lower_is_better, True],
            key=lambda series: series if series.name != "profile_name" else series.map(base_order.get),
        ).reset_index(drop=True)

        max_value = float(panel_data["_metric_value"].max())
        if max_value <= 0:
            max_value = 1.0

        parts.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" rx="10" ry="10" fill="#fffaf0" stroke="#eadfcb" stroke-width="1.2" />'
        )
        suffix = "（越低越好）" if lower_is_better else "（越高越好）"
        parts.append(
            f'<text x="{x0 + 16:.2f}" y="{y0 + 26:.2f}" font-size="16" font-family="Arial, sans-serif" font-weight="700" fill="#14213d">{html.escape(title + suffix)}</text>'
        )

        ticks = [0.0, 0.5, 1.0]
        for frac in ticks:
            x = left + plot_width * frac
            tick_value = max_value * frac
            parts.append(f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="#ece5d8" stroke-width="1" />')
            parts.append(
                f'<text x="{x:.2f}" y="{bottom + 18:.2f}" font-size="11" text-anchor="middle" font-family="Arial, sans-serif" fill="#5c5c5c">{_format_axis_value(tick_value, value_kind)}</text>'
            )

        for row_idx, row in panel_data.iterrows():
            y = top + row_idx * (bar_height + bar_gap)
            label_y = y + bar_height * 0.75
            name = str(row["profile_name"])
            metric_value = float(row["_metric_value"])
            bar_width = plot_width * metric_value / max_value
            fill = series_colors.get(name, PLOT_COLORS[row_idx % len(PLOT_COLORS)])
            stroke = "#111827" if "当前V2" in name else "none"
            stroke_width = 1.5 if "当前V2" in name else 0.0
            parts.append(
                f'<text x="{left - 8:.2f}" y="{label_y:.2f}" font-size="12" text-anchor="end" font-family="Arial, sans-serif" fill="#1f2933">{html.escape(name)}</text>'
            )
            parts.append(
                f'<rect x="{left:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="5" ry="5" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
            )
            parts.append(
                f'<text x="{left + bar_width + 8:.2f}" y="{label_y:.2f}" font-size="12" font-family="Arial, sans-serif" fill="#1f2933">{_format_value(metric_value, value_kind)}</text>'
            )

    parts.append(
        f'<text x="{width - 40}" y="{height - 18}" font-size="11" text-anchor="end" font-family="Arial, sans-serif" fill="#6b7280">回撤深度按绝对值展示，数值越小越好</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def format_percent_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].map(lambda v: _format_value(float(v), "pct"))
    return out


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The default fill_method='pad' in DataFrame.pct_change is deprecated",
        category=FutureWarning,
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    validate_config(config)
    universe = load_universe(resolve_path(config, config["paths"]["universe"]))
    prices = load_price_data(Path(args.prices))
    valuation_data = load_valuation_data(Path(args.valuation)) if args.valuation else pd.DataFrame()
    selection = select_representatives(universe, config["bucket_order"], available_symbols=prices.columns)
    bucket_prices = build_bucket_price_frame(prices, selection.bucket_to_symbol, config["bucket_order"])
    signals = compute_signals(bucket_prices=bucket_prices, valuation_data=valuation_data, config=config)
    returns = bucket_prices.pct_change().fillna(0.0)

    results = pd.read_csv(args.grid_results)
    required_cols = {
        "rank_order",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "sharpe",
        "calmar",
        "monthly_win_rate",
        "is_current_v2",
        *BUCKETS,
    }
    missing = required_cols.difference(results.columns)
    if missing:
        raise ValueError(f"grid_results is missing required columns: {sorted(missing)}")

    if args.candidate_top_n > 0:
        profiles = build_candidate_profiles(results, config["strategic_weights"], args.candidate_top_n)

        candidate_metric_rows = []
        for profile in profiles:
            cfg = copy.deepcopy(config)
            cfg["strategic_weights"] = profile.weights
            strategy_weights, _ = compute_strategy_weights(signals, cfg)
            metrics = fast_backtest_metrics(returns, strategy_weights, cfg)
            candidate_metric_rows.append(
                {
                    "profile_name": profile.profile_name,
                    "selected_by": "/".join(profile.selected_by),
                    "is_current": profile.is_current,
                    **metrics,
                    **profile.weights,
                }
            )
        winner_table, profiles = build_profiles_from_metrics(pd.DataFrame(candidate_metric_rows), config["strategic_weights"])
    else:
        winner_table, profiles = build_profiles(results, config["strategic_weights"])

    comparison_levels = pd.DataFrame()
    comparison_metrics_rows = []
    weights_rows = []
    profile_metric_lookup: Dict[str, Dict[str, float]] = {}

    for profile in profiles:
        cfg = copy.deepcopy(config)
        cfg["strategic_weights"] = profile.weights
        strategy_weights, _ = compute_strategy_weights(signals, cfg)
        backtest = run_backtest(bucket_prices, strategy_weights, cfg, label="strategy")
        metrics = summarize_levels(backtest.levels["strategy_index"], backtest.levels["strategy_return"])
        profile.metrics = metrics
        profile_metric_lookup[profile.profile_name] = metrics

        display_name = profile.profile_name
        if profile.is_current and "当前V2" not in display_name:
            display_name = f"{display_name}(当前V2)"

        comparison_levels[display_name] = backtest.levels["strategy_index"]
        comparison_metrics_rows.append(
            {
                "profile_name": display_name,
                "selected_by": "/".join(profile.selected_by),
                "is_current": profile.is_current,
                **metrics,
                **profile.weights,
            }
        )
        weights_rows.append(
            {
                "profile_name": display_name,
                "selected_by": "/".join(profile.selected_by),
                **profile.weights,
            }
        )

    if not winner_table.empty:
        metric_cols = ["total_return", "annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar", "monthly_win_rate"]
        for idx, row in winner_table.iterrows():
            metrics = profile_metric_lookup.get(str(row["profile_name"]).replace("(当前V2)", ""))
            if metrics is None:
                continue
            for col in metric_cols:
                if col in metrics:
                    winner_table.loc[idx, col] = metrics[col]

    comparison_levels.index.name = "date"
    comparison_levels.to_csv(output_dir / "profile_index_levels.csv", index=True)
    drawdown_levels = comparison_levels.div(comparison_levels.cummax()) - 1.0
    drawdown_levels.index.name = "date"
    drawdown_levels.to_csv(output_dir / "profile_drawdowns.csv", index=True)

    weights_table = pd.DataFrame(weights_rows)
    weights_table.to_csv(output_dir / "profile_weights.csv", index=False)

    metrics_table = pd.DataFrame(comparison_metrics_rows)
    metrics_table = metrics_table.sort_values("annual_return", ascending=False).reset_index(drop=True)
    metrics_table.to_csv(output_dir / "profile_metrics.csv", index=False)

    winner_table.to_csv(output_dir / "criterion_winners.csv", index=False)

    profile_colors = _profile_color_map(comparison_levels.columns)
    build_line_chart_svg(comparison_levels, "全样本净值对比", output_dir / "profile_nav.svg", value_kind="level", color_map=profile_colors)
    build_line_chart_svg(drawdown_levels, "全样本回撤对比", output_dir / "profile_drawdown.svg", value_kind="pct", color_map=profile_colors)
    build_metric_panel_svg(metrics_table, list(comparison_levels.columns), output_dir / "profile_metric_panels.svg", color_map=profile_colors)

    winner_cols = ["criterion", "profile_name"]
    if "rank_order" in winner_table.columns:
        winner_cols.append("rank_order")
    winner_cols += ["annual_return", "annual_volatility", "max_drawdown", "sharpe", "calmar", "monthly_win_rate"] + BUCKETS
    winner_md = format_percent_columns(winner_table[winner_cols], ["annual_return", "annual_volatility", "max_drawdown", "monthly_win_rate"] + BUCKETS)
    winner_md["sharpe"] = winner_md["sharpe"].map(lambda v: _format_value(float(v), "num"))
    winner_md["calmar"] = winner_md["calmar"].map(lambda v: _format_value(float(v), "num"))
    if "rank_order" in winner_md.columns:
        winner_md["rank_order"] = winner_md["rank_order"].map(lambda v: _format_value(float(v), "rank"))

    metrics_md = format_percent_columns(
        metrics_table[["profile_name", "selected_by", "total_return", "annual_return", "annual_volatility", "max_drawdown", "monthly_win_rate"]],
        ["total_return", "annual_return", "annual_volatility", "max_drawdown", "monthly_win_rate"],
    )
    for column in ["sharpe", "calmar"]:
        metrics_md[column] = metrics_table[column].map(lambda v: _format_value(float(v), "num"))
    metrics_md = metrics_md[
        [
            "profile_name",
            "selected_by",
            "total_return",
            "annual_return",
            "annual_volatility",
            "max_drawdown",
            "sharpe",
            "calmar",
            "monthly_win_rate",
        ]
    ]

    weights_md = format_percent_columns(weights_table, BUCKETS)
    winner_md = _rename_columns(winner_md, REPORT_LABELS)
    metrics_md = _rename_columns(metrics_md, REPORT_LABELS)
    weights_md = _rename_columns(weights_md, REPORT_LABELS)

    sample_start = comparison_levels.index.min().strftime("%Y-%m-%d")
    sample_end = comparison_levels.index.max().strftime("%Y-%m-%d")
    top_return = metrics_table.sort_values("annual_return", ascending=False).iloc[0]
    top_sharpe = metrics_table.sort_values("sharpe", ascending=False).iloc[0]
    shallowest_drawdown = metrics_table.sort_values("max_drawdown", ascending=False).iloc[0]
    current_row = metrics_table.loc[metrics_table["is_current"] == True].iloc[0]

    report_lines = [
        "# 权重方案对比",
        "",
        "## 概览",
        "",
        f"- 样本区间：`{sample_start}` 至 `{sample_end}`",
        f"- 对比方案数：`{len(metrics_table)}`",
        f"- 选优口径：单指标冠军来自全样本网格结果，图表与指标按标准回放复核，`当前V2` 作为对照基准",
        f"- 展示组内复核后收益最高：`{top_return['profile_name']}`，累计收益 `{top_return['total_return']:.2%}`，年化 `{top_return['annual_return']:.2%}`",
        f"- 展示组内复核后夏普最高：`{top_sharpe['profile_name']}`，夏普 `{top_sharpe['sharpe']:.2f}`，年化 `{top_sharpe['annual_return']:.2%}`",
        f"- 展示组内复核后回撤最浅：`{shallowest_drawdown['profile_name']}`，最大回撤 `{shallowest_drawdown['max_drawdown']:.2%}`",
        f"- 当前V2：累计收益 `{current_row['total_return']:.2%}`，年化 `{current_row['annual_return']:.2%}`，最大回撤 `{current_row['max_drawdown']:.2%}`",
        "",
        "## 图表",
        "",
        f"![全样本净值对比]({(output_dir / 'profile_nav.svg').resolve()})",
        "",
        f"![全样本回撤对比]({(output_dir / 'profile_drawdown.svg').resolve()})",
        "",
        f"![主要指标对比]({(output_dir / 'profile_metric_panels.svg').resolve()})",
        "",
        "## 单指标冠军",
        "",
        _to_markdown_table(winner_md),
        "",
        "## 方案指标对比",
        "",
        _to_markdown_table(metrics_md),
        "",
        "## 权重表",
        "",
        _to_markdown_table(weights_md),
        "",
        "## 文件",
        "",
        f"- [criterion_winners.csv]({(output_dir / 'criterion_winners.csv').resolve()})",
        f"- [profile_metrics.csv]({(output_dir / 'profile_metrics.csv').resolve()})",
        f"- [profile_weights.csv]({(output_dir / 'profile_weights.csv').resolve()})",
        f"- [profile_index_levels.csv]({(output_dir / 'profile_index_levels.csv').resolve()})",
        f"- [profile_drawdowns.csv]({(output_dir / 'profile_drawdowns.csv').resolve()})",
        f"- [profile_nav.svg]({(output_dir / 'profile_nav.svg').resolve()})",
        f"- [profile_drawdown.svg]({(output_dir / 'profile_drawdown.svg').resolve()})",
        f"- [profile_metric_panels.svg]({(output_dir / 'profile_metric_panels.svg').resolve()})",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"profiles={len(profiles)}")
    print(f"criteria={len(winner_table)}")
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
