#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, Iterable, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.config import load_config, resolve_path  # noqa: E402
from etf_fof_index.data import load_price_data  # noqa: E402
from etf_fof_index.report import _summary_metrics, summarize_period_metrics  # noqa: E402
from etf_fof_index.signals import build_bucket_price_frame  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402


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

DISPLAY_NAME = "12M收益优先(进攻版)"
DISPLAY_MAP = {
    "动态季度滚动": DISPLAY_NAME,
    "当前V2": "当前V2",
}
COLOR_MAP = {
    DISPLAY_NAME: "#c1121f",
    "当前V2": "#243b53",
}
ASSET_COLOR_MAP = {
    DISPLAY_NAME: "#c1121f",
    "沪深300": "#15616d",
    "中证A500": "#1d4ed8",
    "低波": "#6b8e23",
    "红利": "#9c6644",
    "5Y债": "#6d597a",
    "10Y债": "#4f772d",
    "黄金": "#f4a261",
    "货币": "#8d99ae",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone HTML dashboard for the max-return rolling strategy.")
    parser.add_argument(
        "--detail-dir",
        default=str(ROOT / "output" / "rolling_quarterly_v2_return_candidate_maxret12"),
    )
    parser.add_argument(
        "--output-html",
        default=str(ROOT / "output" / "rolling_quarterly_v2_return_candidate_maxret12" / "dashboard.html"),
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2_index_proxy.csv"))
    parser.add_argument("--strategy-name", default=DISPLAY_NAME)
    return parser.parse_args()


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_num(value: float) -> str:
    return f"{value:.2f}"


def _weight_cell(value: float) -> str:
    fill = max(0.0, min(float(value), 1.0))
    alpha = 0.12 + fill * 0.42
    bg = f"rgba(193, 18, 31, {alpha:.3f})"
    return f'<td class="weight-cell" style="background:{bg}">{value:.1%}</td>'


def _embed_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _html_table(headers: Iterable[str], rows: Iterable[Iterable[str]], table_class: str = "data-table") -> str:
    head = "".join(f"<th>{html.escape(str(col))}</th>" for col in headers)
    body_rows = []
    for row in rows:
        cells = "".join(str(cell) for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f'<table class="{table_class}"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _rename_strategy_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "strategy" in out.columns:
        out["strategy"] = out["strategy"].map(lambda value: DISPLAY_MAP.get(str(value), str(value)))
    out.columns = [DISPLAY_MAP.get(str(col), str(col)) for col in out.columns]
    return out


def _build_summary_cards(metrics: pd.DataFrame) -> str:
    cards: List[str] = []
    for _, row in metrics.iterrows():
        name = str(row["strategy"])
        color = COLOR_MAP[name]
        is_dynamic = name == DISPLAY_NAME
        cards.append(
            "\n".join(
                [
                    f'<section class="summary-card" style="border-top-color:{color}">',
                    f'<div class="summary-title">{html.escape(name)}</div>',
                    f'<div class="summary-note">{html.escape("进攻版主策略" if is_dynamic else "对照基准")}</div>',
                    '<div class="metric-grid">',
                    f'<div><span>年化收益</span><strong>{_format_pct(float(row["annual_return"]))}</strong></div>',
                    f'<div><span>最大回撤</span><strong>{_format_pct(float(row["max_drawdown"]))}</strong></div>',
                    f'<div><span>夏普</span><strong>{_format_num(float(row["sharpe"]))}</strong></div>',
                    f'<div><span>月胜率</span><strong>{_format_pct(float(row["monthly_win_rate"]))}</strong></div>',
                    "</div>",
                    "</section>",
                ]
            )
        )
    return "".join(cards)


def _build_comparison_table(metrics: pd.DataFrame) -> str:
    headers = ["策略", "规则", "观察窗", "累计收益", "年化收益", "年化波动", "最大回撤", "夏普", "Calmar", "月胜率", "年化换手代理"]
    rows = []
    for _, row in metrics.iterrows():
        lookback_label = "-" if row["strategy"] == "当前V2" else f"{int(row['lookback_months'])}M"
        rows.append(
            [
                f"<td>{html.escape(str(row['strategy']))}</td>",
                f"<td>{html.escape(str(row['selection_rule_label']) if row['strategy'] == DISPLAY_NAME else '基准')}</td>",
                f"<td>{lookback_label}</td>",
                f"<td>{_format_pct(float(row['total_return']))}</td>",
                f"<td>{_format_pct(float(row['annual_return']))}</td>",
                f"<td>{_format_pct(float(row['annual_volatility']))}</td>",
                f"<td>{_format_pct(float(row['max_drawdown']))}</td>",
                f"<td>{_format_num(float(row['sharpe']))}</td>",
                f"<td>{_format_num(float(row['calmar']))}</td>",
                f"<td>{_format_pct(float(row['monthly_win_rate']))}</td>",
                f"<td>{_format_pct(float(row['annualized_turnover_proxy']))}</td>",
            ]
        )
    return _html_table(headers, rows)


def _build_weight_summary_table(weights: pd.DataFrame) -> str:
    latest = weights.iloc[-1]
    avg = weights.mean()
    min_w = weights.min()
    max_w = weights.max()
    headers = ["资产桶", "最新权重", "区间均值", "区间最小", "区间最大"]
    rows = []
    for bucket in [col for col in weights.columns if col not in {"selected_candidate_index"}]:
        rows.append(
            [
                f"<td>{BUCKET_LABELS.get(bucket, bucket)}</td>",
                _weight_cell(float(latest[bucket])),
                _weight_cell(float(avg[bucket])),
                f"<td>{float(min_w[bucket]):.1%}</td>",
                f"<td>{float(max_w[bucket]):.1%}</td>",
            ]
        )
    return _html_table(headers, rows, table_class="data-table compact")


def _build_selection_table(selection_counts: pd.DataFrame) -> str:
    top = selection_counts.head(10).copy()
    headers = ["候选编号", "出现次数"]
    rows = []
    for _, row in top.iterrows():
        rows.append([f"<td>{int(row['selected_candidate_index'])}</td>", f"<td>{int(row['count'])}</td>"])
    return _html_table(headers, rows, table_class="data-table compact narrow")


def _build_decision_table(decisions: pd.DataFrame) -> str:
    tail = decisions.tail(8).copy()
    headers = ["调仓信号日", "执行日", "观察窗年化", "观察窗回撤", "候选编号", "沪深300", "中证A500", "低波", "红利", "5Y债", "10Y债", "黄金", "货币"]
    rows = []
    for _, row in tail.iterrows():
        rows.append(
            [
                f"<td>{html.escape(str(row['rebalance_signal_date'])[:10])}</td>",
                f"<td>{html.escape(str(row['execution_date'])[:10])}</td>",
                f"<td>{_format_pct(float(row['lookback_annual_return']))}</td>",
                f"<td>{_format_pct(float(row['lookback_max_drawdown']))}</td>",
                f"<td>{int(row['selected_candidate_index'])}</td>",
                _weight_cell(float(row['equity_core_csi300'])),
                _weight_cell(float(row['equity_core_csia500'])),
                _weight_cell(float(row['equity_defensive_lowvol'])),
                _weight_cell(float(row['equity_defensive_dividend'])),
                _weight_cell(float(row['rate_bond_5y'])),
                _weight_cell(float(row['rate_bond_10y'])),
                _weight_cell(float(row['gold'])),
                _weight_cell(float(row['money_market'])),
            ]
        )
    return _html_table(headers, rows, table_class="data-table compact")


def _build_period_metric_table(period_metrics: pd.DataFrame, value_column: str, period_label: str) -> str:
    pivot = period_metrics.pivot(index="period", columns="strategy", values=value_column)
    pivot = pivot.reindex(columns=[DISPLAY_NAME, "当前V2"]).sort_index(ascending=False)
    headers = [period_label, *pivot.columns]
    rows = []
    for period, row in pivot.iterrows():
        rows.append(
            [
                f"<td>{html.escape(str(period))}</td>",
                f"<td>{_format_pct(float(row[DISPLAY_NAME]))}</td>",
                f"<td>{_format_pct(float(row['当前V2']))}</td>",
            ]
        )
    return _html_table(headers, rows, table_class="data-table compact period-table")


def _build_period_sections(index_levels: pd.DataFrame, detail_dir: Path, sample_end: str) -> str:
    yearly = _rename_strategy_column(summarize_period_metrics(index_levels, freq="YE"))
    monthly = _rename_strategy_column(summarize_period_metrics(index_levels, freq="ME"))
    yearly.to_csv(detail_dir / "yearly_period_metrics.csv", index=False)
    monthly.to_csv(detail_dir / "monthly_period_metrics.csv", index=False)

    sections = []
    for section_id, title, note, period_label, metrics in [
        (
            "yearly-breakdown",
            "年度收益 / 波动 / 回撤对比",
            f"波动按各年份内日收益折算为年化波动，回撤按年初重置后计算；最新年份为截至 {sample_end} 的部分期间。",
            "年份",
            yearly,
        ),
        (
            "monthly-breakdown",
            "月度收益 / 波动 / 回撤对比",
            f"波动按各月份内日收益折算为年化波动，回撤按月初重置后计算；最新月份为截至 {sample_end} 的部分期间。",
            "月份",
            monthly,
        ),
    ]:
        sections.append(
            "\n".join(
                [
                    f'<section class="section" id="{section_id}">',
                    f"<h2>{html.escape(title)}</h2>",
                    f'<p class="section-note">{html.escape(note)}</p>',
                    '<div class="period-table-stack">',
                    '<div class="panel-card">',
                    "<h3>收益对比</h3>",
                    '<div class="table-wrap">',
                    _build_period_metric_table(metrics, "period_return", period_label),
                    "</div>",
                    "</div>",
                    '<div class="panel-card">',
                    "<h3>波动对比</h3>",
                    '<div class="table-wrap">',
                    _build_period_metric_table(metrics, "annualized_volatility", period_label),
                    "</div>",
                    "</div>",
                    '<div class="panel-card">',
                    "<h3>回撤对比</h3>",
                    '<div class="table-wrap">',
                    _build_period_metric_table(metrics, "max_drawdown", period_label),
                    "</div>",
                    "</div>",
                    "</div>",
                    "</section>",
                ]
            )
        )
    return "".join(sections)


def _format_axis_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.0%}" if abs(value) >= 0.1 else f"{value:.1%}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _build_line_chart_svg(
    levels: pd.DataFrame,
    title: str,
    output_path: Path,
    color_map: Dict[str, str],
    value_kind: str = "level",
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
        color = color_map.get(column, "#5c5c5c")
        stroke_width = 4 if column == DISPLAY_NAME else 2.6
        pts = [f"{x_pos(i):.2f},{y_pos(float(v)):.2f}" for i, v in enumerate(levels[column])]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" points="{" ".join(pts)}" />')
        parts.append(
            f'<text x="{pad_left + plot_width + 28}" y="{pad_top + 28 + idx * 26}" font-size="13" font-family="Arial, sans-serif" fill="#1f2933">{html.escape(column)}</text>'
        )
        parts.append(
            f'<line x1="{pad_left + plot_width + 4}" y1="{pad_top + 22 + idx * 26}" x2="{pad_left + plot_width + 22}" y2="{pad_top + 22 + idx * 26}" stroke="{color}" stroke-width="4" />'
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


def _load_asset_comparison_levels(
    config_path: Path,
    price_path: Path,
    strategy_levels: pd.Series,
    sample_start: str,
    sample_end: str,
) -> pd.DataFrame:
    config = load_config(config_path)
    universe = load_universe(resolve_path(config, config["paths"]["universe"]))
    prices = load_price_data(price_path)
    selection = select_representatives(universe, config["bucket_order"], available_symbols=prices.columns)
    bucket_prices = build_bucket_price_frame(prices, selection.bucket_to_symbol, config["bucket_order"]).ffill()
    bucket_prices = bucket_prices.rename(columns={bucket: BUCKET_LABELS.get(bucket, bucket) for bucket in bucket_prices.columns})
    window = bucket_prices.loc[(bucket_prices.index >= pd.Timestamp(sample_start)) & (bucket_prices.index <= pd.Timestamp(sample_end))].copy()
    merged = pd.concat([strategy_levels.rename(DISPLAY_NAME), window], axis=1, join="inner").sort_index()
    merged = merged.div(merged.iloc[0])
    merged.index.name = "date"
    return merged


def _build_asset_metrics_table(levels: pd.DataFrame) -> pd.DataFrame:
    returns = levels.pct_change(fill_method=None).fillna(0.0)
    rows = []
    for column in levels.columns:
        metrics = _summary_metrics(levels[column], returns[column])
        rows.append({"标的": column, **metrics})
    out = pd.DataFrame(rows)
    out["sort_key"] = out["标的"].map(lambda value: 0 if value == DISPLAY_NAME else 1)
    out = out.sort_values(["sort_key", "annual_return"], ascending=[True, False]).drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def _build_asset_metrics_html(metrics: pd.DataFrame) -> str:
    headers = ["标的", "累计收益", "年化收益", "年化波动", "最大回撤", "夏普", "Calmar", "月胜率"]
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            [
                f"<td>{html.escape(str(row['标的']))}</td>",
                f"<td>{_format_pct(float(row['total_return']))}</td>",
                f"<td>{_format_pct(float(row['annual_return']))}</td>",
                f"<td>{_format_pct(float(row['annual_volatility']))}</td>",
                f"<td>{_format_pct(float(row['max_drawdown']))}</td>",
                f"<td>{_format_num(float(row['sharpe']))}</td>",
                f"<td>{_format_num(float(row['calmar']))}</td>",
                f"<td>{_format_pct(float(row['monthly_win_rate']))}</td>",
            ]
        )
    return _html_table(headers, rows, table_class="data-table compact")


def main() -> None:
    args = parse_args()
    detail_dir = Path(args.detail_dir)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(detail_dir / "comparison_metrics.csv")
    metrics["strategy"] = metrics["strategy"].map(lambda value: DISPLAY_MAP.get(str(value), str(value)))

    index_levels = pd.read_csv(detail_dir / "comparison_index_levels.csv", index_col="date", parse_dates=["date"])
    index_levels = index_levels.rename(columns=DISPLAY_MAP)
    decisions = pd.read_csv(
        detail_dir / "rolling_decisions.csv",
        parse_dates=["rebalance_signal_date", "execution_date", "lookback_start", "lookback_end", "hold_start_signal_date", "hold_end_signal_date"],
    )
    weights = pd.read_csv(detail_dir / "dynamic_target_weights.csv", index_col="date", parse_dates=["date"])
    selection_counts = pd.read_csv(detail_dir / "selection_counts.csv")

    sample_start = index_levels.index.min().strftime("%Y-%m-%d")
    sample_end = index_levels.index.max().strftime("%Y-%m-%d")
    dynamic_row = metrics.loc[metrics["strategy"] == DISPLAY_NAME].iloc[0]
    current_row = metrics.loc[metrics["strategy"] == "当前V2"].iloc[0]
    latest_decision = decisions.iloc[-1]
    asset_levels = _load_asset_comparison_levels(
        config_path=Path(args.config),
        price_path=Path(args.prices),
        strategy_levels=index_levels[DISPLAY_NAME],
        sample_start=sample_start,
        sample_end=sample_end,
    )
    asset_drawdowns = asset_levels.div(asset_levels.cummax()) - 1.0
    asset_metrics = _build_asset_metrics_table(asset_levels)

    asset_levels.to_csv(detail_dir / "asset_comparison_levels.csv", index=True)
    asset_drawdowns.to_csv(detail_dir / "asset_comparison_drawdowns.csv", index=True)
    asset_metrics.to_csv(detail_dir / "asset_comparison_metrics.csv", index=False)
    _build_line_chart_svg(asset_levels, "收益优先策略 vs 底层资产净值对比", detail_dir / "asset_nav_comparison.svg", ASSET_COLOR_MAP, value_kind="level")
    _build_line_chart_svg(asset_drawdowns, "收益优先策略 vs 底层资产回撤对比", detail_dir / "asset_drawdown_comparison.svg", ASSET_COLOR_MAP, value_kind="pct")

    nav_svg = _embed_svg(detail_dir / "nav_comparison.svg")
    drawdown_svg = _embed_svg(detail_dir / "drawdown_comparison.svg")
    asset_nav_svg = _embed_svg(detail_dir / "asset_nav_comparison.svg")
    asset_drawdown_svg = _embed_svg(detail_dir / "asset_drawdown_comparison.svg")

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(str(args.strategy_name))} 独立结果页</title>
  <style>
    :root {{
      --bg: #f7f2eb;
      --panel: #fffaf5;
      --panel-strong: #fff4ec;
      --line: #eed8c7;
      --text: #1b263b;
      --muted: #5f5f5f;
      --accent: #c1121f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(193, 18, 31, 0.12), transparent 22rem),
        linear-gradient(180deg, #fffaf5 0%, var(--bg) 56%, #efe2d5 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 24px 64px;
    }}
    .hero {{
      padding: 28px 32px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 250, 245, 0.92);
      box-shadow: 0 24px 80px rgba(27, 38, 59, 0.08);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.15;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      max-width: 980px;
    }}
    .hero-meta {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .hero-meta div {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .hero-meta span {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .hero-meta strong {{
      font-size: 18px;
    }}
    .anchor-links {{
      margin-top: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .anchor-links a {{
      text-decoration: none;
      color: var(--text);
      background: #fff4ec;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .section {{
      margin-top: 26px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}
    .section-note {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 5px solid var(--accent);
      border-radius: 18px;
      padding: 18px 18px 16px;
      box-shadow: 0 12px 40px rgba(27, 38, 59, 0.05);
    }}
    .summary-title {{
      font-weight: 700;
      font-size: 18px;
      line-height: 1.3;
    }}
    .summary-note {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .metric-grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric-grid span, .mini-metrics span {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .metric-grid strong, .mini-metrics strong {{
      font-size: 18px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart-card, .panel-card {{
      background: rgba(255, 250, 245, 0.94);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 46px rgba(27, 38, 59, 0.05);
    }}
    .chart-card {{
      padding: 10px;
      overflow: hidden;
    }}
    .chart-card svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .mini-metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .mini-metrics div {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: 1.7fr 0.9fr;
      gap: 16px;
      align-items: start;
    }}
    .panel-card {{
      padding: 14px;
    }}
    .panel-card h3 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .period-table-stack {{
      display: grid;
      gap: 16px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 8px 10px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #f0e0d4;
      padding: 10px 10px;
      text-align: right;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .data-table th:first-child, .data-table td:first-child {{
      text-align: left;
    }}
    .data-table th:nth-child(2), .data-table td:nth-child(2),
    .data-table th:nth-child(3), .data-table td:nth-child(3) {{
      text-align: left;
    }}
    .data-table thead th {{
      position: sticky;
      top: 0;
      background: #fff6ef;
      z-index: 1;
      color: #364152;
      font-size: 12px;
    }}
    .compact th, .compact td {{
      padding: 8px 8px;
      font-size: 12px;
    }}
    .period-table th, .period-table td {{
      text-align: right;
    }}
    .period-table th:first-child, .period-table td:first-child {{
      text-align: left;
    }}
    .narrow {{
      max-width: 420px;
    }}
    .weight-cell {{
      font-variant-numeric: tabular-nums;
    }}
    .footer {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 1200px) {{
      .hero-meta, .mini-metrics, .chart-grid, .panel-grid {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 860px) {{
      .summary-grid, .hero-meta, .mini-metrics, .chart-grid, .panel-grid {{
        grid-template-columns: 1fr;
      }}
      .wrap {{
        padding: 18px 14px 42px;
      }}
      .hero {{
        padding: 22px 18px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{html.escape(str(args.strategy_name))} 独立结果页</h1>
      <p>这份页面只展示收益优先滚动版本和当前 V2 的对比结果。规则上不做回撤护栏，直接按观察窗收益选组合，用于单独观察收益最大化方案的收益、波动、回撤与季度决策。</p>
      <div class="hero-meta">
        <div><span>样本区间</span><strong>{html.escape(sample_start)} 至 {html.escape(sample_end)}</strong></div>
        <div><span>观察窗</span><strong>{int(dynamic_row['lookback_months'])} 个月</strong></div>
        <div><span>规则</span><strong>{html.escape(str(dynamic_row['selection_rule_label']))}</strong></div>
        <div><span>相对当前年化提升</span><strong>{_format_pct(float(dynamic_row['annual_return'] - current_row['annual_return']))}</strong></div>
      </div>
      <div class="anchor-links">
        <a href="#charts">图表</a>
        <a href="#summary">指标总表</a>
        <a href="#asset-compare">底层资产</a>
        <a href="#yearly-breakdown">年度拆解</a>
        <a href="#monthly-breakdown">月度拆解</a>
        <a href="#detail">调仓细节</a>
      </div>
    </section>

    <section class="section">
      <h2>摘要卡片</h2>
      <div class="summary-grid">
        {_build_summary_cards(metrics)}
      </div>
    </section>

    <section class="section" id="charts">
      <h2>核心图表</h2>
      <div class="chart-grid">
        <div class="chart-card">{nav_svg}</div>
        <div class="chart-card">{drawdown_svg}</div>
      </div>
    </section>

    <section class="section" id="summary">
      <h2>指标总表</h2>
      <div class="table-wrap">
        {_build_comparison_table(metrics)}
      </div>
    </section>

    <section class="section" id="asset-compare">
      <h2>底层资产对比</h2>
      <p class="section-note">以下对比使用同一套底层资产桶代表标的：沪深300、中证A500、低波、红利、5Y债、10Y债、黄金、货币。口径与策略样本区间对齐，便于看这套进攻版到底是在战胜哪类资产、又输给哪类资产。</p>
      <div class="chart-grid">
        <div class="chart-card">{asset_nav_svg}</div>
        <div class="chart-card">{asset_drawdown_svg}</div>
      </div>
      <div class="panel-card" style="margin-top:16px;">
        <h3>底层资产指标表</h3>
        <div class="table-wrap">
          {_build_asset_metrics_html(asset_metrics)}
        </div>
      </div>
    </section>

    {_build_period_sections(index_levels, detail_dir, sample_end)}

    <section class="section" id="detail">
      <h2>调仓细节</h2>
      <div class="mini-metrics">
        <div><span>策略年化</span><strong>{_format_pct(float(dynamic_row['annual_return']))}</strong></div>
        <div><span>策略回撤</span><strong>{_format_pct(float(dynamic_row['max_drawdown']))}</strong></div>
        <div><span>最近执行日</span><strong>{str(latest_decision['execution_date'])[:10]}</strong></div>
        <div><span>年化换手代理</span><strong>{_format_pct(float(dynamic_row['annualized_turnover_proxy']))}</strong></div>
      </div>
      <div class="panel-grid">
        <div class="panel-card">
          <h3>动态权重摘要</h3>
          {_build_weight_summary_table(weights)}
        </div>
        <div class="panel-card">
          <h3>高频候选组合</h3>
          {_build_selection_table(selection_counts)}
        </div>
      </div>
      <div class="panel-card" style="margin-top:16px;">
        <h3>最近 8 次季度决策</h3>
        {_build_decision_table(decisions)}
      </div>
    </section>

    <div class="footer">
      源文件来自 `{html.escape(str(detail_dir.relative_to(ROOT)))}`；这是收益优先版本的独立静态 HTML 页面。
    </div>
  </div>
</body>
</html>
"""

    output_html.write_text(html_doc, encoding="utf-8")
    print(f"output_html={output_html.resolve()}")


if __name__ == "__main__":
    main()
