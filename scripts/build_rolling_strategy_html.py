#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, Iterable, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.config import load_config  # noqa: E402
from run_quarterly_rolling_weight_strategy import run_study  # noqa: E402


DETAIL_SPECS = [
    {
        "name": "12M回撤优先(正式版)",
        "selection_rule": "min_drawdown",
        "lookback_months": 12,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_official",
        "note": "正式版",
    },
    {
        "name": "6M回撤优先(极致降回撤)",
        "selection_rule": "min_drawdown",
        "lookback_months": 6,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_mindd6",
        "note": "纯压回撤",
    },
    {
        "name": "36M夏普护栏(均衡版)",
        "selection_rule": "sharpe_guard",
        "lookback_months": 36,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_sharpe36",
        "note": "更均衡",
    },
    {
        "name": "36MCalmar护栏(防守版)",
        "selection_rule": "calmar_guard",
        "lookback_months": 36,
        "detail_dir": ROOT / "output" / "rolling_quarterly_v2_risk_candidate_calmar36",
        "note": "更防守",
    },
]

COLOR_MAP = {
    "当前V2": "#243b53",
    "12M回撤优先(正式版)": "#d1495b",
    "6M回撤优先(极致降回撤)": "#edae49",
    "36M夏普护栏(均衡版)": "#00798c",
    "36MCalmar护栏(防守版)": "#4f772d",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML dashboard for rolling strategy comparison.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2_index_proxy.csv"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    parser.add_argument("--drawdown-band", type=float, default=0.02)
    parser.add_argument(
        "--comparison-dir",
        default=str(ROOT / "output" / "rolling_quarterly_strategy_comparison"),
    )
    parser.add_argument(
        "--output-html",
        default=str(ROOT / "output" / "rolling_quarterly_strategy_comparison" / "dashboard.html"),
    )
    return parser.parse_args()


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_num(value: float) -> str:
    return f"{value:.2f}"


def _weight_cell(value: float) -> str:
    fill = max(0.0, min(float(value), 1.0))
    alpha = 0.12 + fill * 0.42
    bg = f"rgba(0, 121, 140, {alpha:.3f})"
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


def _read_metrics_row(detail_dir: Path) -> Dict[str, float]:
    metrics = pd.read_csv(detail_dir / "comparison_metrics.csv")
    dynamic = metrics.loc[metrics["strategy"] == "动态季度滚动"].iloc[0]
    return dynamic.to_dict()


def _ensure_six_month_detail(args: argparse.Namespace) -> Path:
    spec = next(item for item in DETAIL_SPECS if item["lookback_months"] == 6)
    detail_dir = Path(spec["detail_dir"])
    metrics_file = detail_dir / "comparison_metrics.csv"
    if metrics_file.exists():
        return detail_dir
    run_study(
        config_path=Path(args.config),
        price_path=Path(args.prices),
        output_dir=detail_dir,
        valuation_path=Path(args.valuation) if args.valuation else None,
        lookback_months=6,
        selection_rule="min_drawdown",
        drawdown_band=float(args.drawdown_band),
        write_outputs=True,
    )
    return detail_dir


def _load_current_weights(config_path: Path) -> Dict[str, float]:
    config = load_config(config_path)
    return {bucket: float(config["strategic_weights"][bucket]) for bucket in BUCKET_LABELS}


def _build_summary_cards(metrics: pd.DataFrame) -> str:
    cards: List[str] = []
    for _, row in metrics.iterrows():
        name = str(row["strategy"])
        color = COLOR_MAP[name]
        cards.append(
            "\n".join(
                [
                    f'<section class="summary-card" style="border-top-color:{color}">',
                    f'<div class="summary-title">{html.escape(name)}</div>',
                    f'<div class="summary-note">{html.escape(str(row["note"]))}</div>',
                    '<div class="metric-grid">',
                    f'<div><span>年化收益</span><strong>{_format_pct(float(row["annual_return"]))}</strong></div>',
                    f'<div><span>最大回撤</span><strong>{_format_pct(float(row["max_drawdown"]))}</strong></div>',
                    f'<div><span>夏普</span><strong>{_format_num(float(row["sharpe"]))}</strong></div>',
                    f'<div><span>相对当前回撤改善</span><strong>{_format_pct(float(row["drawdown_improvement"]))}</strong></div>',
                    '</div>',
                    '</section>',
                ]
            )
        )
    return "".join(cards)


def _build_comparison_table(metrics: pd.DataFrame) -> str:
    headers = ["策略", "定位", "规则", "观察窗", "年化收益", "年化波动", "最大回撤", "夏普", "相对当前年化变化", "相对当前回撤改善", "相对当前夏普变化"]
    rows = []
    for _, row in metrics.iterrows():
        lookback_label = "-" if int(row["lookback_months"]) == 0 else f"{int(row['lookback_months'])}M"
        rows.append(
            [
                f"<td>{html.escape(str(row['strategy']))}</td>",
                f"<td>{html.escape(str(row['note']))}</td>",
                f"<td>{html.escape(str(row['selection_rule_label']))}</td>",
                f"<td>{lookback_label}</td>",
                f"<td>{_format_pct(float(row['annual_return']))}</td>",
                f"<td>{_format_pct(float(row['annual_volatility']))}</td>",
                f"<td>{_format_pct(float(row['max_drawdown']))}</td>",
                f"<td>{_format_num(float(row['sharpe']))}</td>",
                f"<td>{_format_pct(float(row['annual_return_delta']))}</td>",
                f"<td>{_format_pct(float(row['drawdown_improvement']))}</td>",
                f"<td>{_format_num(float(row['sharpe_delta']))}</td>",
            ]
        )
    return _html_table(headers, rows)


def _build_weight_summary_table(weights: pd.DataFrame, common_start: str) -> str:
    filtered = weights.loc[weights.index >= pd.Timestamp(common_start)].copy()
    latest = filtered.iloc[-1]
    avg = filtered.mean()
    min_w = filtered.min()
    max_w = filtered.max()
    headers = ["资产桶", "最新权重", "区间均值", "区间最小", "区间最大"]
    rows = []
    for bucket in [col for col in filtered.columns if col not in {"selected_candidate_index"}]:
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


def _build_decision_table(decisions: pd.DataFrame) -> str:
    tail = decisions.tail(8).copy()
    headers = ["调仓信号日", "执行日", "观察窗回撤", "观察窗年化", "候选编号", "沪深300", "中证A500", "低波", "红利", "5Y债", "10Y债", "黄金", "货币"]
    rows = []
    for _, row in tail.iterrows():
        rows.append(
            [
                f"<td>{html.escape(str(row['rebalance_signal_date'])[:10])}</td>",
                f"<td>{html.escape(str(row['execution_date'])[:10])}</td>",
                f"<td>{_format_pct(float(row['lookback_max_drawdown']))}</td>",
                f"<td>{_format_pct(float(row['lookback_annual_return']))}</td>",
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


def _build_selection_table(selection_counts: pd.DataFrame) -> str:
    top = selection_counts.head(8).copy()
    headers = ["候选编号", "出现次数"]
    rows = []
    for _, row in top.iterrows():
        rows.append([f"<td>{int(row['selected_candidate_index'])}</td>", f"<td>{int(row['count'])}</td>"])
    return _html_table(headers, rows, table_class="data-table compact narrow")


def _build_current_weights_table(current_weights: Dict[str, float]) -> str:
    headers = ["资产桶", "当前V2静态权重"]
    rows = []
    for bucket, label in BUCKET_LABELS.items():
        rows.append([f"<td>{label}</td>", _weight_cell(current_weights[bucket])])
    return _html_table(headers, rows, table_class="data-table compact narrow")


def _build_detail_sections(metrics: pd.DataFrame, common_start: str, config_path: Path) -> str:
    sections: List[str] = []
    current_weights = _load_current_weights(config_path)
    current_section = [
        '<section class="strategy-panel" id="current-v2">',
        '<div class="panel-head">',
        '<h2>当前V2</h2>',
        '<p>基准组合，权重固定不滚动。</p>',
        '</div>',
        '<div class="panel-grid single">',
        '<div class="panel-card">',
        '<h3>静态权重</h3>',
        _build_current_weights_table(current_weights),
        '</div>',
        '</div>',
        '</section>',
    ]
    sections.append("".join(current_section))

    for spec in DETAIL_SPECS:
        detail_dir = Path(spec["detail_dir"])
        decisions = pd.read_csv(detail_dir / "rolling_decisions.csv", parse_dates=["rebalance_signal_date", "execution_date", "lookback_start", "lookback_end", "hold_start_signal_date", "hold_end_signal_date"])
        weights = pd.read_csv(detail_dir / "dynamic_target_weights.csv", index_col="date", parse_dates=["date"])
        selection_counts = pd.read_csv(detail_dir / "selection_counts.csv")
        dynamic = metrics.loc[metrics["strategy"] == spec["name"]].iloc[0]
        latest_decision = decisions.iloc[-1]
        sections.append(
            "\n".join(
                [
                    f'<section class="strategy-panel" id="{html.escape(spec["name"])}">',
                    '<div class="panel-head">',
                    f'<h2>{html.escape(spec["name"])}</h2>',
                    f'<p>{html.escape(spec["note"])}，规则为 {html.escape(str(dynamic["selection_rule_label"]))}，观察窗 {int(dynamic["lookback_months"])} 个月。</p>',
                    '</div>',
                    '<div class="mini-metrics">',
                    f'<div><span>共同样本年化</span><strong>{_format_pct(float(dynamic["annual_return"]))}</strong></div>',
                    f'<div><span>共同样本回撤</span><strong>{_format_pct(float(dynamic["max_drawdown"]))}</strong></div>',
                    f'<div><span>共同样本夏普</span><strong>{_format_num(float(dynamic["sharpe"]))}</strong></div>',
                    f'<div><span>最近执行日</span><strong>{str(latest_decision["execution_date"])[:10]}</strong></div>',
                    '</div>',
                    '<div class="panel-grid">',
                    '<div class="panel-card">',
                    '<h3>动态权重摘要</h3>',
                    _build_weight_summary_table(weights, common_start),
                    '</div>',
                    '<div class="panel-card">',
                    '<h3>高频候选组合</h3>',
                    _build_selection_table(selection_counts),
                    '</div>',
                    '</div>',
                    '<div class="panel-card full-width">',
                    '<h3>最近 8 次季度决策</h3>',
                    _build_decision_table(decisions),
                    '</div>',
                    '</section>',
                ]
            )
        )
    return "".join(sections)


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    _ensure_six_month_detail(args)

    metrics = pd.read_csv(comparison_dir / "comparison_table.csv")
    metrics = metrics.sort_values("lookback_months").copy()
    order = ["当前V2", *[spec["name"] for spec in DETAIL_SPECS]]
    metrics["strategy"] = pd.Categorical(metrics["strategy"], categories=order, ordered=True)
    metrics = metrics.sort_values("strategy").reset_index(drop=True)
    sample_start = str(metrics["sample_start"].iloc[0])
    sample_end = str(metrics["sample_end"].iloc[0])

    nav_svg = _embed_svg(comparison_dir / "common_window_nav.svg")
    drawdown_svg = _embed_svg(comparison_dir / "common_window_drawdown.svg")
    scatter_svg = _embed_svg(comparison_dir / "risk_return_scatter.svg")
    delta_svg = _embed_svg(comparison_dir / "delta_vs_current.svg")

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>动态季度滚动方案对比</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --panel: #fffaf0;
      --panel-strong: #fff7e8;
      --line: #e6dbc3;
      --text: #14213d;
      --muted: #5c5c5c;
      --accent: #d1495b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(237, 174, 73, 0.18), transparent 22rem),
        linear-gradient(180deg, #fcf7ed 0%, var(--bg) 58%, #efe6d6 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1520px;
      margin: 0 auto;
      padding: 28px 24px 64px;
    }}
    .hero {{
      padding: 28px 32px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 250, 240, 0.9);
      box-shadow: 0 24px 80px rgba(20, 33, 61, 0.08);
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
    .section {{
      margin-top: 26px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 16px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 5px solid var(--accent);
      border-radius: 18px;
      padding: 18px 18px 16px;
      box-shadow: 0 12px 40px rgba(20, 33, 61, 0.05);
    }}
    .summary-title {{
      font-weight: 700;
      font-size: 17px;
      line-height: 1.3;
      min-height: 44px;
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
    .chart-card, .panel-card, .strategy-panel {{
      background: rgba(255, 250, 240, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 46px rgba(20, 33, 61, 0.05);
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
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #efe5d2;
      padding: 10px 10px;
      text-align: right;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .data-table th:first-child, .data-table td:first-child {{
      text-align: left;
    }}
    .data-table th:nth-child(2), .data-table td:nth-child(2),
    .data-table th:nth-child(3), .data-table td:nth-child(3),
    .data-table th:nth-child(4), .data-table td:nth-child(4) {{
      text-align: left;
    }}
    .data-table thead th {{
      position: sticky;
      top: 0;
      background: #fff8ed;
      z-index: 1;
      color: #364152;
      font-size: 12px;
    }}
    .compact th, .compact td {{
      padding: 8px 8px;
      font-size: 12px;
    }}
    .narrow {{
      max-width: 420px;
    }}
    .weight-cell {{
      font-variant-numeric: tabular-nums;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 8px 10px;
    }}
    .strategy-panel {{
      padding: 20px;
      margin-top: 18px;
    }}
    .panel-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .panel-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .panel-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      max-width: 860px;
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
    .panel-grid.single {{
      grid-template-columns: 1fr;
    }}
    .panel-card {{
      padding: 14px;
    }}
    .panel-card h3 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .full-width {{
      margin-top: 16px;
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
      background: #fff7e8;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .footer {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 1200px) {{
      .summary-grid, .hero-meta, .mini-metrics, .chart-grid, .panel-grid {{
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
      .panel-head {{
        display: block;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>动态季度滚动方案对比</h1>
      <p>这份本地 HTML 报告把当前 V2、正式版 12M、极致降回撤 6M、36M 夏普护栏和 36M Calmar 护栏放到同一页。对比口径统一到共同样本区间，并补充了每个动态方案的最新权重摘要、候选组合出现频次和最近 8 次季度决策。</p>
      <div class="hero-meta">
        <div><span>共同样本区间</span><strong>{html.escape(sample_start)} 至 {html.escape(sample_end)}</strong></div>
        <div><span>调仓频率</span><strong>每 3 个月</strong></div>
        <div><span>候选来源</span><strong>5% 权重网格</strong></div>
        <div><span>当前更均衡候选</span><strong>36M夏普护栏</strong></div>
      </div>
      <div class="anchor-links">
        <a href="#charts">总图表</a>
        <a href="#table">总表</a>
        <a href="#current-v2">当前V2</a>
        <a href="#12M回撤优先(正式版)">12M正式版</a>
        <a href="#6M回撤优先(极致降回撤)">6M回撤优先</a>
        <a href="#36M夏普护栏(均衡版)">36M夏普护栏</a>
        <a href="#36MCalmar护栏(防守版)">36MCalmar护栏</a>
      </div>
    </section>

    <section class="section">
      <h2>摘要卡片</h2>
      <div class="summary-grid">
        {_build_summary_cards(metrics)}
      </div>
    </section>

    <section class="section" id="charts">
      <h2>总图表</h2>
      <div class="chart-grid">
        <div class="chart-card">{nav_svg}</div>
        <div class="chart-card">{drawdown_svg}</div>
        <div class="chart-card">{scatter_svg}</div>
        <div class="chart-card">{delta_svg}</div>
      </div>
    </section>

    <section class="section" id="table">
      <h2>总表</h2>
      <div class="table-wrap">
        {_build_comparison_table(metrics)}
      </div>
    </section>

    <section class="section">
      <h2>动态权重与季度决策</h2>
      {_build_detail_sections(metrics, sample_start, Path(args.config))}
    </section>

    <div class="footer">
      源文件来自 `output/rolling_quarterly_strategy_comparison/` 及各策略明细目录；这份页面为本地静态 HTML，可直接打开查看。
    </div>
  </div>
</body>
</html>
"""

    output_html.write_text(html_doc, encoding="utf-8")
    print(f"output_html={output_html.resolve()}")
    print(f"sample_start={sample_start}")
    print(f"sample_end={sample_end}")


if __name__ == "__main__":
    main()
