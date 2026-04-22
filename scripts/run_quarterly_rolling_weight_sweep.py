#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.backtest import run_backtest  # noqa: E402
from etf_fof_index.config import load_config, resolve_path, validate_config  # noqa: E402
from etf_fof_index.data import load_price_data, load_valuation_data  # noqa: E402
from etf_fof_index.report import _summary_metrics  # noqa: E402
from etf_fof_index.rolling import (  # noqa: E402
    build_quarterly_rolling_target_weights_from_windows,
    prepare_rebalance_windows,
    selection_rule_label,
)
from etf_fof_index.signals import build_bucket_price_frame, compute_signals  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402
from run_quarterly_rolling_weight_strategy import _summarize_backtest, _run_matrix_backtest_detail, run_study  # noqa: E402
from run_weight_grid_research_v2_matrix import (  # noqa: E402
    BUCKETS,
    add_current_v2,
    compute_signal_deltas,
    compute_target_weight_tensor,
    enumerate_weight_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep rolling quarterly selection rules and lookback windows.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v2.yaml"))
    parser.add_argument("--prices", default=str(ROOT / "data" / "input" / "prices_v2_index_proxy.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "rolling_quarterly_v2_rule_sweep"))
    parser.add_argument("--valuation", help="Optional valuation file.")
    parser.add_argument("--lookback-min", type=int, default=6)
    parser.add_argument("--lookback-max", type=int, default=36)
    parser.add_argument("--lookback-step", type=int, default=3)
    parser.add_argument("--selection-rules", default="min_drawdown,sharpe_guard,calmar_guard")
    parser.add_argument("--drawdown-band", type=float, default=0.02)
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


def _format_percent_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].map(lambda value: f"{float(value):.2%}")
    return out


def _format_num_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = out[column].map(lambda value: f"{float(value):.2f}")
    return out


def _common_window_rows(results: List[dict]) -> pd.DataFrame:
    common_start = max(pd.Timestamp(result["sample_start"]) for result in results)
    common_end = min(pd.Timestamp(result["sample_end"]) for result in results)
    rows = []

    for result in results:
        index_frame = result["comparison_index"].loc[common_start:common_end].copy()
        dynamic = _summary_metrics(
            index_frame["动态季度滚动"],
            index_frame["动态季度滚动"].pct_change(fill_method=None).fillna(0.0),
        )
        current = _summary_metrics(
            index_frame["当前V2"],
            index_frame["当前V2"].pct_change(fill_method=None).fillna(0.0),
        )
        rows.append(
            {
                "selection_rule": result["selection_rule"],
                "selection_rule_label": result["selection_rule_label"],
                "lookback_months": result["lookback_months"],
                "sample_start": common_start.strftime("%Y-%m-%d"),
                "sample_end": common_end.strftime("%Y-%m-%d"),
                "dynamic_total_return": dynamic["total_return"],
                "dynamic_annual_return": dynamic["annual_return"],
                "dynamic_annual_volatility": dynamic["annual_volatility"],
                "dynamic_max_drawdown": dynamic["max_drawdown"],
                "dynamic_sharpe": dynamic["sharpe"],
                "current_total_return": current["total_return"],
                "current_annual_return": current["annual_return"],
                "current_annual_volatility": current["annual_volatility"],
                "current_max_drawdown": current["max_drawdown"],
                "current_sharpe": current["sharpe"],
                "drawdown_improvement": dynamic["max_drawdown"] - current["max_drawdown"],
                "annual_return_delta": dynamic["annual_return"] - current["annual_return"],
                "sharpe_delta": dynamic["sharpe"] - current["sharpe"],
            }
        )

    return pd.DataFrame(rows).sort_values(["selection_rule", "lookback_months"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rules = [item.strip() for item in str(args.selection_rules).split(",") if item.strip()]
    lookbacks = list(range(int(args.lookback_min), int(args.lookback_max) + 1, int(args.lookback_step)))

    config = load_config(Path(args.config))
    validate_config(config)
    universe = load_universe(resolve_path(config, config["paths"]["universe"]))
    prices = load_price_data(Path(args.prices))
    valuation_data = load_valuation_data(Path(args.valuation)) if args.valuation else pd.DataFrame()
    selection = select_representatives(universe, config["bucket_order"], available_symbols=prices.columns)
    bucket_prices = build_bucket_price_frame(prices, selection.bucket_to_symbol, config["bucket_order"]).ffill()
    signals = compute_signals(bucket_prices=bucket_prices, valuation_data=valuation_data, config=config)
    returns = bucket_prices.pct_change(fill_method=None).fillna(0.0)

    candidate_frame = add_current_v2(enumerate_weight_grid(), config)
    base_weights = candidate_frame[BUCKETS].to_numpy(dtype=float)
    signal_dates, signal_deltas = compute_signal_deltas(config, signals)
    target_tensor = compute_target_weight_tensor(base_weights, signal_deltas, config)
    run_index, ret_store, turnover_store = _run_matrix_backtest_detail(returns, signal_dates, target_tensor, base_weights, config)

    results = []
    native_rows = []
    detail_root = output_dir / "details"
    current_cache = {}
    window_cache = {}

    for lookback in lookbacks:
        window_cache[lookback] = prepare_rebalance_windows(
            signal_dates=signal_dates,
            run_index=run_index,
            ret_store=ret_store,
            turnover_store=turnover_store,
            lookback_months=int(lookback),
        )

    for selection_rule in rules:
        for lookback in lookbacks:
            rolling_result = build_quarterly_rolling_target_weights_from_windows(
                windows=window_cache[int(lookback)],
                signal_dates=signal_dates,
                target_tensor=target_tensor,
                base_weight_frame=candidate_frame[BUCKETS],
                buckets=BUCKETS,
                selection_rule=selection_rule,
                drawdown_band=float(args.drawdown_band),
            )
            current_start = rolling_result.target_weights.index.min()
            current_target_weights = pd.DataFrame(target_tensor[:, 0, :], index=signal_dates, columns=BUCKETS).loc[current_start:]
            current_target_weights = current_target_weights.loc[rolling_result.target_weights.index]

            dynamic_config = copy.deepcopy(config)
            dynamic_config["strategic_weights"] = rolling_result.decision_table.iloc[0][BUCKETS].astype(float).to_dict()
            dynamic_backtest = run_backtest(bucket_prices, rolling_result.target_weights[BUCKETS], dynamic_config, label="dynamic")

            if lookback not in current_cache:
                current_backtest = run_backtest(bucket_prices, current_target_weights[BUCKETS], config, label="current")
                current_cache[lookback] = current_backtest
            else:
                current_backtest = current_cache[lookback]

            comparison_levels = dynamic_backtest.levels.join(current_backtest.levels, how="inner")
            comparison_index = pd.DataFrame(
                {
                    "动态季度滚动": comparison_levels["dynamic_index"],
                    "当前V2": comparison_levels["current_index"],
                }
            )
            metrics_table = pd.DataFrame(
                [
                    {
                        "strategy": "动态季度滚动",
                        "selection_rule": selection_rule,
                        "selection_rule_label": selection_rule_label(selection_rule),
                        "drawdown_band": float(args.drawdown_band),
                        "lookback_months": int(lookback),
                        **_summarize_backtest(comparison_levels[["dynamic_index", "dynamic_return", "dynamic_turnover"]], "dynamic"),
                    },
                    {
                        "strategy": "当前V2",
                        "selection_rule": selection_rule,
                        "selection_rule_label": selection_rule_label(selection_rule),
                        "drawdown_band": float(args.drawdown_band),
                        "lookback_months": int(lookback),
                        **_summarize_backtest(comparison_levels[["current_index", "current_return", "current_turnover"]], "current"),
                    },
                ]
            )
            dynamic = metrics_table.loc[metrics_table["strategy"] == "动态季度滚动"].iloc[0]
            current = metrics_table.loc[metrics_table["strategy"] == "当前V2"].iloc[0]
            result = {
                "selection_rule": selection_rule,
                "selection_rule_label": selection_rule_label(selection_rule),
                "lookback_months": int(lookback),
                "sample_start": comparison_index.index.min().strftime("%Y-%m-%d"),
                "sample_end": comparison_index.index.max().strftime("%Y-%m-%d"),
                "comparison_index": comparison_index,
            }
            results.append(result)
            native_rows.append(
                {
                    "selection_rule": selection_rule,
                    "selection_rule_label": selection_rule_label(selection_rule),
                    "lookback_months": int(lookback),
                    "sample_start": comparison_index.index.min().strftime("%Y-%m-%d"),
                    "sample_end": comparison_index.index.max().strftime("%Y-%m-%d"),
                    "dynamic_total_return": dynamic["total_return"],
                    "dynamic_annual_return": dynamic["annual_return"],
                    "dynamic_annual_volatility": dynamic["annual_volatility"],
                    "dynamic_max_drawdown": dynamic["max_drawdown"],
                    "dynamic_sharpe": dynamic["sharpe"],
                    "current_total_return": current["total_return"],
                    "current_annual_return": current["annual_return"],
                    "current_annual_volatility": current["annual_volatility"],
                    "current_max_drawdown": current["max_drawdown"],
                    "current_sharpe": current["sharpe"],
                    "drawdown_improvement": dynamic["max_drawdown"] - current["max_drawdown"],
                    "annual_return_delta": dynamic["annual_return"] - current["annual_return"],
                    "sharpe_delta": dynamic["sharpe"] - current["sharpe"],
                    "turnover_delta": dynamic["annualized_turnover_proxy"] - current["annualized_turnover_proxy"],
                }
            )

    native_summary = pd.DataFrame(native_rows).sort_values(["selection_rule", "lookback_months"]).reset_index(drop=True)
    common_summary = _common_window_rows(results)

    native_summary.to_csv(output_dir / "summary.csv", index=False)
    common_summary.to_csv(output_dir / "common_window_summary.csv", index=False)

    best_native = native_summary.sort_values(
        ["drawdown_improvement", "annual_return_delta", "sharpe_delta"],
        ascending=[False, False, False],
    ).groupby("selection_rule", as_index=False).head(1).reset_index(drop=True)
    best_native.to_csv(output_dir / "best_by_rule.csv", index=False)

    overall_best = common_summary.sort_values(
        ["drawdown_improvement", "annual_return_delta", "sharpe_delta"],
        ascending=[False, False, False],
    ).iloc[0]

    for _, row in best_native.iterrows():
        combo_dir = detail_root / f"{row['selection_rule']}_lookback{int(row['lookback_months'])}"
        run_study(
            config_path=Path(args.config),
            price_path=Path(args.prices),
            output_dir=combo_dir,
            valuation_path=Path(args.valuation) if args.valuation else None,
            lookback_months=int(row["lookback_months"]),
            selection_rule=str(row["selection_rule"]),
            drawdown_band=float(args.drawdown_band),
            write_outputs=True,
        )

    summary_md = native_summary[
        [
            "selection_rule_label",
            "lookback_months",
            "sample_start",
            "sample_end",
            "dynamic_annual_return",
            "dynamic_max_drawdown",
            "dynamic_sharpe",
            "drawdown_improvement",
            "annual_return_delta",
            "sharpe_delta",
            "turnover_delta",
        ]
    ].copy()
    summary_md = _format_percent_columns(
        summary_md,
        ["dynamic_annual_return", "dynamic_max_drawdown", "drawdown_improvement", "annual_return_delta", "turnover_delta"],
    )
    summary_md = _format_num_columns(summary_md, ["dynamic_sharpe", "sharpe_delta"])

    common_md = common_summary[
        [
            "selection_rule_label",
            "lookback_months",
            "sample_start",
            "sample_end",
            "dynamic_annual_return",
            "dynamic_max_drawdown",
            "dynamic_sharpe",
            "drawdown_improvement",
            "annual_return_delta",
            "sharpe_delta",
        ]
    ].copy()
    common_md = _format_percent_columns(
        common_md,
        ["dynamic_annual_return", "dynamic_max_drawdown", "drawdown_improvement", "annual_return_delta"],
    )
    common_md = _format_num_columns(common_md, ["dynamic_sharpe", "sharpe_delta"])

    best_native_md = best_native[
        [
            "selection_rule_label",
            "lookback_months",
            "sample_start",
            "sample_end",
            "dynamic_annual_return",
            "dynamic_max_drawdown",
            "dynamic_sharpe",
            "drawdown_improvement",
            "annual_return_delta",
        ]
    ].copy()
    best_native_md = _format_percent_columns(
        best_native_md,
        ["dynamic_annual_return", "dynamic_max_drawdown", "drawdown_improvement", "annual_return_delta"],
    )
    best_native_md = _format_num_columns(best_native_md, ["dynamic_sharpe"])

    report_lines = [
        "# 动态季度滚动规则扫描",
        "",
        "## 参数",
        "",
        f"- lookback range: `{int(args.lookback_min)}..{int(args.lookback_max)}`，step=`{int(args.lookback_step)}`",
        f"- selection rules: `{', '.join(rules)}`",
        f"- drawdown band: `{float(args.drawdown_band):.2%}`",
        "",
        "## 各规则最佳组合",
        "",
        _to_markdown_table(best_native_md),
        "",
        "## 原始样本汇总",
        "",
        _to_markdown_table(summary_md),
        "",
        "## 共同样本窗口汇总",
        "",
        _to_markdown_table(common_md),
        "",
        "## 总体最佳",
        "",
        f"- 规则：`{overall_best['selection_rule_label']}`",
        f"- 观察窗：`{int(overall_best['lookback_months'])}` 个月",
        f"- 共同样本回撤改善：`{overall_best['drawdown_improvement']:.2%}`",
        f"- 共同样本年化变化：`{overall_best['annual_return_delta']:.2%}`",
        "",
        "## 文件",
        "",
        f"- [summary.csv]({(output_dir / 'summary.csv').resolve()})",
        f"- [common_window_summary.csv]({(output_dir / 'common_window_summary.csv').resolve()})",
        f"- [best_by_rule.csv]({(output_dir / 'best_by_rule.csv').resolve()})",
        f"- [details/]({detail_root.resolve()})",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"output_dir={output_dir.resolve()}")
    print(f"overall_best_rule={overall_best['selection_rule']}")
    print(f"overall_best_lookback={int(overall_best['lookback_months'])}")


if __name__ == "__main__":
    main()
