from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .backtest import run_backtest
from .config import load_config, resolve_path, validate_config
from .data import load_price_data, load_valuation_data, write_frame
from .report import build_report
from .signals import build_bucket_price_frame, compute_signals
from .universe import load_universe, select_representatives
from .weights import compute_baseline_weights, compute_strategy_weights


@dataclass
class PipelineResult:
    output_dir: Path
    selected_symbols: Dict[str, str]


def _export_table(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    write_frame(frame, output_dir / f"{stem}.parquet")
    write_frame(frame, output_dir / f"{stem}.csv")


def run_pipeline(
    config_path: Path,
    price_path: Path,
    output_dir: Path,
    valuation_path: Optional[Path] = None,
) -> PipelineResult:
    config = load_config(config_path)
    validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_path = resolve_path(config, config["paths"]["universe"])
    universe = load_universe(universe_path)
    prices = load_price_data(price_path)
    selection = select_representatives(
        universe=universe,
        buckets=config["bucket_order"],
        available_symbols=prices.columns,
    )

    bucket_prices = build_bucket_price_frame(prices, selection.bucket_to_symbol, config["bucket_order"])
    valuation_data = load_valuation_data(valuation_path)
    signals = compute_signals(bucket_prices=bucket_prices, valuation_data=valuation_data, config=config)
    strategy_weights, diagnostics = compute_strategy_weights(signals, config)
    baseline_weights = compute_baseline_weights(signals, config)

    strategy_backtest = run_backtest(bucket_prices, strategy_weights, config, label="strategy")
    baseline_backtest = run_backtest(bucket_prices, baseline_weights, config, label="baseline")

    index_levels = strategy_backtest.levels.join(baseline_backtest.levels, how="inner")
    holdings = strategy_backtest.holdings.add_prefix("strategy_").join(
        baseline_backtest.holdings.add_prefix("baseline_"), how="inner"
    )
    strategy_export = strategy_weights.join(diagnostics, how="left")

    selection.selected.to_csv(output_dir / "selected_universe.csv", index=False)
    _export_table(signals.reset_index(), output_dir, "signals")
    _export_table(strategy_export, output_dir, "strategy_weights")
    _export_table(baseline_weights, output_dir, "baseline_weights")
    _export_table(index_levels, output_dir, "index_levels")
    _export_table(holdings, output_dir, "holdings")

    report = build_report(
        levels=index_levels,
        selected_universe=selection.selected,
        strategy_weights=strategy_weights,
        diagnostics=diagnostics,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    return PipelineResult(output_dir=output_dir, selected_symbols=selection.bucket_to_symbol)

