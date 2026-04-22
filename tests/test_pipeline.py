from pathlib import Path

import pandas as pd

from etf_fof_index.config import load_config
from etf_fof_index.demo import generate_synthetic_inputs
from etf_fof_index.pipeline import run_pipeline


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config" / "index_v1.yaml"
    config = load_config(config_path)

    prices_path, valuation_path = generate_synthetic_inputs(
        config=config,
        output_dir=tmp_path / "demo_input",
        start="2021-06-21",
        end="2026-04-18",
        seed=42,
    )

    result = run_pipeline(
        config_path=config_path,
        price_path=prices_path,
        output_dir=tmp_path / "run",
        valuation_path=valuation_path,
    )

    assert result.output_dir.exists()
    assert set(result.selected_symbols) == {
        "equity_core",
        "equity_defensive",
        "rate_bond",
        "gold",
        "money_market",
    }

    strategy_weights = pd.read_csv(result.output_dir / "strategy_weights.csv", index_col=0, parse_dates=True)
    weight_columns = ["equity_core", "equity_defensive", "rate_bond", "gold", "money_market"]
    weight_sums = strategy_weights[weight_columns].sum(axis=1)
    assert ((weight_sums - 1.0).abs() < 1e-8).all()

    assert (strategy_weights["equity_core"] >= 0.10 - 1e-8).all()
    assert (strategy_weights["equity_core"] <= 0.35 + 1e-8).all()
    assert (strategy_weights["equity_defensive"] >= 0.05 - 1e-8).all()
    assert (strategy_weights["equity_defensive"] <= 0.25 + 1e-8).all()
    assert (strategy_weights["gold"] >= 0.05 - 1e-8).all()
    assert (strategy_weights["gold"] <= 0.20 + 1e-8).all()

    equity_total = strategy_weights["equity_core"] + strategy_weights["equity_defensive"]
    assert (equity_total >= 0.20 - 1e-8).all()
    assert (equity_total <= 0.50 + 1e-8).all()

    levels = pd.read_csv(result.output_dir / "index_levels.csv", index_col=0, parse_dates=True)
    assert {"strategy_index", "baseline_index", "strategy_return", "baseline_return"}.issubset(levels.columns)
    assert levels["strategy_index"].iloc[-1] > 0
    assert levels["baseline_index"].iloc[-1] > 0

    report = (result.output_dir / "report.md").read_text(encoding="utf-8")
    assert "Performance Summary" in report

