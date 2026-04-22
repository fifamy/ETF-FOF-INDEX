#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from run_quarterly_rolling_weight_strategy import run_study  # noqa: E402


def main() -> None:
    output_dir = ROOT / "output" / "rolling_quarterly_v2_official"
    result = run_study(
        config_path=ROOT / "config" / "index_v2.yaml",
        price_path=ROOT / "data" / "input" / "prices_v2_index_proxy.csv",
        output_dir=output_dir,
        valuation_path=None,
        lookback_months=12,
        selection_rule="min_drawdown",
        drawdown_band=0.02,
        write_outputs=True,
    )
    print(f"output_dir={Path(output_dir).resolve()}")
    print(f"sample_start={result['sample_start']}")
    print(f"sample_end={result['sample_end']}")
    print("lookback_months=12")
    print("selection_rule=min_drawdown")


if __name__ == "__main__":
    main()
