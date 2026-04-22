#!/usr/bin/env python3
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from etf_fof_index.config import load_config  # noqa: E402
from etf_fof_index.demo import generate_synthetic_inputs  # noqa: E402
from etf_fof_index.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    config_path = ROOT / "config" / "index_v1.yaml"
    config = load_config(config_path)
    demo_dir = ROOT / "output" / "demo_input"
    output_dir = ROOT / "output" / "demo_run"

    prices_path, valuation_path = generate_synthetic_inputs(
        config=config,
        output_dir=demo_dir,
        start="2021-06-21",
        end="2026-04-18",
        seed=7,
    )

    result = run_pipeline(
        config_path=config_path,
        price_path=prices_path,
        output_dir=output_dir,
        valuation_path=valuation_path,
    )
    print(f"Demo selected symbols: {result.selected_symbols}")
    print(f"Demo output written to: {result.output_dir}")


if __name__ == "__main__":
    main()

