#!/usr/bin/env python3
import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from etf_fof_index.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ETF-FOF index research pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--prices", required=True, help="Path to price data CSV or directory.")
    parser.add_argument("--valuation", help="Optional valuation signal file.")
    parser.add_argument("--output", default=str(ROOT / "output" / "research_run"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        config_path=Path(args.config),
        price_path=Path(args.prices),
        output_dir=Path(args.output),
        valuation_path=Path(args.valuation) if args.valuation else None,
    )
    print(f"Selected symbols: {result.selected_symbols}")
    print(f"Output written to: {result.output_dir}")


if __name__ == "__main__":
    main()

