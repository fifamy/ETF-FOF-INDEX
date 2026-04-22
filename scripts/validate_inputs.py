#!/usr/bin/env python3
import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from etf_fof_index.config import load_config, resolve_path  # noqa: E402
from etf_fof_index.data import load_price_data, load_valuation_data  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real research inputs for the ETF-FOF index.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v1.yaml"), help="Path to YAML config.")
    parser.add_argument("--prices", required=True, help="Path to price data CSV or directory.")
    parser.add_argument("--valuation", help="Optional valuation signal file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    universe_path = resolve_path(config, config["paths"]["universe"])
    universe = load_universe(universe_path)
    prices = load_price_data(Path(args.prices))
    selection = select_representatives(universe, config["bucket_order"], available_symbols=prices.columns)

    print("Selected symbols by bucket:")
    for bucket, symbol in selection.bucket_to_symbol.items():
        print(f"- {bucket}: {symbol}")

    print("\nPrice data summary:")
    print(f"- rows: {len(prices)}")
    print(f"- start: {prices.index.min().date()}")
    print(f"- end: {prices.index.max().date()}")
    print(f"- columns: {len(prices.columns)}")

    missing_symbols = [symbol for symbol in selection.bucket_to_symbol.values() if symbol not in prices.columns]
    if missing_symbols:
        raise SystemExit(f"Missing required symbols in price data: {missing_symbols}")

    for symbol in selection.bucket_to_symbol.values():
        series = prices[symbol]
        if series.isna().all():
            raise SystemExit(f"Price series is empty for {symbol}")
        if (series.dropna() <= 0).any():
            raise SystemExit(f"Non-positive price detected for {symbol}")
        print(
            f"- {symbol}: non-null={series.notna().sum()}, "
            f"missing={series.isna().sum()}, latest={series.dropna().iloc[-1]:.4f}"
        )

    if args.valuation:
        valuation = load_valuation_data(Path(args.valuation))
        print("\nValuation data summary:")
        print(f"- rows: {len(valuation)}")
        print(f"- start: {valuation.index.min().date()}")
        print(f"- end: {valuation.index.max().date()}")
        print(f"- columns: {', '.join(valuation.columns)}")
        invalid_columns = [bucket for bucket in valuation.columns if bucket not in config["bucket_order"]]
        if invalid_columns:
            raise SystemExit(f"Unknown valuation buckets: {invalid_columns}")
        print("- valuation range check: all values clipped to [-1, 1] by loader")

    print("\nValidation passed.")


if __name__ == "__main__":
    main()

