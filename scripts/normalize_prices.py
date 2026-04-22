#!/usr/bin/env python3
import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from etf_fof_index.config import load_config, resolve_path  # noqa: E402
from etf_fof_index.normalize import load_raw_csv, normalize_price_export  # noqa: E402
from etf_fof_index.universe import load_universe, select_representatives  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw vendor ETF price exports into project format.")
    parser.add_argument("--config", default=str(ROOT / "config" / "index_v1.yaml"), help="Path to YAML config.")
    parser.add_argument("--input", required=True, help="Raw input CSV exported from Wind/Choice or another vendor.")
    parser.add_argument("--output", required=True, help="Normalized wide CSV output path.")
    parser.add_argument("--date-column", help="Explicit date column name.")
    parser.add_argument("--symbol-column", help="Explicit symbol column name for long input.")
    parser.add_argument("--value-column", help="Explicit value column name for long input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    universe_path = resolve_path(config, config["paths"]["universe"])
    universe = load_universe(universe_path)
    selection = select_representatives(universe, config["bucket_order"])

    raw = load_raw_csv(Path(args.input))
    normalized = normalize_price_export(
        raw,
        selection.bucket_to_symbol.values(),
        date_column=args.date_column,
        symbol_column=args.symbol_column,
        value_column=args.value_column,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.reset_index().to_csv(output_path, index=False)

    print("Normalized symbols:")
    for bucket, symbol in selection.bucket_to_symbol.items():
        print(f"- {bucket}: {symbol}")
    print(f"Rows written: {len(normalized)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()

