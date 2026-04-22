#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402

from etf_fof_index.data import load_price_data  # noqa: E402


PROXY_SPECS: List[Dict[str, str]] = [
    {
        "bucket": "equity_core_csi300",
        "symbol": "510300.SH",
        "proxy_type": "index",
        "proxy_code": "000300.SH",
        "proxy_name": "沪深300指数",
    },
    {
        "bucket": "equity_core_csia500",
        "symbol": "563360.SH",
        "proxy_type": "index",
        "proxy_code": "000510.SH",
        "proxy_name": "中证A500指数",
    },
    {
        "bucket": "equity_defensive_lowvol",
        "symbol": "512890.SH",
        "proxy_type": "index",
        "proxy_code": "h30269.CSI",
        "proxy_name": "中证红利低波动指数",
    },
    {
        "bucket": "equity_defensive_dividend",
        "symbol": "510880.SH",
        "proxy_type": "index",
        "proxy_code": "000015.SH",
        "proxy_name": "上证红利指数",
    },
    {
        "bucket": "rate_bond_5y",
        "symbol": "511010.SH",
        "proxy_type": "index",
        "proxy_code": "000140.SH",
        "proxy_name": "上证5年期国债指数",
    },
    {
        "bucket": "rate_bond_10y",
        "symbol": "511260.SH",
        "proxy_type": "index",
        "proxy_code": "h01077.SH",
        "proxy_name": "上证10年期国债指数(净价)",
    },
    {
        "bucket": "gold",
        "symbol": "518880.SH",
        "proxy_type": "etf",
        "proxy_code": "518880.SH",
        "proxy_name": "黄金ETF华安",
    },
    {
        "bucket": "money_market",
        "symbol": "511880.SH",
        "proxy_type": "etf",
        "proxy_code": "511880.SH",
        "proxy_name": "银华日利ETF",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V2 full-sample price panel using index proxies where available.")
    parser.add_argument("--etf-prices", default=str(ROOT / "data" / "input" / "prices_v2.csv"))
    parser.add_argument("--index-prices", default=str(ROOT / "output" / "etf_index_download" / "index_daily_prices.csv.gz"))
    parser.add_argument("--output", default=str(ROOT / "data" / "input" / "prices_v2_index_proxy.csv"))
    parser.add_argument("--mapping-output", default=str(ROOT / "output" / "weight_grid_v2_index_proxy" / "proxy_mapping.csv"))
    return parser.parse_args()


def load_index_prices(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, compression="gzip", usecols=["index_windcode", "trade_dt", "close"])
    raw["trade_dt"] = pd.to_datetime(raw["trade_dt"], format="%Y%m%d")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    prices = raw.pivot_table(index="trade_dt", columns="index_windcode", values="close", aggfunc="last")
    prices.index.name = "date"
    return prices.sort_index()


def main() -> None:
    args = parse_args()
    etf_prices = load_price_data(Path(args.etf_prices))
    index_prices = load_index_prices(Path(args.index_prices))

    out = pd.DataFrame(index=etf_prices.index.union(index_prices.index).sort_values())
    mapping_rows = []

    for spec in PROXY_SPECS:
        symbol = spec["symbol"]
        proxy_type = spec["proxy_type"]
        proxy_code = spec["proxy_code"]

        if proxy_type == "index":
            if proxy_code not in index_prices.columns:
                raise ValueError(f"Missing index series for {proxy_code}.")
            series = index_prices[proxy_code]
        else:
            if symbol not in etf_prices.columns:
                raise ValueError(f"Missing ETF series for {symbol}.")
            series = etf_prices[symbol]

        out[symbol] = series.reindex(out.index)

        non_null = out[symbol].dropna()
        mapping_rows.append(
            {
                "bucket": spec["bucket"],
                "symbol": symbol,
                "proxy_type": proxy_type,
                "proxy_code": proxy_code,
                "proxy_name": spec["proxy_name"],
                "start_date": non_null.index.min().date().isoformat() if not non_null.empty else "",
                "end_date": non_null.index.max().date().isoformat() if not non_null.empty else "",
                "observations": int(non_null.shape[0]),
            }
        )

    out.index.name = "date"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(output_path, index=False)

    mapping_path = Path(args.mapping_output)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    print(f"output={output_path.resolve()}")
    print(f"mapping={mapping_path.resolve()}")
    print(f"start={out.dropna(how='all').index.min().date()}")
    print(f"end={out.dropna(how='all').index.max().date()}")


if __name__ == "__main__":
    main()
