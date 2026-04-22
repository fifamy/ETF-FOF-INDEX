from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _load_price_directory(path: Path) -> pd.DataFrame:
    frames = []
    for file_path in sorted(path.glob("*.csv")):
        frame = pd.read_csv(file_path)
        if not {"date", "adj_close"}.issubset(frame.columns):
            raise ValueError(f"{file_path} must contain 'date' and 'adj_close'.")
        symbol = file_path.stem
        tmp = frame[["date", "adj_close"]].copy()
        tmp["symbol"] = symbol
        frames.append(tmp)
    if not frames:
        raise ValueError(f"No CSV files found under {path}.")
    long_df = pd.concat(frames, ignore_index=True)
    return _pivot_long_prices(long_df)


def _pivot_long_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "adj_close"}
    if not required.issubset(frame.columns):
        raise ValueError("Long price data must contain date,symbol,adj_close.")
    prices = frame.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    return _finalize_prices(prices)


def _finalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.astype(float)
    prices = prices.dropna(axis=1, how="all")
    return prices


def load_price_data(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return _load_price_directory(path)

    frame = pd.read_csv(path)
    if {"date", "symbol", "adj_close"}.issubset(frame.columns):
        return _pivot_long_prices(frame)

    if "date" not in frame.columns:
        raise ValueError("Wide price data must contain a 'date' column.")

    prices = frame.set_index("date")
    return _finalize_prices(prices)


def load_valuation_data(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if {"date", "bucket", "value"}.issubset(frame.columns):
        values = frame.pivot_table(index="date", columns="bucket", values="value", aggfunc="last")
    elif "date" in frame.columns:
        values = frame.set_index("date")
    else:
        raise ValueError("Valuation data must be wide with date or long with date,bucket,value.")

    values.index = pd.to_datetime(values.index)
    values = values.sort_index()
    values = values.astype(float)
    return values.clip(-1.0, 1.0)


def write_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path)
    elif output_path.suffix == ".csv":
        frame.to_csv(output_path, index=True)
    else:
        raise ValueError(f"Unsupported output suffix: {output_path.suffix}")

