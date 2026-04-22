from __future__ import annotations

from typing import Dict, List

import pandas as pd


def month_end_trading_days(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    if index.empty:
        return []
    grouped = pd.Series(index=index, data=index).groupby(index.to_period("M"))
    return [group.iloc[-1] for _, group in grouped]


def build_bucket_price_frame(prices: pd.DataFrame, bucket_to_symbol: Dict[str, str], bucket_order: List[str]) -> pd.DataFrame:
    bucket_prices = pd.DataFrame(index=prices.index)
    for bucket in bucket_order:
        symbol = bucket_to_symbol[bucket]
        if symbol not in prices.columns:
            raise ValueError(f"Price data does not contain selected symbol {symbol}.")
        bucket_prices[bucket] = prices[symbol]
    return bucket_prices


def compute_signals(
    bucket_prices: pd.DataFrame,
    valuation_data: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    signal_cfg = config["signals"]
    short_window = int(signal_cfg["momentum_windows_days"]["short"])
    long_window = int(signal_cfg["momentum_windows_days"]["long"])
    short_weight = float(signal_cfg["momentum_weights"]["short"])
    long_weight = float(signal_cfg["momentum_weights"]["long"])
    vol_window = int(signal_cfg["vol_window_days"])
    drawdown_window = int(signal_cfg["drawdown_window_days"])

    returns = bucket_prices.pct_change(fill_method=None).fillna(0.0)
    momentum_short = bucket_prices / bucket_prices.shift(short_window) - 1.0
    momentum_long = bucket_prices / bucket_prices.shift(long_window) - 1.0
    composite_momentum = momentum_short * short_weight + momentum_long * long_weight
    vol_20d = returns.rolling(vol_window).std() * (252 ** 0.5)
    drawdown_60d = bucket_prices / bucket_prices.rolling(drawdown_window).max() - 1.0

    month_ends = month_end_trading_days(bucket_prices.index)
    signal_dates = [
        date
        for date in month_ends
        if date in composite_momentum.index and not composite_momentum.loc[date].isna().any()
    ]
    if not signal_dates:
        raise ValueError("No signal dates available. Extend the price history.")

    valuation_scores = pd.DataFrame(0.0, index=signal_dates, columns=bucket_prices.columns)
    if not valuation_data.empty:
        aligned = valuation_data.reindex(bucket_prices.index).ffill()
        if not aligned.empty:
            aligned = aligned.reindex(columns=bucket_prices.columns, fill_value=0.0)
            valuation_scores = aligned.loc[signal_dates].fillna(0.0).clip(-1.0, 1.0)

    records = []
    for date in signal_dates:
        snapshot = pd.DataFrame(
            {
                "price": bucket_prices.loc[date],
                "momentum_short": momentum_short.loc[date],
                "momentum_long": momentum_long.loc[date],
                "composite_momentum": composite_momentum.loc[date],
                "valuation_score": valuation_scores.loc[date],
                "vol_20d": vol_20d.loc[date],
                "drawdown_60d": drawdown_60d.loc[date],
            }
        )
        snapshot["date"] = date
        snapshot["bucket"] = snapshot.index
        records.append(snapshot.reset_index(drop=True))

    signals = pd.concat(records, ignore_index=True)
    signals = signals.set_index(["date", "bucket"]).sort_index()
    return signals
