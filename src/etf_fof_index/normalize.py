from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DATE_ALIASES = {
    "date",
    "日期",
    "交易日期",
    "trade_date",
    "tradedate",
    "trade_dt",
    "TRADE_DT",
}

SYMBOL_ALIASES = {
    "symbol",
    "代码",
    "证券代码",
    "基金代码",
    "wind_code",
    "s_info_windcode",
    "ticker",
    "sec_code",
}

VALUE_ALIASES = {
    "adj_close",
    "复权收盘价",
    "后复权收盘价",
    "前复权收盘价",
    "收盘价(后复权)",
    "收盘价（后复权）",
    "s_dq_adjclose",
    "S_DQ_ADJCLOSE",
    "单位净值",
    "复权单位净值",
}


def _normalize_name(name: object) -> str:
    return str(name).strip()


def _parse_vendor_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.isin(["", "nan", "None", "<NA>"]))
    yyyymmdd = text.str.extract(r"(\d{8})", expand=False)
    parsed = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(text.loc[fallback_mask], errors="coerce")
    return parsed


def _locate_column(columns: Iterable[object], aliases: Iterable[str], explicit_name: Optional[str] = None) -> Optional[str]:
    normalized = {_normalize_name(column): column for column in columns}
    if explicit_name is not None:
        if explicit_name not in normalized:
            raise ValueError(f"Column '{explicit_name}' not found in input data.")
        return explicit_name

    alias_set = {alias.lower() for alias in aliases}
    for column in columns:
        clean = _normalize_name(column)
        if clean.lower() in alias_set:
            return clean
    return None


def canonicalize_symbol(label: object, allowed_symbols: Iterable[str]) -> Optional[str]:
    raw = _normalize_name(label)
    if not raw:
        return None

    allowed = list(allowed_symbols)
    allowed_set = set(allowed)
    upper = raw.upper().replace(" ", "")

    if upper in allowed_set:
        return upper

    match = re.search(r"(\d{6})\.(SH|SZ)", upper)
    if match:
        candidate = f"{match.group(1)}.{match.group(2)}"
        if candidate in allowed_set:
            return candidate

    match = re.search(r"(SH|SZ)(\d{6})", upper)
    if match:
        candidate = f"{match.group(2)}.{match.group(1)}"
        if candidate in allowed_set:
            return candidate

    digits = re.findall(r"\d{6}", upper)
    if digits:
        digit_to_matches = {}
        for symbol in allowed:
            digit_to_matches.setdefault(symbol[:6], []).append(symbol)
        for digit in digits:
            matches = digit_to_matches.get(digit, [])
            if len(matches) == 1:
                return matches[0]

    for symbol in allowed:
        if symbol in upper:
            return symbol
    return None


def normalize_price_export(
    raw: pd.DataFrame,
    selected_symbols: Iterable[str],
    *,
    date_column: Optional[str] = None,
    symbol_column: Optional[str] = None,
    value_column: Optional[str] = None,
) -> pd.DataFrame:
    selected = list(selected_symbols)
    date_col = _locate_column(raw.columns, DATE_ALIASES, explicit_name=date_column)
    if date_col is None:
        raise ValueError("Could not identify the date column.")

    symbol_col = _locate_column(raw.columns, SYMBOL_ALIASES, explicit_name=symbol_column)
    value_col = _locate_column(raw.columns, VALUE_ALIASES, explicit_name=value_column)

    if symbol_col and value_col:
        return _normalize_long_prices(raw, selected, date_col, symbol_col, value_col)

    return _normalize_wide_prices(raw, selected, date_col)


def _normalize_long_prices(
    raw: pd.DataFrame,
    selected_symbols: Iterable[str],
    date_col: str,
    symbol_col: str,
    value_col: str,
) -> pd.DataFrame:
    frame = raw[[date_col, symbol_col, value_col]].copy()
    frame.columns = ["date", "symbol", "adj_close"]
    frame["symbol"] = frame["symbol"].map(lambda value: canonicalize_symbol(value, selected_symbols))
    frame = frame[frame["symbol"].notna()].copy()
    if frame.empty:
        raise ValueError("No selected symbols were matched in the long-format input.")

    frame["date"] = _parse_vendor_dates(frame["date"])
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["date", "adj_close"])
    frame = frame.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    return _finalize_normalized_prices(frame, selected_symbols)


def _normalize_wide_prices(raw: pd.DataFrame, selected_symbols: Iterable[str], date_col: str) -> pd.DataFrame:
    frame = raw.copy()
    date_values = _parse_vendor_dates(frame[date_col])
    frame = frame.drop(columns=[date_col])
    frame.index = date_values

    mapped_columns = {}
    for column in frame.columns:
        mapped = canonicalize_symbol(column, selected_symbols)
        if mapped is None:
            continue
        mapped_columns.setdefault(mapped, []).append(column)

    if not mapped_columns:
        raise ValueError("Could not match any wide-format columns to the selected symbols.")

    normalized = pd.DataFrame(index=date_values)
    for symbol, columns in mapped_columns.items():
        if len(columns) == 1:
            series = frame[columns[0]]
        else:
            series = frame[columns].bfill(axis=1).iloc[:, 0]
        normalized[symbol] = pd.to_numeric(series, errors="coerce")

    return _finalize_normalized_prices(normalized, selected_symbols)


def _finalize_normalized_prices(prices: pd.DataFrame, selected_symbols: Iterable[str]) -> pd.DataFrame:
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]

    selected = list(selected_symbols)
    for symbol in selected:
        if symbol not in frame.columns:
            frame[symbol] = pd.NA

    frame = frame[selected]
    frame.index.name = "date"
    return frame


def load_raw_csv(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)
