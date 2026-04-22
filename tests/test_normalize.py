from pathlib import Path

import pandas as pd

from etf_fof_index.normalize import normalize_price_export


SELECTED = ["510300.SH", "512890.SH", "511010.SH", "518880.SH", "511990.SH"]


def test_normalize_long_vendor_export() -> None:
    raw = pd.DataFrame(
        {
            "TRADE_DT": ["2021-06-21", "2021-06-21", "2021-06-22", "2021-06-22"],
            "S_INFO_WINDCODE": ["510300.SH", "512890.SH", "510300.SH", "512890.SH"],
            "S_DQ_ADJCLOSE": [5.238, 1.024, 5.251, 1.028],
        }
    )

    normalized = normalize_price_export(raw, SELECTED)
    assert list(normalized.columns) == SELECTED
    assert float(normalized.loc[pd.Timestamp("2021-06-21"), "510300.SH"]) == 5.238
    assert pd.isna(normalized.loc[pd.Timestamp("2021-06-21"), "511010.SH"])


def test_normalize_wide_vendor_export() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2021-06-21", "2021-06-22"],
            "华泰柏瑞沪深300ETF(510300.SH)": [5.238, 5.251],
            "512890": [1.024, 1.028],
            "SH511010": [139.82, 139.77],
            "518880": [3.846, 3.859],
            "511990": [100.014, 100.018],
        }
    )

    normalized = normalize_price_export(raw, SELECTED)
    assert list(normalized.columns) == SELECTED
    assert float(normalized.loc[pd.Timestamp("2021-06-22"), "511010.SH"]) == 139.77
    assert float(normalized.loc[pd.Timestamp("2021-06-21"), "511990.SH"]) == 100.014

