#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETF 对应指数及指数行情批量下载程序

用法：
1. 在另一台能连 Wind Oracle 的电脑上，新建一个 Python 文件
2. 把本文件内容整体粘贴进去，或直接复制本仓库中的这个文件
3. 确认已安装 pandas + oracledb 或 cx_Oracle
4. 在 PyCharm 里直接运行

默认会导出：
- etf_tracking_index_history.csv
- etf_tracking_index_asof.csv
- index_description.csv
- index_daily_prices.csv.gz
- index_latest_snapshot.csv
- etf_index_bridge_snapshot.csv
"""

import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import oracledb as oracle_driver
except Exception:
    oracle_driver = None

if oracle_driver is None:
    try:
        import cx_Oracle as oracle_driver  # type: ignore
    except Exception:
        oracle_driver = None


# =========================
# 运行参数：直接改这里
# =========================
DB_CONN_STR = "chaxun/chaxun123@10.3.80.205/winddata"
START_DATE = "20100101"
END_DATE = "20260420"
INPUT_ETF_FILE = "./output/etf_bulk_download_research/etf_master_snapshot.csv"
OUTPUT_DIR = "./output/etf_index_download"
BATCH_SIZE = 500

os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"


DESCRIPTION_TABLE_SPECS = [
    {
        "table_name": "AIndexDescription",
        "source_table": "AIndexDescription",
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "index_code": ["S_INFO_CODE"],
            "index_short_name": ["S_INFO_NAME"],
            "index_name": ["S_INFO_COMPNAME"],
            "index_exchange": ["S_INFO_EXCHMARKET"],
            "index_base_period": ["S_INFO_INDEX_BASEPER"],
            "index_base_point": ["S_INFO_INDEX_BASEPT"],
            "index_list_date": ["S_INFO_LISTDATE"],
            "index_weights_rule": ["S_INFO_INDEX_WEIGHTSRULE"],
            "index_publisher": ["S_INFO_PUBLISHER"],
            "index_type_code": ["S_INFO_INDEXCODE"],
        },
    },
    {
        "table_name": "CBIndexDescription",
        "source_table": "CBIndexDescription",
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "index_code": ["S_INFO_CODE"],
            "index_short_name": ["S_INFO_NAME"],
            "index_name": ["S_INFO_COMPNAME"],
            "index_exchange": ["S_INFO_EXCHMARKET"],
            "index_base_period": ["S_INFO_INDEX_BASEPER"],
            "index_base_point": ["S_INFO_INDEX_BASEPT"],
            "index_list_date": ["S_INFO_LISTDATE"],
            "index_weights_rule": ["S_INFO_INDEX_WEIGHTSRULE"],
            "index_publisher": ["S_INFO_PUBLISHER"],
            "index_type_code": ["INCOME_PROCESSING_METHOD"],
        },
    },
    {
        "table_name": "CMFIndexDescription",
        "source_table": "CMFIndexDescription",
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "index_code": ["S_INFO_CODE"],
            "index_short_name": ["S_INFO_NAME"],
            "index_name": ["S_INFO_COMPNAME"],
            "index_exchange": ["S_INFO_EXCHMARKET"],
            "index_base_period": ["S_INFO_INDEX_BASEPER"],
            "index_base_point": ["S_INFO_INDEX_BASEPT"],
            "index_list_date": ["S_INFO_LISTDATE"],
            "index_weights_rule": ["S_INFO_INDEX_WEIGHTSRULE"],
            "index_publisher": ["S_INFO_PUBLISHER"],
            "index_type_code": ["INDEX_INTRO"],
        },
    },
    {
        "table_name": "CMFMGIndexDescription",
        "source_table": "CMFMGIndexDescription",
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "index_code": ["S_INFO_CODE"],
            "index_short_name": ["S_INFO_COMPNAME"],
            "index_name": ["S_INFO_COMPNAME"],
            "index_exchange": [],
            "index_base_period": ["S_INFO_INDEX_BASEPER"],
            "index_base_point": ["S_INFO_INDEX_BASEPT"],
            "index_list_date": ["S_INFO_LISTDATE"],
            "index_weights_rule": [],
            "index_publisher": ["S_INFO_PUBLISHER"],
            "index_type_code": ["S_INFO_INDEXTYPE"],
        },
    },
    {
        "table_name": "NEEQIndexDescription",
        "source_table": "NEEQIndexDescription",
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "index_code": ["S_INFO_CODE"],
            "index_short_name": ["S_INFO_NAME"],
            "index_name": ["S_INFO_COMPNAME"],
            "index_exchange": ["S_INFO_EXCHMARKET"],
            "index_base_period": ["S_INFO_INDEX_BASEPER"],
            "index_base_point": ["S_INFO_INDEX_BASEPT"],
            "index_list_date": ["S_INFO_LISTDATE"],
            "index_weights_rule": [],
            "index_publisher": ["S_INFO_PUBLISHER"],
            "index_type_code": [],
        },
    },
    {
        "table_name": "CFutureIndexDescription",
        "source_table": "CFutureIndexDescription",
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "index_code": ["S_INFO_CODE"],
            "index_short_name": ["S_INFO_NAME"],
            "index_name": ["S_INFO_COMPNAME"],
            "index_exchange": ["S_INFO_EXCHMARKET"],
            "index_base_period": ["S_INFO_INDEX_BASEPER"],
            "index_base_point": ["S_INFO_INDEX_BASEPT"],
            "index_list_date": ["S_INFO_LISTDATE"],
            "index_weights_rule": ["S_INFO_INDEX_WEIGHTSRULE"],
            "index_publisher": ["S_INFO_PUBLISHER"],
            "index_type_code": ["S_INFO_INDEXCODE"],
        },
    },
]


PRICE_TABLE_SPECS = [
    {
        "table_name": "AIndexEODPrices",
        "source_table": "AIndexEODPrices",
        "priority": 1,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": ["CRNCY_CODE", "CRNY_CODE"],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": ["S_DQ_CHANGE"],
            "pct_change": ["S_DQ_PCTCHANGE"],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "CSIAIndexEODPrices",
        "source_table": "CSIAIndexEODPrices",
        "priority": 2,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": ["CRNCY_CODE", "CRNY_CODE"],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": ["S_DQ_CHANGE"],
            "pct_change": ["S_DQ_PCTCHANGE"],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "AIndexIndustriesEODCITICS",
        "source_table": "AIndexIndustriesEODCITICS",
        "priority": 3,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": ["CRNCY_CODE"],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": ["S_DQ_CHANGE"],
            "pct_change": ["S_DQ_PCTCHANGE"],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "ASWSIndexEOD",
        "source_table": "ASWSIndexEOD",
        "priority": 4,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": [],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": [],
            "pct_change": [],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "CBIndexEODPrices",
        "source_table": "CBIndexEODPrices",
        "priority": 5,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": ["CRNCY_CODE"],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": ["S_DQ_CHANGE"],
            "pct_change": ["S_DQ_PCTCHANGE"],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "CMFIndexEOD",
        "source_table": "CMFIndexEOD",
        "priority": 6,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": ["CRNCY_CODE"],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": [],
            "pct_change": [],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "CSITotalBondIndeEODPrice",
        "source_table": "CSITotalBondIndeEODPrice",
        "priority": 7,
        "amount_multiplier_to_cny": 10000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": [],
            "preclose": [],
            "open": [],
            "high": [],
            "low": [],
            "close": ["S_DQ_INDEXVALUE"],
            "change": [],
            "pct_change": [],
            "volume_hand": [],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "ThirdPartyIndexEOD",
        "source_table": "ThirdPartyIndexEOD",
        "priority": 8,
        "amount_multiplier_to_cny": 1.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": [],
            "preclose": [],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": [],
            "pct_change": [],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
    {
        "table_name": "NEEQIndexEODPrices",
        "source_table": "NEEQIndexEODPrices",
        "priority": 9,
        "amount_multiplier_to_cny": 1000.0,
        "field_map": {
            "index_windcode": ["S_INFO_WINDCODE"],
            "trade_dt": ["TRADE_DT"],
            "crncy_code": ["CRNCY_CODE"],
            "preclose": ["S_DQ_PRECLOSE"],
            "open": ["S_DQ_OPEN"],
            "high": ["S_DQ_HIGH"],
            "low": ["S_DQ_LOW"],
            "close": ["S_DQ_CLOSE"],
            "change": ["S_DQ_CHANGE"],
            "pct_change": ["S_DQ_PCTCHANGE"],
            "volume_hand": ["S_DQ_VOLUME"],
            "amount": ["S_DQ_AMOUNT"],
        },
    },
]


def ensure_driver() -> None:
    if oracle_driver is None:
        raise RuntimeError("未找到 oracledb / cx_Oracle，请先在当前 Python 环境中安装其中一个。")


def connect_oracle(conn_str: str):
    ensure_driver()
    return oracle_driver.connect(conn_str)


def fetch_dataframe(cursor, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
    cursor.execute(sql, params or {})
    rows = cursor.fetchall()
    cols = [str(d[0]).lower() for d in cursor.description]
    return pd.DataFrame(rows, columns=cols)


def write_dataframe(df: pd.DataFrame, path: Path, compress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def series_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})


def normalize_date(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if len(text) >= 8 and text[:8].isdigit():
        return text[:8]
    return None


def extract_code6(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if len(text) >= 6 and text[:6].isdigit():
        return text[:6]
    return None


def chunks(values: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


def build_in_clause(values: Sequence[str], prefix: str) -> Tuple[str, Dict[str, str]]:
    binds = []
    params = {}
    for i, value in enumerate(values):
        key = f"{prefix}{i}"
        binds.append(f":{key}")
        params[key] = value
    return ", ".join(binds), params


def get_table_columns(cursor, table_name: str) -> set:
    sql_candidates = [
        "SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE UPPER(TABLE_NAME)=UPPER(:t)",
        "SELECT COLUMN_NAME FROM USER_TAB_COLUMNS WHERE UPPER(TABLE_NAME)=UPPER(:t)",
    ]
    for sql in sql_candidates:
        try:
            df = fetch_dataframe(cursor, sql, {"t": table_name})
            if not df.empty:
                return set(df.iloc[:, 0].astype(str).str.upper())
        except Exception:
            continue
    try:
        df = fetch_dataframe(cursor, f"SELECT * FROM {table_name} WHERE 1=0")
        return {str(column).upper() for column in df.columns}
    except Exception:
        return set()


def resolve_col(cols: set, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def resolve_cols(cols: set, candidates: Iterable[str]) -> Optional[str]:
    return resolve_col(cols, candidates)


def load_input_etfs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"输入 ETF 文件不存在: {path}")

    df = pd.read_csv(path)
    if "windcode" not in df.columns:
        raise RuntimeError("输入 ETF 文件缺少 windcode 列。")

    for col in ["windcode", "code6", "short_name", "full_name", "tracking_index_windcode"]:
        if col in df.columns:
            df[col] = series_strip(df[col])

    if "code6" not in df.columns:
        df["code6"] = df["windcode"].apply(extract_code6)

    keep_cols = [col for col in ["windcode", "code6", "short_name", "full_name", "tracking_index_windcode"] if col in df.columns]
    df = df[keep_cols].copy()
    df = df[df["windcode"].notna()].drop_duplicates(subset=["windcode"], keep="first").reset_index(drop=True)
    return df


def fetch_tracking_index_history(cursor, etf_windcodes: Sequence[str], batch_size: int) -> pd.DataFrame:
    table_name = "ChinaMutualFundTrackingIndex"
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame()

    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE", "F_INFO_WINDCODE"])
    index_col = resolve_col(cols, ["S_INFO_INDEXWINDCODE"])
    entry_col = resolve_col(cols, ["ENTRY_DT"])
    remove_col = resolve_col(cols, ["REMOVE_DT"])
    if windcode_col is None or index_col is None:
        raise RuntimeError("ChinaMutualFundTrackingIndex 缺少 ETF/指数映射字段。")

    fields = [
        f"{windcode_col} AS windcode",
        f"{index_col} AS tracking_index_windcode",
        f"{entry_col} AS entry_dt" if entry_col else "NULL AS entry_dt",
        f"{remove_col} AS remove_dt" if remove_col else "NULL AS remove_dt",
    ]

    frames = []
    for batch in chunks(list(etf_windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "w")
        sql = f"""
        SELECT {", ".join(fields)}
        FROM {table_name}
        WHERE {windcode_col} IN ({in_clause})
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    out["windcode"] = series_strip(out["windcode"])
    out["tracking_index_windcode"] = series_strip(out["tracking_index_windcode"])
    out["entry_dt"] = out["entry_dt"].apply(normalize_date)
    out["remove_dt"] = out["remove_dt"].apply(normalize_date)
    out["code6"] = out["windcode"].apply(extract_code6)
    out["index_code6"] = out["tracking_index_windcode"].apply(extract_code6)
    out = out[out["windcode"].notna() & out["tracking_index_windcode"].notna()].copy()
    out = out.sort_values(["windcode", "entry_dt", "remove_dt", "tracking_index_windcode"]).drop_duplicates().reset_index(drop=True)
    return out


def build_tracking_index_asof(history: pd.DataFrame, etfs: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    if history.empty:
        out = etfs.copy()
        out["tracking_index_windcode"] = None
        out["entry_dt"] = None
        out["remove_dt"] = None
        out["mapping_status"] = "missing"
        return out

    work = history.copy()
    work["entry_dt"] = work["entry_dt"].apply(normalize_date)
    work["remove_dt"] = work["remove_dt"].apply(normalize_date)

    has_started = work["entry_dt"].isna() | (work["entry_dt"] <= asof_date)
    not_removed = work["remove_dt"].isna() | (work["remove_dt"] == "") | (work["remove_dt"] > asof_date)
    active = work[has_started & not_removed].copy()
    active = active.sort_values(["windcode", "entry_dt", "tracking_index_windcode"]).drop_duplicates("windcode", keep="last")
    active["mapping_status"] = "active"

    fallback = work[has_started].copy()
    fallback = fallback.sort_values(["windcode", "entry_dt", "tracking_index_windcode"]).drop_duplicates("windcode", keep="last")
    fallback["mapping_status"] = "fallback_latest_before_asof"

    raw_fallback = None
    if "tracking_index_windcode" in etfs.columns:
        raw_fallback = etfs[["windcode", "tracking_index_windcode"]].rename(
            columns={"tracking_index_windcode": "tracking_index_windcode_raw"}
        )

    out = etfs.drop(columns=["tracking_index_windcode"], errors="ignore").copy()
    out = out.merge(active[["windcode", "tracking_index_windcode", "entry_dt", "remove_dt", "mapping_status"]], on="windcode", how="left")

    missing_mask = out["tracking_index_windcode"].isna()
    if missing_mask.any():
        tmp = out.loc[missing_mask, ["windcode"]].merge(
            fallback[["windcode", "tracking_index_windcode", "entry_dt", "remove_dt", "mapping_status"]],
            on="windcode",
            how="left",
        )
        for col in ["tracking_index_windcode", "entry_dt", "remove_dt", "mapping_status"]:
            out.loc[missing_mask, col] = tmp[col].values

    if raw_fallback is not None:
        out = out.merge(raw_fallback, on="windcode", how="left")
        mask = out["tracking_index_windcode"].isna() & out["tracking_index_windcode_raw"].notna()
        out.loc[mask, "tracking_index_windcode"] = out.loc[mask, "tracking_index_windcode_raw"]
        out.loc[mask, "mapping_status"] = "fallback_from_etf_snapshot"
        out = out.drop(columns=["tracking_index_windcode_raw"])

    out["index_code6"] = out["tracking_index_windcode"].apply(extract_code6)
    out.loc[out["tracking_index_windcode"].isna(), "mapping_status"] = "missing"
    return out.sort_values(["mapping_status", "windcode"]).reset_index(drop=True)


def fetch_index_description_from_table(cursor, table_spec: Dict[str, object], index_windcodes: Sequence[str], batch_size: int) -> pd.DataFrame:
    table_name = str(table_spec["table_name"])
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame()

    field_map = {
        target: resolve_cols(cols, candidates)
        for target, candidates in dict(table_spec["field_map"]).items()
    }
    windcode_col = field_map.get("index_windcode")
    if windcode_col is None:
        return pd.DataFrame()

    fields = [
        f"{source} AS {target}" if source else f"NULL AS {target}"
        for target, source in field_map.items()
    ]
    fields.append(f"'{table_spec['source_table']}' AS source_table")

    frames = []
    for batch in chunks(list(index_windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "i")
        sql = f"""
        SELECT {", ".join(fields)}
        FROM {table_name}
        WHERE {windcode_col} IN ({in_clause})
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    for col in ["index_windcode", "index_code", "index_short_name", "index_name", "index_exchange", "index_weights_rule", "index_publisher", "index_type_code", "source_table"]:
        if col in out.columns:
            out[col] = series_strip(out[col])
    for col in ["index_base_period", "index_list_date"]:
        if col in out.columns:
            out[col] = out[col].apply(normalize_date)
    out["index_code6"] = out["index_windcode"].apply(extract_code6)
    out = out[out["index_windcode"].notna()].drop_duplicates(subset=["index_windcode"], keep="first").reset_index(drop=True)
    return out


def fetch_index_description(cursor, index_windcodes: Sequence[str], batch_size: int) -> pd.DataFrame:
    frames = []
    for spec in DESCRIPTION_TABLE_SPECS:
        tmp = fetch_index_description_from_table(cursor, spec, index_windcodes, batch_size)
        if not tmp.empty:
            frames.append(tmp)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    out["source_priority"] = out["source_table"].map(
        {str(spec["source_table"]): i for i, spec in enumerate(DESCRIPTION_TABLE_SPECS, start=1)}
    ).fillna(999)
    out = out.sort_values(["source_priority", "index_windcode"]).drop_duplicates(subset=["index_windcode"], keep="first")
    return out.drop(columns=["source_priority"], errors="ignore").reset_index(drop=True)


def fetch_index_daily_prices_from_table(
    cursor,
    table_spec: Dict[str, object],
    index_windcodes: Sequence[str],
    batch_size: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    table_name = str(table_spec["table_name"])
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame()

    field_map = {
        target: resolve_cols(cols, candidates)
        for target, candidates in dict(table_spec["field_map"]).items()
    }
    windcode_col = field_map.get("index_windcode")
    date_col = field_map.get("trade_dt")
    if windcode_col is None or date_col is None:
        return pd.DataFrame()

    select_fields = [
        f"{source} AS {target}" if source else f"NULL AS {target}"
        for target, source in field_map.items()
    ]
    select_fields.append(f"'{table_spec['source_table']}' AS source_table")

    frames = []
    for batch in chunks(list(index_windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "i")
        params["start_date"] = start_date
        params["end_date"] = end_date
        sql = f"""
        SELECT {", ".join(select_fields)}
        FROM {table_name}
        WHERE {date_col} BETWEEN :start_date AND :end_date
          AND {windcode_col} IN ({in_clause})
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    for col in ["index_windcode", "crncy_code", "source_table"]:
        if col in out.columns:
            out[col] = series_strip(out[col])
    out["trade_dt"] = out["trade_dt"].apply(normalize_date)
    for col in ["preclose", "open", "high", "low", "close", "change", "pct_change", "volume_hand", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    multiplier = float(table_spec["amount_multiplier_to_cny"])
    out["amount_cny"] = out["amount"] * multiplier if "amount" in out.columns else pd.NA
    out["index_code6"] = out["index_windcode"].apply(extract_code6)
    out["source_priority"] = int(table_spec["priority"])
    return out


def fetch_index_daily_prices(
    cursor,
    index_windcodes: Sequence[str],
    batch_size: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = []
    for spec in PRICE_TABLE_SPECS:
        tmp = fetch_index_daily_prices_from_table(cursor, spec, index_windcodes, batch_size, start_date, end_date)
        if not tmp.empty:
            frames.append(tmp)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    out = (
        out.sort_values(["source_priority", "index_windcode", "trade_dt"])
        .drop_duplicates(subset=["index_windcode", "trade_dt"], keep="first")
        .reset_index(drop=True)
    )
    return out


def build_index_latest_snapshot(index_daily_prices: pd.DataFrame) -> pd.DataFrame:
    if index_daily_prices.empty:
        return pd.DataFrame()

    work = index_daily_prices.sort_values(["index_windcode", "trade_dt"], na_position="last")
    latest = work.drop_duplicates(subset=["index_windcode"], keep="last").copy()
    latest = latest.rename(
        columns={
                "trade_dt": "latest_trade_dt",
                "close": "latest_close",
                "pct_change": "latest_pct_change",
                "volume_hand": "latest_volume_hand",
                "amount_cny": "latest_amount_cny",
                "source_table": "latest_price_source_table",
            }
        )
    summary = work.groupby("index_windcode", sort=False).size().rename("price_obs_count").reset_index()
    out = latest.merge(summary, on="index_windcode", how="left")
    return out


def build_etf_index_bridge_snapshot(
    etfs: pd.DataFrame,
    tracking_asof: pd.DataFrame,
    index_description: pd.DataFrame,
    index_latest_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    left = etfs.drop(columns=["tracking_index_windcode"], errors="ignore").copy()
    out = left.merge(
        tracking_asof[
            [
                "windcode",
                "tracking_index_windcode",
                "entry_dt",
                "remove_dt",
                "mapping_status",
                "index_code6",
            ]
        ],
        on="windcode",
        how="left",
    )

    if not index_description.empty:
        out = out.merge(index_description, left_on="tracking_index_windcode", right_on="index_windcode", how="left")
    if not index_latest_snapshot.empty:
        keep_cols = [
            col
            for col in [
                "index_windcode",
                "latest_trade_dt",
                "latest_close",
                "latest_pct_change",
                "latest_volume_hand",
                "latest_amount_cny",
                "latest_price_source_table",
                "price_obs_count",
            ]
            if col in index_latest_snapshot.columns
        ]
        out = out.merge(index_latest_snapshot[keep_cols], left_on="tracking_index_windcode", right_on="index_windcode", how="left")
    for col in ["index_windcode_x", "index_windcode_y", "index_code6_x", "index_code6_y"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out.sort_values(["mapping_status", "windcode"]).reset_index(drop=True)
    return out


def build_index_coverage_snapshot(
    tracking_asof: pd.DataFrame,
    index_description: pd.DataFrame,
    index_latest_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    mapped = tracking_asof[tracking_asof["tracking_index_windcode"].notna()].copy()
    if mapped.empty:
        return pd.DataFrame()

    coverage = (
        mapped[["tracking_index_windcode"]]
        .drop_duplicates()
        .rename(columns={"tracking_index_windcode": "index_windcode"})
        .reset_index(drop=True)
    )
    if not index_description.empty:
        coverage = coverage.merge(
            index_description[["index_windcode", "index_name", "index_short_name", "source_table"]],
            on="index_windcode",
            how="left",
        ).rename(columns={"source_table": "description_source_table"})
    if not index_latest_snapshot.empty:
        keep_cols = [col for col in ["index_windcode", "latest_trade_dt", "latest_close", "latest_price_source_table"] if col in index_latest_snapshot.columns]
        coverage = coverage.merge(index_latest_snapshot[keep_cols], on="index_windcode", how="left")

    coverage["has_description"] = coverage["index_name"].notna() | coverage["index_short_name"].notna()
    coverage["has_latest_price"] = coverage["latest_trade_dt"].notna()
    coverage["coverage_status"] = "complete"
    coverage.loc[coverage["has_description"] & ~coverage["has_latest_price"], "coverage_status"] = "missing_latest_price"
    coverage.loc[~coverage["has_description"] & coverage["has_latest_price"], "coverage_status"] = "missing_description"
    coverage.loc[~coverage["has_description"] & ~coverage["has_latest_price"], "coverage_status"] = "missing_both"
    return coverage.sort_values(["coverage_status", "index_windcode"]).reset_index(drop=True)


def main() -> None:
    input_file = Path(INPUT_ETF_FILE).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("开始下载 ETF 对应指数及相关数据")
    print(f"连接串: {DB_CONN_STR}")
    print(f"起始日期: {START_DATE}")
    print(f"结束日期: {END_DATE}")
    print(f"输入 ETF 文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    etfs = load_input_etfs(input_file)
    etf_windcodes = etfs["windcode"].dropna().astype(str).unique().tolist()
    print(f"读取 ETF 数量: {len(etf_windcodes)}")

    conn = connect_oracle(DB_CONN_STR)
    cursor = conn.cursor()

    try:
        print("1/5 下载 ETF-指数映射历史...")
        tracking_history = fetch_tracking_index_history(cursor, etf_windcodes, BATCH_SIZE)
        print(f"映射历史记录数: {len(tracking_history)}")

        print("2/5 生成截至结束日的 ETF-指数映射快照...")
        tracking_asof = build_tracking_index_asof(tracking_history, etfs, END_DATE)
        mapped_index_windcodes = tracking_asof["tracking_index_windcode"].dropna().astype(str).unique().tolist()
        print(f"截至 {END_DATE} 已映射 ETF 数量: {int(tracking_asof['tracking_index_windcode'].notna().sum())}")
        print(f"对应指数数量: {len(mapped_index_windcodes)}")

        print("3/5 下载指数基础资料...")
        index_description = fetch_index_description(cursor, mapped_index_windcodes, BATCH_SIZE)
        print(f"指数基础资料条数: {len(index_description)}")

        print("4/5 下载指数日行情...")
        index_daily_prices = fetch_index_daily_prices(
            cursor,
            mapped_index_windcodes,
            BATCH_SIZE,
            START_DATE,
            END_DATE,
        )
        print(f"指数日行情记录数: {len(index_daily_prices)}")

    finally:
        cursor.close()
        conn.close()

    print("5/5 生成桥接快照...")
    index_latest_snapshot = build_index_latest_snapshot(index_daily_prices)
    bridge_snapshot = build_etf_index_bridge_snapshot(etfs, tracking_asof, index_description, index_latest_snapshot)
    index_coverage_snapshot = build_index_coverage_snapshot(tracking_asof, index_description, index_latest_snapshot)

    print("写出文件...")
    write_dataframe(tracking_history, output_dir / "etf_tracking_index_history.csv")
    write_dataframe(tracking_asof, output_dir / "etf_tracking_index_asof.csv")
    write_dataframe(index_description, output_dir / "index_description.csv")
    write_dataframe(index_daily_prices, output_dir / "index_daily_prices.csv.gz", compress=True)
    write_dataframe(index_latest_snapshot, output_dir / "index_latest_snapshot.csv")
    write_dataframe(bridge_snapshot, output_dir / "etf_index_bridge_snapshot.csv")
    write_dataframe(index_coverage_snapshot, output_dir / "index_coverage_snapshot.csv")

    print("\n下载完成，输出文件如下：")
    for file_path in sorted(output_dir.glob("*")):
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
