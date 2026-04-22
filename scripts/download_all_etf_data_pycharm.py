#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全市场 ETF 数据批量下载程序

用法：
1. 在另一台能连 Wind Oracle 的电脑上，新建一个 Python 文件
2. 把本文件内容整体粘贴进去
3. 确认已安装 pandas + oracledb 或 cx_Oracle
4. 在 PyCharm 里直接运行

默认会导出：
- etf_universe.csv
- etf_sector_membership.csv
- etf_latest_nav.csv
- etf_latest_float_share.csv
- etf_latest_tracking.csv
- etf_master_snapshot.csv
- etf_daily_prices.csv.gz
- etf_daily_money_flow.csv.gz（可选）
"""

import math
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
OUTPUT_DIR = "./output/etf_bulk_download"
WITH_MONEY_FLOW = True
BATCH_SIZE = 500

os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"


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


def series_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def empty_numeric_series(index) -> pd.Series:
    return pd.Series(index=index, dtype="float64")


def normalize_date(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if len(text) >= 8 and text[:8].isdigit():
        return text[:8]
    return None


def code_suffix(windcode: str) -> Optional[str]:
    if pd.isna(windcode):
        return None
    text = str(windcode).upper().strip()
    if "." not in text:
        return None
    return text.split(".")[-1]


def is_exchange_listed_etf_candidate(windcode: str) -> bool:
    return code_suffix(windcode) in {"SH", "SZ"}


def contains_etf_text(text: Optional[str]) -> bool:
    if text is None or pd.isna(text):
        return False
    return "ETF" in str(text).upper()


def is_linked_fund(full_name: Optional[str], short_name: Optional[str]) -> bool:
    for value in [full_name, short_name]:
        if value is not None and pd.notna(value) and "联接" in str(value):
            return True
    return False


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


def sql_expr_or_null(column_name: Optional[str]) -> str:
    return column_name if column_name else "NULL"


def aggregate_sector_map(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["windcode", "inner_code", "outer_code", "sector_names", "sector_codes", "sector_count"]
        )

    work = raw.copy()
    for col in ["windcode", "inner_code", "outer_code", "sector_code", "sector_name"]:
        if col in work.columns:
            work[col] = series_strip(work[col])

    def join_unique(values: pd.Series) -> Optional[str]:
        cleaned = [str(v) for v in values.dropna().astype(str).tolist() if str(v).strip()]
        unique = sorted(set(cleaned))
        return "|".join(unique) if unique else None

    grouped = (
        work.groupby("windcode", dropna=False)
        .agg(
            inner_code=("inner_code", lambda s: next((x for x in s if pd.notna(x)), None)),
            outer_code=("outer_code", lambda s: next((x for x in s if pd.notna(x)), None)),
            sector_names=("sector_name", join_unique),
            sector_codes=("sector_code", join_unique),
            sector_count=("sector_code", lambda s: int(pd.Series(s).dropna().nunique())),
        )
        .reset_index()
    )
    return grouped


def load_etf_sector_membership(cursor) -> pd.DataFrame:
    table_name = "ChinaETFInvestClass"
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame(columns=["windcode", "sector_code", "sector_name", "inner_code", "outer_code"])

    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE"])
    sector_code_col = resolve_col(cols, ["S_INFO_SECTOR"])
    sector_name_col = resolve_col(cols, ["S_INFO_NAME"])
    inner_code_col = resolve_col(cols, ["S_INFO_INNERCODE"])
    outer_code_col = resolve_col(cols, ["S_INFO_OUTERCODE"])

    if windcode_col is None:
        return pd.DataFrame(columns=["windcode", "sector_code", "sector_name", "inner_code", "outer_code"])

    sql = f"""
    SELECT
        {sql_expr_or_null(windcode_col)} AS windcode,
        {sql_expr_or_null(sector_code_col)} AS sector_code,
        {sql_expr_or_null(sector_name_col)} AS sector_name,
        {sql_expr_or_null(inner_code_col)} AS inner_code,
        {sql_expr_or_null(outer_code_col)} AS outer_code
    FROM {table_name}
    """
    return fetch_dataframe(cursor, sql)


def load_etf_description(cursor) -> pd.DataFrame:
    table_name = "ChinaMutualFundDescription"
    cols = get_table_columns(cursor, table_name)
    if not cols:
        raise RuntimeError("未找到 ChinaMutualFundDescription 表。")

    windcode_col = resolve_col(cols, ["F_INFO_WINDCODE", "S_INFO_WINDCODE"])
    full_name_col = resolve_col(cols, ["F_INFO_FULLNAME", "F_INFO_CHINESENAME"])
    short_name_col = resolve_col(cols, ["F_INFO_NAME", "F_INFO_SHORTNAME"])
    manager_col = resolve_col(cols, ["F_INFO_CORP_FUNDMANAGEMENTCOMP", "F_INFO_MGRCOMP"])
    custodian_col = resolve_col(cols, ["F_INFO_CUSTODIANBANK"])
    invest_type_col = resolve_col(cols, ["F_INFO_FIRSTINVESTTYPE"])
    setup_date_col = resolve_col(cols, ["F_INFO_SETUPDATE"])
    maturity_col = resolve_col(cols, ["F_INFO_MATURITYDATE"])
    manage_fee_col = resolve_col(cols, ["F_INFO_MANAGEMENTFEERATIO"])
    cust_fee_col = resolve_col(cols, ["F_INFO_CUSTODIANFEERATIO"])
    front_code_col = resolve_col(cols, ["F_INFO_FRONT_CODE"])
    backend_code_col = resolve_col(cols, ["F_INFO_BACKEND_CODE"])
    currency_col = resolve_col(cols, ["CRNY_CODE", "CRNCY_CODE"])

    sql = f"""
    SELECT
        {windcode_col} AS windcode,
        {full_name_col} AS full_name,
        {short_name_col} AS short_name,
        {manager_col if manager_col else 'NULL'} AS manager_name,
        {custodian_col if custodian_col else 'NULL'} AS custodian_name,
        {invest_type_col if invest_type_col else 'NULL'} AS invest_type_name,
        {setup_date_col if setup_date_col else 'NULL'} AS setup_date,
        {maturity_col if maturity_col else 'NULL'} AS maturity_date,
        {manage_fee_col if manage_fee_col else 'NULL'} AS management_fee_pct,
        {cust_fee_col if cust_fee_col else 'NULL'} AS custodian_fee_pct,
        {front_code_col if front_code_col else 'NULL'} AS front_code,
        {backend_code_col if backend_code_col else 'NULL'} AS backend_code,
        {currency_col if currency_col else 'NULL'} AS currency_code
    FROM {table_name}
    WHERE UPPER({windcode_col}) LIKE '%.SH'
       OR UPPER({windcode_col}) LIKE '%.SZ'
       OR UPPER({short_name_col}) LIKE '%ETF%'
       OR UPPER({full_name_col}) LIKE '%ETF%'
    """
    return fetch_dataframe(cursor, sql)


def load_issue_info(cursor, windcodes: Sequence[str], batch_size: int) -> pd.DataFrame:
    table_name = "ChinaMutualFundIssue"
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame()

    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE", "F_INFO_WINDCODE"])
    issue_date_col = resolve_col(cols, ["F_ISSUE_DATE"])
    invest_type_code_col = resolve_col(cols, ["F_INFO_INVESTYPE"])
    fund_type_code_col = resolve_col(cols, ["F_INFO_TYPE"])
    manage_fee_col = resolve_col(cols, ["F_INFO_MANAFEERATIO"])
    cust_fee_col = resolve_col(cols, ["F_INFO_CUSTFEERATIO"])
    issue_unit_col = resolve_col(cols, ["F_ISSUE_TOTALUNIT"])

    fields = [
        f"{windcode_col} AS windcode",
        f"{issue_date_col} AS issue_date" if issue_date_col else "NULL AS issue_date",
        f"{invest_type_code_col} AS invest_type_code" if invest_type_code_col else "NULL AS invest_type_code",
        f"{fund_type_code_col} AS fund_type_code" if fund_type_code_col else "NULL AS fund_type_code",
        f"{manage_fee_col} AS issue_management_fee_pct" if manage_fee_col else "NULL AS issue_management_fee_pct",
        f"{cust_fee_col} AS issue_custodian_fee_pct" if cust_fee_col else "NULL AS issue_custodian_fee_pct",
        f"{issue_unit_col} AS issue_total_unit_100m" if issue_unit_col else "NULL AS issue_total_unit_100m",
    ]

    frames = []
    for batch in chunks(list(windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "w")
        sql = f"""
        SELECT {", ".join(fields)}
        FROM {table_name}
        WHERE {windcode_col} IN ({in_clause})
        """
        frames.append(fetch_dataframe(cursor, sql, params))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["windcode"])


def build_etf_universe(cursor, batch_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    desc = load_etf_description(cursor)
    sectors_raw = load_etf_sector_membership(cursor)
    sectors_agg = aggregate_sector_map(sectors_raw)

    for col in [
        "windcode",
        "full_name",
        "short_name",
        "manager_name",
        "custodian_name",
        "invest_type_name",
        "front_code",
        "backend_code",
        "currency_code",
    ]:
        if col in desc.columns:
            desc[col] = series_strip(desc[col])

    if "setup_date" in desc.columns:
        desc["setup_date"] = desc["setup_date"].apply(normalize_date)
    if "maturity_date" in desc.columns:
        desc["maturity_date"] = desc["maturity_date"].apply(normalize_date)
    if "management_fee_pct" in desc.columns:
        desc["management_fee_pct"] = safe_numeric(desc["management_fee_pct"])
    if "custodian_fee_pct" in desc.columns:
        desc["custodian_fee_pct"] = safe_numeric(desc["custodian_fee_pct"])

    universe = desc.merge(sectors_agg, on="windcode", how="outer")
    universe["source_desc"] = universe["short_name"].notna() | universe["full_name"].notna()
    universe["source_sector_map"] = universe["sector_names"].notna()
    universe["exchange_suffix"] = universe["windcode"].apply(code_suffix)
    universe["is_exchange_listed"] = universe["windcode"].apply(is_exchange_listed_etf_candidate)
    universe["is_etf_name"] = universe["short_name"].apply(contains_etf_text) | universe["full_name"].apply(contains_etf_text)
    universe["is_linked_fund"] = universe.apply(
        lambda row: is_linked_fund(row.get("full_name"), row.get("short_name")), axis=1
    )

    universe = universe[universe["windcode"].notna()].copy()
    universe = universe[universe["is_exchange_listed"]].copy()
    universe = universe[(universe["source_sector_map"]) | (universe["is_etf_name"])].copy()
    universe = universe[~universe["is_linked_fund"]].copy()
    universe = universe.drop_duplicates(subset=["windcode"], keep="first").reset_index(drop=True)

    issue = load_issue_info(cursor, universe["windcode"].tolist(), batch_size)
    if not issue.empty:
        issue["windcode"] = series_strip(issue["windcode"])
        issue = issue.drop_duplicates(subset=["windcode"], keep="first")
        universe = universe.merge(issue, on="windcode", how="left")

    issue_mgmt = (
        safe_numeric(universe["issue_management_fee_pct"])
        if "issue_management_fee_pct" in universe.columns
        else empty_numeric_series(universe.index)
    )
    issue_cust = (
        safe_numeric(universe["issue_custodian_fee_pct"])
        if "issue_custodian_fee_pct" in universe.columns
        else empty_numeric_series(universe.index)
    )
    management_fee = safe_numeric(universe["management_fee_pct"]).fillna(issue_mgmt)
    custodian_fee = safe_numeric(universe["custodian_fee_pct"]).fillna(issue_cust)
    universe["total_fee_pct"] = management_fee + custodian_fee

    return universe.sort_values("windcode").reset_index(drop=True), sectors_raw


def fetch_latest_by_windcode(
    cursor,
    table_name: str,
    windcodes: Sequence[str],
    date_col_candidates: Iterable[str],
    select_cols: Iterable[str],
    batch_size: int,
    asof_date: str,
) -> pd.DataFrame:
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame()

    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE", "F_INFO_WINDCODE"])
    date_col = resolve_col(cols, list(date_col_candidates))
    if windcode_col is None or date_col is None:
        return pd.DataFrame()

    available_cols = [col for col in select_cols if col in cols]
    if not available_cols:
        return pd.DataFrame()

    frames = []
    for batch in chunks(list(windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "w")
        params["asof_date"] = asof_date
        sql = f"""
        SELECT *
        FROM (
            SELECT
                {windcode_col} AS windcode,
                {date_col} AS record_date,
                {", ".join(available_cols)},
                ROW_NUMBER() OVER (
                    PARTITION BY {windcode_col}
                    ORDER BY {date_col} DESC
                ) AS rn
            FROM {table_name}
            WHERE {date_col} <= :asof_date
              AND {windcode_col} IN ({in_clause})
        )
        WHERE rn = 1
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "rn" in out.columns:
        out = out.drop(columns=["rn"])
    return out


def fetch_latest_tracking(cursor, universe: pd.DataFrame, batch_size: int, asof_date: str) -> pd.DataFrame:
    table_name = "ChinaETFTrackPerformance"
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame()

    date_col = resolve_col(cols, ["TRADE_DT", "ANN_DT", "PRICE_DATE"])
    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE", "F_INFO_WINDCODE"])
    select_cols = [
        col
        for col in [
            "S_INFO_INDEXWINDCODE",
            "TRACKERROR_1M",
            "TRACKERROR_3M",
            "TRACKERROR_6M",
            "TRACKERROR_1Y",
            "TRACKERROR_3Y",
            "TRACKERROR_5Y",
            "INFORRATIO_1M",
            "INFORRATIO_3M",
            "INFORRATIO_6M",
            "INFORRATIO_1Y",
            "INFORRATIO_3Y",
            "INFORRATIO_6Y",
        ]
        if col in cols
    ]
    if date_col is None or not select_cols:
        return pd.DataFrame()

    if windcode_col is not None:
        return fetch_latest_by_windcode(
            cursor=cursor,
            table_name=table_name,
            windcodes=universe["windcode"].tolist(),
            date_col_candidates=[date_col],
            select_cols=select_cols,
            batch_size=batch_size,
            asof_date=asof_date,
        )

    frames = []
    code_maps = [
        ("inner_code", resolve_col(cols, ["S_INFO_INNERCODE"])),
        ("outer_code", resolve_col(cols, ["S_INFO_OUTERCODE"])),
    ]
    for universe_col, track_col in code_maps:
        if track_col is None or universe_col not in universe.columns:
            continue
        code_values = [code for code in universe[universe_col].dropna().astype(str).unique().tolist() if code]
        if not code_values:
            continue

        for batch in chunks(code_values, batch_size):
            in_clause, params = build_in_clause(batch, "c")
            params["asof_date"] = asof_date
            sql = f"""
            SELECT *
            FROM (
                SELECT
                    {track_col} AS matched_code,
                    {date_col} AS record_date,
                    {", ".join(select_cols)},
                    ROW_NUMBER() OVER (
                        PARTITION BY {track_col}
                        ORDER BY {date_col} DESC
                    ) AS rn
                FROM {table_name}
                WHERE {date_col} <= :asof_date
                  AND {track_col} IN ({in_clause})
            )
            WHERE rn = 1
            """
            tmp = fetch_dataframe(cursor, sql, params)
            if tmp.empty:
                continue
            mapper = universe[[universe_col, "windcode"]].dropna().drop_duplicates().copy()
            mapper[universe_col] = mapper[universe_col].astype(str)
            tmp["matched_code"] = tmp["matched_code"].astype(str)
            tmp = tmp.merge(mapper, left_on="matched_code", right_on=universe_col, how="left")
            tmp = tmp.drop(columns=[universe_col, "rn"], errors="ignore")
            frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["windcode", "record_date"], ascending=[True, False]).drop_duplicates("windcode", keep="first")
    return out


def fetch_daily_prices(cursor, windcodes: Sequence[str], batch_size: int, start_date: str, end_date: str) -> pd.DataFrame:
    table_name = "ChinaClosedFundEODPrice"
    cols = get_table_columns(cursor, table_name)
    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE", "F_INFO_WINDCODE"])
    date_col = resolve_col(cols, ["TRADE_DT"])
    if windcode_col is None or date_col is None:
        raise RuntimeError("未找到 ChinaClosedFundEODPrice 的 Wind代码/交易日期字段。")

    select_cols = [col for col in ["S_DQ_CLOSE", "S_DQ_ADJCLOSE", "S_DQ_VOLUME", "S_DQ_AMOUNT"] if col in cols]
    if not select_cols:
        raise RuntimeError("ChinaClosedFundEODPrice 缺少核心行情字段。")

    frames = []
    for batch in chunks(list(windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "w")
        params["start_date"] = start_date
        params["end_date"] = end_date
        sql = f"""
        SELECT
            {windcode_col} AS windcode,
            {date_col} AS trade_dt,
            {", ".join(select_cols)}
        FROM {table_name}
        WHERE {date_col} BETWEEN :start_date AND :end_date
          AND {windcode_col} IN ({in_clause})
        ORDER BY {windcode_col}, {date_col}
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["windcode"] = series_strip(out["windcode"])
    out["trade_dt"] = out["trade_dt"].apply(normalize_date)
    out = out.rename(
        columns={
            "s_dq_close": "close",
            "s_dq_adjclose": "adj_close",
            "s_dq_volume": "volume_hand",
            "s_dq_amount": "amount_thousand_cny",
        }
    )
    keep_cols = ["windcode", "trade_dt", "close", "adj_close", "volume_hand", "amount_thousand_cny"]
    out = out[[col for col in keep_cols if col in out.columns]].copy()
    if "close" in out.columns:
        out["close"] = safe_numeric(out["close"])
    if "adj_close" in out.columns:
        out["adj_close"] = safe_numeric(out["adj_close"])
    if "volume_hand" in out.columns:
        out["volume_hand"] = safe_numeric(out["volume_hand"])
    if "amount_thousand_cny" in out.columns:
        out["amount_thousand_cny"] = safe_numeric(out["amount_thousand_cny"])
        out["amount_cny"] = out["amount_thousand_cny"] * 1000.0
    return out.sort_values(["windcode", "trade_dt"]).reset_index(drop=True)


def fetch_daily_money_flow(cursor, windcodes: Sequence[str], batch_size: int, start_date: str, end_date: str) -> pd.DataFrame:
    table_name = "ChinaETFMoneyFlow"
    cols = get_table_columns(cursor, table_name)
    windcode_col = resolve_col(cols, ["S_INFO_WINDCODE", "F_INFO_WINDCODE"])
    date_col = resolve_col(cols, ["TRADE_DT"])
    if windcode_col is None or date_col is None:
        return pd.DataFrame()

    select_cols = [
        col
        for col in [
            "BUY_VALUE_EXLARGE_ORDER",
            "SELL_VALUE_EXLARGE_ORDER",
            "BUY_VALUE_LARGE_ORDER",
            "SELL_VALUE_LARGE_ORDER",
            "BUY_VALUE_MED_ORDER",
            "SELL_VALUE_MED_ORDER",
            "BUY_VALUE_SMALL_ORDER",
            "SELL_VALUE_SMALL_ORDER",
        ]
        if col in cols
    ]
    if not select_cols:
        return pd.DataFrame()

    frames = []
    for batch in chunks(list(windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "w")
        params["start_date"] = start_date
        params["end_date"] = end_date
        sql = f"""
        SELECT
            {windcode_col} AS windcode,
            {date_col} AS trade_dt,
            {", ".join(select_cols)}
        FROM {table_name}
        WHERE {date_col} BETWEEN :start_date AND :end_date
          AND {windcode_col} IN ({in_clause})
        ORDER BY {windcode_col}, {date_col}
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["windcode"] = series_strip(out["windcode"])
    out["trade_dt"] = out["trade_dt"].apply(normalize_date)
    return out.sort_values(["windcode", "trade_dt"]).reset_index(drop=True)


def build_master_snapshot(
    universe: pd.DataFrame,
    latest_nav: pd.DataFrame,
    latest_float_share: pd.DataFrame,
    latest_tracking: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    snapshot = universe.copy()

    if not latest_nav.empty:
        latest_nav = latest_nav.rename(columns={"record_date": "nav_date"})
        snapshot = snapshot.merge(latest_nav, on="windcode", how="left")

    if not latest_float_share.empty:
        float_df = latest_float_share.rename(
            columns={"record_date": "float_share_date", "f_unit_floatshare": "float_share"}
        )
        keep_cols = [col for col in ["windcode", "float_share_date", "float_share"] if col in float_df.columns]
        snapshot = snapshot.merge(float_df[keep_cols], on="windcode", how="left")

    if not latest_tracking.empty:
        tracking_df = latest_tracking.rename(
            columns={"record_date": "tracking_date", "s_info_indexwindcode": "tracking_index_windcode"}
        )
        snapshot = snapshot.merge(tracking_df, on="windcode", how="left")

    if not daily_prices.empty:
        work = daily_prices.copy()
        work["trade_dt"] = pd.to_datetime(work["trade_dt"])
        work = work.sort_values(["windcode", "trade_dt"])

        latest_idx = work.groupby("windcode")["trade_dt"].idxmax()
        latest = work.loc[latest_idx, ["windcode", "trade_dt", "close", "adj_close", "volume_hand", "amount_cny"]].copy()
        latest = latest.rename(
            columns={
                "trade_dt": "latest_trade_dt",
                "close": "latest_close",
                "adj_close": "latest_adj_close",
                "volume_hand": "latest_volume_hand",
                "amount_cny": "latest_amount_cny",
            }
        )
        snapshot = snapshot.merge(latest, on="windcode", how="left")

        rows = []
        for windcode, grp in work.groupby("windcode"):
            grp = grp.sort_values("trade_dt")
            tail20 = grp.tail(20)
            tail60 = grp.tail(60)
            rows.append(
                {
                    "windcode": windcode,
                    "price_obs_count": int(len(grp)),
                    "avg_turnover_20d_cny": float(tail20["amount_cny"].mean()) if "amount_cny" in tail20.columns else math.nan,
                    "avg_turnover_60d_cny": float(tail60["amount_cny"].mean()) if "amount_cny" in tail60.columns else math.nan,
                    "avg_volume_20d_hand": float(tail20["volume_hand"].mean()) if "volume_hand" in tail20.columns else math.nan,
                }
            )
        summary_df = pd.DataFrame(rows)
        snapshot = snapshot.merge(summary_df, on="windcode", how="left")

    if "netasset_total" in snapshot.columns:
        snapshot["aum_cny"] = safe_numeric(snapshot["netasset_total"])
    elif "f_prt_netasset" in snapshot.columns:
        snapshot["aum_cny"] = safe_numeric(snapshot["f_prt_netasset"])
    else:
        snapshot["aum_cny"] = math.nan

    if "total_fee_pct" in snapshot.columns:
        snapshot["total_fee_bp"] = safe_numeric(snapshot["total_fee_pct"]) * 100.0

    if "trackerror_1y" in snapshot.columns:
        snapshot["tracking_error_1y_pct"] = safe_numeric(snapshot["trackerror_1y"])

    return snapshot


def write_dataframe(df: pd.DataFrame, path: Path, compress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def main() -> None:
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("开始下载全市场 ETF 数据")
    print(f"连接串: {DB_CONN_STR}")
    print(f"起始日期: {START_DATE}")
    print(f"结束日期: {END_DATE}")
    print(f"输出目录: {output_dir}")
    print(f"下载资金流向: {WITH_MONEY_FLOW}")
    print("=" * 80)

    conn = connect_oracle(DB_CONN_STR)
    cursor = conn.cursor()

    try:
        print("1/7 识别全市场 ETF 宇宙...")
        universe, sector_membership = build_etf_universe(cursor, BATCH_SIZE)
        windcodes = universe["windcode"].dropna().astype(str).tolist()
        print(f"识别到 ETF 数量: {len(windcodes)}")

        print("2/7 下载 ETF 最新净值/规模...")
        latest_nav = fetch_latest_by_windcode(
            cursor=cursor,
            table_name="ChinaMutualFundNAV",
            windcodes=windcodes,
            date_col_candidates=["PRICE_DATE", "ANN_DATE", "NAV_DATE"],
            select_cols=[
                "F_NAV_UNIT",
                "F_NAV_ACCUMULATED",
                "F_NAV_ADJUSTED",
                "F_PRT_NETASSET",
                "NETASSET_TOTAL",
                "F_ASSET_MERGEDSHARESORNOT",
            ],
            batch_size=BATCH_SIZE,
            asof_date=END_DATE,
        )

        print("3/7 下载 ETF 最新场内流通份额...")
        latest_float_share = fetch_latest_by_windcode(
            cursor=cursor,
            table_name="ChinaMutualFundFloatShare",
            windcodes=windcodes,
            date_col_candidates=["TRADE_DT", "END_DT"],
            select_cols=["F_UNIT_FLOATSHARE", "S_INFO_INNERCODE", "S_INFO_OUTERCODE"],
            batch_size=BATCH_SIZE,
            asof_date=END_DATE,
        )

        print("4/7 下载 ETF 最新跟踪表现...")
        latest_tracking = fetch_latest_tracking(cursor, universe, BATCH_SIZE, END_DATE)

        print("5/7 下载 ETF 全历史日行情...")
        daily_prices = fetch_daily_prices(cursor, windcodes, BATCH_SIZE, START_DATE, END_DATE)

        if WITH_MONEY_FLOW:
            print("6/7 下载 ETF 资金流向历史...")
            daily_money_flow = fetch_daily_money_flow(cursor, windcodes, BATCH_SIZE, START_DATE, END_DATE)
        else:
            print("6/7 跳过 ETF 资金流向历史...")
            daily_money_flow = pd.DataFrame()

        print("7/7 生成 ETF 汇总快照...")
        master_snapshot = build_master_snapshot(universe, latest_nav, latest_float_share, latest_tracking, daily_prices)

    finally:
        cursor.close()
        conn.close()

    print("写出文件...")
    write_dataframe(universe, output_dir / "etf_universe.csv")
    write_dataframe(sector_membership, output_dir / "etf_sector_membership.csv")
    write_dataframe(latest_nav, output_dir / "etf_latest_nav.csv")
    write_dataframe(latest_float_share, output_dir / "etf_latest_float_share.csv")
    write_dataframe(latest_tracking, output_dir / "etf_latest_tracking.csv")
    write_dataframe(master_snapshot, output_dir / "etf_master_snapshot.csv")
    write_dataframe(daily_prices, output_dir / "etf_daily_prices.csv.gz", compress=True)
    if WITH_MONEY_FLOW and not daily_money_flow.empty:
        write_dataframe(daily_money_flow, output_dir / "etf_daily_money_flow.csv.gz", compress=True)

    print("\n下载完成，输出文件如下：")
    for file_path in sorted(output_dir.glob("*")):
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
