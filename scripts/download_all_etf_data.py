#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量下载全市场 ETF 数据。

设计目标：
1. 参考资料中的 analyze_000493_share_scale_merge.py，沿用 Oracle + pandas 的工作流
2. 基于 ChinaETFInvestClass + ChinaMutualFundDescription 识别全市场上市 ETF
3. 导出后续评分所需的核心表：
   - ETF 基础信息
   - ETF 板块/分类映射
   - ETF 最新净值与规模
   - ETF 最新场内流通份额
   - ETF 最新跟踪表现
   - ETF 全历史日行情（复权收盘价、成交额、成交量）
   - ETF 汇总快照（master snapshot）

建议在 PyCharm 中直接运行：

python scripts/download_all_etf_data.py \
  --conn "chaxun/chaxun123@10.3.80.205/winddata" \
  --start-date 20100101 \
  --end-date 20260420 \
  --output-dir output/etf_bulk_download
"""

import argparse
import datetime as dt
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from _bootstrap import bootstrap

    bootstrap()
except Exception:
    pass

try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError("缺少 pandas，请先在运行环境中安装 pandas。") from exc

try:
    import oracledb as oracle_driver
except Exception:
    oracle_driver = None

if oracle_driver is None:
    try:
        import cx_Oracle as oracle_driver  # type: ignore
    except Exception:
        oracle_driver = None


os.environ.setdefault("NLS_LANG", "SIMPLIFIED CHINESE_CHINA.UTF8")

DEFAULT_CONN_STR = "chaxun/chaxun123@10.3.80.205/winddata"
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = dt.date.today().strftime("%Y%m%d")
DEFAULT_OUTPUT_DIR = "output/etf_bulk_download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载全市场 ETF 数据并汇总为可评分数据包。")
    parser.add_argument(
        "--conn",
        default=os.environ.get("WIND_ORACLE_CONN", DEFAULT_CONN_STR),
        help=(
            "Oracle 连接串。默认使用资料脚本中的连接串 "
            f"{DEFAULT_CONN_STR}；也可通过环境变量 WIND_ORACLE_CONN 覆盖。"
        ),
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="日行情起始日期，格式 YYYYMMDD。")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="日行情终止日期，格式 YYYYMMDD。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument(
        "--with-money-flow",
        action="store_true",
        help="是否额外下载 ETF 资金流向历史。默认不下载，以减少体量。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Oracle IN 子句批量大小，默认 500。",
    )
    return parser.parse_args()


def ensure_driver() -> None:
    if oracle_driver is None:
        raise RuntimeError("未找到 oracledb/cx_Oracle，请在 PyCharm 环境中安装其中一个。")


def connect_oracle(conn_str: str):
    ensure_driver()
    return oracle_driver.connect(conn_str)


def fetch_dataframe(cursor, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
    cursor.execute(sql, params or {})
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return pd.DataFrame(rows, columns=cols)


def series_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})


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


def normalize_date(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
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
    upper = str(text).upper()
    return "ETF" in upper


def is_linked_fund(full_name: Optional[str], short_name: Optional[str]) -> bool:
    fields = [full_name, short_name]
    return any(value is not None and "联接" in str(value) for value in fields)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def empty_numeric_series(index) -> pd.Series:
    return pd.Series(index=index, dtype="float64")


def aggregate_sector_map(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["windcode", "inner_code", "outer_code", "sector_names", "sector_codes", "sector_count"]
        )

    work = raw.copy()
    work["windcode"] = series_strip(work["windcode"])
    work["inner_code"] = series_strip(work["inner_code"])
    work["outer_code"] = series_strip(work["outer_code"])
    work["sector_code"] = series_strip(work["sector_code"])
    work["sector_name"] = series_strip(work["sector_name"])

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
        raise RuntimeError("未找到 ChinaMutualFundDescription。")

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
        {manager_col} AS manager_name,
        {custodian_col} AS custodian_name,
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

    desc["windcode"] = series_strip(desc["windcode"])
    desc["full_name"] = series_strip(desc["full_name"])
    desc["short_name"] = series_strip(desc["short_name"])
    desc["manager_name"] = series_strip(desc["manager_name"])
    desc["custodian_name"] = series_strip(desc["custodian_name"])
    desc["invest_type_name"] = series_strip(desc["invest_type_name"])
    desc["setup_date"] = desc["setup_date"].apply(normalize_date)
    desc["maturity_date"] = desc["maturity_date"].apply(normalize_date)
    desc["management_fee_pct"] = safe_numeric(desc["management_fee_pct"])
    desc["custodian_fee_pct"] = safe_numeric(desc["custodian_fee_pct"])
    desc["front_code"] = series_strip(desc["front_code"])
    desc["backend_code"] = series_strip(desc["backend_code"])
    desc["currency_code"] = series_strip(desc["currency_code"])

    universe = desc.merge(sectors_agg, on="windcode", how="outer")
    universe["source_desc"] = universe["short_name"].notna() | universe["full_name"].notna()
    universe["source_sector_map"] = universe["sector_names"].notna()
    universe["exchange_suffix"] = universe["windcode"].apply(code_suffix)
    universe["is_exchange_listed"] = universe["windcode"].apply(is_exchange_listed_etf_candidate)
    universe["is_etf_name"] = universe["short_name"].apply(contains_etf_text) | universe["full_name"].apply(contains_etf_text)
    universe["is_linked_fund"] = universe.apply(lambda row: is_linked_fund(row.get("full_name"), row.get("short_name")), axis=1)

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
    alias_cols = ",\n                ".join(available_cols)
    if not alias_cols:
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
                {alias_cols},
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
    if not out.empty and "RN" in out.columns:
        out = out.drop(columns=["RN"])
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
        codes = [code for code in universe[universe_col].dropna().astype(str).unique().tolist() if code]
        if not codes:
            continue
        for batch in chunks(codes, batch_size):
            in_clause, params = build_in_clause(batch, "c")
            params["asof_date"] = asof_date
            sql = f"""
            SELECT *
            FROM (
                SELECT
                    {track_col} AS matched_code,
                    {date_col} AS record_date,
                    {', '.join(select_cols)},
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
            mapper = universe[[universe_col, "windcode"]].dropna().drop_duplicates()
            mapper[universe_col] = mapper[universe_col].astype(str)
            tmp["MATCHED_CODE"] = tmp["MATCHED_CODE"].astype(str)
            tmp = tmp.merge(mapper, left_on="MATCHED_CODE", right_on=universe_col, how="left")
            frames.append(tmp.drop(columns=[universe_col, "RN"], errors="ignore"))

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
            {', '.join(select_cols)}
        FROM {table_name}
        WHERE {date_col} BETWEEN :start_date AND :end_date
          AND {windcode_col} IN ({in_clause})
        ORDER BY {windcode_col}, {date_col}
        """
        frames.append(fetch_dataframe(cursor, sql, params))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["windcode"] = series_strip(out["WINDCODE"])
    out["trade_dt"] = out["TRADE_DT"].apply(normalize_date)
    rename_map = {
        "S_DQ_CLOSE": "close",
        "S_DQ_ADJCLOSE": "adj_close",
        "S_DQ_VOLUME": "volume_hand",
        "S_DQ_AMOUNT": "amount_thousand_cny",
    }
    out = out.rename(columns=rename_map)
    keep_cols = ["windcode", "trade_dt"] + [col for col in rename_map.values() if col in out.columns]
    out = out[keep_cols].copy()
    out["close"] = safe_numeric(out["close"]) if "close" in out.columns else math.nan
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
            {', '.join(select_cols)}
        FROM {table_name}
        WHERE {date_col} BETWEEN :start_date AND :end_date
          AND {windcode_col} IN ({in_clause})
        ORDER BY {windcode_col}, {date_col}
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["windcode"] = series_strip(out["WINDCODE"])
    out["trade_dt"] = out["TRADE_DT"].apply(normalize_date)
    return out.rename(columns={col: col.lower() for col in out.columns}).sort_values(["windcode", "trade_dt"])


def build_master_snapshot(
    universe: pd.DataFrame,
    latest_nav: pd.DataFrame,
    latest_float_share: pd.DataFrame,
    latest_tracking: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    snapshot = universe.copy()

    if not latest_nav.empty:
        latest_nav = latest_nav.rename(columns={column.lower(): column.lower() for column in latest_nav.columns})
        latest_nav.columns = [str(column).lower() for column in latest_nav.columns]
        latest_nav = latest_nav.rename(columns={"record_date": "nav_date"})
        snapshot = snapshot.merge(latest_nav, on="windcode", how="left")

    if not latest_float_share.empty:
        float_df = latest_float_share.copy()
        float_df.columns = [str(column).lower() for column in float_df.columns]
        float_df = float_df.rename(columns={"record_date": "float_share_date", "f_unit_floatshare": "float_share"})
        snapshot = snapshot.merge(float_df[["windcode", "float_share_date", "float_share"]], on="windcode", how="left")

    if not latest_tracking.empty:
        tracking_df = latest_tracking.copy()
        tracking_df.columns = [str(column).lower() for column in tracking_df.columns]
        tracking_df = tracking_df.rename(columns={"record_date": "tracking_date", "s_info_indexwindcode": "tracking_index_windcode"})
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

        summaries = []
        for windcode, grp in work.groupby("windcode"):
            grp = grp.sort_values("trade_dt")
            tail20 = grp.tail(20)
            tail60 = grp.tail(60)
            summaries.append(
                {
                    "windcode": windcode,
                    "price_obs_count": int(len(grp)),
                    "avg_turnover_20d_cny": float(tail20["amount_cny"].mean()) if "amount_cny" in tail20 else math.nan,
                    "avg_turnover_60d_cny": float(tail60["amount_cny"].mean()) if "amount_cny" in tail60 else math.nan,
                    "avg_volume_20d_hand": float(tail20["volume_hand"].mean()) if "volume_hand" in tail20 else math.nan,
                }
            )
        summary_df = pd.DataFrame(summaries)
        snapshot = snapshot.merge(summary_df, on="windcode", how="left")

    nav_total = safe_numeric(snapshot["netasset_total"]) if "netasset_total" in snapshot.columns else empty_numeric_series(snapshot.index)
    nav_single = safe_numeric(snapshot["f_prt_netasset"]) if "f_prt_netasset" in snapshot.columns else empty_numeric_series(snapshot.index)
    latest_price = (
        safe_numeric(snapshot["latest_adj_close"])
        if "latest_adj_close" in snapshot.columns
        else empty_numeric_series(snapshot.index)
    )
    if "latest_close" in snapshot.columns:
        latest_price = latest_price.fillna(safe_numeric(snapshot["latest_close"]))

    float_share = safe_numeric(snapshot["float_share"]) if "float_share" in snapshot.columns else empty_numeric_series(snapshot.index)
    issue_total_unit = (
        safe_numeric(snapshot["issue_total_unit_100m"]) * 1e8
        if "issue_total_unit_100m" in snapshot.columns
        else empty_numeric_series(snapshot.index)
    )

    aum_from_float = float_share * latest_price
    aum_from_issue = issue_total_unit * latest_price

    snapshot["aum_nav_total_cny"] = nav_total
    snapshot["aum_nav_single_cny"] = nav_single
    snapshot["aum_float_price_est_cny"] = aum_from_float
    snapshot["aum_issue_price_est_cny"] = aum_from_issue

    snapshot["aum_cny"] = nav_total
    snapshot["aum_source"] = "nav_total"
    mask = snapshot["aum_cny"].isna() & nav_single.notna()
    snapshot.loc[mask, "aum_cny"] = nav_single[mask]
    snapshot.loc[mask, "aum_source"] = "nav_single"
    mask = snapshot["aum_cny"].isna() & aum_from_float.notna()
    snapshot.loc[mask, "aum_cny"] = aum_from_float[mask]
    snapshot.loc[mask, "aum_source"] = "float_share_x_price"
    mask = snapshot["aum_cny"].isna() & aum_from_issue.notna()
    snapshot.loc[mask, "aum_cny"] = aum_from_issue[mask]
    snapshot.loc[mask, "aum_source"] = "issue_share_x_price"
    snapshot.loc[snapshot["aum_cny"].isna(), "aum_source"] = "missing"

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
    args = parse_args()
    if not args.conn:
        raise RuntimeError("请通过 --conn 或环境变量 WIND_ORACLE_CONN 提供 Oracle 连接串。")

    print(f"使用 Oracle 连接串: {args.conn}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_oracle(args.conn)
    cursor = conn.cursor()
    try:
        print("1/7 识别全市场 ETF 宇宙...")
        universe, sector_membership = build_etf_universe(cursor, args.batch_size)
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
            batch_size=args.batch_size,
            asof_date=args.end_date,
        )

        print("3/7 下载 ETF 最新场内流通份额...")
        latest_float_share = fetch_latest_by_windcode(
            cursor=cursor,
            table_name="ChinaMutualFundFloatShare",
            windcodes=windcodes,
            date_col_candidates=["TRADE_DT", "END_DT"],
            select_cols=["F_UNIT_FLOATSHARE", "S_INFO_INNERCODE", "S_INFO_OUTERCODE"],
            batch_size=args.batch_size,
            asof_date=args.end_date,
        )

        print("4/7 下载 ETF 最新跟踪表现...")
        latest_tracking = fetch_latest_tracking(cursor, universe, args.batch_size, args.end_date)

        print("5/7 下载 ETF 全历史日行情...")
        daily_prices = fetch_daily_prices(cursor, windcodes, args.batch_size, args.start_date, args.end_date)

        daily_money_flow = pd.DataFrame()
        if args.with_money_flow:
            print("6/7 下载 ETF 资金流向历史...")
            daily_money_flow = fetch_daily_money_flow(cursor, windcodes, args.batch_size, args.start_date, args.end_date)
        else:
            print("6/7 跳过 ETF 资金流向历史（未开启 --with-money-flow）...")

        print("7/7 生成 ETF 汇总快照...")
        master_snapshot = build_master_snapshot(universe, latest_nav, latest_float_share, latest_tracking, daily_prices)
    finally:
        cursor.close()
        conn.close()

    write_dataframe(universe, output_dir / "etf_universe.csv")
    write_dataframe(sector_membership, output_dir / "etf_sector_membership.csv")
    write_dataframe(latest_nav, output_dir / "etf_latest_nav.csv")
    write_dataframe(latest_float_share, output_dir / "etf_latest_float_share.csv")
    write_dataframe(latest_tracking, output_dir / "etf_latest_tracking.csv")
    write_dataframe(master_snapshot, output_dir / "etf_master_snapshot.csv")
    write_dataframe(daily_prices, output_dir / "etf_daily_prices.csv.gz", compress=True)
    if args.with_money_flow and not daily_money_flow.empty:
        write_dataframe(daily_money_flow, output_dir / "etf_daily_money_flow.csv.gz", compress=True)

    print("\n下载完成。输出文件：")
    for file_path in sorted(output_dir.glob("*")):
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
