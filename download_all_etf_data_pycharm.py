#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全市场 ETF 数据批量下载程序

用法：
1. 在另一台能连 Wind Oracle 的电脑上，新建一个 Python 文件
2. 把本文件内容整体粘贴进去，或直接复制本仓库中的这个文件
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
import re
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


def extract_code6(value) -> Optional[str]:
    if pd.isna(value):
        return None
    match = re.search(r"(\d{6})", str(value))
    return match.group(1) if match else None


def code_suffix(windcode: str) -> Optional[str]:
    if pd.isna(windcode):
        return None
    text = str(windcode).upper().strip()
    if "." not in text:
        return None
    return text.split(".")[-1]


def windcode_priority(windcode: str) -> int:
    suffix = code_suffix(windcode)
    if suffix == "OF":
        return 1
    if suffix in {"SH", "SZ"}:
        return 2
    return 9


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


def explode_pipe_separated_values(series: pd.Series) -> List[str]:
    if series.empty:
        return []
    values = (
        series.dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values.ne("")]
    return values.drop_duplicates().tolist()


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


def regex_code6_sql(column_name: str) -> str:
    return f"REGEXP_SUBSTR(TO_CHAR({column_name}), '[0-9]{{6}}')"


def aggregate_sector_map(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["code6", "sector_names", "sector_codes", "sector_count"]
        )

    work = raw.copy()
    for col in ["windcode", "inner_code", "outer_code", "sector_code", "sector_name"]:
        if col in work.columns:
            work[col] = series_strip(work[col])
    work["code6"] = work["windcode"].apply(extract_code6)
    work = work[work["code6"].notna()].copy()

    def join_unique(values: pd.Series) -> Optional[str]:
        cleaned = [str(v) for v in values.dropna().astype(str).tolist() if str(v).strip()]
        unique = sorted(set(cleaned))
        return "|".join(unique) if unique else None

    grouped = (
        work.groupby("code6", dropna=False)
        .agg(
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


def build_code_mapping(desc: pd.DataFrame) -> pd.DataFrame:
    if desc.empty:
        return pd.DataFrame(
            columns=[
                "code6",
                "primary_windcode",
                "listed_windcodes",
                "of_windcodes",
                "all_windcodes",
                "front_codes",
                "backend_codes",
                "full_names",
                "short_names",
                "manager_names",
                "listed_code_count",
                "all_code_count",
            ]
        )

    work = desc.copy()
    work["code6"] = work["windcode"].apply(extract_code6)
    work = work[work["code6"].notna()].copy()
    work["windcode_suffix"] = work["windcode"].apply(code_suffix)
    work["windcode_priority"] = work["windcode"].apply(windcode_priority)

    def join_unique(values: pd.Series) -> Optional[str]:
        cleaned = [str(v) for v in values.dropna().astype(str).tolist() if str(v).strip()]
        unique = sorted(set(cleaned))
        return "|".join(unique) if unique else None

    rows = []
    for code6, grp in work.groupby("code6", dropna=False):
        grp = grp.sort_values(["windcode_priority", "windcode"])
        listed = grp[grp["windcode_suffix"].isin(["SH", "SZ"])]
        of_codes = grp[grp["windcode_suffix"].eq("OF")]
        primary = listed.iloc[0]["windcode"] if not listed.empty else grp.iloc[0]["windcode"]
        rows.append(
            {
                "code6": code6,
                "primary_windcode": primary,
                "listed_windcodes": join_unique(listed["windcode"]),
                "of_windcodes": join_unique(of_codes["windcode"]),
                "all_windcodes": join_unique(grp["windcode"]),
                "front_codes": join_unique(grp["front_code"]) if "front_code" in grp.columns else None,
                "backend_codes": join_unique(grp["backend_code"]) if "backend_code" in grp.columns else None,
                "full_names": join_unique(grp["full_name"]) if "full_name" in grp.columns else None,
                "short_names": join_unique(grp["short_name"]) if "short_name" in grp.columns else None,
                "manager_names": join_unique(grp["manager_name"]) if "manager_name" in grp.columns else None,
                "listed_code_count": int(listed["windcode"].nunique()),
                "all_code_count": int(grp["windcode"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values("code6").reset_index(drop=True)


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


def build_etf_universe(cursor, batch_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    desc["code6"] = desc["windcode"].apply(extract_code6)
    desc["windcode_suffix"] = desc["windcode"].apply(code_suffix)
    desc["windcode_priority"] = desc["windcode"].apply(windcode_priority)

    code_mapping = build_code_mapping(desc)

    universe = desc.merge(sectors_agg, on="code6", how="left")
    universe["source_desc"] = universe["short_name"].notna() | universe["full_name"].notna()
    universe["source_sector_map"] = universe["sector_names"].notna()
    universe["exchange_suffix"] = universe["windcode"].apply(code_suffix)
    universe["is_exchange_listed"] = universe["windcode"].apply(is_exchange_listed_etf_candidate)
    universe["is_etf_name"] = universe["short_name"].apply(contains_etf_text) | universe["full_name"].apply(contains_etf_text)
    universe["is_linked_fund"] = universe.apply(
        lambda row: is_linked_fund(row.get("full_name"), row.get("short_name")), axis=1
    )

    universe = universe[universe["windcode"].notna()].copy()
    universe = universe[universe["code6"].notna()].copy()
    universe = universe[universe["is_exchange_listed"]].copy()
    universe = universe[(universe["source_sector_map"]) | (universe["is_etf_name"])].copy()
    universe = universe[~universe["is_linked_fund"]].copy()
    universe = universe.sort_values(["code6", "windcode_priority", "windcode"]).drop_duplicates(
        subset=["code6"], keep="first"
    ).reset_index(drop=True)

    etf_code6_values = universe["code6"].dropna().astype(str).unique().tolist()
    code_mapping = code_mapping[code_mapping["code6"].isin(etf_code6_values)].copy()
    code_mapping = code_mapping.sort_values("code6").reset_index(drop=True)

    all_windcodes = explode_pipe_separated_values(code_mapping["all_windcodes"])
    issue = load_issue_info(cursor, all_windcodes, batch_size)
    if not issue.empty:
        issue["windcode"] = series_strip(issue["windcode"])
        issue["code6"] = issue["windcode"].apply(extract_code6)
        issue = issue.sort_values(["code6", "windcode"]).drop_duplicates(subset=["code6"], keep="first")
        universe = universe.merge(issue.drop(columns=["windcode"]), on="code6", how="left")

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

    universe = universe.merge(code_mapping, on="code6", how="left")

    return universe.sort_values("windcode").reset_index(drop=True), sectors_raw, code_mapping


def fetch_latest_by_code6(
    cursor,
    table_name: str,
    code6_values: Sequence[str],
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
    for batch in chunks(list(code6_values), batch_size):
        in_clause, params = build_in_clause(batch, "c")
        params["asof_date"] = asof_date
        code6_expr = regex_code6_sql(windcode_col)
        sql = f"""
        SELECT *
        FROM (
            SELECT
                {code6_expr} AS code6,
                {windcode_col} AS source_windcode,
                {date_col} AS record_date,
                {", ".join(available_cols)},
                ROW_NUMBER() OVER (
                    PARTITION BY {code6_expr}
                    ORDER BY {date_col} DESC,
                             CASE
                                 WHEN UPPER({windcode_col}) LIKE '%.OF' THEN 1
                                 WHEN UPPER({windcode_col}) LIKE '%.SH' THEN 2
                                 WHEN UPPER({windcode_col}) LIKE '%.SZ' THEN 2
                                 ELSE 9
                             END ASC
                ) AS rn
            FROM {table_name}
            WHERE {date_col} <= :asof_date
              AND {code6_expr} IN ({in_clause})
        )
        WHERE rn = 1
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    out = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    if "rn" in out.columns:
        out = out.drop(columns=["rn"])
    return out


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
    aggregate_select = ",\n                ".join(
        [f"MAX({col}) KEEP (DENSE_RANK LAST ORDER BY {date_col}) AS {col}" for col in available_cols]
    )
    for batch in chunks(list(windcodes), batch_size):
        in_clause, params = build_in_clause(batch, "w")
        params["asof_date"] = asof_date
        sql = f"""
        SELECT
            {windcode_col} AS windcode,
            MAX({date_col}) AS record_date,
            {aggregate_select}
        FROM {table_name}
        WHERE {date_col} <= :asof_date
          AND {windcode_col} IN ({in_clause})
        GROUP BY {windcode_col}
        """
        try:
            frames.append(fetch_dataframe(cursor, sql, params))
            continue
        except Exception:
            # 兼容少数表字段不适合 KEEP 聚合的情况，退回窗口函数版本。
            pass

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

    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    out = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    if "rn" in out.columns:
        out = out.drop(columns=["rn"])
    return out


def collapse_latest_windcode_rows_to_code6(df: pd.DataFrame, windcode_col: str = "windcode") -> pd.DataFrame:
    if df.empty or windcode_col not in df.columns:
        return df.copy()

    out = df.copy()
    out[windcode_col] = series_strip(out[windcode_col])
    out["code6"] = out[windcode_col].apply(extract_code6)
    out = out[out["code6"].notna()].copy()
    if out.empty:
        return out

    if "record_date" in out.columns:
        out["record_date"] = out["record_date"].apply(normalize_date)
    out["windcode_priority"] = out[windcode_col].apply(windcode_priority)
    out = (
        out.sort_values(
            ["code6", "record_date", "windcode_priority", windcode_col],
            ascending=[True, False, True, True],
            na_position="last",
        )
        .drop_duplicates(subset=["code6"], keep="first")
        .reset_index(drop=True)
    )
    out = out.rename(columns={windcode_col: "source_windcode"})
    return out.drop(columns=["windcode_priority"], errors="ignore")


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
            windcodes=universe["windcode"].dropna().astype(str).unique().tolist(),
            date_col_candidates=[date_col],
            select_cols=select_cols,
            batch_size=batch_size,
            asof_date=asof_date,
        )

    code6_values = universe["code6"].dropna().astype(str).unique().tolist()

    frames = []
    code_maps = [
        ("inner_code", resolve_col(cols, ["S_INFO_INNERCODE"])),
        ("outer_code", resolve_col(cols, ["S_INFO_OUTERCODE"])),
    ]
    for _, track_col in code_maps:
        if track_col is None:
            continue

        for batch in chunks(code6_values, batch_size):
            in_clause, params = build_in_clause(batch, "c")
            params["asof_date"] = asof_date
            code6_expr = regex_code6_sql(track_col)
            sql = f"""
            SELECT *
            FROM (
                SELECT
                    {code6_expr} AS code6,
                    {track_col} AS matched_code,
                    {date_col} AS record_date,
                    {", ".join(select_cols)},
                    ROW_NUMBER() OVER (
                        PARTITION BY {code6_expr}
                        ORDER BY {date_col} DESC
                    ) AS rn
                FROM {table_name}
                WHERE {date_col} <= :asof_date
                  AND {code6_expr} IN ({in_clause})
            )
            WHERE rn = 1
            """
            tmp = fetch_dataframe(cursor, sql, params)
            if tmp.empty:
                continue
            tmp = tmp.drop(columns=["rn"], errors="ignore")
            frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["code6", "record_date"], ascending=[True, False]).drop_duplicates("code6", keep="first")
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
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["windcode"] = series_strip(out["windcode"])
    out["code6"] = out["windcode"].apply(extract_code6)
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
        """
        frames.append(fetch_dataframe(cursor, sql, params))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["windcode"] = series_strip(out["windcode"])
    out["code6"] = out["windcode"].apply(extract_code6)
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
        snapshot = snapshot.merge(latest_nav, on="code6", how="left")

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
        work = daily_prices.sort_values(["windcode", "trade_dt"], na_position="last")
        latest = work.drop_duplicates(subset=["windcode"], keep="last")[
            ["windcode", "trade_dt", "close", "adj_close", "volume_hand", "amount_cny"]
        ].copy()
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

        summary_df = work.groupby("windcode", sort=False).size().rename("price_obs_count").reset_index()
        summary_df["avg_turnover_20d_cny"] = math.nan
        summary_df["avg_turnover_60d_cny"] = math.nan
        summary_df["avg_volume_20d_hand"] = math.nan
        if "amount_cny" in work.columns or "volume_hand" in work.columns:
            tail20 = work.groupby("windcode", sort=False).tail(20)
            summary20 = tail20.groupby("windcode", sort=False).agg(
                avg_turnover_20d_cny=("amount_cny", "mean") if "amount_cny" in tail20.columns else ("windcode", "size"),
                avg_volume_20d_hand=("volume_hand", "mean") if "volume_hand" in tail20.columns else ("windcode", "size"),
            ).reset_index()
            if "amount_cny" not in tail20.columns:
                summary20["avg_turnover_20d_cny"] = math.nan
            if "volume_hand" not in tail20.columns:
                summary20["avg_volume_20d_hand"] = math.nan
            summary20 = summary20[["windcode", "avg_turnover_20d_cny", "avg_volume_20d_hand"]]
            summary_df = summary_df.merge(summary20, on="windcode", how="left")

            if "amount_cny" in work.columns:
                tail60 = work.groupby("windcode", sort=False).tail(60)
                summary60 = tail60.groupby("windcode", sort=False)["amount_cny"].mean().rename(
                    "avg_turnover_60d_cny"
                ).reset_index()
                summary_df = summary_df.merge(summary60, on="windcode", how="left")
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
        universe, sector_membership, code_mapping = build_etf_universe(cursor, BATCH_SIZE)
        windcodes = universe["windcode"].dropna().astype(str).tolist()
        code6_values = universe["code6"].dropna().astype(str).tolist()
        print(f"识别到 ETF 数量: {len(windcodes)}")
        print(f"归并后的基金代码数量: {len(code6_values)}")

        print("2/7 下载 ETF 最新净值/规模...")
        nav_windcodes = explode_pipe_separated_values(universe["all_windcodes"])
        print(f"净值查询 Wind 代码数: {len(nav_windcodes)}")
        # NAV 大表对 code6 做函数过滤会变慢，改成精确 windcode 过滤后再本地归并。
        latest_nav = fetch_latest_by_windcode(
            cursor=cursor,
            table_name="ChinaMutualFundNAV",
            windcodes=nav_windcodes,
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
        latest_nav = collapse_latest_windcode_rows_to_code6(latest_nav)

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
        tracking_windcodes = universe["windcode"].dropna().astype(str).unique().tolist()
        print(f"跟踪表现查询 Wind 代码数: {len(tracking_windcodes)}")
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
    write_dataframe(code_mapping, output_dir / "etf_code_mapping.csv")
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
