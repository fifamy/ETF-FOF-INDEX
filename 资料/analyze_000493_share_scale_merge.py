#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析 000493 不同份额的规模：
1) 各份额原始规模字段（F_PRT_NETASSET / NETASSET_TOTAL）
2) 选取规模（按主程序规则：统一使用 NETASSET_TOTAL）
3) 币种去重 + A/C去重后的最终规模（若选中份额规模为空，回退发行规模并组内聚合）
"""

import os
import re
import argparse
import pandas as pd

try:
    import cx_Oracle as ora
except Exception:
    ora = None


os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"
DB_CONN_STR = "chaxun/chaxun123@10.3.80.205/winddata"
DEFAULT_CODE = "000493.OF"
ASOF_CUR = "20251231"
ASOF_PREV = "20241231"


def fetch_dataframe(cursor, sql: str, params=None) -> pd.DataFrame:
    cursor.execute(sql, params or {})
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return pd.DataFrame(rows, columns=cols)


def series_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})


def get_table_columns(cursor, table_name: str):
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
        return set([str(c).upper() for c in df.columns])
    except Exception:
        return set()


def resolve_col(cols: set, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def extract_code6(code: str):
    if pd.isna(code):
        return None
    m = re.search(r"(\d{6})", str(code))
    return m.group(1) if m else None


def nav_code_suffix_priority(code: str) -> int:
    text = "" if pd.isna(code) else str(code).upper()
    if text.endswith(".SH") or text.endswith(".SZ"):
        return 1
    if text.endswith(".OF"):
        return 2
    return 9


def is_initial_yes(v) -> bool:
    text = "" if pd.isna(v) else str(v).strip()
    return text in {"是", "1", "Y", "YES", "True", "TRUE"}


def detect_share_class(short_name: str) -> str:
    if pd.isna(short_name):
        return "OTHER"
    name = str(short_name).strip()
    if "LOF" in name.upper():
        if re.search(r"A类$|A份额$|A$", name):
            return "A"
        if re.search(r"C类$|C份额$|C$", name):
            return "C"
        return "LOF"
    if re.search(r"A类$|A份额$|A$", name):
        return "A"
    if re.search(r"C类$|C份额$|C$", name):
        return "C"
    if re.search(r"E类$|E份额$|E$", name):
        return "E"
    if re.search(r"Y类$|Y份额$|Y$", name):
        return "Y"
    if re.search(r"I类$|I份额$|I$", name):
        return "I"
    if re.search(r"R类$|R份额$|R$", name):
        return "R"
    return "OTHER"


def share_priority(share_class: str) -> int:
    return {"A": 1, "OTHER": 2, "E": 3, "Y": 4, "I": 5, "R": 6, "LOF": 8, "C": 9}.get(share_class, 50)


def initial_priority(is_initial) -> int:
    if pd.isna(is_initial):
        return 9
    text = str(is_initial).strip()
    return 1 if text in {"是", "1", "Y", "YES", "True", "TRUE"} else 9


def normalize_fund_fullname(name: str) -> str:
    if pd.isna(name):
        return None
    text = str(name).strip()
    patterns = [
        r"\(LOF\)$", r"（LOF）$", r"[A-Z]类份额$", r"[A-Z]份额$", r"[A-Z]类$",
        r"\([A-Z]\)$", r"（[A-Z]）$", r"[A-Z]$", r"人民币$", r"美元$", r"LOF$",
    ]
    changed = True
    while changed:
        old = text
        for p in patterns:
            text = re.sub(p, "", text).strip()
        changed = text != old
    generic_patterns = [
        r"被动式指数证券投资基金$", r"指数证券投资基金$", r"混合型证券投资基金$",
        r"股票型证券投资基金$", r"债券型证券投资基金$", r"证券投资基金$", r"基金$",
    ]
    changed = True
    while changed:
        old = text
        for p in generic_patterns:
            text = re.sub(p, "", text).strip()
        changed = text != old
    return text


def normalize_fund_shortname(name: str) -> str:
    if pd.isna(name):
        return None
    text = str(name).strip()
    patterns = [
        r"[A-Z]类份额$", r"[A-Z]份额$", r"[A-Z]类$", r"\([A-Z]\)$", r"（[A-Z]）$",
        r"[A-Z]$", r"人民币$", r"美元$", r"LOF$", r"基金LOF$", r"ETF联接$", r"ETF$", r"联接$",
    ]
    changed = True
    while changed:
        old = text
        for p in patterns:
            text = re.sub(p, "", text).strip()
        changed = text != old
    return text


def build_fund_group_key(full_name: str, short_name: str) -> str:
    full_key = normalize_fund_fullname(full_name)
    short_key = normalize_fund_shortname(short_name)
    return full_key or short_key or short_name or full_name


def currency_priority(short_name: str) -> int:
    if pd.isna(short_name):
        return 50
    name = str(short_name).strip()
    if "人民币" in name:
        return 1
    if "美元" in name:
        return 9
    return 2


def normalize_currency_key(name: str) -> str:
    if pd.isna(name):
        return None
    text = str(name).strip()
    patterns = [
        r"人民币$", r"美元$", r"\(人民币\)$", r"（人民币）$", r"\(美元\)$", r"（美元）$",
        r"\(美元现汇\)$", r"（美元现汇）$", r"\(美元现钞\)$", r"（美元现钞）$", r"\(USD\)$", r"（USD）$",
    ]
    changed = True
    while changed:
        old = text
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
        changed = text != old
    return text


def deduplicate_currency(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["证券简称"] = work["证券简称"].fillna("")
    work["币种优先级"] = work["证券简称"].apply(currency_priority)
    work["币种标记"] = work["证券简称"].apply(lambda x: "人民币" if "人民币" in str(x) else ("美元" if "美元" in str(x) else None))
    work["币种组key"] = work["证券简称"].apply(normalize_currency_key)
    part_currency = work[work["币种标记"].notna()].copy()
    part_other = work[work["币种标记"].isna()].copy()
    if not part_currency.empty:
        part_currency = part_currency.sort_values(["币种组key", "币种优先级", "证券简称"])
        part_currency = part_currency.drop_duplicates(subset=["币种组key"], keep="first").copy()
    merged = pd.concat([part_currency, part_other], ignore_index=True)
    return merged.drop(columns=["币种优先级", "币种标记", "币种组key"], errors="ignore")


def deduplicate_ac(df: pd.DataFrame, aggregate_cols=None, fallback_source_cols=None, always_aggregate_cols=None) -> pd.DataFrame:
    """
    份额归并（不限A/C）：同一基金全称下的多个份额统一处理。
    """
    work = df.copy()
    aggregate_cols = aggregate_cols or []
    fallback_source_cols = fallback_source_cols or {}
    always_aggregate_cols = set(always_aggregate_cols or [])
    for col in aggregate_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["份额类型"] = work["证券简称"].apply(detect_share_class)
    work["份额优先级"] = work["份额类型"].apply(share_priority)
    work["初始基金优先级"] = work["是否为初始基金"].apply(initial_priority)
    work["基金分组key"] = work.apply(lambda x: build_fund_group_key(x.get("基金全称"), x.get("证券简称")), axis=1)
    work["基金分组key"] = work["基金分组key"].fillna(work["证券简称"])

    parts = []
    for _, grp in work.groupby("基金分组key", dropna=False):
        grp = grp.copy().sort_values(["初始基金优先级", "份额优先级", "证券简称"])
        chosen = grp.head(1).copy()
        for col in aggregate_cols:
            chosen_val = pd.to_numeric(chosen[col], errors="coerce").iloc[0]
            need_aggregate = (col in always_aggregate_cols) or pd.isna(chosen_val)
            if not need_aggregate:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if vals.empty:
                fallback_col = fallback_source_cols.get(col)
                if fallback_col and fallback_col in grp.columns:
                    vals = pd.to_numeric(grp[fallback_col], errors="coerce").dropna()
            if vals.empty:
                chosen.loc[:, col] = float("nan")
                continue
            sorted_vals = vals.sort_values()
            unique_vals = []
            for v in sorted_vals:
                if not unique_vals or abs(float(v) - unique_vals[-1]) > 1e-8:
                    unique_vals.append(float(v))
            if len(unique_vals) == 1:
                chosen.loc[:, col] = unique_vals[0]
            else:
                vmax = max(unique_vals)
                total_unique = sum(unique_vals)
                remain = total_unique - vmax
                chosen.loc[:, col] = vmax if abs(vmax - remain) <= 1e-6 else total_unique
        parts.append(chosen)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=work.columns)
    return out.drop(columns=["份额类型", "份额优先级", "初始基金优先级", "基金分组key"], errors="ignore")


def load_nav_selected_for_codes(cursor, codes, asof_date: str, out_col: str):
    codes = [c for c in codes if c]
    if not codes:
        return pd.DataFrame(columns=["F_INFO_CODE6", out_col, "NAV_WINDCODE"])

    code6_list = []
    for c in codes:
        c6 = extract_code6(c)
        if c6:
            code6_list.append(c6)
    code6_list = sorted(set(code6_list))
    if not code6_list:
        return pd.DataFrame(columns=["F_INFO_CODE6", out_col, "NAV_WINDCODE"])

    nav_table = "ChinaMutualFundNAV"
    nav_cols = get_table_columns(cursor, nav_table)
    code_col = resolve_col(nav_cols, ["F_INFO_WINDCODE", "S_INFO_WINDCODE"])
    date_col = resolve_col(nav_cols, ["TRADE_DT", "PRICE_DATE", "NAV_DATE", "ANN_DT", "REPORT_PERIOD", "ENDDATE"])
    if code_col is None or date_col is None or "F_PRT_NETASSET" not in nav_cols:
        return pd.DataFrame(columns=["F_INFO_CODE6", out_col, "NAV_WINDCODE"])

    total_expr = "NETASSET_TOTAL" if "NETASSET_TOTAL" in nav_cols else "NULL AS NETASSET_TOTAL"
    binds = []
    params = {"asof_date": asof_date}
    for i, c in enumerate(code6_list):
        k = f"c{i}"
        params[k] = c
        binds.append(f":{k}")
    in_clause = ", ".join(binds)

    sql_nav = f"""
    SELECT
        {code_col} AS F_INFO_WINDCODE,
        {date_col} AS NAV_DT,
        F_PRT_NETASSET,
        {total_expr}
    FROM (
        SELECT
            {code_col},
            {date_col},
            F_PRT_NETASSET,
            {total_expr},
            ROW_NUMBER() OVER (
                PARTITION BY {code_col}
                ORDER BY {date_col} DESC
            ) AS RN
        FROM {nav_table}
        WHERE {date_col} <= :asof_date
          AND SUBSTR({code_col}, 1, 6) IN ({in_clause})
    )
    WHERE RN = 1
    """
    nav_df = fetch_dataframe(cursor, sql_nav, params)
    if nav_df.empty:
        return pd.DataFrame(columns=["F_INFO_CODE6", "NAV_WINDCODE", out_col, "NAV_DT", "F_PRT_NETASSET", "NETASSET_TOTAL", "IS_GROUP_INITIAL"])

    nav_df.columns = ["F_INFO_WINDCODE", "NAV_DT", "F_PRT_NETASSET", "NETASSET_TOTAL"]
    nav_df["F_INFO_WINDCODE"] = series_strip(nav_df["F_INFO_WINDCODE"])
    nav_df["F_INFO_CODE6"] = nav_df["F_INFO_WINDCODE"].apply(extract_code6)
    nav_df["NAV_DT"] = pd.to_numeric(nav_df["NAV_DT"], errors="coerce")
    nav_df["F_PRT_NETASSET"] = pd.to_numeric(nav_df["F_PRT_NETASSET"], errors="coerce")
    nav_df["NETASSET_TOTAL"] = pd.to_numeric(nav_df["NETASSET_TOTAL"], errors="coerce")

    wc_df = fetch_dataframe(cursor, """
        SELECT S_INFO_WINDCODE, S_INFO_COMPCODE, SECURITY_STATUS
        FROM WindCustomCode
        WHERE S_INFO_SECURITIESTYPES = 'J'
    """)
    wc_df.columns = ["F_INFO_WINDCODE", "S_INFO_COMPCODE", "SECURITY_STATUS"]
    wc_df["F_INFO_WINDCODE"] = series_strip(wc_df["F_INFO_WINDCODE"])
    wc_df["S_INFO_COMPCODE"] = series_strip(wc_df["S_INFO_COMPCODE"])
    wc_df["SECURITY_STATUS"] = series_strip(wc_df["SECURITY_STATUS"])

    desc_df = fetch_dataframe(cursor, """
        SELECT F_INFO_WINDCODE, F_INFO_ISINITIAL
        FROM ChinaMutualFundDescription
    """)
    desc_df.columns = ["F_INFO_WINDCODE", "F_INFO_ISINITIAL"]
    desc_df["F_INFO_WINDCODE"] = series_strip(desc_df["F_INFO_WINDCODE"])
    desc_df["F_INFO_ISINITIAL"] = series_strip(desc_df["F_INFO_ISINITIAL"])
    desc_df["IS_INITIAL_FLAG"] = desc_df["F_INFO_ISINITIAL"].apply(is_initial_yes)

    wc_active = wc_df[wc_df["SECURITY_STATUS"] != "101002000"].copy()
    comp_join = wc_active.merge(desc_df[["F_INFO_WINDCODE", "IS_INITIAL_FLAG"]], on="F_INFO_WINDCODE", how="left")
    comp_initial = comp_join[comp_join["IS_INITIAL_FLAG"] == True].copy()
    comp_initial = comp_initial.sort_values(["S_INFO_COMPCODE", "F_INFO_WINDCODE"]).drop_duplicates(
        subset=["S_INFO_COMPCODE"], keep="first"
    )
    comp_initial = comp_initial[["S_INFO_COMPCODE", "F_INFO_WINDCODE"]].rename(columns={"F_INFO_WINDCODE": "INITIAL_WINDCODE"})

    code_to_initial = wc_df[["F_INFO_WINDCODE", "S_INFO_COMPCODE"]].merge(comp_initial, on="S_INFO_COMPCODE", how="left")
    nav_df = nav_df.merge(code_to_initial[["F_INFO_WINDCODE", "INITIAL_WINDCODE"]], on="F_INFO_WINDCODE", how="left")
    nav_df["IS_GROUP_INITIAL"] = nav_df["F_INFO_WINDCODE"] == nav_df["INITIAL_WINDCODE"]

    nav_df["NAV_CHOSEN"] = nav_df["NETASSET_TOTAL"]
    nav_df["NAV_SUFFIX_PRIORITY"] = nav_df["F_INFO_WINDCODE"].apply(nav_code_suffix_priority)
    nav_df = nav_df.sort_values(["F_INFO_CODE6", "NAV_SUFFIX_PRIORITY", "NAV_DT"], ascending=[True, True, False])
    nav_df = nav_df.drop_duplicates(subset=["F_INFO_CODE6"], keep="first").copy()
    nav_df[out_col] = nav_df["NAV_CHOSEN"] / 1e8
    nav_df["NAV_WINDCODE"] = nav_df["F_INFO_WINDCODE"]
    return nav_df[["F_INFO_CODE6", "NAV_WINDCODE", "NAV_DT", "F_PRT_NETASSET", "NETASSET_TOTAL", "IS_GROUP_INITIAL", out_col]]


def load_transformation_flag(cursor, code6_list):
    """
    从 ChinaMutualFundTransformation 识别发生过转型的基金（按6位代码）。
    动态扫描所有包含 WINDCODE 的字段，适配不同库表结构。
    """
    if not code6_list:
        return pd.DataFrame(columns=["F_INFO_CODE6", "是否发生转型"])

    table_name = "ChinaMutualFundTransformation"
    cols = get_table_columns(cursor, table_name)
    if not cols:
        return pd.DataFrame(columns=["F_INFO_CODE6", "是否发生转型"])

    windcode_cols = sorted([c for c in cols if "WINDCODE" in c])
    if not windcode_cols:
        return pd.DataFrame(columns=["F_INFO_CODE6", "是否发生转型"])

    code6_set = set([c for c in code6_list if c])
    parts = []
    for c in windcode_cols:
        try:
            sql = f"""
            SELECT DISTINCT
                {c} AS F_INFO_WINDCODE
            FROM {table_name}
            WHERE {c} IS NOT NULL
            """
            tmp = fetch_dataframe(cursor, sql)
            if tmp.empty:
                continue
            tmp.columns = ["F_INFO_WINDCODE"]
            tmp["F_INFO_WINDCODE"] = series_strip(tmp["F_INFO_WINDCODE"])
            tmp["F_INFO_CODE6"] = tmp["F_INFO_WINDCODE"].apply(extract_code6)
            tmp = tmp[tmp["F_INFO_CODE6"].isin(code6_set)].copy()
            if not tmp.empty:
                parts.append(tmp[["F_INFO_CODE6"]])
        except Exception:
            continue

    if not parts:
        return pd.DataFrame(columns=["F_INFO_CODE6", "是否发生转型"])

    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["F_INFO_CODE6"], keep="first")
    out["是否发生转型"] = 1
    return out[["F_INFO_CODE6", "是否发生转型"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default=DEFAULT_CODE, help="基金代码，默认 000493.OF")
    parser.add_argument("--asof_cur", default=ASOF_CUR, help="最新日期，默认 20251231")
    parser.add_argument("--asof_prev", default=ASOF_PREV, help="历史日期，默认 20241231")
    parser.add_argument("--conn", default=DB_CONN_STR, help="Oracle连接串")
    args, _ = parser.parse_known_args()

    if ora is None:
        raise RuntimeError("cx_Oracle 未安装")

    conn = ora.connect(args.conn)
    cursor = conn.cursor()
    try:
        # 先定位同组全称
        df_seed = fetch_dataframe(cursor, """
            SELECT F_INFO_FULLNAME
            FROM ChinaMutualFundDescription
            WHERE F_INFO_WINDCODE = :code
        """, {"code": args.code})
        if df_seed.empty or pd.isna(df_seed.iloc[0, 0]):
            print(f"未找到代码：{args.code}")
            return
        full_name = str(df_seed.iloc[0, 0]).strip()

        df_desc = fetch_dataframe(cursor, """
            SELECT
                F_INFO_WINDCODE,
                F_INFO_NAME,
                F_INFO_FULLNAME,
                F_INFO_SETUPDATE,
                F_INFO_ISINITIAL,
                F_ISSUE_TOTALUNIT
            FROM ChinaMutualFundDescription
            WHERE F_INFO_FULLNAME = :full_name
            ORDER BY F_INFO_NAME, F_INFO_WINDCODE
        """, {"full_name": full_name})
    finally:
        cursor.close()
        conn.close()

    if df_desc.empty:
        print(f"未找到同组份额：{args.code}")
        return

    df_desc.columns = ["F_INFO_WINDCODE", "证券简称", "基金全称", "基金成立日", "是否为初始基金", "发行总规模 [单位] 亿元"]
    for c in ["F_INFO_WINDCODE", "证券简称", "基金全称", "是否为初始基金"]:
        df_desc[c] = series_strip(df_desc[c])
    df_desc["F_INFO_CODE6"] = df_desc["F_INFO_WINDCODE"].apply(extract_code6)
    df_desc["基金成立日"] = pd.to_datetime(df_desc["基金成立日"].astype(str), format="%Y%m%d", errors="coerce")
    df_desc["发行总规模 [单位] 亿元"] = pd.to_numeric(df_desc["发行总规模 [单位] 亿元"], errors="coerce")

    # 再开连接拿 NAV 规模 + 转型标识
    conn = ora.connect(args.conn)
    cursor = conn.cursor()
    try:
        codes = df_desc["F_INFO_WINDCODE"].dropna().astype(str).tolist()
        code6_list = sorted(set(df_desc["F_INFO_CODE6"].dropna().astype(str).tolist()))
        cur_df = load_nav_selected_for_codes(cursor, codes, args.asof_cur, f"选取规模_亿元_{args.asof_cur}")
        prev_df = load_nav_selected_for_codes(cursor, codes, args.asof_prev, f"选取规模_亿元_{args.asof_prev}")
        transform_df = load_transformation_flag(cursor, code6_list)
    finally:
        cursor.close()
        conn.close()

    merged = df_desc.merge(cur_df, on="F_INFO_CODE6", how="left")
    merged = merged.merge(
        prev_df.rename(
            columns={
                "NAV_WINDCODE": "NAV_WINDCODE_PREV",
                "NAV_DT": "NAV_DT_PREV",
                "F_PRT_NETASSET": "F_PRT_NETASSET_PREV",
                "NETASSET_TOTAL": "NETASSET_TOTAL_PREV",
                "IS_GROUP_INITIAL": "IS_GROUP_INITIAL_PREV",
            }
        ),
        on="F_INFO_CODE6",
        how="left",
    )
    merged = merged.merge(transform_df, on="F_INFO_CODE6", how="left")
    merged["是否发生转型"] = pd.to_numeric(merged["是否发生转型"], errors="coerce").fillna(0).astype(int)

    print("=== 000493 同组份额（原始 + 选取规模）===")
    cols_show = [
        "F_INFO_WINDCODE", "证券简称", "是否为初始基金", "基金成立日", "NAV_WINDCODE",
        "NAV_DT", "F_PRT_NETASSET", "NETASSET_TOTAL", "IS_GROUP_INITIAL", f"选取规模_亿元_{args.asof_cur}",
        "NAV_WINDCODE_PREV", "NAV_DT_PREV", "F_PRT_NETASSET_PREV", "NETASSET_TOTAL_PREV", "IS_GROUP_INITIAL_PREV", f"选取规模_亿元_{args.asof_prev}",
        "发行总规模 [单位] 亿元", "是否发生转型",
    ]
    print(merged[cols_show].to_string(index=False))

    # 币种去重 + A/C 去重（同主程序）
    work = merged.rename(
        columns={
            f"选取规模_亿元_{args.asof_cur}": "基金规模合计[交易日期] 2025-12-31",
            f"选取规模_亿元_{args.asof_prev}": "基金规模合计[交易日期] 2024-12-31",
        }
    ).copy()
    after_currency = deduplicate_currency(work)
    after_merge = deduplicate_ac(
        after_currency,
        aggregate_cols=[
            "基金规模合计[交易日期] 2025-12-31",
            "基金规模合计[交易日期] 2024-12-31",
            "发行总规模 [单位] 亿元",
        ],
        fallback_source_cols={
            "基金规模合计[交易日期] 2025-12-31": "发行总规模 [单位] 亿元",
        },
        always_aggregate_cols=[
            "发行总规模 [单位] 亿元",
        ],
    )

    print("\n=== 币种去重后份额 ===")
    print(after_currency[["F_INFO_WINDCODE", "证券简称", "是否为初始基金", "基金规模合计[交易日期] 2025-12-31", "基金规模合计[交易日期] 2024-12-31"]].to_string(index=False))

    print("\n=== 最终保留份额与最终规模（A/C归并后）===")
    print(after_merge[["F_INFO_WINDCODE", "证券简称", "是否为初始基金", "基金规模合计[交易日期] 2025-12-31", "基金规模合计[交易日期] 2024-12-31", "发行总规模 [单位] 亿元"]].to_string(index=False))

    # 对齐主程序：2025年及以后成立基金，2024规模通常置空；若发生转型则保留
    after_merge["是否发生转型"] = pd.to_numeric(after_merge.get("是否发生转型"), errors="coerce").fillna(0).astype(int)
    after_merge["2024规模_主程序口径"] = pd.to_numeric(after_merge["基金规模合计[交易日期] 2024-12-31"], errors="coerce")
    founded_2025_or_later = pd.to_datetime(after_merge["基金成立日"], errors="coerce") >= pd.Timestamp("2025-01-01")
    after_merge.loc[founded_2025_or_later & (~after_merge["是否发生转型"].eq(1)), "2024规模_主程序口径"] = float("nan")

    print("\n=== 转型规则检查（2024规模是否应保留）===")
    print(after_merge[["F_INFO_WINDCODE", "证券简称", "基金成立日", "是否发生转型", "基金规模合计[交易日期] 2024-12-31", "2024规模_主程序口径"]].to_string(index=False))


if __name__ == "__main__":
    main()
