#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402


GROUP_SPECS: List[Dict[str, object]] = [
    {
        "bucket": "equity_core",
        "benchmark_group": "CSI300",
        "match_any": [r"沪深300"],
        "exclude_any": [r"红利低波", r"港股", r"医药ETF", r"证券ETF", r"证券保险", r"300价值", r"300成长"],
        "top_n": 3,
        "notes": "中国权益核心宽基，优先沪深300主流ETF。",
    },
    {
        "bucket": "equity_core",
        "benchmark_group": "CSI_A500",
        "match_any": [r"中证A500", r"\bA500ETF\b", r"A500ETF"],
        "exclude_any": [r"港股"],
        "top_n": 3,
        "notes": "中国权益核心宽基替代，A500系列ETF。",
    },
    {
        "bucket": "equity_defensive",
        "benchmark_group": "CSI_DIV_LOWVOL",
        "match_any": [r"红利低波"],
        "exclude_any": [r"港股"],
        "top_n": 3,
        "notes": "权益防御层，优先红利低波。",
    },
    {
        "bucket": "equity_defensive",
        "benchmark_group": "CSI_DIVIDEND",
        "match_any": [r"红利ETF", r"中证红利"],
        "exclude_any": [r"低波", r"港股", r"央企"],
        "top_n": 3,
        "notes": "权益防御替代层，优先纯红利风格。",
    },
    {
        "bucket": "rate_bond",
        "benchmark_group": "SSE_5Y_GOV",
        "match_any": [r"5年.*国债", r"上证5年期国债"],
        "exclude_any": [],
        "top_n": 2,
        "notes": "利率债中等久期桶。",
    },
    {
        "bucket": "rate_bond",
        "benchmark_group": "SSE_10Y_GOV",
        "match_any": [r"10年.*国债", r"十年国债"],
        "exclude_any": [],
        "top_n": 2,
        "notes": "利率债长久期替代桶。",
    },
    {
        "bucket": "gold",
        "benchmark_group": "DOMESTIC_SPOT_GOLD",
        "match_any": [r"黄金ETF", r"金ETF", r"上海金ETF"],
        "exclude_any": [r"港股", r"矿业", r"有色"],
        "top_n": 3,
        "notes": "国内黄金现货工具。",
    },
    {
        "bucket": "money_market",
        "benchmark_group": "CASH_MGMT",
        "money_only": True,
        "exclude_any": [],
        "top_n": 3,
        "notes": "场内货币ETF现金管理桶。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial ETF asset pool candidates from cleaned research dataset.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "output" / "etf_bulk_download_research" / "etf_master_snapshot.csv"),
        help="Cleaned research master snapshot path.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "output" / "etf_bulk_download_research" / "asset_pool_candidates_initial.csv"),
        help="Output candidate CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "output" / "etf_bulk_download_research" / "asset_pool_candidates_initial.md"),
        help="Output markdown summary path.",
    )
    return parser.parse_args()


def text_match(series: pd.Series, patterns: List[str]) -> pd.Series:
    if not patterns:
        return pd.Series(True, index=series.index)
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask = mask | series.str.contains(pattern, regex=True, na=False)
    return mask


def coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(math.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["name_text"] = (
        work["short_name"].fillna("").astype(str)
        + " "
        + work["full_name"].fillna("").astype(str)
        + " "
        + work["sector_names"].fillna("").astype(str)
    )
    work["exchange"] = work["windcode"].astype(str).str[-2:].map({"SH": "SSE", "SZ": "SZSE"}).fillna("")
    work["listed_date"] = pd.to_datetime(work["setup_date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    work["avg_turnover_1m_cny"] = coerce_numeric(work, "avg_turnover_20d_cny")
    work["aum_cny"] = coerce_numeric(work, "aum_cny")
    work["total_fee_bp"] = coerce_numeric(work, "total_fee_bp")
    work["tracking_error_1y_pct"] = coerce_numeric(work, "tracking_error_1y_pct")
    work["listed_days"] = coerce_numeric(work, "listed_days")
    work["is_money_market_etf"] = work["is_money_market_etf"].fillna(False).astype(str).eq("True") | work["is_money_market_etf"].fillna(False).astype(bool)

    rows: List[Dict[str, object]] = []
    tier_names = ["primary", "backup_1", "backup_2", "backup_3"]

    for spec in GROUP_SPECS:
        if spec.get("money_only"):
            subset = work[work["is_money_market_etf"]].copy()
        else:
            subset = work.copy()
        subset = subset[text_match(subset["name_text"], spec.get("match_any", []))].copy()
        exclude_any = spec.get("exclude_any", [])
        if exclude_any:
            subset = subset[~text_match(subset["name_text"], exclude_any)].copy()
        if subset.empty:
            continue

        subset["tracking_sort"] = subset["tracking_error_1y_pct"].fillna(999.0)
        subset = subset.sort_values(
            ["avg_turnover_1m_cny", "aum_cny", "tracking_sort", "listed_days"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

        for rank, (_, row) in enumerate(subset.head(int(spec["top_n"])).iterrows()):
            rows.append(
                {
                    "bucket": spec["bucket"],
                    "benchmark_group": spec["benchmark_group"],
                    "tier": tier_names[rank],
                    "symbol": row["windcode"],
                    "name": row["short_name"],
                    "exchange": row["exchange"],
                    "listed_date": row["listed_date"],
                    "is_etf": 1,
                    "is_qdii": 1 if "QDII" in str(row.get("full_name", "")) else 0,
                    "is_leveraged_inverse": 1 if re.search(r"杠杆|反向|做空", str(row.get("short_name", ""))) else 0,
                    "avg_turnover_1m_cny": row["avg_turnover_1m_cny"],
                    "aum_cny": row["aum_cny"],
                    "total_fee_bp": row["total_fee_bp"],
                    "tracking_error_1y_pct": row["tracking_error_1y_pct"],
                    "tracking_proxy_score": "" if pd.notna(row["tracking_error_1y_pct"]) else (90 - rank * 5),
                    "has_primary_market_maker": "",
                    "has_option_underlying": "",
                    "bucket_fit_score": max(95 - rank * 5, 80),
                    "structure_score": max(90 - rank * 5, 75),
                    "notes": spec["notes"],
                }
            )

    return pd.DataFrame(rows)


def render_markdown(candidates: pd.DataFrame) -> str:
    lines = [
        "# Initial Asset Pool Candidates From Research Dataset",
        "",
        f"- total_rows: `{len(candidates)}`",
        "",
    ]
    for group, sub in candidates.groupby(["bucket", "benchmark_group"], sort=False):
        bucket, benchmark_group = group
        lines.append(f"## {bucket} / {benchmark_group}")
        lines.append("")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['tier']}: `{row['symbol']}` {row['name']} | turnover_1m={row['avg_turnover_1m_cny']:.0f} | aum={row['aum_cny']:.0f} | fee_bp={row['total_fee_bp']:.1f}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    candidates = build_candidates(df)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_csv, index=False)
    output_md.write_text(render_markdown(candidates), encoding="utf-8")
    print(f"rows={len(candidates)}")
    print(f"output_csv={output_csv}")
    print(f"output_md={output_md}")


if __name__ == "__main__":
    main()
