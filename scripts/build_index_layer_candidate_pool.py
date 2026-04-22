#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402


INDEX_GROUP_SPECS: List[Dict[str, object]] = [
    {
        "bucket": "equity_core",
        "benchmark_group": "CSI300",
        "match_any": [r"沪深300指数"],
        "exclude_any": [r"红利", r"成长", r"价值", r"医药", r"金融地产", r"ESG", r"自由现金流"],
        "top_n": 3,
        "notes": "中国权益核心宽基，先选指数，再选同指数ETF。",
    },
    {
        "bucket": "equity_core",
        "benchmark_group": "CSI_A500",
        "match_any": [r"中证A500指数"],
        "exclude_any": [],
        "top_n": 3,
        "notes": "中国权益核心宽基替代，A500指数族。",
    },
    {
        "bucket": "equity_defensive",
        "benchmark_group": "CSI_DIV_LOWVOL",
        "match_any": [r"红利低波", r"低波动"],
        "exclude_any": [r"港股"],
        "top_n": 3,
        "notes": "权益防御层，优先红利低波指数。",
    },
    {
        "bucket": "equity_defensive",
        "benchmark_group": "CSI_DIVIDEND",
        "match_any": [r"红利指数", r"中证红利", r"红利质量", r"央企红利", r"国企红利"],
        "exclude_any": [r"低波", r"港股"],
        "top_n": 4,
        "notes": "权益防御替代层，优先纯红利和红利质量指数。",
    },
    {
        "bucket": "rate_bond",
        "benchmark_group": "SSE_5Y_GOV",
        "match_any": [r"5年.*国债"],
        "exclude_any": [r"地方政府债"],
        "top_n": 2,
        "notes": "利率债中等久期桶。",
    },
    {
        "bucket": "rate_bond",
        "benchmark_group": "SSE_10Y_GOV",
        "match_any": [r"10年.*国债", r"5-10年期国债活跃券"],
        "exclude_any": [r"地方政府债"],
        "top_n": 2,
        "notes": "利率债长久期桶。",
    },
    {
        "bucket": "gold",
        "benchmark_group": "DOMESTIC_SPOT_GOLD",
        "match_index_code_any": [r"Au9999\\.SGE$", r"SHAU\\.SGE$"],
        "match_etf_name_any": [r"黄金ETF", r"金ETF", r"上海金ETF"],
        "exclude_etf_name_any": [r"黄金股"],
        "top_n": 4,
        "notes": "黄金现货层目前仍需补指数源，但ETF工具层已可筛选。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build index-layer candidate pool from ETF-index research base.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "output" / "etf_index_download" / "etf_index_research_base.csv"),
        help="ETF-index research base CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "etf_index_download"),
        help="Output directory.",
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


def build_index_representative_pool(base: pd.DataFrame) -> pd.DataFrame:
    work = base[base["tracking_index_windcode"].notna()].copy()
    work["turnover20"] = coerce_numeric(work, "avg_turnover_20d_cny")
    work["aum"] = coerce_numeric(work, "aum_cny")
    work["tracking_error_1y_pct"] = coerce_numeric(work, "tracking_error_1y_pct")
    work["tracking_sort"] = work["tracking_error_1y_pct"].fillna(999.0)

    reps = (
        work.sort_values(
            ["tracking_index_windcode", "turnover20", "aum", "tracking_sort"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(subset=["tracking_index_windcode"], keep="first")
        .copy()
    )

    members = (
        work.groupby("tracking_index_windcode", dropna=False)
        .agg(
            etf_count=("windcode", "nunique"),
            all_etf_windcodes=("windcode", lambda s: "|".join(sorted(pd.Series(s).dropna().astype(str).unique().tolist()))),
            all_etf_names=("short_name", lambda s: "|".join(sorted(pd.Series(s).dropna().astype(str).unique().tolist()))),
            max_turnover_20d_cny=("turnover20", "max"),
            max_aum_cny=("aum", "max"),
        )
        .reset_index()
    )

    keep_cols = [
        "tracking_index_windcode",
        "index_name",
        "index_short_name",
        "index_exchange",
        "description_source_table",
        "latest_price_source_table",
        "coverage_status",
        "index_suffix_family",
        "asset_bucket_hint",
        "windcode",
        "short_name",
        "avg_turnover_20d_cny",
        "aum_cny",
        "total_fee_bp",
        "tracking_error_1y_pct",
        "research_lane",
    ]
    reps = reps[[col for col in keep_cols if col in reps.columns]].copy()
    reps = reps.rename(
        columns={
            "windcode": "representative_etf",
            "short_name": "representative_etf_name",
            "avg_turnover_20d_cny": "representative_turnover_20d_cny",
            "aum_cny": "representative_aum_cny",
            "total_fee_bp": "representative_total_fee_bp",
            "tracking_error_1y_pct": "representative_tracking_error_1y_pct",
            "research_lane": "representative_research_lane",
        }
    )
    reps = reps.merge(members, on="tracking_index_windcode", how="left")
    return reps.sort_values(["asset_bucket_hint", "tracking_index_windcode"]).reset_index(drop=True)


def build_bucket_candidates(index_pool: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    tier_names = ["primary", "backup_1", "backup_2", "backup_3", "backup_4"]
    ready_index_pool = index_pool[index_pool["representative_research_lane"].eq("domestic_index_ready")].copy()

    for spec in INDEX_GROUP_SPECS:
        if spec["benchmark_group"] == "DOMESTIC_SPOT_GOLD":
            subset = base.copy()
            subset["turnover20"] = coerce_numeric(subset, "avg_turnover_20d_cny")
            subset["aum"] = coerce_numeric(subset, "aum_cny")
            subset = subset[
                subset["tracking_index_windcode"].fillna("").astype(str).str.contains("|".join(spec.get("match_index_code_any", [])), regex=True)
                | text_match(subset["short_name"].fillna("").astype(str), list(spec.get("match_etf_name_any", [])))
            ].copy()
            exclude = list(spec.get("exclude_etf_name_any", []))
            if exclude:
                subset = subset[~text_match(subset["short_name"].fillna("").astype(str), exclude)].copy()
            subset = subset.sort_values(["turnover20", "aum"], ascending=[False, False]).reset_index(drop=True)
            for rank, (_, row) in enumerate(subset.head(int(spec["top_n"])).iterrows()):
                rows.append(
                    {
                        "selection_layer": "etf_exception",
                        "bucket": spec["bucket"],
                        "benchmark_group": spec["benchmark_group"],
                        "tier": tier_names[rank],
                        "index_windcode": row.get("tracking_index_windcode"),
                        "index_name": row.get("index_name"),
                        "index_short_name": row.get("index_short_name"),
                        "symbol": row.get("windcode"),
                        "name": row.get("short_name"),
                        "avg_turnover_1m_cny": row.get("avg_turnover_20d_cny"),
                        "aum_cny": row.get("aum_cny"),
                        "total_fee_bp": row.get("total_fee_bp"),
                        "tracking_error_1y_pct": row.get("tracking_error_1y_pct"),
                        "coverage_status": row.get("coverage_status"),
                        "research_lane": row.get("research_lane"),
                        "notes": spec["notes"],
                    }
                )
            continue

        subset = ready_index_pool.copy()
        name_text = subset["index_name"].fillna("").astype(str) + " " + subset["index_short_name"].fillna("").astype(str)
        subset = subset[text_match(name_text, list(spec.get("match_any", [])))].copy()
        exclude_any = list(spec.get("exclude_any", []))
        if exclude_any:
            subset_name_text = subset["index_name"].fillna("").astype(str) + " " + subset["index_short_name"].fillna("").astype(str)
            subset = subset[~text_match(subset_name_text, exclude_any)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["representative_turnover_20d_cny", "representative_aum_cny", "representative_tracking_error_1y_pct"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(subset.head(int(spec["top_n"])).iterrows()):
            rows.append(
                {
                    "selection_layer": "index_layer",
                    "bucket": spec["bucket"],
                    "benchmark_group": spec["benchmark_group"],
                    "tier": tier_names[rank],
                    "index_windcode": row.get("tracking_index_windcode"),
                    "index_name": row.get("index_name"),
                    "index_short_name": row.get("index_short_name"),
                    "symbol": row.get("representative_etf"),
                    "name": row.get("representative_etf_name"),
                    "avg_turnover_1m_cny": row.get("representative_turnover_20d_cny"),
                    "aum_cny": row.get("representative_aum_cny"),
                    "total_fee_bp": row.get("representative_total_fee_bp"),
                    "tracking_error_1y_pct": row.get("representative_tracking_error_1y_pct"),
                    "coverage_status": row.get("coverage_status"),
                    "research_lane": row.get("representative_research_lane"),
                    "notes": spec["notes"],
                }
            )

    # Money market remains ETF-layer exception because mapping coverage is absent.
    money = base[base["research_lane"].eq("money_market_no_index_mapping")].copy()
    if not money.empty:
        money["turnover20"] = coerce_numeric(money, "avg_turnover_20d_cny")
        money["aum"] = coerce_numeric(money, "aum_cny")
        money["fee_bp"] = coerce_numeric(money, "total_fee_bp")
        money = money.sort_values(["turnover20", "aum", "fee_bp"], ascending=[False, False, True]).reset_index(drop=True)
        for rank, (_, row) in enumerate(money.head(4).iterrows()):
            rows.append(
                {
                    "selection_layer": "etf_exception",
                    "bucket": "money_market",
                    "benchmark_group": "CASH_MGMT",
                    "tier": tier_names[rank],
                    "index_windcode": row.get("tracking_index_windcode"),
                    "index_name": row.get("index_name"),
                    "index_short_name": row.get("index_short_name"),
                    "symbol": row.get("windcode"),
                    "name": row.get("short_name"),
                    "avg_turnover_1m_cny": row.get("avg_turnover_20d_cny"),
                    "aum_cny": row.get("aum_cny"),
                    "total_fee_bp": row.get("total_fee_bp"),
                    "tracking_error_1y_pct": row.get("tracking_error_1y_pct"),
                    "coverage_status": row.get("coverage_status"),
                    "research_lane": row.get("research_lane"),
                    "notes": "货币ETF当前无稳定指数映射，保留ETF层筛选。",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        mask = out["selection_layer"].eq("index_layer")
        out = out[~mask | out["research_lane"].eq("domestic_index_ready")].copy()
    return out


def render_markdown(index_pool: pd.DataFrame, bucket_candidates: pd.DataFrame) -> str:
    lines = [
        "# Index Layer Candidate Pool",
        "",
        f"- index_representative_count: `{len(index_pool)}`",
        f"- bucket_candidate_rows: `{len(bucket_candidates)}`",
        "",
        "## Bucket Candidates",
        "",
    ]
    for group, sub in bucket_candidates.groupby(["bucket", "benchmark_group"], sort=False):
        bucket, benchmark_group = group
        lines.append(f"## {bucket} / {benchmark_group}")
        lines.append("")
        for _, row in sub.iterrows():
            lines.append(
                f"- {row['tier']}: `{row['symbol']}` {row['name']} | index=`{row.get('index_windcode')}` {row.get('index_short_name') or row.get('index_name')} | layer={row['selection_layer']} | lane={row['research_lane']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_pool = build_index_representative_pool(base)
    bucket_candidates = build_bucket_candidates(index_pool, base)

    index_pool_path = output_dir / "index_representative_pool.csv"
    bucket_candidates_path = output_dir / "asset_bucket_candidates_from_index_layer.csv"
    summary_path = output_dir / "index_layer_candidate_pool_summary.md"

    index_pool.to_csv(index_pool_path, index=False)
    bucket_candidates.to_csv(bucket_candidates_path, index=False)
    summary_path.write_text(render_markdown(index_pool, bucket_candidates), encoding="utf-8")

    print(f"index_representative_rows={len(index_pool)}")
    print(f"bucket_candidate_rows={len(bucket_candidates)}")
    print(f"index_pool={index_pool_path}")
    print(f"bucket_candidates={bucket_candidates_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
