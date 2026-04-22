#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402


TIER_BUCKET_FIT = {
    "primary": 95,
    "backup_1": 90,
    "backup_2": 85,
    "backup_3": 80,
    "backup_4": 75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scoring input from index-layer bucket candidates.")
    parser.add_argument(
        "--candidates",
        default=str(ROOT / "output" / "etf_index_download" / "asset_bucket_candidates_from_index_layer.csv"),
        help="Bucket candidates from index layer.",
    )
    parser.add_argument(
        "--etf-master",
        default=str(ROOT / "output" / "etf_bulk_download_research" / "etf_master_snapshot.csv"),
        help="ETF master snapshot.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "etf_index_download" / "asset_pool_candidates_scoring_index_layer.csv"),
        help="Output scoring input CSV.",
    )
    return parser.parse_args()


def infer_exchange(symbol: pd.Series) -> pd.Series:
    suffix = symbol.fillna("").astype(str).str.upper().str[-2:]
    return suffix.map({"SH": "SSE", "SZ": "SZSE"}).fillna("")


def to_bool_int(series: pd.Series) -> pd.Series:
    text = series.fillna(False).astype(str).str.lower()
    return text.isin(["true", "1", "yes"]).astype(int)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    etf_master = pd.read_csv(args.etf_master)

    keep_master_cols = [
        col
        for col in [
            "windcode",
            "setup_date",
            "full_name",
            "avg_turnover_20d_cny",
            "aum_cny",
            "total_fee_bp",
            "tracking_error_1y_pct",
        ]
        if col in etf_master.columns
    ]
    master = etf_master[keep_master_cols].copy().rename(columns={"windcode": "symbol"})

    out = candidates.copy()
    out = out.merge(master, on="symbol", how="left", suffixes=("", "_master"))

    out["exchange"] = infer_exchange(out["symbol"])
    out["listed_date"] = pd.to_datetime(out["setup_date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    out["is_etf"] = 1
    out["is_qdii"] = out.get("full_name", pd.Series("", index=out.index)).fillna("").astype(str).str.contains("QDII", regex=False).astype(int)
    out["is_leveraged_inverse"] = out["name"].fillna("").astype(str).str.contains("杠杆|反向|做空", regex=True).astype(int)

    out["avg_turnover_1m_cny"] = pd.to_numeric(out["avg_turnover_1m_cny"], errors="coerce").fillna(
        pd.to_numeric(out.get("avg_turnover_20d_cny_master"), errors="coerce")
    )
    out["aum_cny"] = pd.to_numeric(out["aum_cny"], errors="coerce").fillna(pd.to_numeric(out.get("aum_cny_master"), errors="coerce"))
    out["total_fee_bp"] = pd.to_numeric(out["total_fee_bp"], errors="coerce").fillna(
        pd.to_numeric(out.get("total_fee_bp_master"), errors="coerce")
    )
    out["tracking_error_1y_pct"] = pd.to_numeric(out["tracking_error_1y_pct"], errors="coerce").fillna(
        pd.to_numeric(out.get("tracking_error_1y_pct_master"), errors="coerce")
    )
    out["tracking_deviation_daily_pct"] = pd.NA

    out["tracking_proxy_score"] = pd.NA
    gold_mask = out["bucket"].eq("gold")
    money_mask = out["bucket"].eq("money_market")
    out.loc[gold_mask & out["tracking_error_1y_pct"].isna(), "tracking_proxy_score"] = out.loc[
        gold_mask & out["tracking_error_1y_pct"].isna(), "tier"
    ].map({"primary": 90, "backup_1": 85, "backup_2": 82, "backup_3": 78, "backup_4": 75})
    out.loc[money_mask, "tracking_proxy_score"] = 90

    out["has_primary_market_maker"] = 0
    out["has_option_underlying"] = 0
    out.loc[(out["bucket"] == "equity_core") & (out["benchmark_group"] == "CSI300"), "has_option_underlying"] = 1
    out.loc[(out["bucket"] == "gold") & out["symbol"].isin(["518880.SH", "159934.SZ", "159937.SZ"]), "has_primary_market_maker"] = 1
    out.loc[(out["bucket"] == "money_market") & out["symbol"].isin(["511880.SH", "511990.SH"]), "has_primary_market_maker"] = 1

    out["bucket_fit_score"] = out["tier"].map(TIER_BUCKET_FIT).fillna(70).astype(float)
    out["structure_score"] = out["bucket_fit_score"] - 5
    out.loc[out["selection_layer"].eq("etf_exception"), "structure_score"] = out.loc[
        out["selection_layer"].eq("etf_exception"), "structure_score"
    ] - 5
    out["structure_score"] = out["structure_score"].clip(lower=60)

    out["notes"] = (
        out.get("notes", pd.Series("", index=out.index)).fillna("").astype(str)
        + " | layer="
        + out.get("selection_layer", pd.Series("", index=out.index)).fillna("").astype(str)
        + " | lane="
        + out.get("research_lane", pd.Series("", index=out.index)).fillna("").astype(str)
    )

    keep_cols = [
        "bucket",
        "benchmark_group",
        "tier",
        "symbol",
        "name",
        "exchange",
        "listed_date",
        "is_etf",
        "is_qdii",
        "is_leveraged_inverse",
        "avg_turnover_1m_cny",
        "aum_cny",
        "total_fee_bp",
        "tracking_error_1y_pct",
        "tracking_deviation_daily_pct",
        "tracking_proxy_score",
        "has_primary_market_maker",
        "has_option_underlying",
        "bucket_fit_score",
        "structure_score",
        "notes",
    ]
    output = out[keep_cols].copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"rows={len(output)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
