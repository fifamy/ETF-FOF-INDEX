#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ETF-index research base table from downloaded bridge files.")
    parser.add_argument(
        "--etf-master",
        default=str(ROOT / "output" / "etf_bulk_download_research" / "etf_master_snapshot.csv"),
        help="Cleaned ETF research master snapshot.",
    )
    parser.add_argument(
        "--index-dir",
        default=str(ROOT / "output" / "etf_index_download"),
        help="Directory containing ETF-index bridge outputs.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "output" / "etf_index_download" / "etf_index_research_base.csv"),
        help="Output research base CSV.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "output" / "etf_index_download" / "etf_index_research_base_summary.md"),
        help="Output markdown summary.",
    )
    return parser.parse_args()


def normalize_bool(series: pd.Series) -> pd.Series:
    text = series.fillna(False).astype(str).str.lower()
    return text.isin(["true", "1", "yes"])


def suffix_family(index_windcode: pd.Series) -> pd.Series:
    text = index_windcode.fillna("").astype(str).str.upper()
    family = pd.Series("other", index=index_windcode.index, dtype="object")
    for suffix, label in [
        (".SH", "cn_exchange"),
        (".SZ", "cn_exchange"),
        (".CSI", "cn_index"),
        (".CNI", "cn_index"),
        (".CS", "cn_bond"),
        (".HI", "hk_index"),
        (".GI", "global_index"),
        (".FI", "third_party"),
        (".SPI", "third_party"),
        (".SGE", "commodity_index"),
        (".DCE", "commodity_index"),
        (".SHF", "commodity_index"),
        (".CZC", "commodity_index"),
        (".SG", "global_index"),
    ]:
        family = family.mask(text.str.endswith(suffix), label)
    return family


def infer_asset_bucket_hint(df: pd.DataFrame) -> pd.Series:
    text = (
        df.get("sector_names", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("short_name", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("index_name", pd.Series("", index=df.index)).fillna("").astype(str)
    ).str.upper()

    bucket = pd.Series("other", index=df.index, dtype="object")
    bucket = bucket.mask(text.str.contains("货币型ETF|货币ETF|快线|现金"), "money_market")
    bucket = bucket.mask(text.str.contains("债券型ETF|国债|政金债|信用债|科创债"), "bond")
    bucket = bucket.mask(text.str.contains("商品型ETF|黄金|上海金|石油|豆粕|有色|能源"), "commodity")
    bucket = bucket.mask(text.str.contains("跨境ETF|恒生|纳指|标普|港股|巴西|沙特"), "cross_border")
    bucket = bucket.mask(text.str.contains("股票型ETF|红利|低波|沪深300|中证A500|上证50|中证500"), "equity")
    return bucket


def assign_research_lane(df: pd.DataFrame) -> pd.Series:
    lane = pd.Series("review_needed", index=df.index, dtype="object")
    lane = lane.mask(df["is_money_market_etf"] & df["tracking_index_windcode"].isna(), "money_market_no_index_mapping")
    lane = lane.mask(df["coverage_status"].eq("complete"), "domestic_index_ready")
    lane = lane.mask(df["coverage_status"].eq("missing_latest_price"), "mapped_index_missing_price")
    lane = lane.mask(
        df["coverage_status"].eq("missing_both") & df["index_suffix_family"].isin(["hk_index", "global_index", "third_party", "commodity_index"]),
        "external_index_family_needs_extra_source",
    )
    lane = lane.mask(df["coverage_status"].eq("missing_both"), "mapped_index_missing_both")
    lane = lane.mask(df["tracking_index_windcode"].isna() & ~df["is_money_market_etf"], "unmapped_non_money_etf")
    return lane


def build_summary(base: pd.DataFrame) -> str:
    lines = [
        "# ETF Index Research Base Summary",
        "",
        f"- total_rows: `{len(base)}`",
        f"- mapped_rows: `{int(base['tracking_index_windcode'].notna().sum())}`",
        f"- unmapped_rows: `{int(base['tracking_index_windcode'].isna().sum())}`",
        "",
        "## Research Lane Counts",
        "",
    ]
    for lane, count in base["research_lane"].value_counts(dropna=False).items():
        lines.append(f"- {lane}: `{int(count)}`")
    lines.extend(["", "## Coverage Status Counts", ""])
    for status, count in base["coverage_status"].fillna("missing_mapping").value_counts(dropna=False).items():
        lines.append(f"- {status}: `{int(count)}`")
    lines.extend(["", "## Index Suffix Family Counts", ""])
    for family, count in base["index_suffix_family"].value_counts(dropna=False).items():
        lines.append(f"- {family}: `{int(count)}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    etf_master = pd.read_csv(args.etf_master)
    index_dir = Path(args.index_dir)
    bridge = pd.read_csv(index_dir / "etf_index_bridge_snapshot.csv")
    coverage = pd.read_csv(index_dir / "index_coverage_snapshot.csv")

    keep_etf_cols = [
        col
        for col in [
            "windcode",
            "code6",
            "short_name",
            "full_name",
            "sector_names",
            "setup_date",
            "aum_cny",
            "avg_turnover_20d_cny",
            "total_fee_bp",
            "tracking_error_1y_pct",
            "is_money_market_etf",
            "research_eligible",
        ]
        if col in etf_master.columns
    ]
    base = etf_master[keep_etf_cols].copy()
    if "is_money_market_etf" in base.columns:
        base["is_money_market_etf"] = normalize_bool(base["is_money_market_etf"])
    if "research_eligible" in base.columns:
        base["research_eligible"] = normalize_bool(base["research_eligible"])

    keep_bridge_cols = [
        col
        for col in [
            "windcode",
            "tracking_index_windcode",
            "entry_dt",
            "remove_dt",
            "mapping_status",
            "index_code",
            "index_short_name",
            "index_name",
            "index_exchange",
            "index_list_date",
            "index_publisher",
            "index_type_code",
            "source_table",
            "latest_trade_dt",
            "latest_close",
            "latest_pct_change",
            "latest_volume_hand",
            "latest_amount_cny",
            "latest_price_source_table",
            "price_obs_count",
        ]
        if col in bridge.columns
    ]
    bridge = bridge[keep_bridge_cols].copy().rename(columns={"source_table": "description_source_table"})
    coverage = coverage.copy()

    base = base.merge(bridge, on="windcode", how="left")
    if "index_windcode" in coverage.columns:
        coverage = coverage.rename(columns={"index_windcode": "tracking_index_windcode"})
    base = base.merge(
        coverage[
            [
                col
                for col in [
                    "tracking_index_windcode",
                    "coverage_status",
                    "has_description",
                    "has_latest_price",
                ]
                if col in coverage.columns
            ]
        ].drop_duplicates(subset=["tracking_index_windcode"]),
        on="tracking_index_windcode",
        how="left",
    )

    base["index_suffix_family"] = suffix_family(base["tracking_index_windcode"])
    base["asset_bucket_hint"] = infer_asset_bucket_hint(base)
    base["coverage_status"] = base["coverage_status"].fillna("missing_mapping")
    base["research_lane"] = assign_research_lane(base)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_csv, index=False)
    output_md.write_text(build_summary(base), encoding="utf-8")

    print(f"rows={len(base)}")
    print(f"output_csv={output_csv}")
    print(f"output_md={output_md}")


if __name__ == "__main__":
    main()
