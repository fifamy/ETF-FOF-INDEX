#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from _bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ETF bulk download outputs for research usage.")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "output" / "etf_bulk_download"),
        help="Raw ETF bulk download directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "etf_bulk_download_research"),
        help="Cleaned research dataset directory.",
    )
    parser.add_argument(
        "--as-of-date",
        default="",
        help="Optional YYYYMMDD override. Defaults to max latest_trade_dt / trade_dt found in input.",
    )
    parser.add_argument(
        "--min-listed-days",
        type=int,
        default=365,
        help="Minimum listed days for research eligibility.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_date_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "None", "<NA>"]))
    cleaned = cleaned.str.extract(r"(\d{8})", expand=False)
    return cleaned


def to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(normalize_date_series(series), format="%Y%m%d", errors="coerce")


def infer_as_of_date(master: pd.DataFrame, prices: pd.DataFrame, explicit: str) -> str:
    if explicit:
        return explicit
    if not master.empty and "latest_trade_dt" in master.columns:
        latest = normalize_date_series(master["latest_trade_dt"]).dropna()
        if not latest.empty:
            return str(latest.max())
    if not prices.empty and "trade_dt" in prices.columns:
        latest = normalize_date_series(prices["trade_dt"]).dropna()
        if not latest.empty:
            return str(latest.max())
    raise ValueError("Unable to infer as_of_date from input files.")


def build_reason_columns(master: pd.DataFrame) -> pd.DataFrame:
    reason_cols = [
        "is_delisted_name",
        "is_matured",
        "missing_current_price",
        "missing_aum",
        "missing_tracking_non_money",
        "listed_days_lt_threshold",
    ]
    labels = {
        "is_delisted_name": "delisted_name",
        "is_matured": "matured",
        "missing_current_price": "missing_current_price",
        "missing_aum": "missing_aum",
        "missing_tracking_non_money": "missing_tracking_non_money",
        "listed_days_lt_threshold": "listed_days_lt_threshold",
    }

    def collect_reasons(row: pd.Series, columns: Iterable[str]) -> str:
        reasons = [labels[column] for column in columns if pd.notna(row[column]) and bool(row[column])]
        return "|".join(reasons)

    master["active_exclusion_reason"] = master.apply(
        lambda row: collect_reasons(row, ["is_delisted_name", "is_matured"]),
        axis=1,
    )
    master["research_exclusion_reason"] = master.apply(
        lambda row: collect_reasons(row, reason_cols),
        axis=1,
    )
    return master


def prepare_master(master: pd.DataFrame, as_of_date: str, min_listed_days: int) -> pd.DataFrame:
    out = master.copy()
    for column in [
        "short_name",
        "full_name",
        "sector_names",
        "latest_trade_dt",
        "nav_date",
        "tracking_date",
        "float_share_date",
        "maturity_date",
        "setup_date",
    ]:
        if column in out.columns:
            out[column] = normalize_date_series(out[column]) if column.endswith("_date") else out[column].astype("string")

    as_of_ts = pd.to_datetime(as_of_date, format="%Y%m%d")
    setup_dt = to_datetime_yyyymmdd(out["setup_date"]) if "setup_date" in out.columns else pd.Series(pd.NaT, index=out.index)
    maturity_dt = to_datetime_yyyymmdd(out["maturity_date"]) if "maturity_date" in out.columns else pd.Series(pd.NaT, index=out.index)

    short_name = out["short_name"].fillna("")
    full_name = out["full_name"].fillna("")
    sector_names = out["sector_names"].fillna("")
    latest_trade_dt = normalize_date_series(out["latest_trade_dt"]) if "latest_trade_dt" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    tracking_date = normalize_date_series(out["tracking_date"]) if "tracking_date" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    aum = pd.to_numeric(out["aum_cny"], errors="coerce") if "aum_cny" in out.columns else pd.Series(pd.NA, index=out.index, dtype="float64")

    out["is_money_market_etf"] = sector_names.str.contains("货币型ETF", na=False)
    out["is_delisted_name"] = short_name.str.contains("退市", na=False) | full_name.str.contains("退市", na=False)
    out["is_matured"] = maturity_dt.notna() & (maturity_dt < as_of_ts)
    out["is_active"] = ~(out["is_delisted_name"] | out["is_matured"])
    out["listed_days"] = (as_of_ts - setup_dt).dt.days
    out["listed_days_ok"] = out["listed_days"].fillna(-1) >= min_listed_days
    out["has_current_price"] = latest_trade_dt.eq(as_of_date)
    out["has_aum"] = aum.gt(0)
    out["tracking_available"] = tracking_date.notna()
    out["tracking_required"] = ~out["is_money_market_etf"]
    out["tracking_ok"] = out["tracking_available"] | out["is_money_market_etf"]
    out["missing_current_price"] = ~out["has_current_price"]
    out["missing_aum"] = ~out["has_aum"]
    out["missing_tracking_non_money"] = out["tracking_required"] & ~out["tracking_available"]
    out["listed_days_lt_threshold"] = ~out["listed_days_ok"]
    out["research_eligible"] = (
        out["is_active"]
        & out["has_current_price"]
        & out["has_aum"]
        & out["tracking_ok"]
        & out["listed_days_ok"]
    )
    return build_reason_columns(out)


def filter_by_keys(df: pd.DataFrame, column: str, values: set[str]) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df.copy()
    return df[df[column].astype("string").isin(values)].copy()


def coalesce_duplicate_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base in ["avg_turnover_20d_cny", "avg_turnover_60d_cny", "avg_volume_20d_hand"]:
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        if x_col in out.columns or y_col in out.columns:
            combined = pd.Series(pd.NA, index=out.index, dtype="object")
            if x_col in out.columns:
                combined = pd.to_numeric(out[x_col], errors="coerce")
            if y_col in out.columns:
                y_vals = pd.to_numeric(out[y_col], errors="coerce")
                if x_col in out.columns:
                    combined = combined.fillna(y_vals)
                else:
                    combined = y_vals
            out[base] = combined
            out = out.drop(columns=[col for col in [x_col, y_col] if col in out.columns])
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def write_summary(
    path: Path,
    as_of_date: str,
    min_listed_days: int,
    master_all: pd.DataFrame,
    active_master: pd.DataFrame,
    research_master: pd.DataFrame,
) -> None:
    lines = [
        "# ETF Research Dataset Cleaning Summary",
        "",
        f"- as_of_date: `{as_of_date}`",
        f"- min_listed_days: `{min_listed_days}`",
        f"- raw_master_count: `{len(master_all)}`",
        f"- active_count: `{len(active_master)}`",
        f"- research_eligible_count: `{len(research_master)}`",
        f"- money_market_count_in_research: `{int(research_master['is_money_market_etf'].sum())}`",
        "",
        "## Exclusion Counts",
        "",
    ]
    for column in [
        "is_delisted_name",
        "is_matured",
        "missing_current_price",
        "missing_aum",
        "missing_tracking_non_money",
        "listed_days_lt_threshold",
    ]:
        lines.append(f"- {column}: `{int(master_all[column].sum())}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Money market ETFs are allowed to have missing tracking metrics.",
            "- Delisted / matured ETFs are removed from the research dataset.",
            "- Research dataset also removes ETFs without current price, without AUM, or listed for less than the threshold.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    universe = load_csv(input_dir / "etf_universe.csv")
    code_mapping = load_csv(input_dir / "etf_code_mapping.csv")
    latest_nav = load_csv(input_dir / "etf_latest_nav.csv")
    latest_float_share = load_csv(input_dir / "etf_latest_float_share.csv")
    latest_tracking = load_csv(input_dir / "etf_latest_tracking.csv")
    master = load_csv(input_dir / "etf_master_snapshot.csv")
    sector_membership = load_csv(input_dir / "etf_sector_membership.csv")
    daily_prices = load_csv(input_dir / "etf_daily_prices.csv.gz")
    daily_money_flow = load_csv(input_dir / "etf_daily_money_flow.csv.gz")

    master = coalesce_duplicate_metric_columns(master)

    as_of_date = infer_as_of_date(master, daily_prices, args.as_of_date)
    master_flags = prepare_master(master, as_of_date=as_of_date, min_listed_days=args.min_listed_days)

    active_master = master_flags[master_flags["is_active"]].copy()
    research_master = master_flags[master_flags["research_eligible"]].copy()

    research_windcodes = set(research_master["windcode"].astype("string"))
    research_code6 = set(research_master["code6"].astype("string"))

    universe_research = filter_by_keys(universe, "windcode", research_windcodes)
    code_mapping_research = filter_by_keys(code_mapping, "code6", research_code6)
    latest_nav_research = filter_by_keys(latest_nav, "code6", research_code6)
    latest_float_share_research = filter_by_keys(latest_float_share, "windcode", research_windcodes)
    latest_tracking_research = filter_by_keys(latest_tracking, "code6", research_code6)
    sector_membership_research = filter_by_keys(sector_membership, "windcode", research_windcodes)
    daily_prices_research = filter_by_keys(daily_prices, "windcode", research_windcodes)
    daily_money_flow_research = filter_by_keys(daily_money_flow, "windcode", research_windcodes)

    exclusions = master_flags[~master_flags["research_eligible"]].copy()

    write_csv(master_flags, output_dir / "etf_master_with_flags.csv")
    write_csv(active_master, output_dir / "etf_master_active_snapshot.csv")
    write_csv(research_master, output_dir / "etf_master_snapshot.csv")
    write_csv(exclusions, output_dir / "etf_research_exclusions.csv")
    write_csv(universe_research, output_dir / "etf_universe.csv")
    write_csv(code_mapping_research, output_dir / "etf_code_mapping.csv")
    write_csv(latest_nav_research, output_dir / "etf_latest_nav.csv")
    write_csv(latest_float_share_research, output_dir / "etf_latest_float_share.csv")
    write_csv(latest_tracking_research, output_dir / "etf_latest_tracking.csv")
    write_csv(sector_membership_research, output_dir / "etf_sector_membership.csv")
    write_csv(daily_prices_research, output_dir / "etf_daily_prices.csv.gz")
    write_csv(daily_money_flow_research, output_dir / "etf_daily_money_flow.csv.gz")
    write_summary(
        output_dir / "cleaning_summary.md",
        as_of_date=as_of_date,
        min_listed_days=args.min_listed_days,
        master_all=master_flags,
        active_master=active_master,
        research_master=research_master,
    )

    print(f"as_of_date={as_of_date}")
    print(f"raw_count={len(master_flags)}")
    print(f"active_count={len(active_master)}")
    print(f"research_eligible_count={len(research_master)}")
    print(f"money_market_in_research={int(research_master['is_money_market_etf'].sum())}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
