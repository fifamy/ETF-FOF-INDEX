from pathlib import Path

import pandas as pd

from etf_fof_index.scoring import load_scoring_config, score_candidate_pool


def test_score_candidate_pool_ranks_and_filters() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_scoring_config(repo_root / "config" / "asset_pool_scoring_v1.yaml")

    frame = pd.DataFrame(
        [
            {
                "bucket": "equity_core",
                "benchmark_group": "CSI300",
                "symbol": "510300.SH",
                "exchange": "SSE",
                "listed_date": "2012-05-28",
                "is_etf": 1,
                "is_qdii": 0,
                "is_leveraged_inverse": 0,
                "avg_turnover_1m_cny": 1_200_000_000,
                "aum_cny": 350_000_000_000,
                "total_fee_bp": 50,
                "tracking_error_1y_pct": 0.20,
                "tracking_deviation_daily_pct": 0.05,
                "tracking_proxy_score": 95,
                "has_primary_market_maker": 1,
                "has_option_underlying": 1,
                "bucket_fit_score": 95,
                "structure_score": 95,
            },
            {
                "bucket": "equity_core",
                "benchmark_group": "CSI300",
                "symbol": "159919.SZ",
                "exchange": "SZSE",
                "listed_date": "2012-05-07",
                "is_etf": 1,
                "is_qdii": 0,
                "is_leveraged_inverse": 0,
                "avg_turnover_1m_cny": 600_000_000,
                "aum_cny": 180_000_000_000,
                "total_fee_bp": 50,
                "tracking_error_1y_pct": 0.28,
                "tracking_deviation_daily_pct": 0.07,
                "tracking_proxy_score": 90,
                "has_primary_market_maker": 0,
                "has_option_underlying": 1,
                "bucket_fit_score": 95,
                "structure_score": 90,
            },
            {
                "bucket": "equity_core",
                "benchmark_group": "CSI_A500",
                "symbol": "563220.SH",
                "exchange": "SSE",
                "listed_date": "2024-10-15",
                "is_etf": 1,
                "is_qdii": 0,
                "is_leveraged_inverse": 0,
                "avg_turnover_1m_cny": 20_000_000,
                "aum_cny": 8_000_000_000,
                "total_fee_bp": 15,
                "tracking_error_1y_pct": 0.35,
                "tracking_deviation_daily_pct": 0.10,
                "tracking_proxy_score": 75,
                "has_primary_market_maker": 1,
                "has_option_underlying": 0,
                "bucket_fit_score": 85,
                "structure_score": 75,
            },
        ]
    )

    scored = score_candidate_pool(frame, config)
    passed = scored[scored["hard_filter_pass"]].set_index("symbol")
    failed = scored[~scored["hard_filter_pass"]].set_index("symbol")

    assert "510300.SH" in passed.index
    assert "159919.SZ" in passed.index
    assert "563220.SH" in failed.index
    assert "turnover_below_threshold" in failed.loc["563220.SH", "hard_filter_reasons"]
    assert passed.loc["510300.SH", "final_score"] > passed.loc["159919.SZ", "final_score"]
    assert int(passed.loc["510300.SH", "group_rank"]) == 1

