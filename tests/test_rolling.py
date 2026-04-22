import numpy as np
import pandas as pd

from etf_fof_index.rolling import build_quarterly_rolling_target_weights, quarterly_rebalance_dates, rank_candidates


def test_quarterly_rebalance_dates_filters_to_quarter_end() -> None:
    signal_dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-09-29",
            "2020-06-30",
        ]
    )

    actual = quarterly_rebalance_dates(pd.DatetimeIndex(signal_dates))

    assert list(actual) == list(pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-29"]))


def test_rank_candidates_prefers_shallower_drawdown_then_return() -> None:
    metrics = pd.DataFrame(
        [
            {"annual_return": 0.08, "annual_volatility": 0.10, "max_drawdown": -0.08, "sharpe": 0.80, "calmar": 1.00},
            {"annual_return": 0.12, "annual_volatility": 0.12, "max_drawdown": -0.08, "sharpe": 1.00, "calmar": 1.20},
            {"annual_return": 0.20, "annual_volatility": 0.18, "max_drawdown": -0.12, "sharpe": 1.10, "calmar": 1.30},
        ]
    )

    ranked = rank_candidates(metrics, selection_rule="min_drawdown")

    assert int(ranked.iloc[0]["candidate_index"]) == 1
    assert int(ranked.iloc[1]["candidate_index"]) == 0
    assert int(ranked.iloc[2]["candidate_index"]) == 2


def test_rank_candidates_max_return_prefers_higher_return_then_quality() -> None:
    metrics = pd.DataFrame(
        [
            {"annual_return": 0.10, "annual_volatility": 0.08, "max_drawdown": -0.06, "sharpe": 1.25, "calmar": 1.67},
            {"annual_return": 0.15, "annual_volatility": 0.11, "max_drawdown": -0.09, "sharpe": 1.36, "calmar": 1.67},
            {"annual_return": 0.15, "annual_volatility": 0.10, "max_drawdown": -0.08, "sharpe": 1.50, "calmar": 1.88},
        ]
    )

    ranked = rank_candidates(metrics, selection_rule="max_return")

    assert int(ranked.iloc[0]["candidate_index"]) == 2
    assert int(ranked.iloc[1]["candidate_index"]) == 1
    assert int(ranked.iloc[2]["candidate_index"]) == 0


def test_rank_candidates_sharpe_guard_prefers_higher_sharpe_within_drawdown_band() -> None:
    metrics = pd.DataFrame(
        [
            {"annual_return": 0.10, "annual_volatility": 0.08, "max_drawdown": -0.050, "sharpe": 1.25, "calmar": 2.00},
            {"annual_return": 0.12, "annual_volatility": 0.09, "max_drawdown": -0.058, "sharpe": 1.60, "calmar": 2.10},
            {"annual_return": 0.20, "annual_volatility": 0.15, "max_drawdown": -0.090, "sharpe": 1.80, "calmar": 2.20},
        ]
    )

    ranked = rank_candidates(metrics, selection_rule="sharpe_guard", drawdown_band=0.01)

    assert int(ranked.iloc[0]["candidate_index"]) == 1
    assert int(ranked.iloc[1]["candidate_index"]) == 0


def test_rank_candidates_calmar_guard_prefers_higher_calmar_within_drawdown_band() -> None:
    metrics = pd.DataFrame(
        [
            {"annual_return": 0.09, "annual_volatility": 0.07, "max_drawdown": -0.050, "sharpe": 1.30, "calmar": 1.80},
            {"annual_return": 0.11, "annual_volatility": 0.09, "max_drawdown": -0.057, "sharpe": 1.20, "calmar": 2.10},
            {"annual_return": 0.16, "annual_volatility": 0.14, "max_drawdown": -0.090, "sharpe": 1.10, "calmar": 2.50},
        ]
    )

    ranked = rank_candidates(metrics, selection_rule="calmar_guard", drawdown_band=0.01)

    assert int(ranked.iloc[0]["candidate_index"]) == 1
    assert int(ranked.iloc[1]["candidate_index"]) == 0


def test_build_quarterly_rolling_target_weights_assigns_selected_candidate_until_next_quarter() -> None:
    signal_dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
            "2020-05-31",
            "2020-06-30",
        ]
    )
    run_index = pd.date_range("2020-01-01", "2020-06-30", freq="D")
    n_days = len(run_index)
    n_cfg = 2

    ret_store = np.zeros((n_days, n_cfg), dtype=float)
    turnover_store = np.zeros((n_days, n_cfg), dtype=float)
    ret_store[(run_index >= "2020-02-01") & (run_index <= "2020-03-31"), 0] = 0.0010
    ret_store[(run_index >= "2020-02-01") & (run_index <= "2020-03-31"), 1] = -0.0030
    ret_store[(run_index >= "2020-04-01") & (run_index <= "2020-06-30"), 0] = -0.0020
    ret_store[(run_index >= "2020-04-01") & (run_index <= "2020-06-30"), 1] = 0.0010

    target_tensor = np.array(
        [
            [[0.80, 0.20], [0.20, 0.80]],
            [[0.80, 0.20], [0.20, 0.80]],
            [[0.80, 0.20], [0.20, 0.80]],
            [[0.80, 0.20], [0.20, 0.80]],
            [[0.80, 0.20], [0.20, 0.80]],
            [[0.80, 0.20], [0.20, 0.80]],
        ],
        dtype=float,
    )
    base_weight_frame = pd.DataFrame(
        [
            {"asset_a": 0.80, "asset_b": 0.20},
            {"asset_a": 0.20, "asset_b": 0.80},
        ]
    )

    result = build_quarterly_rolling_target_weights(
        signal_dates=pd.DatetimeIndex(signal_dates),
        run_index=pd.DatetimeIndex(run_index),
        ret_store=ret_store,
        turnover_store=turnover_store,
        target_tensor=target_tensor,
        base_weight_frame=base_weight_frame,
        buckets=["asset_a", "asset_b"],
        lookback_months=2,
    )

    assert list(result.target_weights.index) == list(pd.to_datetime(["2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30"]))
    assert (result.target_weights.loc["2020-03-31", ["asset_a", "asset_b"]].to_numpy() == np.array([0.80, 0.20])).all()
    assert (result.target_weights.loc["2020-04-30", ["asset_a", "asset_b"]].to_numpy() == np.array([0.80, 0.20])).all()
    assert (result.target_weights.loc["2020-06-30", ["asset_a", "asset_b"]].to_numpy() == np.array([0.20, 0.80])).all()
    assert int(result.decision_table.iloc[0]["selected_candidate_index"]) == 0
    assert int(result.decision_table.iloc[1]["selected_candidate_index"]) == 1
    assert result.decision_table["selection_rule"].eq("min_drawdown").all()
