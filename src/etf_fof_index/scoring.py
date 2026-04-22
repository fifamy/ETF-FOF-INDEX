from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "bucket",
    "symbol",
    "exchange",
    "listed_date",
    "is_etf",
    "is_qdii",
    "is_leveraged_inverse",
    "bucket_fit_score",
}


def load_scoring_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_candidate_pool(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Candidate pool missing columns: {sorted(missing)}")
    return frame


def _to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _linear_higher_better(value: float, lower: float, upper: float) -> float:
    if pd.isna(value):
        return math.nan
    if upper <= lower:
        return 100.0
    scaled = (value - lower) / (upper - lower)
    return float(max(0.0, min(1.0, scaled)) * 100.0)


def _linear_lower_better(value: float, best: float, worst: float) -> float:
    if pd.isna(value):
        return math.nan
    if worst <= best:
        return 100.0
    scaled = (worst - value) / (worst - best)
    return float(max(0.0, min(1.0, scaled)) * 100.0)


def _log_higher_better(value: float, lower: float, upper: float) -> float:
    if pd.isna(value):
        return math.nan
    if value <= 0 or lower <= 0 or upper <= lower:
        return math.nan
    scaled = (math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower))
    return float(max(0.0, min(1.0, scaled)) * 100.0)


def _tracking_score(row: pd.Series, anchors: Dict[str, Any]) -> float:
    components: List[float] = []
    te = row.get("tracking_error_1y_pct")
    td = row.get("tracking_deviation_daily_pct")
    proxy = row.get("tracking_proxy_score")

    if not pd.isna(te):
        params = anchors["tracking_error_1y_pct"]
        components.append(_linear_lower_better(float(te), float(params["best"]), float(params["worst"])))
    if not pd.isna(td):
        params = anchors["tracking_deviation_daily_pct"]
        components.append(_linear_lower_better(float(td), float(params["best"]), float(params["worst"])))
    if components:
        return float(sum(components) / len(components))
    if not pd.isna(proxy):
        return float(proxy)
    return math.nan


def _liquidity_score(row: pd.Series, anchors: Dict[str, Any]) -> float:
    turnover_anchor = anchors["avg_turnover_1m_cny"]
    turnover_score = _log_higher_better(
        float(row.get("avg_turnover_1m_cny")),
        float(turnover_anchor["min"]),
        float(turnover_anchor["target"]),
    )
    support_score = 0.0
    if _to_bool(row.get("has_primary_market_maker")):
        support_score += 50.0
    if _to_bool(row.get("has_option_underlying")):
        support_score += 50.0

    if pd.isna(turnover_score):
        return support_score if support_score > 0 else math.nan
    return float(turnover_score * 0.8 + support_score * 0.2)


def _scale_score(row: pd.Series, anchors: Dict[str, Any]) -> float:
    params = anchors["aum_cny"]
    return _log_higher_better(float(row.get("aum_cny")), float(params["min"]), float(params["target"]))


def _fee_score(row: pd.Series, anchors: Dict[str, Any]) -> float:
    params = anchors["total_fee_bp"]
    return _linear_lower_better(float(row.get("total_fee_bp")), float(params["best"]), float(params["worst"]))


def _tenure_score(row: pd.Series, anchors: Dict[str, Any], as_of_date: pd.Timestamp) -> float:
    listed_date = pd.to_datetime(row.get("listed_date"), errors="coerce")
    if pd.isna(listed_date):
        return math.nan
    listed_days = float((as_of_date - listed_date).days)
    params = anchors["listed_days"]
    return _linear_higher_better(listed_days, float(params["min"]), float(params["target"]))


def _hard_filter_reasons(row: pd.Series, config: Dict[str, Any], as_of_date: pd.Timestamp) -> List[str]:
    filters = config["hard_filters"]
    reasons: List[str] = []

    exchange = str(row.get("exchange", "")).strip().upper()
    if exchange not in {item.upper() for item in filters["exchanges"]}:
        reasons.append("exchange_not_allowed")
    if bool(filters["require_is_etf"]) and not _to_bool(row.get("is_etf")):
        reasons.append("not_etf")
    if bool(filters["exclude_qdii"]) and _to_bool(row.get("is_qdii")):
        reasons.append("is_qdii")
    if bool(filters["exclude_leveraged_inverse"]) and _to_bool(row.get("is_leveraged_inverse")):
        reasons.append("is_leveraged_inverse")

    listed_date = pd.to_datetime(row.get("listed_date"), errors="coerce")
    if pd.isna(listed_date):
        reasons.append("missing_listed_date")
    else:
        listed_days = (as_of_date - listed_date).days
        if listed_days < int(filters["min_listed_days"]):
            reasons.append("listed_days_below_threshold")

    turnover = row.get("avg_turnover_1m_cny")
    if pd.isna(turnover):
        reasons.append("missing_turnover")
    elif float(turnover) < float(filters["min_avg_turnover_1m_cny"]):
        reasons.append("turnover_below_threshold")

    aum = row.get("aum_cny")
    if pd.isna(aum):
        reasons.append("missing_aum")
    elif float(aum) < float(filters["min_aum_cny"]):
        reasons.append("aum_below_threshold")

    fit = row.get("bucket_fit_score")
    if pd.isna(fit):
        reasons.append("missing_bucket_fit_score")
    elif float(fit) < float(filters["min_bucket_fit_score"]):
        reasons.append("bucket_fit_below_threshold")

    return reasons


def score_candidate_pool(frame: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    working = frame.copy()
    working["listed_date"] = pd.to_datetime(working["listed_date"], errors="coerce")
    as_of_date = pd.Timestamp(config["as_of_date"])
    anchors = config["anchors"]

    working["listed_days"] = (as_of_date - working["listed_date"]).dt.days
    working["liquidity_score"] = working.apply(lambda row: _liquidity_score(row, anchors), axis=1)
    working["scale_score"] = working.apply(lambda row: _scale_score(row, anchors), axis=1)
    working["fee_score"] = working.apply(lambda row: _fee_score(row, anchors), axis=1)
    working["tracking_score"] = working.apply(lambda row: _tracking_score(row, anchors), axis=1)
    working["tenure_score"] = working.apply(lambda row: _tenure_score(row, anchors, as_of_date), axis=1)
    working["bucket_fit_score"] = pd.to_numeric(working["bucket_fit_score"], errors="coerce")
    working["structure_score"] = pd.to_numeric(working.get("structure_score"), errors="coerce")

    reasons: List[str] = []
    passes: List[bool] = []
    final_scores: List[float] = []
    metric_coverage: List[float] = []

    for _, row in working.iterrows():
        row_reasons = _hard_filter_reasons(row, config, as_of_date)
        passes.append(len(row_reasons) == 0)
        reasons.append("|".join(row_reasons))

        bucket = row["bucket"]
        bucket_weights = config["weights"].get(bucket)
        if bucket_weights is None:
            raise ValueError(f"No scoring weights configured for bucket '{bucket}'.")

        total = 0.0
        used_weight = 0.0
        for metric, weight in bucket_weights.items():
            weight_value = float(weight)
            metric_value = row.get(metric)
            if weight_value <= 0 or pd.isna(metric_value):
                continue
            total += float(metric_value) * weight_value
            used_weight += weight_value

        if used_weight > 0:
            final_scores.append(total / used_weight)
            metric_coverage.append(used_weight)
        else:
            final_scores.append(math.nan)
            metric_coverage.append(0.0)

    working["hard_filter_pass"] = passes
    working["hard_filter_reasons"] = reasons
    working["score_metric_weight_coverage"] = metric_coverage
    working["final_score"] = final_scores

    if "benchmark_group" in working.columns:
        working["group_rank"] = (
            working[working["hard_filter_pass"]]
            .groupby(["bucket", "benchmark_group"])["final_score"]
            .rank(ascending=False, method="dense")
        )
    else:
        working["group_rank"] = math.nan

    working["bucket_rank"] = (
        working[working["hard_filter_pass"]]
        .groupby("bucket")["final_score"]
        .rank(ascending=False, method="dense")
    )
    return working.sort_values(["bucket", "hard_filter_pass", "final_score"], ascending=[True, False, False])

