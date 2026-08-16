from datetime import date, timedelta

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from modeling.pipelines.modeling.nodes import inference


def compute_predict_window(lookback_weeks: int) -> tuple[str, str, str]:
    """Narrow rolling window covering the current week, freshly fetched.

    Calls can only ever be fetched through today (next week hasn't happened yet), but
    events/weather need a separately *extended* end date reaching into next week —
    scheduled events and forecast weather for next week are legitimately knowable now.
    Deeper lag history (beyond this narrow window) comes from modeling_data instead —
    see build_next_week_features.
    """
    today = date.today()
    end_date = today.isoformat()
    start_date = (today - timedelta(weeks=lookback_weeks)).isoformat()
    next_week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=1)
    forecast_end_date = (next_week_start + timedelta(days=6)).isoformat()
    return start_date, end_date, forecast_end_date


def build_next_week_features(
    features: pd.DataFrame, event_features: pd.DataFrame, weather_features: pd.DataFrame,
    modeling_data: pd.DataFrame, target_col: str, max_lag_weeks: int, year_offset_weeks: int,
) -> pd.DataFrame:
    """Look up each lag feature directly from historical call counts — the "_ly"
    (last-year) lags in particular reach back further than this narrow inference fetch
    covers. Recent weeks come from the freshly fetched `features` (more current than
    modeling_data might be); anything further back falls back to modeling_data, which
    spans years. A lookup with no match (e.g. before a board's earliest history) comes
    back NaN — left as-is, matching training: XGBoost handles missing values natively.
    """
    current_week = features[features["week_start"] == features["week_start"].max()].copy()
    next_week_start = current_week["week_start"].iloc[0] + timedelta(weeks=1)
    boards = current_week["board_key"].to_numpy()

    history = (
        pd.concat([modeling_data[["board_key", "week_start", target_col]], features[["board_key", "week_start", target_col]]])
        .drop_duplicates(subset=["board_key", "week_start"], keep="last")
        .set_index(["board_key", "week_start"])[target_col]
    )

    def lag_lookup(weeks_back: int) -> np.ndarray:
        keys = list(zip(boards, [next_week_start - timedelta(weeks=weeks_back)] * len(boards)))
        return np.log1p(history.reindex(keys).to_numpy())

    next_week = pd.DataFrame({"board_key": boards, "week_start": next_week_start})
    next_week["ft_week_of_year"] = pd.Timestamp(next_week_start).isocalendar().week
    for lag in range(1, max_lag_weeks + 1):
        next_week[f"ft_lag_{lag}"] = lag_lookup(lag)
    for lag in range(1, max_lag_weeks + 1):
        next_week[f"ft_lag_{lag}_ly"] = lag_lookup(year_offset_weeks + lag)

    # Category set must match what the model was trained on, not just whatever boards
    # happen to appear in this narrow window — a board with no recent calls would
    # otherwise be silently dropped from the categories, shifting XGBoost's categorical
    # codes out of alignment with what the model actually learned.
    board_categories = sorted(modeling_data["board_key"].unique())
    next_week["ft_board_key"] = pd.Categorical(next_week["board_key"], categories=board_categories)

    next_week = next_week.merge(event_features, on=["board_key", "week_start"], how="left")
    next_week["ft_event_count"] = next_week["ft_event_count"].fillna(0)
    next_week = next_week.merge(weather_features, on="week_start", how="left")
    return next_week


def rank_districts(
    model: XGBRegressor, next_week_features: pd.DataFrame, feature_cols: list, target_col: str,
) -> pd.DataFrame:
    """Predictions + rank for every district (not just a top-k slice) — features stay
    attached so this one table is the complete, self-contained inference result."""
    pred_col = f"pred_{target_col}"
    scored = inference(model, next_week_features, feature_cols, target_col)
    scored = scored.sort_values(pred_col, ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))
    return scored


def add_shap_reasons(
    model: XGBRegressor, ranked_districts: pd.DataFrame, feature_cols: list, top_reasons: int, reason_map: dict,
) -> pd.DataFrame:
    """Top-N *distinct* reasons per district, by SHAP contribution strength. Features
    that map to the same reason (e.g. both lag features -> "recent historical patterns")
    count once toward the cap, so the same information is never repeated."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(ranked_districts[feature_cols], check_additivity=False)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=ranked_districts.index)

    def reasons_for(idx):
        contributions = shap_df.loc[idx]
        positive = contributions[contributions > 0].sort_values(ascending=False)
        reasons = []
        for feature in positive.index:
            reason = reason_map.get(feature, feature)
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) == top_reasons:
                break
        return reasons

    result = ranked_districts.copy()
    result["reasons"] = [reasons_for(idx) for idx in ranked_districts.index]
    return result
