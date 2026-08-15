from datetime import date, timedelta

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from modeling.pipelines.modeling.nodes import inference


def compute_predict_window(lookback_weeks: int) -> tuple[str, str, str]:
    """Narrow rolling window: enough history for lag_1/lag_2 plus the current week.

    Calls can only ever be fetched through today (next week hasn't happened yet), but
    events/weather need a separately *extended* end date reaching into next week —
    scheduled events and forecast weather for next week are legitimately knowable now.
    """
    today = date.today()
    end_date = today.isoformat()
    start_date = (today - timedelta(weeks=lookback_weeks)).isoformat()
    next_week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=1)
    forecast_end_date = (next_week_start + timedelta(days=6)).isoformat()
    return start_date, end_date, forecast_end_date


def build_next_week_features(
    features: pd.DataFrame, event_features: pd.DataFrame, weather_features: pd.DataFrame,
    modeling_data: pd.DataFrame, target_col: str,
) -> pd.DataFrame:
    """Shift the current (most recent) week's row one week forward.

    The model learned a calendar-agnostic mapping: lag_1 (last week's calls), lag_2 (two
    weeks back), events scheduled for the target week, and the weather forecast for the
    target week -> that week's actual calls. To predict *next* week we construct exactly
    that: lag_1 becomes this week's own calls, lag_2 becomes this week's former lag_1,
    and event/weather features are freshly joined at the real next-week date rather than
    reused from this week's row.
    """
    current_week = features[features["week_start"] == features["week_start"].max()].copy()
    next_week_start = current_week["week_start"].iloc[0] + timedelta(weeks=1)

    next_week = pd.DataFrame({
        "board_key": current_week["board_key"].to_numpy(),
        "week_start": next_week_start,
        "ft_week_of_year": pd.Timestamp(next_week_start).isocalendar().week,
        "ft_lag_1": np.log1p(current_week[target_col].to_numpy()),
        "ft_lag_2": current_week["ft_lag_1"].to_numpy(),
    })
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
