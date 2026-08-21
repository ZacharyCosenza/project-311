from datetime import date, timedelta

import numpy as np
import pandas as pd
import shap

from modeling.pipelines.features.nodes import group_feature_cols
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
    modeling_data: pd.DataFrame, complaint_type_groups: dict, max_lag_weeks: int, year_offset_weeks: int,
) -> pd.DataFrame:
    """Look up each group's lag features directly from that group's own historical
    calls — the "_ly" (last-year) lags in particular reach back further than this
    narrow inference fetch covers. Recent weeks come from the freshly fetched
    `features` (more current than modeling_data might be); anything further back falls
    back to modeling_data, which spans years. A lookup with no match (e.g. before a
    board's earliest history) comes back NaN — left as-is, matching training: XGBoost
    handles missing values natively.
    """
    current_week = features[features["week_start"] == features["week_start"].max()].copy()
    next_week_start = current_week["week_start"].iloc[0] + timedelta(weeks=1)
    boards = current_week["board_key"].to_numpy()

    next_week = pd.DataFrame({"board_key": boards, "week_start": next_week_start})
    next_week["ft_week_of_year"] = pd.Timestamp(next_week_start).isocalendar().week

    for group in [*complaint_type_groups.keys(), "other"]:
        target_col = f"tgt_{group}"
        history = (
            pd.concat([
                modeling_data[["board_key", "week_start", target_col]],
                features[["board_key", "week_start", target_col]],
            ])
            .drop_duplicates(subset=["board_key", "week_start"], keep="last")
            .set_index(["board_key", "week_start"])[target_col]
        )

        def lag_lookup(weeks_back: int) -> np.ndarray:
            keys = list(zip(boards, [next_week_start - timedelta(weeks=weeks_back)] * len(boards)))
            return np.log1p(history.reindex(keys).to_numpy())

        for lag in range(1, max_lag_weeks + 1):
            next_week[f"ft_lag_{lag}_{group}"] = lag_lookup(lag)
        for lag in range(1, max_lag_weeks + 1):
            next_week[f"ft_lag_{lag}_ly_{group}"] = lag_lookup(year_offset_weeks + lag)

    # Category set must match what the models were trained on, not just whatever boards
    # happen to appear in this narrow window — a board with no recent calls would
    # otherwise be silently dropped from the categories, shifting XGBoost's categorical
    # codes out of alignment with what the models actually learned.
    board_categories = sorted(modeling_data["board_key"].unique())
    next_week["ft_board_key"] = pd.Categorical(next_week["board_key"], categories=board_categories)

    next_week = next_week.merge(event_features, on=["board_key", "week_start"], how="left")
    next_week["ft_event_count"] = next_week["ft_event_count"].fillna(0)
    next_week = next_week.merge(weather_features, on="week_start", how="left")
    return next_week


def _winsorize_isolated_outliers(
    df: pd.DataFrame, value_col: str, z_threshold: float, min_corroborators: int,
) -> pd.DataFrame:
    """Cap a board-week to that board's own historical median when it's an extreme
    outlier (modified z-score) AND isolated (no other board elevated the same week).
    Magnitude alone can't tell a data artifact from a real citywide event — both can
    show one category dominating a board's total — but only the artifact is isolated;
    see docs/delta-eda for the two real cases this was tuned against.
    """
    df = df.copy()
    df[value_col] = df[value_col].astype(float)  # median/winsorized values are never whole call counts
    median = df.groupby("board_key")[value_col].transform("median")
    mad = df.groupby("board_key")[value_col].transform(lambda s: (s - s.median()).abs().median())
    mad = mad.where(mad > 0, df.groupby("board_key")[value_col].transform("std")).fillna(1.0)
    modified_z = 0.6745 * (df[value_col] - median) / mad

    elevated = modified_z > 3
    corroborators = elevated.groupby(df["week_start"]).transform("sum").astype(int) - elevated.astype(int)
    is_artifact = (modified_z > z_threshold) & (corroborators < min_corroborators)

    df.loc[is_artifact, value_col] = median[is_artifact]
    return df


def compute_call_deltas(
    ranked_districts: pd.DataFrame, modeling_data: pd.DataFrame, target_col: str,
    delta_baseline_weeks: int, outlier_z_threshold: float, outlier_min_corroborators: int,
) -> pd.DataFrame:
    """delta_{target_col} = prediction minus each board's own trailing
    delta_baseline_weeks average of actual calls (4 weeks — the lowest-noise choice
    from a sensitivity sweep, see docs/delta-eda). delta_rank only ranks positive
    deltas — a predicted decrease gets no rank at all, never a top-5 slot.
    """
    pred_col = f"pred_{target_col}"
    delta_col = f"delta_{target_col}"

    history = _winsorize_isolated_outliers(
        modeling_data[["board_key", "week_start", target_col]], target_col,
        outlier_z_threshold, outlier_min_corroborators,
    )
    baseline = (
        history.sort_values("week_start")
        .groupby("board_key")[target_col]
        .apply(lambda s: s.tail(delta_baseline_weeks).mean())
    )

    result = ranked_districts.copy()
    result[delta_col] = result[pred_col] - result["board_key"].map(baseline)

    positive = result[delta_col] > 0
    result["delta_rank"] = np.nan
    result.loc[positive, "delta_rank"] = result.loc[positive, delta_col].rank(ascending=False, method="first")
    return result


def rank_districts(
    models: dict, next_week_features: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, max_lag_weeks: int, target_col: str,
) -> pd.DataFrame:
    """Predictions + rank for every district (not just a top-k slice) — one prediction
    per group, summed into pred_{target_col} so everything downstream (tweet copy, the
    district map) sees the same single total-calls number it always did, regardless of
    how many models are actually behind it.
    """
    pred_col = f"pred_{target_col}"
    scored = next_week_features.copy()
    scored[pred_col] = 0.0
    for group, model in models.items():
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        group_scored = inference(model, scored, feature_cols, f"tgt_{group}")
        scored[pred_col] += group_scored[f"pred_tgt_{group}"]
    scored = scored.sort_values(pred_col, ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))
    return scored


def add_shap_reasons(
    models: dict, ranked_districts: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, max_lag_weeks: int, top_reasons: int, reason_map: dict,
) -> pd.DataFrame:
    """Top-N *distinct* reasons per district, by pooled SHAP contribution strength
    across every group's model — no single model sees the whole picture. Feature names
    are stripped back to their un-suffixed form (ft_lag_1_noise -> ft_lag_1) and summed
    before pooling, since reason_map already treats every group's ft_lag_1 identically
    — this avoids needing 10x the reason_map entries.
    """
    contributions = []
    for group, model in models.items():
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(ranked_districts[feature_cols], check_additivity=False)
        shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=ranked_districts.index)
        suffix = f"_{group}"
        shap_df.columns = [c[: -len(suffix)] if c.endswith(suffix) else c for c in shap_df.columns]
        contributions.append(shap_df)

    pooled = pd.concat(contributions, axis=1)
    pooled = pooled.T.groupby(level=0).sum().T

    def reasons_for(idx):
        row = pooled.loc[idx]
        positive = row[row > 0].sort_values(ascending=False)
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
