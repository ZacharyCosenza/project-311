import os
from datetime import date, timedelta

import numpy as np
import pytest

from modeling.pipelines.features.nodes import (
    drop_incomplete_rows,
    featurize_events,
    featurize_lags,
    featurize_weather,
    join_features,
)
from modeling.pipelines.modeling.nodes import compute_metrics, training
from modeling.pipelines.raw.nodes import fetch_calls_weekly, fetch_events_weekly, fetch_weather_weekly
from modeling.pipelines.target.nodes import build_target

CALLS_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
EVENTS_URL = "https://data.cityofnewyork.us/resource/bkfu-528j.json"
EVENT_INCLUDE_TYPES = [
    "Parade", "Street Festival", "Single Block Festival", "Block Party",
    "Farmers Market", "Street Event", "Religious Event", "Plaza Event",
    "Plaza Partner Event", "Athletic Race / Tour", "Open Street Partner Event",
    "Health Fair", "Sidewalk Sale",
]
TARGET_COL = "tgt_calls"
SPLIT_COL = "split"
STRATIFY_COL = "week_start"
FEATURE_COLS = [
    "ft_week_of_year",
    "ft_lag_1", "ft_lag_2", "ft_lag_3", "ft_lag_4", "ft_lag_5", "ft_lag_6",
    "ft_lag_1_ly", "ft_lag_2_ly", "ft_lag_3_ly", "ft_lag_4_ly", "ft_lag_5_ly", "ft_lag_6_ly",
    "ft_event_count",
    "ft_lag1_temp_max", "ft_lag1_temp_min", "ft_lag1_had_rain", "ft_lag1_had_snow",
    "ft_pred_temp_max", "ft_pred_temp_min", "ft_pred_had_rain", "ft_pred_had_snow",
    "ft_board_key",
]
CATEGORICAL_FEATURES = ["ft_board_key"]
MAX_LAG_WEEKS = 6
YEAR_OFFSET_WEEKS = 52
MODEL_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, enable_categorical=True)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="requires external API access")
@pytest.mark.timeout(900)
def test_pipeline_smoke():
    end = date.today().isoformat()
    start = (date.today() - timedelta(weeks=YEAR_OFFSET_WEEKS + MAX_LAG_WEEKS + 6)).isoformat()

    raw_dir = "data/dev/00_raw"
    retries = 3
    backoff_seconds = 5.0
    calls = fetch_calls_weekly(
        start_date=start, end_date=end, calls_url=CALLS_URL, raw_dir=raw_dir,
        retries=retries, backoff_seconds=backoff_seconds,
    )
    events = fetch_events_weekly(
        start_date=start, end_date=end, events_url=EVENTS_URL,
        event_include_types=EVENT_INCLUDE_TYPES, raw_dir=raw_dir,
        retries=retries, backoff_seconds=backoff_seconds,
    )
    weather_lag1, weather_pred = fetch_weather_weekly(
        start_date=start, end_date=end,
        weather_lat=40.7812, weather_lon=-73.9665,
        weather_daily_vars="temperature_2m_max,temperature_2m_min,rain_sum,snowfall_sum",
        weather_archive_url="https://archive-api.open-meteo.com/v1/archive",
        weather_forecast_url="https://historical-forecast-api.open-meteo.com/v1/forecast",
        raw_dir=raw_dir, retries=retries, backoff_seconds=backoff_seconds,
    )
    assert not calls.empty

    target = build_target(calls, target_col=TARGET_COL)
    assert not target.empty

    lag_features = featurize_lags(
        target, target_col=TARGET_COL, max_lag_weeks=MAX_LAG_WEEKS, year_offset_weeks=YEAR_OFFSET_WEEKS,
    )
    event_features = featurize_events(events)
    weather_features = featurize_weather(weather_lag1, weather_pred)
    features = join_features(target, lag_features, event_features, weather_features, FEATURE_COLS, CATEGORICAL_FEATURES)
    features = drop_incomplete_rows(features, FEATURE_COLS)
    assert not features.empty

    model, full_df = training(
        features, FEATURE_COLS, CATEGORICAL_FEATURES, TARGET_COL, STRATIFY_COL, SPLIT_COL,
        MODEL_PARAMS, test_size=0.2, val_size=0.2, random_state=42,
    )
    assert SPLIT_COL in full_df.columns

    metrics_df = compute_metrics(model, full_df, FEATURE_COLS, TARGET_COL, SPLIT_COL, ranking_k=5)
    test_metrics = metrics_df[metrics_df["split"] == "test"].set_index("metric")["value"]
    assert np.isfinite(test_metrics["mae"])
    assert np.isfinite(test_metrics["rmse"])
