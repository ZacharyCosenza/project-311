import os
from datetime import date, timedelta

import numpy as np
import pytest

from modeling.pipelines.features.nodes import (
    featurize_events,
    featurize_grouped_lags,
    featurize_weather,
    join_grouped_features,
)
from modeling.pipelines.modeling.model import GroupedCallModel
from modeling.pipelines.raw.nodes import (
    fetch_calls_weekly,
    fetch_calls_weekly_by_group,
    fetch_events_weekly,
    fetch_weather_weekly,
)
from modeling.pipelines.target.nodes import build_grouped_target
from modeling.pipelines.train.nodes import compute_grouped_metrics, drop_incomplete_grouped_rows, train_models

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

# Two groups, not the production eleven: the smoke test is checking that the grouped
# path holds together end to end, and each extra group costs another Socrata round trip.
# "other" comes along for free — build_grouped_target derives it as the residual.
COMPLAINT_TYPE_GROUPS = {
    "noise": ["Noise", "Noise - Residential", "Noise - Street/Sidewalk"],
    "pests_health": ["Rodent"],
}
INVALID_BOARD_KEYS = ["0 Unspecified"]
SHARED_FEATURE_COLS = [
    "ft_week_of_year",
    "ft_event_count",
    "ft_lag1_temp_max", "ft_lag1_temp_min", "ft_lag1_had_rain", "ft_lag1_had_snow",
    "ft_pred_temp_max", "ft_pred_temp_min", "ft_pred_had_rain", "ft_pred_had_snow",
    "ft_board_key",
]
MAX_LAG_WEEKS = 6
YEAR_OFFSET_WEEKS = 52
RANKING_K = 5
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
    calls_by_group = fetch_calls_weekly_by_group(
        start_date=start, end_date=end, calls_url=CALLS_URL,
        complaint_type_groups=COMPLAINT_TYPE_GROUPS, raw_dir=raw_dir,
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

    target = build_grouped_target(calls, calls_by_group, COMPLAINT_TYPE_GROUPS, INVALID_BOARD_KEYS)
    assert not target.empty

    lag_features = featurize_grouped_lags(target, COMPLAINT_TYPE_GROUPS, MAX_LAG_WEEKS, YEAR_OFFSET_WEEKS)
    event_features = featurize_events(events)
    weather_features = featurize_weather(weather_lag1, weather_pred)
    features = drop_incomplete_grouped_rows(
        join_grouped_features(target, lag_features, event_features, weather_features)
    )
    assert not features.empty

    models, full_df = train_models(
        features, COMPLAINT_TYPE_GROUPS, SHARED_FEATURE_COLS, MAX_LAG_WEEKS, YEAR_OFFSET_WEEKS,
        STRATIFY_COL, SPLIT_COL, MODEL_PARAMS, test_size=0.2, val_size=0.2, random_state=42,
    )
    assert SPLIT_COL in full_df.columns

    expected_groups = [*COMPLAINT_TYPE_GROUPS, "other"]
    assert isinstance(models, GroupedCallModel)
    assert models.groups == expected_groups
    for group in expected_groups:
        assert models.feature_cols(group)[:MAX_LAG_WEEKS + 1] == [
            *(f"ft_lag_{lag}_{group}" for lag in range(1, MAX_LAG_WEEKS + 1)),
            f"ft_lag_ly_{group}",
        ]

    # The total is what inference publishes, so the sum of the heads is the thing worth
    # asserting on, not any single head's output.
    total = models.predict(full_df)
    assert total.shape == (len(full_df),)
    assert np.isfinite(total).all()
    assert (total >= 0).all()

    metrics_df = compute_grouped_metrics(
        models, full_df, COMPLAINT_TYPE_GROUPS, SHARED_FEATURE_COLS, MAX_LAG_WEEKS, SPLIT_COL, RANKING_K,
    )
    assert set(metrics_df["group"]) == set(expected_groups)
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    for group in expected_groups:
        g = test_metrics[test_metrics["group"] == group].set_index("metric")["value"]
        assert np.isfinite(g["mae"])
        assert np.isfinite(g["rmse"])
