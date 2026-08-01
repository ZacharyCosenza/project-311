import os
from datetime import date, timedelta

import numpy as np
import pytest
from pyspark.sql import SparkSession

from modeling import pipeline_features, pipeline_modeling, pipeline_raw, pipeline_target


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="requires external API access")
@pytest.mark.timeout(300)
def test_pipeline_smoke():
    end = date.today().isoformat()
    start = (date.today() - timedelta(weeks=12)).isoformat()

    calls = pipeline_raw.fetch_calls_weekly(start, end)
    events = pipeline_raw.fetch_events_weekly(start, end)
    weather_lag1, weather_pred = pipeline_raw.fetch_weather_weekly(start, end)
    assert not calls.empty

    spine = pipeline_target.build_spine(calls)
    target = pipeline_target.build_target(calls, spine)
    assert not target.empty

    spark = SparkSession.builder.appName("modeling-test").master("local[*]").getOrCreate()
    try:
        features = pipeline_features.build_features(spark, target, events, weather_lag1, weather_pred)
    finally:
        spark.stop()
    assert not features.empty

    model, full_df = pipeline_modeling.training(features)
    assert pipeline_modeling.SPLIT_COL in full_df.columns

    all_metrics = pipeline_modeling.metrics(model, full_df)
    assert "test" in all_metrics
    assert np.isfinite(all_metrics["test"]["mae"])
    assert np.isfinite(all_metrics["test"]["rmse"])
