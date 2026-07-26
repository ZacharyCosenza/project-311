from datetime import date, timedelta

import numpy as np
import pytest
from pyspark.sql import SparkSession

from modeling import utils
from modeling.train import evaluate


@pytest.mark.timeout(300)
def test_pipeline_smoke():
    end = date.today().isoformat()
    start = (date.today() - timedelta(weeks=12)).isoformat()

    calls = utils.fetch_calls_weekly(start, end)
    events = utils.fetch_events_weekly(start, end)
    lag1_weather, pred_weather = utils.fetch_weather_weekly(start, end)
    assert not calls.empty

    spark = SparkSession.builder.appName("modeling-test").master("local[*]").getOrCreate()
    try:
        data = utils.build_features(spark, calls, events, lag1_weather, pred_weather)
    finally:
        spark.stop()
    assert not data.empty

    data = utils.prepare_features(data)
    train, test = utils.chronological_split(data)
    assert not train.empty and not test.empty

    metrics, model, X_train, X_test = evaluate(train, test)

    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert len(model.predict(X_test)) == len(X_test)
