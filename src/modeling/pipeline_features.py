import os
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from modeling import pipeline_target

FEATURES = [
    "ft_week_of_year", "ft_lag_1", "ft_lag_2", "ft_event_count",
    "ft_lag1_temp_max", "ft_lag1_temp_min", "ft_lag1_had_rain", "ft_lag1_had_snow",
    "ft_pred_temp_max", "ft_pred_temp_min", "ft_pred_had_rain", "ft_pred_had_snow",
    "ft_board_key",
]
CATEGORICAL_FEATURES = ["ft_board_key"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

ENV = os.environ.get("MODELING_ENV", "dev")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / ENV / "00_raw"
PROCESSED_DIR = ROOT / "data" / ENV / "01_processed"


def build_features(spark: SparkSession, target: pd.DataFrame, events: pd.DataFrame,
                    weather_lag1: pd.DataFrame, weather_pred: pd.DataFrame) -> pd.DataFrame:
    sdf = spark.createDataFrame(target)
    w = Window.partitionBy("board_key").orderBy("week_start")
    sdf = sdf.withColumn("ft_week_of_year", F.weekofyear("week_start"))
    sdf = sdf.withColumn("ft_lag_1", F.log1p(F.lag(pipeline_target.TARGET_COL, 1).over(w)))
    sdf = sdf.withColumn("ft_lag_2", F.log1p(F.lag(pipeline_target.TARGET_COL, 2).over(w)))

    events = events.rename(columns={"event_count": "ft_event_count"})
    if len(events):
        sdf = sdf.join(spark.createDataFrame(events), on=["board_key", "week_start"], how="left")
    else:
        sdf = sdf.withColumn("ft_event_count", F.lit(0))
    sdf = sdf.fillna({"ft_event_count": 0})
    sdf = sdf.withColumn("ft_event_count", F.log1p(F.col("ft_event_count")))

    weather_lag1 = weather_lag1.rename(columns={c: f"ft_{c}" for c in weather_lag1.columns if c != "week_start"})
    weather_pred = weather_pred.rename(columns={c: f"ft_{c}" for c in weather_pred.columns if c != "week_start"})
    sdf = sdf.join(spark.createDataFrame(weather_lag1), on="week_start", how="left")
    sdf = sdf.join(spark.createDataFrame(weather_pred), on="week_start", how="left")

    sdf = sdf.dropna(subset=["ft_lag_1", "ft_lag_2", "ft_lag1_temp_max", "ft_pred_temp_max"])
    result = sdf.orderBy("board_key", "week_start").toPandas()
    result[NUMERIC_FEATURES] = result[NUMERIC_FEATURES].astype(float)
    result["ft_board_key"] = result["board_key"].astype("category")
    return result


def run(target: pd.DataFrame | None = None, events: pd.DataFrame | None = None,
        weather_lag1: pd.DataFrame | None = None, weather_pred: pd.DataFrame | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if target is None:
        target = pd.read_parquet(PROCESSED_DIR / "target.parquet")
    if events is None:
        events = pd.read_parquet(RAW_DIR / "events_weekly.parquet")
    if weather_lag1 is None:
        weather_lag1 = pd.read_parquet(RAW_DIR / "weather_lag1.parquet")
    if weather_pred is None:
        weather_pred = pd.read_parquet(RAW_DIR / "weather_pred.parquet")

    spark = SparkSession.builder.appName("pipeline-features").master("local[*]").getOrCreate()
    try:
        features = build_features(spark, target, events, weather_lag1, weather_pred)
    finally:
        spark.stop()

    features.to_parquet(PROCESSED_DIR / "features.parquet")
    return features


if __name__ == "__main__":
    run()
