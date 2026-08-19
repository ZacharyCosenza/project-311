import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def _add_lag_columns(sdf, w: Window, target_col: str, max_lag_weeks: int, year_offset_weeks: int, suffix: str = ""):
    """Shared by featurize_lags and featurize_multi_lags — suffix namespaces the output
    columns (e.g. "_noise") so the same logic can run once per target column without
    collisions; empty by default, matching featurize_lags's own column names exactly.
    """
    recent_cols = [f"ft_lag_{lag}{suffix}" for lag in range(1, max_lag_weeks + 1)]
    for lag, col in zip(range(1, max_lag_weeks + 1), recent_cols):
        sdf = sdf.withColumn(col, F.log1p(F.lag(target_col, lag).over(w)))

    ly_cols = [f"ft_lag_{lag}_ly{suffix}" for lag in range(1, max_lag_weeks + 1)]
    for lag, col in zip(range(1, max_lag_weeks + 1), ly_cols):
        sdf = sdf.withColumn(col, F.log1p(F.lag(target_col, year_offset_weeks + lag).over(w)))

    return sdf, recent_cols + ly_cols


def featurize_lags(target: pd.DataFrame, target_col: str, max_lag_weeks: int, year_offset_weeks: int) -> pd.DataFrame:
    """Recent lags (1..max_lag_weeks back) plus the same lags shifted back
    year_offset_weeks (~1 calendar year) — "_ly" columns — so the model can separate a
    recent trend from what this same time of year looked like last year, rather than
    conflating the two. Week-of-year on top captures seasonality more broadly.
    """
    spark = SparkSession.builder.appName("featurize-lags").master("local[*]").getOrCreate()
    try:
        w = Window.partitionBy("board_key").orderBy("week_start")
        sdf = spark.createDataFrame(target).withColumn("ft_week_of_year", F.weekofyear("week_start"))
        sdf, lag_cols = _add_lag_columns(sdf, w, target_col, max_lag_weeks, year_offset_weeks)
        sdf = sdf.select("board_key", "week_start", "ft_week_of_year", *lag_cols)
        return sdf.toPandas()
    finally:
        spark.stop()


def featurize_multi_lags(
    multi_target: pd.DataFrame, complaint_type_groups: dict, max_lag_weeks: int, year_offset_weeks: int,
) -> pd.DataFrame:
    """Same lag logic as featurize_lags, run once per target head (each
    complaint_type_groups key, plus "other") in a single shared Spark session — calling
    featurize_lags itself in a loop would restart the JVM once per head.
    """
    spark = SparkSession.builder.appName("featurize-multi-lags").master("local[*]").getOrCreate()
    try:
        w = Window.partitionBy("board_key").orderBy("week_start")
        sdf = spark.createDataFrame(multi_target).withColumn("ft_week_of_year", F.weekofyear("week_start"))

        all_lag_cols = []
        for head in [*complaint_type_groups.keys(), "other"]:
            sdf, lag_cols = _add_lag_columns(
                sdf, w, f"tgt_{head}", max_lag_weeks, year_offset_weeks, suffix=f"_{head}",
            )
            all_lag_cols.extend(lag_cols)

        sdf = sdf.select("board_key", "week_start", "ft_week_of_year", *all_lag_cols)
        return sdf.toPandas()
    finally:
        spark.stop()


def featurize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Rename and log-transform event counts."""
    df = events.rename(columns={"event_count": "ft_event_count"}).copy()
    df["ft_event_count"] = np.log1p(df["ft_event_count"].fillna(0))
    return df


def featurize_weather(weather_lag1: pd.DataFrame, weather_pred: pd.DataFrame) -> pd.DataFrame:
    """Prefix lag and forecast weather columns with ft_ and merge into one table."""
    lag = weather_lag1.rename(columns={c: f"ft_{c}" for c in weather_lag1.columns if c != "week_start"})
    pred = weather_pred.rename(columns={c: f"ft_{c}" for c in weather_pred.columns if c != "week_start"})
    return lag.merge(pred, on="week_start", how="outer")


def join_features(
    target: pd.DataFrame,
    lag_features: pd.DataFrame,
    event_features: pd.DataFrame,
    weather_features: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
) -> pd.DataFrame:
    """Join all feature groups onto the target spine and finalize types."""
    numeric_features = [f for f in feature_cols if f not in categorical_features]

    df = (
        target
        .merge(lag_features, on=["board_key", "week_start"], how="left")
        .merge(event_features, on=["board_key", "week_start"], how="left")
        .merge(weather_features, on="week_start", how="left")
    )
    df["ft_event_count"] = df["ft_event_count"].fillna(0)
    df = df.sort_values(["board_key", "week_start"]).reset_index(drop=True)
    df[numeric_features] = df[numeric_features].astype(float)
    df["ft_board_key"] = df["board_key"].astype("category")
    return df


def drop_incomplete_rows(features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    lag_cols = [c for c in feature_cols if c.startswith("ft_lag_")]
    return features.dropna(subset=[*lag_cols, "ft_lag1_temp_max", "ft_pred_temp_max"]).reset_index(drop=True)
