import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def featurize_lags(target: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Lag features and week-of-year via Spark window functions."""
    spark = SparkSession.builder.appName("featurize-lags").master("local[*]").getOrCreate()
    try:
        w = Window.partitionBy("board_key").orderBy("week_start")
        sdf = (
            spark.createDataFrame(target)
            .withColumn("ft_week_of_year", F.weekofyear("week_start"))
            .withColumn("ft_lag_1", F.log1p(F.lag(target_col, 1).over(w)))
            .withColumn("ft_lag_2", F.log1p(F.lag(target_col, 2).over(w)))
            .select("board_key", "week_start", "ft_week_of_year", "ft_lag_1", "ft_lag_2")
        )
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
    df = df.dropna(subset=["ft_lag_1", "ft_lag_2", "ft_lag1_temp_max", "ft_pred_temp_max"])
    df = df.sort_values(["board_key", "week_start"]).reset_index(drop=True)
    df[numeric_features] = df[numeric_features].astype(float)
    df["ft_board_key"] = df["board_key"].astype("category")
    return df
