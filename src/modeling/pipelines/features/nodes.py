import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def _add_lag_columns(sdf, w: Window, target_col: str, max_lag_weeks: int, year_offset_weeks: int, suffix: str):
    """suffix namespaces the output columns (e.g. "_noise") so the same logic can run
    once per target column without collisions.
    """
    recent_cols = [f"ft_lag_{lag}{suffix}" for lag in range(1, max_lag_weeks + 1)]
    for lag, col in zip(range(1, max_lag_weeks + 1), recent_cols):
        sdf = sdf.withColumn(col, F.log1p(F.lag(target_col, lag).over(w)))

    # A single column at exactly year_offset_weeks back — the same calendar week last
    # year. The previous fan of six columns sat at year_offset_weeks+1..+6, so it
    # straddled the anniversary without ever landing on it.
    ly_col = f"ft_lag_ly{suffix}"
    sdf = sdf.withColumn(ly_col, F.log1p(F.lag(target_col, year_offset_weeks).over(w)))

    return sdf, [*recent_cols, ly_col]


def featurize_grouped_lags(
    target: pd.DataFrame, complaint_type_groups: dict, max_lag_weeks: int, year_offset_weeks: int,
) -> pd.DataFrame:
    """Recent lags (1..max_lag_weeks back) plus one "_ly" column holding the same
    calendar week a year ago, built once per group (each complaint_type_groups key, plus
    "other") in a single shared Spark session — a session per group would restart the
    JVM eleven times. Week-of-year on top captures seasonality more broadly.
    """
    spark = SparkSession.builder.appName("featurize-grouped-lags").master("local[*]").getOrCreate()
    try:
        w = Window.partitionBy("board_key").orderBy("week_start")
        sdf = spark.createDataFrame(target).withColumn("ft_week_of_year", F.weekofyear("week_start"))

        all_lag_cols = []
        for group in [*complaint_type_groups.keys(), "other"]:
            sdf, lag_cols = _add_lag_columns(
                sdf, w, f"tgt_{group}", max_lag_weeks, year_offset_weeks, suffix=f"_{group}",
            )
            all_lag_cols.extend(lag_cols)

        sdf = sdf.select("board_key", "week_start", "ft_week_of_year", *all_lag_cols)
        return sdf.toPandas()
    finally:
        spark.stop()


def group_feature_cols(group: str, shared_feature_cols: list, max_lag_weeks: int) -> list:
    """The feature list one group's model trains/predicts on: its own lag columns
    (named by featurize_grouped_lags) plus the exogenous columns every group shares.
    Shared by the train and inference pipelines so the naming convention has one
    definition, not two that could drift apart.
    """
    lag_cols = [f"ft_lag_{lag}_{group}" for lag in range(1, max_lag_weeks + 1)]
    lag_cols.append(f"ft_lag_ly_{group}")
    return [*lag_cols, *shared_feature_cols]


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


def join_grouped_features(
    target: pd.DataFrame, lag_features: pd.DataFrame,
    event_features: pd.DataFrame, weather_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join every feature group onto the target spine. The column set is group-driven
    (one set of lag columns per complaint_type_groups key) rather than a fixed list, so
    numeric/categorical casting goes by naming convention: every ft_ column is numeric
    except ft_board_key, the one categorical. Shared by the train and inference
    pipelines.
    """
    df = (
        target
        .merge(lag_features, on=["board_key", "week_start"], how="left")
        .merge(event_features, on=["board_key", "week_start"], how="left")
        .merge(weather_features, on="week_start", how="left")
    )
    df["ft_event_count"] = df["ft_event_count"].fillna(0)
    df = df.sort_values(["board_key", "week_start"]).reset_index(drop=True)
    df["ft_board_key"] = df["board_key"].astype("category")
    numeric_cols = [c for c in df.columns if c.startswith("ft_") and c != "ft_board_key"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


def drop_incomplete_rows(features: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    lag_cols = [c for c in feature_cols if c.startswith("ft_lag_")]
    return features.dropna(subset=[*lag_cols, "ft_lag1_temp_max", "ft_pred_temp_max"]).reset_index(drop=True)
