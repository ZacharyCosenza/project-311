import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from modeling.pipelines.features.nodes import drop_incomplete_rows
from modeling.pipelines.modeling.nodes import compute_metrics, log_to_mlflow, plot_feature_histograms, plot_feature_timeseries, plot_shap_beeswarm


def _head_feature_cols(head: str, shared_feature_cols: list, max_lag_weeks: int) -> list:
    lag_cols = [f"ft_lag_{lag}_{head}" for lag in range(1, max_lag_weeks + 1)]
    lag_cols += [f"ft_lag_{lag}_ly_{head}" for lag in range(1, max_lag_weeks + 1)]
    return [*lag_cols, *shared_feature_cols]


def join_multi_features(
    multi_target: pd.DataFrame, multi_lag_features: pd.DataFrame,
    event_features: pd.DataFrame, weather_features: pd.DataFrame,
) -> pd.DataFrame:
    """Same join as features.join_features, but the column set is group-driven (one
    set of lag columns per complaint_type_groups head) rather than a fixed feature_cols
    list, so numeric/categorical casting goes by naming convention instead: every ft_
    column is numeric except ft_board_key, the one categorical.
    """
    df = (
        multi_target
        .merge(multi_lag_features, on=["board_key", "week_start"], how="left")
        .merge(event_features, on=["board_key", "week_start"], how="left")
        .merge(weather_features, on="week_start", how="left")
    )
    df["ft_event_count"] = df["ft_event_count"].fillna(0)
    df = df.sort_values(["board_key", "week_start"]).reset_index(drop=True)
    df["ft_board_key"] = df["board_key"].astype("category")
    numeric_cols = [c for c in df.columns if c.startswith("ft_") and c != "ft_board_key"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


def drop_incomplete_multi_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper around features.drop_incomplete_rows — the lag-column list is read
    off the dataframe itself (every ft_lag_ column, across all heads) rather than
    passed in, since the multi-head column set is group-driven, not a fixed list.
    """
    lag_cols = [c for c in features.columns if c.startswith("ft_lag_")]
    return drop_incomplete_rows(features, lag_cols)


def train_multi_head(
    features: pd.DataFrame,
    complaint_type_groups: dict,
    shared_feature_cols: list,
    max_lag_weeks: int,
    year_offset_weeks: int,
    stratify_col: str,
    split_col: str,
    model_params: dict,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[dict, pd.DataFrame]:
    """One XGBRegressor per head (each complaint_type_groups key, plus "other"), all
    trained on the same train/val/test split so heads stay comparable — the split
    happens once here rather than reusing modeling.training per head, which would
    otherwise re-split independently for every head and leave each on different rows.
    """
    df = features.copy()
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_state, stratify=train_val[stratify_col],
    )
    df[split_col] = "train"
    df.loc[val.index, split_col] = "val"
    df.loc[test.index, split_col] = "test"

    train_df = df[df[split_col] == "train"]

    models = {}
    for head in [*complaint_type_groups.keys(), "other"]:
        feature_cols = _head_feature_cols(head, shared_feature_cols, max_lag_weeks)
        X, y = train_df[feature_cols], train_df[f"tgt_{head}"]
        model = XGBRegressor(**model_params)
        model.fit(X, np.log1p(y))
        models[head] = model

    return models, df


def compute_multi_metrics(
    models: dict, modeling_data: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, max_lag_weeks: int, split_col: str, ranking_k: int,
) -> pd.DataFrame:
    """Reuses modeling.compute_metrics per head — same metric set (MAE/RMSE/ranking vs.
    a previous-week baseline), computed independently against each head's own target
    column, tagged with a group column so the per-head rows can be told apart.
    """
    frames = []
    for head, model in models.items():
        feature_cols = _head_feature_cols(head, shared_feature_cols, max_lag_weeks)
        m = compute_metrics(model, modeling_data, feature_cols, f"tgt_{head}", split_col, ranking_k)
        m.insert(0, "group", head)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def log_multi_to_mlflow(
    models: dict,
    modeling_data: pd.DataFrame,
    metrics: pd.DataFrame,
    complaint_type_groups: dict,
    shared_feature_cols: list,
    shared_categorical_features: list,
    max_lag_weeks: int,
    split_col: str,
    mlflow_tracking_uri: str,
    mlflow_multihead_experiment: str,
    mlflow_multihead_model_name: str,
    model_params: dict,
    multihead_report_dir: str,
) -> None:
    """One MLflow run per head, in a shared experiment, tagged with its group name —
    reuses modeling.log_to_mlflow unchanged aside from that tag."""
    for head, model in models.items():
        feature_cols = _head_feature_cols(head, shared_feature_cols, max_lag_weeks)
        head_metrics = metrics[metrics["group"] == head].drop(columns="group")
        log_to_mlflow(
            model, modeling_data, head_metrics, feature_cols, shared_categorical_features, split_col,
            mlflow_tracking_uri, mlflow_multihead_experiment, f"{mlflow_multihead_model_name}-{head}",
            model_params, f"{multihead_report_dir}/{head}", extra_tags={"target_group": head},
        )


def plot_multi_histograms(
    models: dict, modeling_data: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, shared_categorical_features: list, max_lag_weeks: int,
    split_col: str, multihead_report_dir: str,
) -> None:
    for head, model in models.items():
        feature_cols = _head_feature_cols(head, shared_feature_cols, max_lag_weeks)
        plot_feature_histograms(
            model, modeling_data, feature_cols, shared_categorical_features, split_col, f"{multihead_report_dir}/{head}",
        )


def plot_multi_timeseries(
    modeling_data: pd.DataFrame, complaint_type_groups: dict, shared_feature_cols: list,
    shared_categorical_features: list, max_lag_weeks: int, multihead_report_dir: str,
) -> None:
    for head in [*complaint_type_groups.keys(), "other"]:
        feature_cols = _head_feature_cols(head, shared_feature_cols, max_lag_weeks)
        plot_feature_timeseries(
            modeling_data, feature_cols, shared_categorical_features, f"{multihead_report_dir}/{head}",
        )


def plot_multi_shap_beeswarm(
    models: dict, modeling_data: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, max_lag_weeks: int, split_col: str, multihead_report_dir: str,
) -> None:
    for head, model in models.items():
        feature_cols = _head_feature_cols(head, shared_feature_cols, max_lag_weeks)
        plot_shap_beeswarm(model, modeling_data, feature_cols, split_col, f"{multihead_report_dir}/{head}")
