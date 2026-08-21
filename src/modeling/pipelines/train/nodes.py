from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from modeling.pipelines.features.nodes import drop_incomplete_rows, group_feature_cols
from modeling.pipelines.modeling.nodes import compute_metrics, log_to_mlflow, plot_feature_histograms, plot_feature_timeseries, plot_shap_beeswarm


def compute_train_end_date() -> str:
    """A constant end_date in parameters.yml would silently stop the training window
    from ever advancing past whatever date was last typed in there — this keeps it at
    today, the same way inference.compute_predict_window does for its own window.
    """
    return date.today().isoformat()


def drop_incomplete_grouped_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper around features.drop_incomplete_rows — the lag-column list is read
    off the dataframe itself rather than passed in, since it's group-driven. Also
    drops rows where tgt_other is null (the one target build_grouped_target doesn't
    zero-fill, and drop_incomplete_rows only checks feature completeness, not target).
    """
    lag_cols = [c for c in features.columns if c.startswith("ft_lag_")]
    return drop_incomplete_rows(features, lag_cols).dropna(subset=["tgt_other"]).reset_index(drop=True)


def train_models(
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
    """One XGBRegressor per group (each complaint_type_groups key, plus "other"), all
    trained on the same train/val/test split so groups stay comparable — the split
    happens once here rather than reusing modeling.training per group, which would
    otherwise re-split independently for every group and leave each on different rows.
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
    for group in [*complaint_type_groups.keys(), "other"]:
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        X, y = train_df[feature_cols], train_df[f"tgt_{group}"]
        model = XGBRegressor(**model_params)
        model.fit(X, np.log1p(y))
        models[group] = model

    return models, df


def compute_grouped_metrics(
    models: dict, modeling_data: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, max_lag_weeks: int, split_col: str, ranking_k: int,
) -> pd.DataFrame:
    """Reuses modeling.compute_metrics per group — same metric set (MAE/RMSE/ranking
    vs. a previous-week baseline), computed independently against each group's own
    target column, tagged with a group column so the per-group rows can be told apart.
    """
    frames = []
    for group, model in models.items():
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        m = compute_metrics(model, modeling_data, feature_cols, f"tgt_{group}", split_col, ranking_k)
        m.insert(0, "group", group)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def log_groups_to_mlflow(
    models: dict,
    modeling_data: pd.DataFrame,
    metrics: pd.DataFrame,
    complaint_type_groups: dict,
    shared_feature_cols: list,
    shared_categorical_features: list,
    max_lag_weeks: int,
    split_col: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    mlflow_model_name: str,
    model_params: dict,
    report_dir: str,
) -> None:
    """One MLflow run per group, in a shared experiment, tagged with its group name —
    reuses modeling.log_to_mlflow unchanged aside from that tag."""
    for group, model in models.items():
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        group_metrics = metrics[metrics["group"] == group].drop(columns="group")
        log_to_mlflow(
            model, modeling_data, group_metrics, feature_cols, shared_categorical_features, split_col,
            mlflow_tracking_uri, mlflow_experiment, f"{mlflow_model_name}-{group}",
            model_params, f"{report_dir}/{group}", extra_tags={"target_group": group},
        )


def plot_metrics_comparison(metrics: pd.DataFrame, report_dir: str) -> None:
    """The per-group plots (histograms, timeseries, beeswarm) each show one head in
    isolation, in its own subfolder — nothing compares heads against each other. This
    is the one chart at the top level of report_dir that does: test MAE improvement
    over the last-week baseline, per group, so a reviewer can see at a glance which
    heads are actually earning their keep without opening all 11 subfolders.
    """
    test = metrics[(metrics["split"] == "test") & (metrics["metric"].isin(["mae", "baseline_mae"]))]
    pivot = test.pivot(index="group", columns="metric", values="value")
    pivot["improvement_pct"] = 100 * (pivot["baseline_mae"] - pivot["mae"]) / pivot["baseline_mae"]
    pivot = pivot.sort_values("improvement_pct")

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pivot) + 1.5))
    colors = ["#93c5fd" if v >= 0 else "#f97316" for v in pivot["improvement_pct"]]
    ax.barh(pivot.index, pivot["improvement_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% improvement over last-week baseline (test MAE)")
    plt.tight_layout()

    out = Path(report_dir) / "metrics_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_grouped_histograms(
    models: dict, modeling_data: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, shared_categorical_features: list, max_lag_weeks: int,
    split_col: str, report_dir: str,
) -> None:
    for group, model in models.items():
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        plot_feature_histograms(
            model, modeling_data, feature_cols, shared_categorical_features, split_col, f"{report_dir}/{group}",
        )


def plot_grouped_timeseries(
    modeling_data: pd.DataFrame, complaint_type_groups: dict, shared_feature_cols: list,
    shared_categorical_features: list, max_lag_weeks: int, report_dir: str,
) -> None:
    for group in [*complaint_type_groups.keys(), "other"]:
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        plot_feature_timeseries(
            modeling_data, feature_cols, shared_categorical_features, f"{report_dir}/{group}",
        )


def plot_grouped_shap_beeswarm(
    models: dict, modeling_data: pd.DataFrame, complaint_type_groups: dict,
    shared_feature_cols: list, max_lag_weeks: int, split_col: str, report_dir: str,
) -> None:
    for group, model in models.items():
        feature_cols = group_feature_cols(group, shared_feature_cols, max_lag_weeks)
        plot_shap_beeswarm(model, modeling_data, feature_cols, split_col, f"{report_dir}/{group}")
