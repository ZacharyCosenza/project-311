import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

matplotlib.use("Agg")


def _pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _split(
    df: pd.DataFrame, test_size: float, val_size: float, random_state: int,
    stratify_col: str, split_col: str,
) -> pd.DataFrame:
    df = df.copy()
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify_col],
    )
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_state, stratify=train_val[stratify_col],
    )
    df[split_col] = "train"
    df.loc[val.index, split_col] = "val"
    df.loc[test.index, split_col] = "test"
    return df


def _input_drift_metrics(X: pd.DataFrame, numeric_features: list) -> dict:
    """Mean and std per numeric feature — tracks input distribution shift."""
    return {
        **{f"input_mean_{col}": float(X[col].mean()) for col in numeric_features},
        **{f"input_std_{col}": float(X[col].std()) for col in numeric_features},
    }


def _shap_drift_metrics(model: XGBRegressor, X: pd.DataFrame, feature_cols: list) -> dict:
    """Mean and std of |SHAP values| per feature — tracks concept drift."""
    explainer = shap.TreeExplainer(model)
    shap_vals = pd.DataFrame(
        explainer.shap_values(X, check_additivity=False), columns=feature_cols
    )
    return {
        **{f"shap_mean_{col}": float(shap_vals[col].abs().mean()) for col in feature_cols},
        **{f"shap_std_{col}": float(shap_vals[col].abs().std()) for col in feature_cols},
    }


def training(
    features: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    target_col: str,
    stratify_col: str,
    split_col: str,
    model_params: dict,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[XGBRegressor, pd.DataFrame]:
    df = _pre_processing(features)
    df = _split(df, test_size, val_size, random_state, stratify_col, split_col)

    train_df = df[df[split_col] == "train"]
    X, y = train_df[feature_cols], train_df[target_col]
    model = XGBRegressor(**model_params)
    model.fit(X, np.log1p(y))
    return model, df


def inference(
    model: XGBRegressor, df: pd.DataFrame, feature_cols: list, target_col: str,
) -> pd.DataFrame:
    result = df.copy()
    result[f"pred_{target_col}"] = np.expm1(model.predict(df[feature_cols]))
    return result


def _ranking_metrics(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> dict:
    def week_metrics(g):
        actual_sorted = g.sort_values(target_col, ascending=False)
        pred_sorted = g.sort_values(score_col, ascending=False)
        pred_top = list(pred_sorted["board_key"].head(k))
        actual_top = set(actual_sorted["board_key"].head(k))

        rel = dict(zip(g["board_key"], g[target_col]))
        dcg = sum(rel[b] / np.log2(i + 2) for i, b in enumerate(pred_top))
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(actual_sorted[target_col].head(k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precision = len(set(pred_top) & actual_top) / k
        top1 = actual_sorted["board_key"].iloc[0]
        rr = 1 / (pred_top.index(top1) + 1) if top1 in pred_top else 0.0
        return pd.Series({"precision": precision, "ndcg": ndcg, "mrr": rr, "n_boards": len(g)})

    weekly = df.groupby("week_start").apply(week_metrics, include_groups=False)
    weekly = weekly[weekly["n_boards"] >= k]
    return {
        f"precision_at_{k}": weekly["precision"].mean(),
        f"ndcg_at_{k}": weekly["ndcg"].mean(),
        f"mrr_at_{k}": weekly["mrr"].mean(),
    }


def compute_metrics(
    model: XGBRegressor,
    modeling_data: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    split_col: str,
    ranking_k: int,
) -> pd.DataFrame:
    pred_col = f"pred_{target_col}"
    prev_week_lookup = modeling_data.set_index(["board_key", "week_start"])[target_col]

    rows = []
    for split_name in sorted(modeling_data[split_col].unique()):
        scored = inference(model, modeling_data[modeling_data[split_col] == split_name], feature_cols, target_col)

        prev_week_key = list(zip(scored["board_key"], scored["week_start"] - timedelta(weeks=1)))
        scored = scored.assign(baseline_prev_week=prev_week_lookup.reindex(prev_week_key).to_numpy())
        valid = scored.dropna(subset=["baseline_prev_week"])

        m = {
            "mae": mean_absolute_error(scored[target_col], scored[pred_col]),
            "rmse": mean_squared_error(scored[target_col], scored[pred_col]) ** 0.5,
            "baseline_mae": mean_absolute_error(valid[target_col], valid["baseline_prev_week"]),
        }
        m.update({f"model_{k}": v for k, v in _ranking_metrics(scored, pred_col, target_col, ranking_k).items()})
        m.update({f"baseline_{k}": v for k, v in _ranking_metrics(valid, "baseline_prev_week", target_col, ranking_k).items()})
        rows.extend({"split": split_name, "metric": metric, "value": v} for metric, v in m.items())

    return pd.DataFrame(rows)


def log_to_mlflow(
    model: XGBRegressor,
    modeling_data: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    split_col: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    mlflow_model_name: str,
    model_params: dict,
    report_dir: str,
) -> None:
    numeric_features = [f for f in feature_cols if f not in categorical_features]
    X_train = modeling_data[modeling_data[split_col] == "train"][feature_cols]

    all_metrics = {
        **{f"{r.split}_{r.metric}": r.value for r in metrics.itertuples()},
        **_input_drift_metrics(X_train, numeric_features),
        **_shap_drift_metrics(model, X_train, feature_cols),
    }

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(mlflow_experiment) is None:
        mlflow.create_experiment(mlflow_experiment, artifact_location=str(Path(report_dir) / "mlruns"))
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_metrics(all_metrics)
        mlflow.log_params(model_params)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_dir = Path(report_dir) / "mlartifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"model_{timestamp}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_param("model_artifact_path", str(artifact_path))
        mlflow.set_tag("trained_at_utc", timestamp)
        mlflow.xgboost.log_model(model, name="model", registered_model_name=mlflow_model_name)


def plot_feature_histograms(
    model: XGBRegressor,
    modeling_data: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    split_col: str,
    report_dir: str,
) -> None:
    from pathlib import Path

    numeric_features = [f for f in feature_cols if f not in categorical_features]
    categorical_mask = [f in categorical_features for f in feature_cols]
    train_df = modeling_data[modeling_data[split_col] == "train"]

    ncols = 4
    nrows = -(-len(numeric_features) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    for ax, col in zip(axes.flat, numeric_features):
        ax.hist(train_df[col], bins=30, color="#93c5fd", alpha=0.7)
        ax.set_title(col, fontsize=9)
        pdp = partial_dependence(
            model, train_df[feature_cols], [col], categorical_features=categorical_mask, kind="average"
        )
        ax2 = ax.twinx()
        ax2.plot(pdp["grid_values"][0], pdp["average"][0], color="tomato", linewidth=2)
        ax2.set_yticks([])
    for ax in axes.flat[len(numeric_features):]:
        ax.axis("off")
    plt.tight_layout()

    out = Path(report_dir) / "feature_histograms.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_feature_timeseries(
    modeling_data: pd.DataFrame, feature_cols: list, categorical_features: list, report_dir: str,
) -> None:
    """Mean ± std of each numeric feature per week (the spine's time column) — a quick
    way to spot distribution drift over time, across all splits (not just train) so the
    time axis stays continuous."""
    numeric_features = [f for f in feature_cols if f not in categorical_features]
    weekly = modeling_data.groupby("week_start")[numeric_features].agg(["mean", "std"])

    ncols = 4
    nrows = -(-len(numeric_features) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    for ax, col in zip(axes.flat, numeric_features):
        mean, std = weekly[(col, "mean")], weekly[(col, "std")]
        ax.plot(mean.index, mean.values, color="#2563eb", linewidth=1.5)
        ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, color="#93c5fd", alpha=0.4)
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
    for ax in axes.flat[len(numeric_features):]:
        ax.axis("off")
    plt.tight_layout()

    out = Path(report_dir) / "feature_timeseries.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_shap_beeswarm(
    model: XGBRegressor, modeling_data: pd.DataFrame, feature_cols: list, split_col: str, report_dir: str,
) -> None:
    train_df = modeling_data[modeling_data[split_col] == "train"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_df[feature_cols], check_additivity=False)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, train_df[feature_cols], show=False)
    plt.tight_layout()

    out = Path(report_dir) / "shap_beeswarm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
