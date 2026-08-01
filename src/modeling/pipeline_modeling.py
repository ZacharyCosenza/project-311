import os
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from modeling import pipeline_features, pipeline_target

FEATURES = pipeline_features.FEATURES
TARGET = pipeline_target.TARGET_COL

STRATIFY_COL = "week_start"
SPLIT_COL = "split"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
RANKING_K = 5

MODEL_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, enable_categorical=True)
CATEGORICAL_MASK = [f in pipeline_features.CATEGORICAL_FEATURES for f in FEATURES]

ENV = os.environ.get("ENV", "dev")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / ENV / "01_processed"
REPORT_DIR = ROOT / "data" / ENV / "02_reporting"


def pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    # Currently a no-op; hook for future row-level filtering (e.g. temporal
    # windowing, dropping specific community districts) before training.
    return df.copy()


def split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[STRATIFY_COL],
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_val[STRATIFY_COL],
    )
    df[SPLIT_COL] = "train"
    df.loc[val.index, SPLIT_COL] = "val"
    df.loc[test.index, SPLIT_COL] = "test"
    return df


def training(df: pd.DataFrame) -> tuple[XGBRegressor, pd.DataFrame]:
    df = pre_processing(df)
    df = split(df)

    train_df = df[df[SPLIT_COL] == "train"]
    X, y = train_df[FEATURES], train_df[TARGET]
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X, np.log1p(y))
    return model, df


def inference(model: XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["pred_tgt_calls"] = np.expm1(model.predict(df[FEATURES]))
    return result


def _ranking_metrics(df: pd.DataFrame, score_col: str, k: int = RANKING_K) -> dict:
    def week_metrics(g):
        actual_sorted = g.sort_values(TARGET, ascending=False)
        pred_sorted = g.sort_values(score_col, ascending=False)
        pred_top = list(pred_sorted["board_key"].head(k))
        actual_top = set(actual_sorted["board_key"].head(k))

        rel = dict(zip(g["board_key"], g[TARGET]))
        dcg = sum(rel[b] / np.log2(i + 2) for i, b in enumerate(pred_top))
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(actual_sorted[TARGET].head(k)))
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


def metrics(model: XGBRegressor, full_df: pd.DataFrame) -> dict[str, dict]:
    prev_week_lookup = full_df.set_index(["board_key", "week_start"])[TARGET]

    result = {}
    for split_name in sorted(full_df[SPLIT_COL].unique()):
        scored = inference(model, full_df[full_df[SPLIT_COL] == split_name])

        prev_week_key = list(zip(scored["board_key"], scored["week_start"] - timedelta(weeks=1)))
        scored = scored.assign(baseline_prev_week=prev_week_lookup.reindex(prev_week_key).to_numpy())
        valid = scored.dropna(subset=["baseline_prev_week"])

        m = {
            "mae": mean_absolute_error(scored[TARGET], scored["pred_tgt_calls"]),
            "rmse": mean_squared_error(scored[TARGET], scored["pred_tgt_calls"]) ** 0.5,
            "baseline_mae": mean_absolute_error(valid[TARGET], valid["baseline_prev_week"]),
        }
        m.update({f"model_{k}": v for k, v in _ranking_metrics(scored, "pred_tgt_calls").items()})
        m.update({f"baseline_{k}": v for k, v in _ranking_metrics(valid, "baseline_prev_week").items()})
        result[split_name] = m

    return result


def write_metrics_csv(all_metrics: dict[str, dict], path: Path):
    rows = [
        {"split": split_name, "metric": k, "value": v}
        for split_name, m in all_metrics.items()
        for k, v in m.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_feature_histograms(model: XGBRegressor, df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, col in zip(axes.flat, pipeline_features.NUMERIC_FEATURES):
        ax.hist(df[col], bins=30, color="#93c5fd", alpha=0.7)
        ax.set_title(col, fontsize=9)
        pdp = partial_dependence(model, df[FEATURES], [col], categorical_features=CATEGORICAL_MASK, kind="average")
        ax2 = ax.twinx()
        ax2.plot(pdp["grid_values"][0], pdp["average"][0], color="tomato", linewidth=2)
        ax2.set_yticks([])
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
