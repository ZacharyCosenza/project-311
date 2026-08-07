from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

matplotlib.use("Agg")


def _pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    # Currently a no-op; hook for future row-level filtering before training.
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


def _ranking_metrics(
    df: pd.DataFrame, score_col: str, target_col: str, k: int,
) -> dict:
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
    categorical_features: list,
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


def plot_feature_histograms(
    model: XGBRegressor,
    modeling_data: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    split_col: str,
) -> plt.Figure:
    numeric_features = [f for f in feature_cols if f not in categorical_features]
    categorical_mask = [f in categorical_features for f in feature_cols]
    train_df = modeling_data[modeling_data[split_col] == "train"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, col in zip(axes.flat, numeric_features):
        ax.hist(train_df[col], bins=30, color="#93c5fd", alpha=0.7)
        ax.set_title(col, fontsize=9)
        pdp = partial_dependence(
            model, train_df[feature_cols], [col], categorical_features=categorical_mask, kind="average"
        )
        ax2 = ax.twinx()
        ax2.plot(pdp["grid_values"][0], pdp["average"][0], color="tomato", linewidth=2)
        ax2.set_yticks([])
    plt.tight_layout()
    return fig
