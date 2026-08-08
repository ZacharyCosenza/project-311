import sys
from datetime import date, timedelta

import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBRegressor

from modeling.pipelines.modeling.nodes import inference


def compute_predict_window(lookback_weeks: int) -> tuple[str, str]:
    """Narrow rolling window: enough history for lag_1/lag_2 plus the current week."""
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(weeks=lookback_weeks)).isoformat()
    return start_date, end_date


def load_latest_model(mlflow_tracking_uri: str, mlflow_model_name: str) -> XGBRegressor:
    # Loads the highest registered version, no manual promotion required.
    # Swap back to `models:/{mlflow_model_name}@champion` once manual gating is wanted again.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{mlflow_model_name}'")
    latest_version = max(int(v.version) for v in versions)
    return mlflow.xgboost.load_model(f"models:/{mlflow_model_name}/{latest_version}")


def select_latest_week(features: pd.DataFrame) -> pd.DataFrame:
    """The dropna in join_features can leave more than one surviving week; keep only the newest."""
    latest_week = features["week_start"].max()
    return features[features["week_start"] == latest_week].copy()


def compute_top_k(
    model: XGBRegressor, latest_features: pd.DataFrame, feature_cols: list, target_col: str, top_k: int,
) -> pd.DataFrame:
    pred_col = f"pred_{target_col}"
    scored = inference(model, latest_features, feature_cols, target_col)
    return scored.sort_values(pred_col, ascending=False).head(top_k)[["board_key", "week_start", pred_col]]


def format_tweet(top_k_districts: pd.DataFrame, target_col: str) -> str:
    pred_col = f"pred_{target_col}"
    week_start = top_k_districts["week_start"].iloc[0]
    lines = [
        f"NYC 311 forecast — week of {week_start}",
        "Predicted top community districts by call volume:",
    ]
    for i, row in enumerate(top_k_districts.itertuples(), start=1):
        lines.append(f"{i}. {row.board_key} — ~{getattr(row, pred_col):,.0f} calls")
    return "\n".join(lines)


def publish_tweet(tweet_text: str) -> None:
    # Placeholder — no Twitter API credentials wired up yet. Prints so it's visible
    # in kedro/argo logs; swap in a real client call (e.g. tweepy) here.
    print(f"[publish_tweet] would post:\n{tweet_text}", file=sys.stderr)
