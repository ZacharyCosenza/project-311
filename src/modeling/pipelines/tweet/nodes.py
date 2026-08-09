import os
import sys
from datetime import date, timedelta

import pandas as pd
import tweepy
from xgboost import XGBRegressor

from modeling.pipelines.modeling.nodes import inference


def compute_predict_window(lookback_weeks: int) -> tuple[str, str]:
    """Narrow rolling window: enough history for lag_1/lag_2 plus the current week."""
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(weeks=lookback_weeks)).isoformat()
    return start_date, end_date


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
    lines = ["TEST_TWEET", f"Week of {week_start}:"]
    for i, row in enumerate(top_k_districts.itertuples(), start=1):
        pred = round(getattr(row, pred_col))
        lines.append(f"{i}. {row.board_key} — {pred:,} calls")
    return "\n".join(lines)


def publish_tweet(tweet_text: str) -> None:
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([api_key, api_secret, access_token, access_token_secret]):
        print(f"[publish_tweet] no Twitter credentials set, not posting. Would post:\n{tweet_text}", file=sys.stderr)
        return

    client = tweepy.Client(
        consumer_key=api_key, consumer_secret=api_secret,
        access_token=access_token, access_token_secret=access_token_secret,
    )
    response = client.create_tweet(text=tweet_text)
    print(f"[publish_tweet] posted: {response.data}", file=sys.stderr)
