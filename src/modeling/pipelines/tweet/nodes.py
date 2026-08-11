import os
import sys
from datetime import date, timedelta

import mlflow
import numpy as np
import pandas as pd
import tweepy
from xgboost import XGBRegressor

from modeling.pipelines.modeling.nodes import inference


def compute_predict_window(lookback_weeks: int) -> tuple[str, str, str]:
    """Narrow rolling window: enough history for lag_1/lag_2 plus the current week.

    Calls can only ever be fetched through today (next week hasn't happened yet), but
    events/weather need a separately *extended* end date reaching into next week —
    scheduled events and forecast weather for next week are legitimately knowable now.
    """
    today = date.today()
    end_date = today.isoformat()
    start_date = (today - timedelta(weeks=lookback_weeks)).isoformat()
    next_week_start = today - timedelta(days=today.weekday()) + timedelta(weeks=1)
    forecast_end_date = (next_week_start + timedelta(days=6)).isoformat()
    return start_date, end_date, forecast_end_date


def build_next_week_features(
    features: pd.DataFrame, event_features: pd.DataFrame, weather_features: pd.DataFrame,
    modeling_data: pd.DataFrame, target_col: str,
) -> pd.DataFrame:
    """Shift the current (most recent) week's row one week forward.

    The model learned a calendar-agnostic mapping: lag_1 (last week's calls), lag_2 (two
    weeks back), events scheduled for the target week, and the weather forecast for the
    target week -> that week's actual calls. To predict *next* week we construct exactly
    that: lag_1 becomes this week's own calls, lag_2 becomes this week's former lag_1,
    and event/weather features are freshly joined at the real next-week date rather than
    reused from this week's row.
    """
    current_week = features[features["week_start"] == features["week_start"].max()].copy()
    next_week_start = current_week["week_start"].iloc[0] + timedelta(weeks=1)

    next_week = pd.DataFrame({
        "board_key": current_week["board_key"].to_numpy(),
        "week_start": next_week_start,
        "ft_week_of_year": pd.Timestamp(next_week_start).isocalendar().week,
        "ft_lag_1": np.log1p(current_week[target_col].to_numpy()),
        "ft_lag_2": current_week["ft_lag_1"].to_numpy(),
    })
    # Category set must match what the model was trained on, not just whatever boards
    # happen to appear in this narrow window — a board with no recent calls would
    # otherwise be silently dropped from the categories, shifting XGBoost's categorical
    # codes out of alignment with what the model actually learned.
    board_categories = sorted(modeling_data["board_key"].unique())
    next_week["ft_board_key"] = pd.Categorical(next_week["board_key"], categories=board_categories)

    next_week = next_week.merge(event_features, on=["board_key", "week_start"], how="left")
    next_week["ft_event_count"] = next_week["ft_event_count"].fillna(0)
    next_week = next_week.merge(weather_features, on="week_start", how="left")
    return next_week


def compute_top_k(
    model: XGBRegressor, next_week_features: pd.DataFrame, feature_cols: list, target_col: str, top_k: int,
) -> pd.DataFrame:
    pred_col = f"pred_{target_col}"
    scored = inference(model, next_week_features, feature_cols, target_col)
    return scored.sort_values(pred_col, ascending=False).head(top_k)[["board_key", "week_start", pred_col]]


def format_tweet(top_k_districts: pd.DataFrame, target_col: str) -> str:
    pred_col = f"pred_{target_col}"
    week_start = top_k_districts["week_start"].iloc[0]
    lines = ["TEST_TWEET", f"Week of {week_start}:"]
    for i, row in enumerate(top_k_districts.itertuples(), start=1):
        pred = round(getattr(row, pred_col))
        lines.append(f"{i}. {row.board_key} — {pred:,} calls")
    return "\n".join(lines)


def log_tweet_to_mlflow(
    tweet_text: str, top_k_districts: pd.DataFrame, target_col: str,
    mlflow_tracking_uri: str, mlflow_tweet_experiment: str,
) -> None:
    pred_col = f"pred_{target_col}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_tweet_experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "week_start": str(top_k_districts["week_start"].iloc[0]),
            "top_board": top_k_districts["board_key"].iloc[0],
            "top_pred_calls": round(top_k_districts[pred_col].iloc[0]),
        })
        mlflow.log_text(tweet_text, "tweet.txt")
        mlflow.log_text(top_k_districts.to_csv(index=False), "top_k_districts.csv")


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
