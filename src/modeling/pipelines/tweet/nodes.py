import os
import sys
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import tweepy
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")


def _load_districts(districts_url: str, raw_dir: str, retries: int, backoff_seconds: float) -> gpd.GeoDataFrame:
    """District boundaries essentially never change, so cache them locally after the
    first successful fetch — later runs don't depend on Socrata being up at all, rather
    than just retrying within a single run."""
    cache_path = Path(raw_dir) / "districts.parquet"

    for attempt in range(1, retries + 1):
        try:
            districts = gpd.read_file(f"{districts_url}?$limit=500")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            districts.to_parquet(cache_path)
            return districts
        except Exception as e:
            print(f"[plot_district_map] attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    if cache_path.exists():
        print("[plot_district_map] falling back to cached district boundaries", file=sys.stderr)
        return gpd.read_parquet(cache_path)
    raise RuntimeError(f"could not fetch district boundaries and no cache exists at {cache_path}")


def format_weekly_summary(inference_results: pd.DataFrame, target_col: str, top_k: int) -> str:
    pred_col = f"pred_{target_col}"
    top = inference_results.sort_values("rank").head(top_k)
    week_start = top["week_start"].iloc[0]
    leader = top.iloc[0]
    others = ", ".join(
        f"{row.board_key} (~{round(getattr(row, pred_col)):,})" for row in top.iloc[1:].itertuples()
    )
    return (
        f"NYC 311 forecast for the week of {week_start}: {leader['board_key']} is expected to "
        f"see the most call activity (~{round(leader[pred_col]):,} calls). "
        f"Also watching: {others}."
    )


def plot_district_map(
    inference_results: pd.DataFrame, target_col: str, districts_url: str, boro_names: dict,
    raw_dir: str, report_dir: str, map_colors: list, districts_retries: int, districts_backoff_seconds: float,
) -> str:
    """Every district shaded by predicted call volume, blue (low) to orange (high)."""
    pred_col = f"pred_{target_col}"
    districts = _load_districts(districts_url, raw_dir, districts_retries, districts_backoff_seconds)
    districts["board_key"] = districts["boro_cd"].apply(lambda s: s[1:].zfill(2) + " " + boro_names[s[0]])
    merged = districts.merge(inference_results[["board_key", pred_col]], on="board_key", how="left")

    cmap = LinearSegmentedColormap.from_list("blue_orange", map_colors)
    fig, ax = plt.subplots(figsize=(8, 8))
    merged.plot(
        column=pred_col, cmap=cmap, legend=True, ax=ax,
        edgecolor="white", linewidth=0.3, missing_kwds={"color": "lightgrey"},
        legend_kwds={"shrink": 0.6},
    )
    ax.set_axis_off()
    plt.tight_layout()

    out = Path(report_dir) / "district_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def select_daily_district(inference_results: pd.DataFrame, weekday_to_rank: dict) -> pd.DataFrame:
    rank = weekday_to_rank.get(date.today().weekday(), 1)
    return inference_results[inference_results["rank"] == rank]


def format_daily_deep_dive(daily_district: pd.DataFrame, target_col: str) -> str:
    pred_col = f"pred_{target_col}"
    row = daily_district.iloc[0]
    reasons = list(row["reasons"])

    if len(reasons) > 1:
        reason_text = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
    elif reasons:
        reason_text = reasons[0]
    else:
        reason_text = "typical activity patterns"

    return (
        f"District spotlight: {row['board_key']} is projected for ~{round(row[pred_col]):,} "
        f"311 calls the week of {row['week_start']}, driven by {reason_text}."
    )


def plot_daily_trend(
    daily_district: pd.DataFrame, modeling_data: pd.DataFrame, target_col: str, report_dir: str,
) -> str:
    """Last 12 weeks of actual call volume for the spotlighted district, plus a dotted
    line out to next week's prediction."""
    pred_col = f"pred_{target_col}"
    row = daily_district.iloc[0]
    history = modeling_data[modeling_data["board_key"] == row["board_key"]].sort_values("week_start").tail(12)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["week_start"], history[target_col], color="#2563eb", linewidth=2, marker="o", markersize=4)
    ax.plot(
        [history["week_start"].iloc[-1], row["week_start"]],
        [history[target_col].iloc[-1], row[pred_col]],
        color="#f97316", linewidth=2, linestyle="--", marker="o", markersize=4,
    )
    ax.set_ylabel("311 calls")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()

    out = Path(report_dir) / "daily_trend.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def log_tweet_to_mlflow(
    tweet_text: str, image_path: str, mlflow_tracking_uri: str, mlflow_experiment: str, report_dir: str,
) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(mlflow_experiment) is None:
        mlflow.create_experiment(mlflow_experiment, artifact_location=str(Path(report_dir) / "mlruns"))
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_text(tweet_text, "tweet.txt")
        mlflow.log_artifact(image_path)


def publish_tweet(tweet_text: str, image_path: str) -> None:
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    has_credentials = all([api_key, api_secret, access_token, access_token_secret])
    is_prod = os.environ.get("KEDRO_ENV") == "prod"

    if not (has_credentials and is_prod):
        reason = "no Twitter credentials set" if not has_credentials else "KEDRO_ENV is not \"prod\""
        print(
            f"[publish_tweet] {reason}, not posting. Would post (with {image_path}):\n{tweet_text}",
            file=sys.stderr,
        )
        return

    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    media = tweepy.API(auth).media_upload(image_path)

    client = tweepy.Client(
        consumer_key=api_key, consumer_secret=api_secret,
        access_token=access_token, access_token_secret=access_token_secret,
    )
    response = client.create_tweet(text=tweet_text, media_ids=[media.media_id])
    print(f"[publish_tweet] posted: {response.data}", file=sys.stderr)
