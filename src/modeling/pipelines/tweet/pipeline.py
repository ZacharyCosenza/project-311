from kedro.pipeline import Pipeline, node, pipeline

from modeling.pipelines.features.nodes import featurize_events, featurize_lags, featurize_weather, join_features
from modeling.pipelines.raw.nodes import fetch_calls_weekly, fetch_events_weekly, fetch_weather_weekly
from modeling.pipelines.target.nodes import build_target

from .nodes import (
    compute_predict_window,
    compute_top_k,
    format_tweet,
    load_latest_model,
    publish_tweet,
    select_latest_week,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_predict_window,
            inputs="params:lookback_weeks",
            outputs=["tweet_start_date", "tweet_end_date"],
            name="compute_predict_window",
        ),
        node(
            func=fetch_calls_weekly,
            inputs=["tweet_start_date", "tweet_end_date", "params:calls_url", "params:raw_dir"],
            outputs="tweet_calls_weekly",
            name="tweet_fetch_calls",
        ),
        node(
            func=fetch_events_weekly,
            inputs=[
                "tweet_start_date", "tweet_end_date", "params:events_url",
                "params:event_include_types", "params:raw_dir",
            ],
            outputs="tweet_events_weekly",
            name="tweet_fetch_events",
        ),
        node(
            func=fetch_weather_weekly,
            inputs=[
                "tweet_start_date", "tweet_end_date",
                "params:weather_lat", "params:weather_lon", "params:weather_daily_vars",
                "params:weather_archive_url", "params:weather_forecast_url",
                "params:raw_dir",
            ],
            outputs=["tweet_weather_lag1", "tweet_weather_pred"],
            name="tweet_fetch_weather",
        ),
        node(
            func=build_target,
            inputs=["tweet_calls_weekly", "params:target_col"],
            outputs="tweet_target",
            name="tweet_build_target",
        ),
        node(
            func=featurize_lags,
            inputs=["tweet_target", "params:target_col"],
            outputs="tweet_lag_features",
            name="tweet_featurize_lags",
        ),
        node(
            func=featurize_events,
            inputs="tweet_events_weekly",
            outputs="tweet_event_features",
            name="tweet_featurize_events",
        ),
        node(
            func=featurize_weather,
            inputs=["tweet_weather_lag1", "tweet_weather_pred"],
            outputs="tweet_weather_features",
            name="tweet_featurize_weather",
        ),
        node(
            func=join_features,
            inputs=[
                "tweet_target", "tweet_lag_features", "tweet_event_features", "tweet_weather_features",
                "params:feature_cols", "params:categorical_features",
            ],
            outputs="tweet_features",
            name="tweet_join_features",
        ),
        node(
            func=select_latest_week,
            inputs="tweet_features",
            outputs="tweet_latest_features",
            name="select_latest_week",
        ),
        node(
            func=load_latest_model,
            inputs=["params:mlflow_tracking_uri", "params:mlflow_model_name"],
            outputs="latest_model",
            name="load_latest_model",
        ),
        node(
            func=compute_top_k,
            inputs=[
                "latest_model", "tweet_latest_features",
                "params:feature_cols", "params:target_col", "params:top_k",
            ],
            outputs="top_k_districts",
            name="compute_top_k",
        ),
        node(
            func=format_tweet,
            inputs=["top_k_districts", "params:target_col"],
            outputs="tweet_text",
            name="format_tweet",
        ),
        node(
            func=publish_tweet,
            inputs="tweet_text",
            outputs=None,
            name="publish_tweet",
        ),
    ])
