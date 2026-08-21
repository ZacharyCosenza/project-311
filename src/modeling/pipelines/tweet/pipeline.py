from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    format_daily_deep_dive,
    format_delta_summary,
    format_weekly_summary,
    log_tweet_to_mlflow,
    plot_daily_trend,
    plot_delta_map,
    plot_district_map,
    publish_tweet,
    select_daily_district,
)


def create_summary_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=format_weekly_summary,
            inputs=["inference_results", "params:target_col", "params:top_k"],
            outputs="tweet_summary_text",
            name="format_weekly_summary",
        ),
        node(
            func=plot_district_map,
            inputs=[
                "inference_results", "params:target_col",
                "params:districts_url", "params:boro_names", "params:raw_dir", "params:report_dir",
                "params:map_colors", "params:districts_retries", "params:districts_backoff_seconds",
            ],
            outputs="district_map_path",
            name="plot_district_map",
        ),
        node(
            func=log_tweet_to_mlflow,
            inputs=[
                "tweet_summary_text", "district_map_path",
                "params:mlflow_tracking_uri", "params:mlflow_tweet_experiment", "params:report_dir",
            ],
            outputs=None,
            name="log_summary_to_mlflow",
        ),
        node(
            func=publish_tweet,
            inputs=["tweet_summary_text", "district_map_path"],
            outputs=None,
            name="publish_summary_tweet",
        ),
    ])


def create_delta_summary_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=format_delta_summary,
            inputs=["inference_results", "params:target_col", "params:top_k"],
            outputs="tweet_delta_summary_text",
            name="format_delta_summary",
        ),
        node(
            func=plot_delta_map,
            inputs=[
                "inference_results", "params:target_col",
                "params:districts_url", "params:boro_names", "params:raw_dir", "params:report_dir",
                "params:map_colors", "params:districts_retries", "params:districts_backoff_seconds",
            ],
            outputs="delta_map_path",
            name="plot_delta_map",
        ),
        node(
            func=log_tweet_to_mlflow,
            inputs=[
                "tweet_delta_summary_text", "delta_map_path",
                "params:mlflow_tracking_uri", "params:mlflow_tweet_delta_experiment", "params:report_dir",
            ],
            outputs=None,
            name="log_delta_summary_to_mlflow",
        ),
        node(
            func=publish_tweet,
            inputs=["tweet_delta_summary_text", "delta_map_path"],
            outputs=None,
            name="publish_delta_summary_tweet",
        ),
    ])


def create_daily_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=select_daily_district,
            inputs=["inference_results", "params:weekday_to_rank"],
            outputs="daily_district",
            name="select_daily_district",
        ),
        node(
            func=format_daily_deep_dive,
            inputs=["daily_district", "params:target_col"],
            outputs="tweet_daily_text",
            name="format_daily_deep_dive",
        ),
        node(
            func=plot_daily_trend,
            inputs=["daily_district", "modeling_data", "params:target_col", "params:report_dir"],
            outputs="daily_trend_path",
            name="plot_daily_trend",
        ),
        node(
            func=log_tweet_to_mlflow,
            inputs=[
                "tweet_daily_text", "daily_trend_path",
                "params:mlflow_tracking_uri", "params:mlflow_tweet_daily_experiment", "params:report_dir",
            ],
            outputs=None,
            name="log_daily_to_mlflow",
        ),
        node(
            func=publish_tweet,
            inputs=["tweet_daily_text", "daily_trend_path"],
            outputs=None,
            name="publish_daily_tweet",
        ),
    ])
