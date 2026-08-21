from kedro.pipeline import Pipeline, node, pipeline

from modeling.pipelines.features.nodes import featurize_events, featurize_grouped_lags, featurize_weather, join_grouped_features
from modeling.pipelines.raw.nodes import fetch_calls_weekly, fetch_calls_weekly_by_group, fetch_events_weekly, fetch_weather_weekly
from modeling.pipelines.target.nodes import build_grouped_target

from .nodes import (
    compute_grouped_metrics,
    compute_train_end_date,
    drop_incomplete_grouped_rows,
    log_groups_to_mlflow,
    plot_grouped_histograms,
    plot_grouped_shap_beeswarm,
    plot_grouped_timeseries,
    plot_metrics_comparison,
    train_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_train_end_date,
            inputs=None,
            outputs="train_end_date",
            name="compute_train_end_date",
        ),
        node(
            func=fetch_calls_weekly,
            inputs=[
                "params:start_date", "train_end_date", "params:calls_url", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="calls_weekly",
            name="fetch_calls",
        ),
        node(
            func=fetch_calls_weekly_by_group,
            inputs=[
                "params:start_date", "train_end_date", "params:calls_url", "params:complaint_type_groups",
                "params:raw_dir", "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="calls_by_group",
            name="fetch_calls_by_group",
        ),
        node(
            func=fetch_events_weekly,
            inputs=[
                "params:start_date", "train_end_date", "params:events_url",
                "params:event_include_types", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="events_weekly",
            name="fetch_events",
        ),
        node(
            func=fetch_weather_weekly,
            inputs=[
                "params:start_date", "train_end_date",
                "params:weather_lat", "params:weather_lon",
                "params:weather_daily_vars",
                "params:weather_archive_url", "params:weather_forecast_url",
                "params:raw_dir", "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs=["weather_lag1", "weather_pred"],
            name="fetch_weather",
        ),
        node(
            func=build_grouped_target,
            inputs=["calls_weekly", "calls_by_group", "params:complaint_type_groups"],
            outputs="target",
            name="build_target",
        ),
        node(
            func=featurize_grouped_lags,
            inputs=["target", "params:complaint_type_groups", "params:max_lag_weeks", "params:year_offset_weeks"],
            outputs="lag_features",
            name="featurize_lags",
        ),
        node(
            func=featurize_events,
            inputs="events_weekly",
            outputs="event_features",
            name="featurize_events",
        ),
        node(
            func=featurize_weather,
            inputs=["weather_lag1", "weather_pred"],
            outputs="weather_features",
            name="featurize_weather",
        ),
        node(
            func=join_grouped_features,
            inputs=["target", "lag_features", "event_features", "weather_features"],
            outputs="joined_features",
            name="join_features",
        ),
        node(
            func=drop_incomplete_grouped_rows,
            inputs="joined_features",
            outputs="features",
            name="drop_incomplete_rows",
        ),
        node(
            func=train_models,
            inputs=[
                "features", "params:complaint_type_groups", "params:shared_feature_cols",
                "params:max_lag_weeks", "params:year_offset_weeks",
                "params:stratify_col", "params:split_col",
                "params:model_params", "params:test_size", "params:val_size", "params:random_state",
            ],
            outputs=["models", "modeling_data"],
            name="train_models",
        ),
        node(
            func=compute_grouped_metrics,
            inputs=[
                "models", "modeling_data", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:max_lag_weeks", "params:split_col", "params:ranking_k",
            ],
            outputs="metrics",
            name="compute_metrics",
        ),
        node(
            func=plot_metrics_comparison,
            inputs=["metrics", "params:report_dir"],
            outputs=None,
            name="plot_metrics_comparison",
        ),
        node(
            func=log_groups_to_mlflow,
            inputs=[
                "models", "modeling_data", "metrics", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:shared_categorical_features", "params:max_lag_weeks",
                "params:split_col", "params:mlflow_tracking_uri", "params:mlflow_experiment",
                "params:mlflow_model_name", "params:model_params", "params:report_dir",
            ],
            outputs=None,
            name="log_to_mlflow",
        ),
        node(
            func=plot_grouped_histograms,
            inputs=[
                "models", "modeling_data", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:shared_categorical_features", "params:max_lag_weeks",
                "params:split_col", "params:report_dir",
            ],
            outputs=None,
            name="plot_histograms",
        ),
        node(
            func=plot_grouped_timeseries,
            inputs=[
                "modeling_data", "params:complaint_type_groups", "params:shared_feature_cols",
                "params:shared_categorical_features", "params:max_lag_weeks", "params:report_dir",
            ],
            outputs=None,
            name="plot_feature_timeseries",
        ),
        node(
            func=plot_grouped_shap_beeswarm,
            inputs=[
                "models", "modeling_data", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:max_lag_weeks", "params:split_col", "params:report_dir",
            ],
            outputs=None,
            name="plot_shap_beeswarm",
        ),
    ])
