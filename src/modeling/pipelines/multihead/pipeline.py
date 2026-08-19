from kedro.pipeline import Pipeline, node, pipeline

from modeling.pipelines.features.nodes import featurize_events, featurize_multi_lags, featurize_weather
from modeling.pipelines.raw.nodes import fetch_calls_weekly, fetch_calls_weekly_by_group, fetch_events_weekly, fetch_weather_weekly
from modeling.pipelines.target.nodes import build_multi_target

from .nodes import (
    compute_multi_metrics,
    drop_incomplete_multi_rows,
    join_multi_features,
    log_multi_to_mlflow,
    plot_multi_histograms,
    plot_multi_shap_beeswarm,
    plot_multi_timeseries,
    train_multi_head,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fetch_calls_weekly,
            inputs=[
                "params:start_date", "params:end_date", "params:calls_url", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="multihead_calls_weekly",
            name="multihead_fetch_calls",
        ),
        node(
            func=fetch_calls_weekly_by_group,
            inputs=[
                "params:start_date", "params:end_date", "params:calls_url", "params:complaint_type_groups",
                "params:raw_dir", "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="multihead_calls_by_group",
            name="multihead_fetch_calls_by_group",
        ),
        node(
            func=fetch_events_weekly,
            inputs=[
                "params:start_date", "params:end_date", "params:events_url",
                "params:event_include_types", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="multihead_events_weekly",
            name="multihead_fetch_events",
        ),
        node(
            func=fetch_weather_weekly,
            inputs=[
                "params:start_date", "params:end_date",
                "params:weather_lat", "params:weather_lon",
                "params:weather_daily_vars",
                "params:weather_archive_url", "params:weather_forecast_url",
                "params:raw_dir", "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs=["multihead_weather_lag1", "multihead_weather_pred"],
            name="multihead_fetch_weather",
        ),
        node(
            func=build_multi_target,
            inputs=["multihead_calls_weekly", "multihead_calls_by_group", "params:complaint_type_groups"],
            outputs="multi_target",
            name="build_multi_target",
        ),
        node(
            func=featurize_multi_lags,
            inputs=["multi_target", "params:complaint_type_groups", "params:max_lag_weeks", "params:year_offset_weeks"],
            outputs="multi_lag_features",
            name="featurize_multi_lags",
        ),
        node(
            func=featurize_events,
            inputs="multihead_events_weekly",
            outputs="multihead_event_features",
            name="multihead_featurize_events",
        ),
        node(
            func=featurize_weather,
            inputs=["multihead_weather_lag1", "multihead_weather_pred"],
            outputs="multihead_weather_features",
            name="multihead_featurize_weather",
        ),
        node(
            func=join_multi_features,
            inputs=["multi_target", "multi_lag_features", "multihead_event_features", "multihead_weather_features"],
            outputs="multi_joined_features",
            name="join_multi_features",
        ),
        node(
            func=drop_incomplete_multi_rows,
            inputs="multi_joined_features",
            outputs="multihead_features",
            name="drop_incomplete_multi_rows",
        ),
        node(
            func=train_multi_head,
            inputs=[
                "multihead_features", "params:complaint_type_groups", "params:shared_feature_cols",
                "params:max_lag_weeks", "params:year_offset_weeks",
                "params:stratify_col", "params:split_col",
                "params:model_params", "params:test_size", "params:val_size", "params:random_state",
            ],
            outputs=["multihead_models", "multihead_modeling_data"],
            name="train_multi_head",
        ),
        node(
            func=compute_multi_metrics,
            inputs=[
                "multihead_models", "multihead_modeling_data", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:max_lag_weeks", "params:split_col", "params:ranking_k",
            ],
            outputs="multihead_metrics",
            name="compute_multi_metrics",
        ),
        node(
            func=log_multi_to_mlflow,
            inputs=[
                "multihead_models", "multihead_modeling_data", "multihead_metrics", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:shared_categorical_features", "params:max_lag_weeks",
                "params:split_col", "params:mlflow_tracking_uri", "params:mlflow_multihead_experiment",
                "params:mlflow_multihead_model_name", "params:model_params", "params:multihead_report_dir",
            ],
            outputs=None,
            name="log_multi_to_mlflow",
        ),
        node(
            func=plot_multi_histograms,
            inputs=[
                "multihead_models", "multihead_modeling_data", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:shared_categorical_features", "params:max_lag_weeks",
                "params:split_col", "params:multihead_report_dir",
            ],
            outputs=None,
            name="plot_multi_histograms",
        ),
        node(
            func=plot_multi_timeseries,
            inputs=[
                "multihead_modeling_data", "params:complaint_type_groups", "params:shared_feature_cols",
                "params:shared_categorical_features", "params:max_lag_weeks", "params:multihead_report_dir",
            ],
            outputs=None,
            name="plot_multi_timeseries",
        ),
        node(
            func=plot_multi_shap_beeswarm,
            inputs=[
                "multihead_models", "multihead_modeling_data", "params:complaint_type_groups",
                "params:shared_feature_cols", "params:max_lag_weeks", "params:split_col", "params:multihead_report_dir",
            ],
            outputs=None,
            name="plot_multi_shap_beeswarm",
        ),
    ])
