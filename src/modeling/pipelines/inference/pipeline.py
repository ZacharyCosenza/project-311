from kedro.pipeline import Pipeline, node, pipeline

from modeling.pipelines.features.nodes import featurize_events, featurize_lags, featurize_weather, join_features
from modeling.pipelines.raw.nodes import fetch_calls_weekly, fetch_events_weekly, fetch_weather_weekly
from modeling.pipelines.target.nodes import build_target

from .nodes import add_shap_reasons, build_next_week_features, compute_predict_window, rank_districts


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_predict_window,
            inputs="params:lookback_weeks",
            outputs=["inference_start_date", "inference_end_date", "inference_forecast_end_date"],
            name="compute_predict_window",
        ),
        node(
            func=fetch_calls_weekly,
            inputs=[
                "inference_start_date", "inference_end_date", "params:calls_url", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="inference_calls_weekly",
            name="inference_fetch_calls",
        ),
        node(
            func=fetch_events_weekly,
            inputs=[
                "inference_start_date", "inference_forecast_end_date", "params:events_url",
                "params:event_include_types", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="inference_events_weekly",
            name="inference_fetch_events",
        ),
        node(
            func=fetch_weather_weekly,
            inputs=[
                "inference_start_date", "inference_end_date",
                "params:weather_lat", "params:weather_lon", "params:weather_daily_vars",
                "params:weather_archive_url", "params:weather_forecast_url",
                "params:raw_dir", "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
                "inference_forecast_end_date",
            ],
            outputs=["inference_weather_lag1", "inference_weather_pred"],
            name="inference_fetch_weather",
        ),
        node(
            func=build_target,
            inputs=["inference_calls_weekly", "params:target_col"],
            outputs="inference_target",
            name="inference_build_target",
        ),
        node(
            func=featurize_lags,
            inputs=["inference_target", "params:target_col"],
            outputs="inference_lag_features",
            name="inference_featurize_lags",
        ),
        node(
            func=featurize_events,
            inputs="inference_events_weekly",
            outputs="inference_event_features",
            name="inference_featurize_events",
        ),
        node(
            func=featurize_weather,
            inputs=["inference_weather_lag1", "inference_weather_pred"],
            outputs="inference_weather_features",
            name="inference_featurize_weather",
        ),
        node(
            func=join_features,
            inputs=[
                "inference_target", "inference_lag_features", "inference_event_features",
                "inference_weather_features", "params:feature_cols", "params:categorical_features",
            ],
            outputs="inference_features",
            name="inference_join_features",
        ),
        node(
            func=build_next_week_features,
            inputs=[
                "inference_features", "inference_event_features", "inference_weather_features",
                "modeling_data", "params:target_col",
            ],
            outputs="inference_next_week_features",
            name="build_next_week_features",
        ),
        node(
            func=rank_districts,
            inputs=["model", "inference_next_week_features", "params:feature_cols", "params:target_col"],
            outputs="ranked_districts",
            name="rank_districts",
        ),
        node(
            func=add_shap_reasons,
            inputs=["model", "ranked_districts", "params:feature_cols", "params:top_reasons", "params:reason_map"],
            outputs="inference_results",
            name="add_shap_reasons",
        ),
    ])
