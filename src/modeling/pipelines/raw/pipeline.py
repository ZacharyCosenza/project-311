from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fetch_calls_weekly, fetch_events_weekly, fetch_weather_weekly


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fetch_calls_weekly,
            inputs=[
                "params:start_date", "params:end_date", "params:calls_url", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="calls_weekly",
            name="fetch_calls",
        ),
        node(
            func=fetch_events_weekly,
            inputs=[
                "params:start_date", "params:end_date", "params:events_url",
                "params:event_include_types", "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs="events_weekly",
            name="fetch_events",
        ),
        node(
            func=fetch_weather_weekly,
            inputs=[
                "params:start_date", "params:end_date",
                "params:weather_lat", "params:weather_lon",
                "params:weather_daily_vars",
                "params:weather_archive_url", "params:weather_forecast_url",
                "params:raw_dir",
                "params:raw_fetch_retries", "params:raw_fetch_backoff_seconds",
            ],
            outputs=["weather_lag1", "weather_pred"],
            name="fetch_weather",
        ),
    ])
