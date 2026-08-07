from kedro.pipeline import Pipeline, node, pipeline

from .nodes import featurize_events, featurize_lags, featurize_weather, join_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=featurize_lags,
            inputs=["target", "params:target_col"],
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
            func=join_features,
            inputs=[
                "target", "lag_features", "event_features", "weather_features",
                "params:feature_cols", "params:categorical_features",
            ],
            outputs="features",
            name="join_features",
        ),
    ])
