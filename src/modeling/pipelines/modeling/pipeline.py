from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_metrics, log_to_mlflow, plot_feature_histograms, training


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=training,
            inputs=[
                "features",
                "params:feature_cols", "params:categorical_features",
                "params:target_col", "params:stratify_col", "params:split_col",
                "params:model_params", "params:test_size", "params:val_size", "params:random_state",
            ],
            outputs=["model", "modeling_data"],
            name="train_model",
        ),
        node(
            func=compute_metrics,
            inputs=[
                "model", "modeling_data",
                "params:feature_cols", "params:target_col", "params:split_col", "params:ranking_k",
            ],
            outputs="metrics",
            name="compute_metrics",
        ),
        node(
            func=log_to_mlflow,
            inputs=[
                "model", "modeling_data", "metrics",
                "params:feature_cols", "params:categorical_features", "params:split_col",
                "params:mlflow_tracking_uri", "params:mlflow_experiment",
            ],
            outputs=None,
            name="log_to_mlflow",
        ),
        node(
            func=plot_feature_histograms,
            inputs=[
                "model", "modeling_data",
                "params:feature_cols", "params:categorical_features",
                "params:split_col", "params:report_dir",
            ],
            outputs=None,
            name="plot_histograms",
        ),
    ])
