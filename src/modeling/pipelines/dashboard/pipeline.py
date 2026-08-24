from kedro.pipeline import Pipeline, node

from .nodes import export_dashboard_json


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=export_dashboard_json,
            inputs=["modeling_data", "inference_results"],
            outputs="dashboard_json",
            name="export_dashboard_json",
        ),
    ])
