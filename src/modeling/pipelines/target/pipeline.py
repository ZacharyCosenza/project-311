from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_target


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_target,
            inputs=["calls_weekly", "params:target_col"],
            outputs="target",
            name="build_target",
        ),
    ])
