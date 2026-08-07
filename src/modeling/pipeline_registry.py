from kedro.pipeline import Pipeline

from modeling.pipelines.features.pipeline import create_pipeline as features_pipeline
from modeling.pipelines.modeling.pipeline import create_pipeline as modeling_pipeline
from modeling.pipelines.raw.pipeline import create_pipeline as raw_pipeline
from modeling.pipelines.target.pipeline import create_pipeline as target_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    raw = raw_pipeline()
    target = target_pipeline()
    features = features_pipeline()
    modeling = modeling_pipeline()
    train = raw + target + features + modeling

    return {
        "__default__": train,
        "raw": raw,
        "target": target,
        "features": features,
        "modeling": modeling,
    }
