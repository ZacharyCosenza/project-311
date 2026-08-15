from kedro.pipeline import Pipeline

from modeling.pipelines.features.pipeline import create_pipeline as features_pipeline
from modeling.pipelines.inference.pipeline import create_pipeline as inference_pipeline
from modeling.pipelines.modeling.pipeline import create_pipeline as modeling_pipeline
from modeling.pipelines.raw.pipeline import create_pipeline as raw_pipeline
from modeling.pipelines.target.pipeline import create_pipeline as target_pipeline
from modeling.pipelines.tweet.pipeline import create_daily_pipeline as tweet_daily_pipeline
from modeling.pipelines.tweet.pipeline import create_summary_pipeline as tweet_summary_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    raw = raw_pipeline()
    target = target_pipeline()
    features = features_pipeline()
    modeling = modeling_pipeline()
    inference = inference_pipeline()
    tweet_summary = tweet_summary_pipeline()
    tweet_daily = tweet_daily_pipeline()
    train = raw + target + features + modeling

    return {
        "__default__": train,
        "raw": raw,
        "target": target,
        "features": features,
        "modeling": modeling,
        "inference": inference,
        "tweet_summary": tweet_summary,
        "tweet_daily": tweet_daily,
    }
