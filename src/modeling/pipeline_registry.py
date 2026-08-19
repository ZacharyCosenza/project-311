from kedro.pipeline import Pipeline

from modeling.pipelines.inference.pipeline import create_pipeline as inference_pipeline
from modeling.pipelines.multihead.pipeline import create_pipeline as multihead_pipeline
from modeling.pipelines.tweet.pipeline import create_daily_pipeline as tweet_daily_pipeline
from modeling.pipelines.tweet.pipeline import create_summary_pipeline as tweet_summary_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    multihead = multihead_pipeline()
    inference = inference_pipeline()
    tweet_summary = tweet_summary_pipeline()
    tweet_daily = tweet_daily_pipeline()

    return {
        "__default__": multihead,
        "multihead": multihead,
        "inference": inference,
        "tweet_summary": tweet_summary,
        "tweet_daily": tweet_daily,
    }
