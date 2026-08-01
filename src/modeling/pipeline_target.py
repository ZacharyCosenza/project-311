import os
from pathlib import Path

import pandas as pd

from modeling import pipeline_raw

TARGET_COL = "tgt_calls"

ENV = os.environ.get("MODELING_ENV", "dev")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / ENV / "01_processed"


def build_spine(calls: pd.DataFrame) -> pd.DataFrame:
    boards = sorted(calls["board_key"].unique())
    weeks = sorted(calls["week_start"].unique())
    return pd.DataFrame(
        [(b, w) for b in boards for w in weeks],
        columns=["board_key", "week_start"],
    )


def build_target(calls: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    target = spine.merge(calls, on=["board_key", "week_start"], how="left")
    target["calls"] = target["calls"].fillna(0)
    return target.rename(columns={"calls": TARGET_COL})


def run(calls: pd.DataFrame | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if calls is None:
        calls = pd.read_parquet(pipeline_raw.RAW_DIR / "calls_weekly.parquet")

    spine = build_spine(calls)
    target = build_target(calls, spine)
    target.to_parquet(PROCESSED_DIR / "target.parquet")
    return target


if __name__ == "__main__":
    run()
