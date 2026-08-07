import pandas as pd


def build_target(calls: pd.DataFrame, target_col: str) -> pd.DataFrame:
    boards = sorted(calls["board_key"].unique())
    weeks = sorted(calls["week_start"].unique())
    spine = pd.DataFrame(
        [(b, w) for b in boards for w in weeks],
        columns=["board_key", "week_start"],
    )
    target = spine.merge(calls, on=["board_key", "week_start"], how="left")
    target["calls"] = target["calls"].fillna(0)
    return target.rename(columns={"calls": target_col})
