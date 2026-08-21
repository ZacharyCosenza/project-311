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


def build_grouped_target(
    calls: pd.DataFrame, calls_by_group: pd.DataFrame, complaint_type_groups: dict, real_board_keys: list,
) -> pd.DataFrame:
    """Wide board x week spine: one tgt_<group> column per complaint_type_groups key,
    plus tgt_other (residual, left null rather than zero-filled where calls doesn't
    cover a week) and tgt_calls (their sum — kept for tweet/nodes.py's
    plot_daily_trend). Spine comes from calls_by_group, the more resilient of the two
    fetches. Both inputs are filtered to real_board_keys first.
    """
    calls = calls[calls["board_key"].isin(real_board_keys)]
    calls_by_group = calls_by_group[calls_by_group["board_key"].isin(real_board_keys)]

    boards = sorted(calls_by_group["board_key"].unique())
    weeks = sorted(calls_by_group["week_start"].unique())
    spine = pd.DataFrame([(b, w) for b in boards for w in weeks], columns=["board_key", "week_start"])

    spine = spine.merge(calls.rename(columns={"calls": "tgt_total"}), on=["board_key", "week_start"], how="left")

    group_totals = pd.Series(0.0, index=spine.index)
    for group in complaint_type_groups:
        g = calls_by_group.loc[calls_by_group["group"] == group, ["board_key", "week_start", "calls"]]
        spine = spine.merge(g.rename(columns={"calls": f"tgt_{group}"}), on=["board_key", "week_start"], how="left")
        spine[f"tgt_{group}"] = spine[f"tgt_{group}"].fillna(0)
        group_totals += spine[f"tgt_{group}"]

    spine["tgt_other"] = (spine["tgt_total"] - group_totals).clip(lower=0)
    spine["tgt_calls"] = spine["tgt_other"] + group_totals
    return spine.drop(columns="tgt_total")
