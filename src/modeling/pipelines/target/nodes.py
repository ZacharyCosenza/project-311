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


def build_multi_target(calls: pd.DataFrame, calls_by_group: pd.DataFrame, complaint_type_groups: dict) -> pd.DataFrame:
    """Wide board x week spine with one tgt_<group> column per complaint_type_groups
    key, plus tgt_other — the residual after subtracting the named groups from calls
    (the existing, unfiltered total), not a separately fetched category. Zero-filled
    like build_target: a board-week absent from calls_by_group had zero calls in that
    group, not missing data.
    """
    boards = sorted(calls["board_key"].unique())
    weeks = sorted(calls["week_start"].unique())
    spine = pd.DataFrame([(b, w) for b in boards for w in weeks], columns=["board_key", "week_start"])

    spine = spine.merge(calls.rename(columns={"calls": "tgt_total"}), on=["board_key", "week_start"], how="left")
    spine["tgt_total"] = spine["tgt_total"].fillna(0)

    group_totals = pd.Series(0.0, index=spine.index)
    for group in complaint_type_groups:
        g = calls_by_group.loc[calls_by_group["group"] == group, ["board_key", "week_start", "calls"]]
        spine = spine.merge(g.rename(columns={"calls": f"tgt_{group}"}), on=["board_key", "week_start"], how="left")
        spine[f"tgt_{group}"] = spine[f"tgt_{group}"].fillna(0)
        group_totals += spine[f"tgt_{group}"]

    spine["tgt_other"] = (spine["tgt_total"] - group_totals).clip(lower=0)
    return spine.drop(columns="tgt_total")
