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


def build_grouped_target(calls: pd.DataFrame, calls_by_group: pd.DataFrame, complaint_type_groups: dict) -> pd.DataFrame:
    """Wide board x week spine with one tgt_<group> column per complaint_type_groups
    key, plus tgt_other and tgt_calls.

    The spine's board/week universe comes from calls_by_group, not calls — ten
    independent, independently-fallback-protected fetches (one per group) are more
    resilient than the single unfiltered fetch_calls_weekly call: previously, a
    fallback on that one fetch (to its own last cached file) capped every group's date
    range to however stale that one cache happened to be, even on days the ten group
    fetches all succeeded fresh.

    tgt_other is still the residual against calls' own total, but only where that
    total actually has a matching board-week — where calls fell further behind than
    calls_by_group, tgt_other is left null rather than guessed at (e.g. zero), and
    such rows are dropped downstream (see drop_incomplete_grouped_rows) like any other
    incomplete row, instead of quietly claiming zero "other" complaints for weeks we
    simply don't have a total for.

    tgt_calls (= tgt_other + the named groups, i.e. calls' own total whenever it isn't
    null) is carried through modeling_data purely so tweet/nodes.py's plot_daily_trend
    — which plots modeling_data[target_col] and predates the grouped model — keeps
    working unchanged.
    """
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
