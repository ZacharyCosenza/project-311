import json
from datetime import date

import pandas as pd


GROUPS = [
    "noise", "parking_vehicles", "heat_hot_water", "housing_conditions",
    "streets_traffic", "sanitation_trash", "water_sewer", "parks_trees",
    "homelessness", "pests_health", "other",
]


def export_dashboard_json(modeling_data: pd.DataFrame, inference_results: pd.DataFrame) -> str:
    weeks = sorted(str(w) for w in modeling_data["week_start"].unique())
    boards = sorted(modeling_data["board_key"].unique())
    week_idx = {w: i for i, w in enumerate(weeks)}

    series: dict[str, dict[str, list]] = {
        "total": {b: [None] * len(weeks) for b in boards},
        **{g: {b: [None] * len(weeks) for b in boards} for g in GROUPS},
    }

    for row in modeling_data.itertuples(index=False):
        w = str(row.week_start)
        b = row.board_key
        if w not in week_idx:
            continue
        wi = week_idx[w]
        series["total"][b][wi] = int(row.tgt_calls) if pd.notna(row.tgt_calls) else None
        for g in GROUPS:
            val = getattr(row, f"tgt_{g}", None)
            series[g][b][wi] = int(val) if val is not None and pd.notna(val) else None

    pred_week = str(inference_results["week_start"].iloc[0])
    prediction = {"week_start": pred_week, "total": {}, "ranks": {}}
    for row in inference_results.itertuples(index=False):
        prediction["total"][row.board_key] = round(float(row.pred_tgt_calls))
        prediction["ranks"][row.board_key] = int(row.rank) if pd.notna(row.rank) else None

    payload = {
        "updated": str(date.today()),
        "weeks": weeks,
        "boards": boards,
        "series": series,
        "prediction": prediction,
    }
    return json.dumps(payload, separators=(",", ":"))
