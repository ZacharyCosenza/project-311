from datetime import date, timedelta

import pandas as pd
import requests


def _iso_year(yr: int, mo: int, woy: int) -> int:
    if mo == 1 and woy >= 52:
        return yr - 1
    if mo == 12 and woy == 1:
        return yr + 1
    return yr


def _iso_week_monday(yr: int, mo: int, woy: int) -> date:
    return date.fromisocalendar(_iso_year(yr, mo, woy), woy, 1)


def fetch_calls_weekly(start_date: str, end_date: str, calls_url: str) -> pd.DataFrame:
    select = (
        "community_board, date_extract_y(created_date) as yr, "
        "date_extract_m(created_date) as mo, date_extract_woy(created_date) as woy, "
        "count(*) as calls"
    )
    where = f"created_date >= '{start_date}' and created_date <= '{end_date}'"
    group = "community_board, yr, mo, woy"
    r = requests.get(
        calls_url,
        params={"$select": select, "$where": where, "$group": group, "$limit": "50000"},
        timeout=300,
    )
    r.raise_for_status()

    df = pd.DataFrame(r.json()).dropna(subset=["community_board"])
    df["yr"] = df["yr"].astype(int)
    df["mo"] = df["mo"].astype(int)
    df["woy"] = df["woy"].astype(int)
    df["calls"] = df["calls"].astype(int)
    df["week_start"] = df.apply(lambda r: _iso_week_monday(r["yr"], r["mo"], r["woy"]), axis=1)
    df["board_key"] = df["community_board"]
    return df.groupby(["board_key", "week_start"])["calls"].sum().reset_index()


def fetch_events_weekly(
    start_date: str, end_date: str, events_url: str, event_include_types: list
) -> pd.DataFrame:
    type_list = ",".join(f"'{t}'" for t in event_include_types)
    where = (
        f"start_date_time <= '{end_date}' and end_date_time >= '{start_date}'"
        f" and event_type in({type_list})"
    )
    select = (
        "community_board, event_borough, "
        "date_extract_y(start_date_time) as syr, date_extract_m(start_date_time) as smo, "
        "date_extract_woy(start_date_time) as swoy, "
        "date_extract_y(end_date_time) as eyr, date_extract_m(end_date_time) as emo, "
        "date_extract_woy(end_date_time) as ewoy, count(*) as n"
    )
    group = "community_board, event_borough, syr, smo, swoy, eyr, emo, ewoy"
    r = requests.get(
        events_url,
        params={"$select": select, "$where": where, "$group": group, "$limit": "50000"},
        timeout=120,
    )
    r.raise_for_status()

    df = pd.DataFrame(r.json()).dropna(subset=["community_board", "event_borough"])
    for col in ["syr", "smo", "swoy", "eyr", "emo", "ewoy", "n"]:
        df[col] = df[col].astype(int)

    df["boards"] = df["community_board"].str.split(",")
    df = df.explode("boards")
    df["boards"] = df["boards"].str.strip()
    df = df[df["boards"].str.fullmatch(r"\d{1,2}")]
    df["board_key"] = df["boards"].str.zfill(2) + " " + df["event_borough"].str.upper()

    df["week_start"] = df.apply(lambda r: _iso_week_monday(r["syr"], r["smo"], r["swoy"]), axis=1)
    df["week_end"] = df.apply(lambda r: _iso_week_monday(r["eyr"], r["emo"], r["ewoy"]), axis=1)
    df["n_weeks"] = df.apply(lambda r: (r["week_end"] - r["week_start"]).days // 7 + 1, axis=1)

    rows = []
    for _, row in df.iterrows():
        for i in range(row["n_weeks"]):
            rows.append({
                "board_key": row["board_key"],
                "week_start": row["week_start"] + timedelta(weeks=i),
                "n": row["n"],
            })
    return pd.DataFrame(rows).groupby(["board_key", "week_start"])["n"].sum().reset_index(name="event_count")


def fetch_weather_weekly(
    start_date: str, end_date: str,
    weather_lat: float, weather_lon: float,
    weather_daily_vars: str,
    weather_archive_url: str, weather_forecast_url: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def fetch(base_url):
        rows = requests.get(base_url, params={
            "latitude": weather_lat, "longitude": weather_lon,
            "start_date": start_date, "end_date": end_date,
            "daily": weather_daily_vars, "timezone": "America/New_York",
        }, timeout=60).json()["daily"]
        w = pd.DataFrame(rows)
        w["time"] = pd.to_datetime(w["time"])
        w["week_start"] = (w["time"] - pd.to_timedelta(w["time"].dt.weekday, unit="D")).dt.date
        return w.groupby("week_start").agg(
            temp_max=("temperature_2m_max", "max"),
            temp_min=("temperature_2m_min", "min"),
            had_rain=("rain_sum", lambda s: int((s > 0).any())),
            had_snow=("snowfall_sum", lambda s: int((s > 0).any())),
        ).reset_index()

    actual = fetch(weather_archive_url)
    forecast = fetch(weather_forecast_url)

    lag1 = actual.rename(columns={
        "temp_max": "lag1_temp_max", "temp_min": "lag1_temp_min",
        "had_rain": "lag1_had_rain", "had_snow": "lag1_had_snow",
    }).copy()
    lag1["week_start"] = lag1["week_start"] + timedelta(weeks=1)

    pred = forecast.rename(columns={
        "temp_max": "pred_temp_max", "temp_min": "pred_temp_min",
        "had_rain": "pred_had_rain", "had_snow": "pred_had_snow",
    })
    return lag1, pred
