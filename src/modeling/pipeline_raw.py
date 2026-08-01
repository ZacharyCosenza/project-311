import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

START_DATE = "2021-01-01"
END_DATE = "2026-07-25"

CALLS_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
EVENTS_URL = "https://data.cityofnewyork.us/resource/bkfu-528j.json"
EVENT_INCLUDE_TYPES = [
    "Parade", "Street Festival", "Single Block Festival", "Block Party",
    "Farmers Market", "Street Event", "Religious Event", "Plaza Event",
    "Plaza Partner Event", "Athletic Race / Tour", "Open Street Partner Event",
    "Health Fair", "Sidewalk Sale",
]
WEATHER_LAT, WEATHER_LON = 40.7812, -73.9665
WEATHER_DAILY_VARS = "temperature_2m_max,temperature_2m_min,rain_sum,snowfall_sum"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

ENV = os.environ.get("MODELING_ENV", "dev")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / ENV / "00_raw"


def iso_year(yr: int, mo: int, woy: int) -> int:
    if mo == 1 and woy >= 52:
        return yr - 1
    if mo == 12 and woy == 1:
        return yr + 1
    return yr


def iso_week_monday(yr: int, mo: int, woy: int) -> date:
    return date.fromisocalendar(iso_year(yr, mo, woy), woy, 1)


def fetch_calls_weekly(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    select = (
        "community_board, date_extract_y(created_date) as yr, "
        "date_extract_m(created_date) as mo, date_extract_woy(created_date) as woy, "
        "count(*) as calls"
    )
    where = f"created_date >= '{start}' and created_date <= '{end}'"
    group = "community_board, yr, mo, woy"
    r = requests.get(CALLS_URL,
                     params={"$select": select, "$where": where, "$group": group, "$limit": "50000"},
                     timeout=300)
    r.raise_for_status()

    df = pd.DataFrame(r.json()).dropna(subset=["community_board"])
    df["yr"] = df["yr"].astype(int)
    df["mo"] = df["mo"].astype(int)
    df["woy"] = df["woy"].astype(int)
    df["calls"] = df["calls"].astype(int)
    df["week_start"] = df.apply(lambda r: iso_week_monday(r["yr"], r["mo"], r["woy"]), axis=1)
    df["board_key"] = df["community_board"]
    return df.groupby(["board_key", "week_start"])["calls"].sum().reset_index()


def fetch_events_weekly(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    type_list = ",".join(f"'{t}'" for t in EVENT_INCLUDE_TYPES)
    where = (
        f"start_date_time <= '{end}' and end_date_time >= '{start}'"
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
    r = requests.get(EVENTS_URL,
                     params={"$select": select, "$where": where, "$group": group, "$limit": "50000"},
                     timeout=120)
    r.raise_for_status()

    df = pd.DataFrame(r.json()).dropna(subset=["community_board", "event_borough"])
    for col in ["syr", "smo", "swoy", "eyr", "emo", "ewoy", "n"]:
        df[col] = df[col].astype(int)

    df["boards"] = df["community_board"].str.split(",")
    df = df.explode("boards")
    df["boards"] = df["boards"].str.strip()
    df = df[df["boards"].str.fullmatch(r"\d{1,2}")]
    df["board_key"] = df["boards"].str.zfill(2) + " " + df["event_borough"].str.upper()

    df["week_start"] = df.apply(lambda r: iso_week_monday(r["syr"], r["smo"], r["swoy"]), axis=1)
    df["week_end"] = df.apply(lambda r: iso_week_monday(r["eyr"], r["emo"], r["ewoy"]), axis=1)
    df["n_weeks"] = df.apply(lambda r: (r["week_end"] - r["week_start"]).days // 7 + 1, axis=1)

    rows = []
    for _, r in df.iterrows():
        for i in range(r["n_weeks"]):
            rows.append({"board_key": r["board_key"], "week_start": r["week_start"] + timedelta(weeks=i), "n": r["n"]})
    return pd.DataFrame(rows).groupby(["board_key", "week_start"])["n"].sum().reset_index(name="event_count")


def fetch_weather_weekly(start: str = START_DATE, end: str = END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    def fetch(base_url):
        rows = requests.get(base_url, params={
            "latitude": WEATHER_LAT, "longitude": WEATHER_LON,
            "start_date": start, "end_date": end,
            "daily": WEATHER_DAILY_VARS, "timezone": "America/New_York",
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

    actual = fetch(WEATHER_ARCHIVE_URL)
    forecast = fetch(WEATHER_FORECAST_URL)

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


def run() -> dict[str, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    calls = fetch_calls_weekly()
    events = fetch_events_weekly()
    weather_lag1, weather_pred = fetch_weather_weekly()

    calls.to_parquet(RAW_DIR / "calls_weekly.parquet")
    events.to_parquet(RAW_DIR / "events_weekly.parquet")
    weather_lag1.to_parquet(RAW_DIR / "weather_lag1.parquet")
    weather_pred.to_parquet(RAW_DIR / "weather_pred.parquet")

    return {"calls": calls, "events": events, "weather_lag1": weather_lag1, "weather_pred": weather_pred}


if __name__ == "__main__":
    run()
