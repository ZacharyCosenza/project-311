import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

RETRIES = 3
BACKOFF_SECONDS = 5.0


def _iso_year(yr: int, mo: int, woy: int) -> int:
    if mo == 1 and woy >= 52:
        return yr - 1
    if mo == 12 and woy == 1:
        return yr + 1
    return yr


def _iso_week_monday(yr: int, mo: int, woy: int) -> date:
    return date.fromisocalendar(_iso_year(yr, mo, woy), woy, 1)


def _with_fallback(fetch, name: str, *fallback_paths: str):
    """Retry `fetch`, falling back to the last saved raw parquet(s) on repeated failure."""
    for attempt in range(1, RETRIES + 1):
        try:
            return fetch()
        except Exception as e:
            print(f"[{name}] attempt {attempt}/{RETRIES} failed: {e}", file=sys.stderr)
            if attempt < RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)

    print(f"[{name}] all {RETRIES} attempts failed, falling back to previous raw data", file=sys.stderr)
    try:
        results = tuple(pd.read_parquet(p) for p in fallback_paths)
    except Exception as e:
        print(f"[{name}] no usable fallback data at {fallback_paths}: {e}", file=sys.stderr)
        raise
    return results[0] if len(results) == 1 else results


def fetch_calls_weekly(start_date: str, end_date: str, calls_url: str, raw_dir: str) -> pd.DataFrame:
    def fetch():
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

    return _with_fallback(fetch, "fetch_calls_weekly", str(Path(raw_dir) / "calls_weekly.parquet"))


def fetch_events_weekly(
    start_date: str, end_date: str, events_url: str, event_include_types: list, raw_dir: str
) -> pd.DataFrame:
    def fetch():
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

    return _with_fallback(fetch, "fetch_events_weekly", str(Path(raw_dir) / "events_weekly.parquet"))


def fetch_weather_weekly(
    start_date: str, end_date: str,
    weather_lat: float, weather_lon: float,
    weather_daily_vars: str,
    weather_archive_url: str, weather_forecast_url: str,
    raw_dir: str,
    forecast_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # forecast_end_date lets the forecast leg reach further ahead than the archive leg —
    # the archive (actual/historical) API can't return data for dates that haven't
    # happened yet, so it always stays bounded by end_date. Defaults to end_date,
    # matching training's behavior where both legs share one range.
    forecast_end_date = forecast_end_date or end_date

    def fetch():
        def fetch_one(base_url, end):
            rows = requests.get(base_url, params={
                "latitude": weather_lat, "longitude": weather_lon,
                "start_date": start_date, "end_date": end,
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

        actual = fetch_one(weather_archive_url, end_date)
        forecast = fetch_one(weather_forecast_url, forecast_end_date)

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

    return _with_fallback(
        fetch, "fetch_weather_weekly",
        str(Path(raw_dir) / "weather_lag1.parquet"), str(Path(raw_dir) / "weather_pred.parquet"),
    )
