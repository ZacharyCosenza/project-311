from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests_cache

BASE_URL = 'https://data.cityofnewyork.us/resource/erm2-nwe9.json'
CACHE_PATH = '../data/00_cache/311'


def _paginated_get(params, limit=50000, max_offset=2_000_000, timeout=120):
    session = requests_cache.CachedSession(CACHE_PATH, expire_after=3600)
    frames = []
    for offset in range(0, max_offset, limit):
        p = dict(params, **{'$limit': str(limit), '$offset': str(offset)})
        r = session.get(BASE_URL, params=p, timeout=timeout)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        print(f'  fetched {offset + len(batch):,}')
        if len(batch) < limit:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_311(days: int = 90) -> pd.DataFrame:
    """Fetch all 311 service requests from the last N days."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return fetch_311_range(start.strftime('%Y-%m-%dT%H:%M:%S'), end.strftime('%Y-%m-%dT%H:%M:%S'))


def fetch_311_range(start: str, end: str) -> pd.DataFrame:
    """Fetch raw 311 service requests between two ISO timestamps."""
    where = f"created_date >= '{start}' and created_date <= '{end}'"
    df = _paginated_get({'$where': where, '$order': 'created_date ASC'})
    if df.empty:
        print('WARNING: no data returned')
        return df
    df['created_date'] = pd.to_datetime(df['created_date'])
    for col in ['latitude', 'longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'location' in df.columns:
        df = df.drop(columns=['location'])
    print(f"311: {len(df):,} records | {df['created_date'].min().date()} to {df['created_date'].max().date()}")
    return df


def iso_year(yr, mo, woy):
    """Resolve the true ISO week-year for a (Gregorian yr, month, ISO week) triple.
    date_extract_y is the Gregorian calendar year, which disagrees with the ISO
    week-year right at year boundaries (e.g. Jan 1 can fall in ISO week 52/53 of
    the prior year; Dec 29-31 can fall in ISO week 1 of the next year). Month
    disambiguates which side of the boundary a given (yr, woy) group is on."""
    if mo == 1 and woy >= 52:
        return yr - 1
    if mo == 12 and woy == 1:
        return yr + 1
    return yr


def iso_week_period(yr, mo, woy):
    """Build the pandas Period('W') for a (Gregorian yr, month, ISO week) triple."""
    return pd.Period(date.fromisocalendar(iso_year(yr, mo, woy), woy, 1), freq='W')


def fetch_311_weekly_by_board(start: str, end: str) -> pd.DataFrame:
    """Fetch 311 call counts aggregated by community_board and ISO week, computed
    server-side so no raw rows are transferred. Returns one row per
    (community_board, year_week) with a 'calls' count."""
    select = (
        'community_board, date_extract_y(created_date) as yr, '
        'date_extract_m(created_date) as mo, '
        'date_extract_woy(created_date) as woy, count(*) as calls'
    )
    where = f"created_date >= '{start}' and created_date <= '{end}'"
    group = 'community_board, yr, mo, woy'
    df = _paginated_get({'$select': select, '$where': where, '$group': group}, timeout=300)
    df = df.dropna(subset=['community_board']).copy()
    df['yr'] = df['yr'].astype(int)
    df['mo'] = df['mo'].astype(int)
    df['woy'] = df['woy'].astype(int)
    df['calls'] = df['calls'].astype(int)
    df['year_week'] = df.apply(lambda r: iso_week_period(r['yr'], r['mo'], r['woy']), axis=1)
    df = df.groupby(['community_board', 'year_week'])['calls'].sum().reset_index()
    df = df.sort_values(['community_board', 'year_week'])
    print(f'311 weekly-by-board: {len(df):,} (board, week) rows | {start} to {end}')
    return df
