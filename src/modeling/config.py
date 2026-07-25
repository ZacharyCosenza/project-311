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

DISTRICTS_URL = "https://data.cityofnewyork.us/resource/5crt-au7u.geojson"
BORO_NAMES = {"1": "MANHATTAN", "2": "BRONX", "3": "BROOKLYN", "4": "QUEENS", "5": "STATEN ISLAND"}

FEATURES = [
    "week_of_year", "lag_1", "lag_2", "event_count",
    "lag1_temp_max", "lag1_temp_min", "lag1_had_rain", "lag1_had_snow",
    "pred_temp_max", "pred_temp_min", "pred_had_rain", "pred_had_snow",
]
TARGET = "calls"

MODEL_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
