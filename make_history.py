"""Build the DAILY weather history used by the chatbot for questions about past days.

Reads the cleaned big table (data/processed/merged.parquet, about 1.5 million hourly rows made by
prepare_data.py) and computes, for each province and each day since Nov 2021:
min / max / mean temperature, total rain, max wind, mean PM2.5 and max US AQI.

Output (about 60,000 rows, a few MB):
  data/processed/daily_history.csv
  live/processed/daily_history.csv   (commit this folder so the online web app has it)

Run once (and again after you download more history):
    python make_history.py
"""
import shutil

import pandas as pd

import config
from features import load_table
from om_client import log

OUT = config.DATA_DIR / "processed" / "daily_history.csv"
LIVE = config.ROOT / "live" / "processed" / "daily_history.csv"


def main():
    log("Reading the cleaned hourly table...")
    df = load_table(config.DATA_DIR / "processed" / "merged")
    df["date"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    agg = {
        "temperature_2m": ["min", "max", "mean"],
        "precipitation": "sum",
        "wind_speed_10m": "max",
        "relative_humidity_2m": "mean",
    }
    if "pm2_5" in df:
        agg["pm2_5"] = "mean"
    if "us_aqi" in df:
        agg["us_aqi"] = "max"
    g = df.groupby(["location", "date"]).agg(agg)
    g.columns = ["tmin", "tmax", "tmean", "rain", "wind_max", "humidity"] + \
                (["pm25"] if "pm2_5" in df else []) + (["aqi_max"] if "us_aqi" in df else [])
    g = g.reset_index().round(1)
    counts = df.groupby(["location", "date"]).size().rename("hours").reset_index()
    g = g.merge(counts, on=["location", "date"])
    g = g[g["hours"] >= 18]  # skip days with too few hourly values
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT, index=False, encoding="utf-8")
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, LIVE)
    log(f"Daily history: {len(g):,} rows, {g['location'].nunique()} provinces, "
        f"{g['date'].min()} to {g['date'].max()} ({LIVE.stat().st_size / 1e6:.1f} MB) -> {LIVE}")


if __name__ == "__main__":
    main()
