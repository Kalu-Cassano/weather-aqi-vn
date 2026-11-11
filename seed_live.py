"""Copy the data collected on this PC into the folder live/, so the cloud version starts
with the same history. Run once before the first commit of live/:

    python seed_live.py
"""
import shutil

import config

LIVE = config.ROOT / "live"
PAIRS = [
    (config.REALTIME_DIR / "weather_current.csv", LIVE / "realtime" / "weather_current.csv"),
    (config.REALTIME_DIR / "aq_current.csv", LIVE / "realtime" / "aq_current.csv"),
    (config.REALTIME_DIR / "om_forecast.csv", LIVE / "realtime" / "om_forecast.csv"),
    (config.REALTIME_DIR / "predictions.csv", LIVE / "realtime" / "predictions.csv"),
    (config.DATA_DIR / "processed" / "etl_report.json", LIVE / "processed" / "etl_report.json"),
    (config.DATA_DIR / "processed" / "monthly_means.csv", LIVE / "processed" / "monthly_means.csv"),
    (config.DATA_DIR / "processed" / "summary_by_province.csv", LIVE / "processed" / "summary_by_province.csv"),
]

for src, dst in PAIRS:
    if not src.exists():
        print(f"MISSING  {src}")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"OK       {dst.relative_to(config.ROOT)}  ({dst.stat().st_size / 1024:.0f} KB)")
print("Done. Now commit the folder live/ to GitHub.")
