"""Cloud mode: keep the web app's data up to date without the home PC.

Every hour, GitHub Actions collects new data, makes forecasts and saves the CSV files
in the folder live/ of the GitHub repository. On Streamlit Community Cloud, this module
downloads those files into data/ (the same place app.py already reads from), at most
once every 5 minutes. On the home PC it does nothing.

Turn it on by setting CLOUD_MODE = "true" in the app's secrets on Streamlit Cloud.

Usage, near the top of app.py (after the imports):
    import cloud_sync
    cloud_sync.sync()
"""
import os
import shutil
import time
import urllib.request
from pathlib import Path

import config

RAW_BASE = os.environ.get(
    "LIVE_DATA_URL",
    "https://raw.githubusercontent.com/Kalu-Cassano/weather-aqi-vn/main/live",
)
LOCAL_LIVE = config.ROOT / "live"
MAX_AGE_SECONDS = 300

FILES = {
    "realtime/weather_current.csv": config.REALTIME_DIR / "weather_current.csv",
    "realtime/aq_current.csv": config.REALTIME_DIR / "aq_current.csv",
    "realtime/om_forecast.csv": config.REALTIME_DIR / "om_forecast.csv",
    "realtime/predictions.csv": config.REALTIME_DIR / "predictions.csv",
    "processed/etl_report.json": config.DATA_DIR / "processed" / "etl_report.json",
    "processed/monthly_means.csv": config.DATA_DIR / "processed" / "monthly_means.csv",
    "processed/summary_by_province.csv": config.DATA_DIR / "processed" / "summary_by_province.csv",
}


def is_cloud() -> bool:
    if os.environ.get("CLOUD_MODE", "").lower() in ("1", "true", "yes"):
        return True
    try:  # Streamlit secrets (root keys are also env vars on Community Cloud, this is a fallback)
        import streamlit as st
        return str(st.secrets.get("CLOUD_MODE", "")).lower() in ("1", "true", "yes")
    except Exception:
        return False


def _fresh(path: Path, max_age: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age


def sync(max_age: int = MAX_AGE_SECONDS) -> None:
    if not is_cloud():
        return
    for rel, dst in FILES.items():
        if _fresh(dst, max_age):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        try:
            url = f"{RAW_BASE}/{rel}?t={int(time.time() // 60)}"
            req = urllib.request.Request(url, headers={"User-Agent": "weather-aqi-vn"})
            with urllib.request.urlopen(req, timeout=20) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(dst)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            # Fall back to the copy inside the deployed repo
            src = LOCAL_LIVE / rel
            if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
