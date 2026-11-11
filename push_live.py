"""Backup for the cloud version: the home PC pushes its newest data to GitHub every hour.

GitHub Actions schedules are "best effort" and can be late or skipped. While the PC is on,
predictor.py calls run() after each hourly forecast, so the web app on Streamlit Cloud
always gets fresh data.

It works in a separate copy of the repository (data/live_repo, ignored by git), so it never
touches the files you are editing in VS Code. Rows from GitHub Actions and from the PC are
merged, so neither side loses data.

Test by hand:  python push_live.py
"""
import os
import subprocess
from datetime import datetime

import pandas as pd

import config
from om_client import log

REPO_DIR = config.DATA_DIR / "live_repo"
FILES = {  # file name -> columns that identify one row
    "weather_current.csv": ["fetched_at", "location"],
    "aq_current.csv": ["fetched_at", "location"],
    "om_forecast.csv": ["fetched_at", "location", "target_time"],
    "predictions.csv": ["made_at", "location", "target", "horizon_h"],
}


def _git(*args, cwd=REPO_DIR):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                       timeout=180, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[:300]}")
    return r.stdout.strip()


def _ensure_clone():
    if (REPO_DIR / ".git").exists():
        return
    url = _git("remote", "get-url", "origin", cwd=config.ROOT)
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    log("push_live: making a separate copy of the repository (first time only)...")
    _git("clone", "--depth", "20", "--branch", "main", url, str(REPO_DIR), cwd=REPO_DIR.parent)


def _merge(name, keys):
    local = config.REALTIME_DIR / name
    remote = REPO_DIR / "live" / "realtime" / name
    if not local.exists():
        return
    frames = [pd.read_csv(local, dtype=str)]
    if remote.exists():
        frames.insert(0, pd.read_csv(remote, dtype=str))
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in keys if k in df.columns]
    df = df.drop_duplicates(subset=keys or None, keep="last").sort_values(df.columns[0], kind="stable")
    remote.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(remote, index=False, encoding="utf-8")


def run():
    if os.environ.get("GITHUB_ACTIONS"):
        return  # never run inside GitHub Actions
    _ensure_clone()
    _git("config", "user.name", "weather-aqi-vn PC")
    _git("config", "user.email", "pc@weather-aqi-vn.local")
    for attempt in range(3):
        _git("fetch", "origin", "main")
        _git("reset", "--hard", "origin/main")
        for name, keys in FILES.items():
            _merge(name, keys)
        _git("add", "live/realtime")
        if not _git("status", "--porcelain", "live/realtime"):
            log("push_live: GitHub already has the newest data.")
            return
        _git("commit", "-m", f"Hourly data from PC {datetime.now():%Y-%m-%d %H:%M} (UTC+7)")
        try:
            _git("push", "origin", "HEAD:main")
            log("push_live: pushed new data to GitHub (the web app updates in a few minutes).")
            return
        except RuntimeError as e:
            log(f"push_live: push was rejected, trying again ({attempt + 1}/3): {e}")
    log("push_live: could not push this hour, will try again next hour.")


if __name__ == "__main__":
    run()
