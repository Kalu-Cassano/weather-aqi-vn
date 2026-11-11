"""DỰ BÁO REAL-TIME bằng mô hình nhóm đã train.

collect_realtime.py tự gọi hàm run() mỗi giờ (khi trong thư mục có file này và đã có
models/meta.json). Không cần khởi động lại collect_realtime.py.

Mỗi lần chạy:
  1. Lấy 48 giờ dữ liệu gần nhất của 34 tỉnh (để tính các đặc trưng lag 24 giờ).
  2. Tạo đặc trưng giống hệt lúc train (dùng chung features.py).
  3. Dự báo nhiệt độ và PM2.5 cho 1, 3, 6 giờ tới.
  4. Ghi thêm vào data/realtime/predictions.csv.

Chạy tay để thử:  python predictor.py
"""
import csv
import json
import os
from datetime import datetime

import pandas as pd

import config
from features import make_features
from om_client import get_json, log

# MODEL_DIR can be changed with an environment variable (GitHub Actions uses cloud_models/)
MODEL_DIR = config.ROOT / os.environ.get("MODEL_DIR", "models")
PRED_FILE = config.REALTIME_DIR / "predictions.csv"
HISTORY_HOURS = 48


def _multi_params():
    return {"latitude": ",".join(str(l[3]) for l in config.LOCATIONS),
            "longitude": ",".join(str(l[4]) for l in config.LOCATIONS),
            "timezone": config.TIMEZONE,
            "past_hours": HISTORY_HOURS, "forecast_hours": 1}


def _to_frame(items, variables):
    items = items if isinstance(items, list) else [items]
    frames = []
    for loc, item in zip(config.LOCATIONS, items):
        h = item.get("hourly", {})
        f = pd.DataFrame({"time": h.get("time", []), **{v: h.get(v) for v in variables}})
        f.insert(0, "location", loc[0])
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def fetch_recent() -> pd.DataFrame:
    p = _multi_params()
    p["hourly"] = ",".join(config.WEATHER_VARS)
    weather = _to_frame(get_json(config.FORECAST_URL, p), config.WEATHER_VARS)
    p = _multi_params()
    p["hourly"] = ",".join(config.AQ_VARS)
    aq = _to_frame(get_json(config.AQ_URL, p), config.AQ_VARS)
    return weather.merge(aq, on=["location", "time"], how="left")


def _load_models(meta):
    models = {}
    if meta["model"] == "XGBoost":
        import xgboost as xgb
        for t in meta["targets"]:
            for h in meta["horizons"]:
                m = xgb.XGBRegressor()
                path = MODEL_DIR / f"{t}_h{h}.json"
                if not path.exists():
                    path = MODEL_DIR / f"{t}_h{h}.ubj"
                m.load_model(path)
                m.set_params(device="cpu")  # dự báo vài chục dòng, CPU là đủ
                models[(t, h)] = m
    else:
        import joblib
        for t in meta["targets"]:
            for h in meta["horizons"]:
                models[(t, h)] = joblib.load(MODEL_DIR / f"{t}_h{h}.joblib")
    return models


def predict(recent: pd.DataFrame, meta: dict, models: dict) -> list:
    feats_df = make_features(recent)
    now_hour = pd.Timestamp(datetime.now()).floor("h")
    # Giờ gốc: giờ đầy đủ gần nhất <= hiện tại, của từng tỉnh
    feats_df = feats_df[feats_df["time"] <= now_hour]
    latest = feats_df.sort_values("time").groupby("location").tail(1)

    made_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    names = {loc[0]: loc[1] for loc in config.LOCATIONS}
    rows = []
    for (target, h), model in models.items():
        cols = meta["features"][target]
        X = latest[cols]
        ok = X.notna().all(axis=1)
        if not ok.any():
            continue
        preds = model.predict(X[ok])
        for (_, r), p in zip(latest[ok].iterrows(), preds):
            rows.append([made_at, r["location"], names[r["location"]],
                         r["time"].strftime("%Y-%m-%dT%H:%M"), target, h,
                         (r["time"] + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M"),
                         round(float(p), 2),
                         None if pd.isna(r[target]) else round(float(r[target]), 2)])
    return rows


def run():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        return  # chưa train xong thì bỏ qua, không báo lỗi
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = _load_models(meta)
    rows = predict(fetch_recent(), meta, models)

    new_file = not PRED_FILE.exists()
    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["made_at", "location", "province", "base_time", "target",
                        "horizon_h", "target_time", "prediction", "value_at_base"])
        w.writerows(rows)
    log(f"Mô hình nhóm: ghi {len(rows)} dự báo vào {PRED_FILE.name}")

    # On the home PC: also push the newest data to GitHub for the cloud web app (backup for
    # GitHub Actions). Any problem here is logged and never stops the hourly collection.
    if not os.environ.get("GITHUB_ACTIONS") and (config.ROOT / "push_live.py").exists():
        try:
            import importlib
            import sys
            import push_live
            if "push_live" in sys.modules:
                push_live = importlib.reload(sys.modules["push_live"])
            push_live.run()
        except Exception as e:
            log(f"push_live error (data collection continues): {e}")


if __name__ == "__main__":
    run()
