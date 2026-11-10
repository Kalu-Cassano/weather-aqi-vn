"""Tạo đặc trưng (features) cho mô hình. Dùng CHUNG cho lúc train và lúc dự báo real-time,
để đảm bảo mô hình nhìn thấy dữ liệu giống hệt nhau ở cả hai lúc.

Ý tưởng: để dự báo giá trị ở thời điểm t + h, mô hình dùng
  - giá trị hiện tại và các giờ trước đó (lag 1, 2, 3, 6, 12, 24 giờ),
  - trung bình / độ lệch chuẩn 24 giờ qua,
  - thời điểm trong ngày và trong năm (mã hóa sin/cos để 23h và 0h nằm gần nhau),
  - vị trí tỉnh (vĩ độ, kinh độ, mã tỉnh).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config

TARGETS = ["temperature_2m", "pm2_5"]
HORIZONS = [1, 3, 6]  # dự báo trước 1, 3 và 6 giờ

TARGET_LAGS = [1, 2, 3, 6, 12, 24]
OTHER_LAGS = [1, 3, 24]

WEATHER_LAG_COLS = ["relative_humidity_2m", "pressure_msl", "wind_speed_10m",
                    "cloud_cover", "precipitation", "dew_point_2m"]
AQ_LAG_COLS = ["pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]

LOC_INDEX = {loc[0]: i for i, loc in enumerate(config.LOCATIONS)}
LOC_LATLON = {loc[0]: (loc[3], loc[4]) for loc in config.LOCATIONS}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df cần có cột: location, time + các biến thời tiết (+ các biến không khí nếu có).
    Mỗi tỉnh phải là chuỗi theo giờ liên tục, không bị thiếu giờ."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["location", "time"]).reset_index(drop=True)
    g = df.groupby("location", sort=False)

    new = {}
    # Thời gian
    hour = df["time"].dt.hour
    doy = df["time"].dt.dayofyear
    new["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    new["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    new["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    new["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # Vị trí
    new["loc_id"] = df["location"].map(LOC_INDEX)
    new["loc_lat"] = df["location"].map(lambda c: LOC_LATLON[c][0])
    new["loc_lon"] = df["location"].map(lambda c: LOC_LATLON[c][1])

    # Lag và thống kê trượt cho 2 biến mục tiêu
    for col in TARGETS:
        if col not in df:
            continue
        for lag in TARGET_LAGS:
            new[f"{col}_lag{lag}"] = g[col].shift(lag)
        new[f"{col}_mean24"] = g[col].transform(lambda s: s.rolling(24, min_periods=24).mean())
        new[f"{col}_std24"] = g[col].transform(lambda s: s.rolling(24, min_periods=24).std())
        new[f"{col}_diff1"] = df[col] - new[f"{col}_lag1"]
        new[f"{col}_diff24"] = df[col] - new[f"{col}_lag24"]

    # Lag cho các biến phụ
    for col in WEATHER_LAG_COLS + AQ_LAG_COLS:
        if col not in df:
            continue
        for lag in OTHER_LAGS:
            new[f"{col}_lag{lag}"] = g[col].shift(lag)

    # Áp suất thay đổi nhanh thường báo hiệu thời tiết đổi
    if "pressure_msl" in df:
        new["pressure_msl_diff3"] = df["pressure_msl"] - g["pressure_msl"].shift(3)

    feat = pd.DataFrame(new, index=df.index).astype("float32")
    return pd.concat([df, feat], axis=1)


def feature_columns(target: str, df_columns) -> list:
    """Danh sách cột đầu vào cho từng mục tiêu.
    - Nhiệt độ: chỉ dùng thời tiết (có dữ liệu nhiều năm).
    - PM2.5: dùng cả thời tiết lẫn chất lượng không khí."""
    base = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "loc_id", "loc_lat", "loc_lon"]
    weather_now = [v for v in config.WEATHER_VARS if v != "weather_code"]
    cols = base + weather_now
    cols += [c for c in df_columns if c.startswith("temperature_2m_")]
    cols += [c for c in df_columns
             if any(c.startswith(w + "_lag") for w in WEATHER_LAG_COLS)]
    if "pressure_msl_diff3" in df_columns:
        cols.append("pressure_msl_diff3")
    if target == "pm2_5":
        aq_now = [v for v in config.AQ_VARS if v != "us_aqi"]
        cols += aq_now
        cols += [c for c in df_columns if c.startswith("pm2_5_")]
        cols += [c for c in df_columns
                 if any(c.startswith(a + "_lag") for a in AQ_LAG_COLS)]
    # bỏ trùng, giữ thứ tự
    seen, out = set(), []
    for c in cols:
        if c in df_columns and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Lưu / đọc bảng dữ liệu (Parquet nếu có pyarrow, nếu không thì pickle)
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, path_no_ext: Path) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    except ImportError:
        p = path_no_ext.with_suffix(".pkl")
        df.to_pickle(p)
    return p


def load_table(path_no_ext: Path) -> pd.DataFrame:
    p = path_no_ext.with_suffix(".parquet")
    if p.exists():
        return pd.read_parquet(p)
    p = path_no_ext.with_suffix(".pkl")
    if p.exists():
        return pd.read_pickle(p)
    raise FileNotFoundError(f"Không thấy {path_no_ext}.parquet hoặc .pkl. Hãy chạy prepare_data.py trước.")
