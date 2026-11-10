"""Thu dữ liệu THỜI GIAN THỰC mỗi giờ cho 34 tỉnh/thành. Hãy để chạy liên tục.

Mỗi vòng (mặc định 60 phút) ghi thêm vào thư mục data/realtime/:
  - weather_current.csv  : thời tiết hiện tại của từng tỉnh
  - aq_current.csv       : chất lượng không khí hiện tại của từng tỉnh
  - om_forecast.csv      : dự báo 1-6 giờ tới của chính Open-Meteo
                           (để sau này so sánh với mô hình nhóm tự train)

Nếu trong thư mục có file predictor.py (sẽ viết ở bước train mô hình),
script sẽ tự gọi nó mỗi vòng để ghi dự báo của mô hình nhóm,
KHÔNG cần tắt script này rồi bật lại.

Cách chạy:
    python collect_realtime.py            # chạy mãi, mỗi giờ một lần
    python collect_realtime.py --once     # chạy 1 vòng để kiểm tra
"""
import argparse
import csv
import importlib
import sys
import time
import traceback
from datetime import datetime, timedelta

import config
from om_client import get_json, log

FORECAST_HOURS = 6


def append_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerows(rows)


def multi_location_params():
    """Gọi 1 request cho cả 34 tỉnh (API cho phép nhiều tọa độ cách nhau bởi dấu phẩy)."""
    return {
        "latitude": ",".join(str(l[3]) for l in config.LOCATIONS),
        "longitude": ",".join(str(l[4]) for l in config.LOCATIONS),
        "timezone": config.TIMEZONE,
    }


def as_list(data):
    # Với nhiều tọa độ, API trả về danh sách; với 1 tọa độ, trả về 1 object
    return data if isinstance(data, list) else [data]


def collect_once():
    fetched_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 1) Thời tiết hiện tại + dự báo của Open-Meteo cho 6 giờ tới
    params = multi_location_params()
    params.update({
        "current": ",".join(config.WEATHER_VARS),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "forecast_hours": FORECAST_HOURS + 1,
    })
    weather = as_list(get_json(config.FORECAST_URL, params))

    cur_rows, fc_rows = [], []
    for loc, item in zip(config.LOCATIONS, weather):
        cur = item.get("current", {})
        cur_rows.append([fetched_at, loc[0], loc[1], cur.get("time"),
                         *[cur.get(v) for v in config.WEATHER_VARS]])
        hourly = item.get("hourly", {})
        for i, t in enumerate(hourly.get("time", [])):
            fc_rows.append([fetched_at, loc[0], t,
                            hourly["temperature_2m"][i],
                            hourly["relative_humidity_2m"][i],
                            hourly["precipitation"][i]])

    append_rows(config.REALTIME_DIR / "weather_current.csv",
                ["fetched_at", "location", "province", "time", *config.WEATHER_VARS],
                cur_rows)
    append_rows(config.REALTIME_DIR / "om_forecast.csv",
                ["fetched_at", "location", "target_time",
                 "temperature_2m", "relative_humidity_2m", "precipitation"],
                fc_rows)

    # 2) Chất lượng không khí hiện tại
    params = multi_location_params()
    params["current"] = ",".join(config.AQ_VARS)
    aq = as_list(get_json(config.AQ_URL, params))
    aq_rows = []
    for loc, item in zip(config.LOCATIONS, aq):
        cur = item.get("current", {})
        aq_rows.append([fetched_at, loc[0], loc[1], cur.get("time"),
                        *[cur.get(v) for v in config.AQ_VARS]])
    append_rows(config.REALTIME_DIR / "aq_current.csv",
                ["fetched_at", "location", "province", "time", *config.AQ_VARS],
                aq_rows)

    log(f"Đã ghi {len(cur_rows)} dòng thời tiết, {len(aq_rows)} dòng AQI, "
        f"{len(fc_rows)} dòng dự báo Open-Meteo.")


def run_predictor():
    """Nếu đã có predictor.py (mô hình nhóm), gọi nó. Nạp lại mỗi vòng để cập nhật code mới."""
    if not (config.ROOT / "predictor.py").exists():
        return
    try:
        if "predictor" in sys.modules:
            module = importlib.reload(sys.modules["predictor"])
        else:
            module = importlib.import_module("predictor")
        module.run()
    except Exception:
        log("Lỗi khi chạy predictor.py (vẫn tiếp tục thu dữ liệu):")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="chỉ chạy 1 vòng rồi thoát")
    ap.add_argument("--interval", type=int, default=60, help="số phút giữa 2 lần thu (mặc định 60)")
    args = ap.parse_args()

    log(f"Bắt đầu thu dữ liệu real-time cho {len(config.LOCATIONS)} tỉnh/thành. "
        f"Lưu tại: {config.REALTIME_DIR}")
    while True:
        try:
            collect_once()
            run_predictor()
        except Exception:
            # Không bao giờ dừng hẳn: lỗi 1 vòng thì ghi log và đợi vòng sau
            log("Vòng này bị lỗi, sẽ thử lại ở vòng sau:")
            traceback.print_exc()
        if args.once:
            break
        # Căn vào phút thứ 5 của mỗi giờ (lúc dữ liệu giờ mới đã sẵn sàng)
        now = datetime.now()
        if args.interval == 60:
            nxt = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
        else:
            nxt = now + timedelta(minutes=args.interval)
        log(f"Lần thu tiếp theo lúc {nxt:%H:%M}.")
        time.sleep(max(1, (nxt - datetime.now()).total_seconds()))


if __name__ == "__main__":
    main()
