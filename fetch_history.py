"""Tải dữ liệu LỊCH SỬ (dùng để train mô hình) cho 34 tỉnh/thành.

- Thời tiết theo giờ: Open-Meteo Historical Weather API (mặc định 5 năm gần nhất).
- Chất lượng không khí theo giờ: Open-Meteo Air Quality API (từ 08/2022).

Mỗi (tỉnh, năm) lưu thành 1 file CSV riêng. Nếu script bị ngắt giữa chừng,
chạy lại sẽ tự bỏ qua các file đã tải xong (tải tiếp, không tải lại từ đầu).

Cách chạy:
    python fetch_history.py              # 5 năm thời tiết + toàn bộ AQI
    python fetch_history.py --years 10   # nhiều dữ liệu hơn (lâu hơn)
    python fetch_history.py --only weather   hoặc   --only aq
"""
import argparse
import csv
import time
from datetime import date, timedelta

import config
from om_client import get_json, log

PAUSE_SECONDS = 3  # nghỉ giữa các lần gọi để không bị giới hạn


def write_csv(path, loc, hourly: dict, variables: list) -> int:
    code, province, place, lat, lon = loc
    times = hourly.get("time", [])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "province", "lat", "lon", "time", *variables])
        for i, t in enumerate(times):
            w.writerow([code, province, lat, lon, t,
                        *[hourly.get(v, [None] * len(times))[i] for v in variables]])
    tmp.replace(path)  # chỉ đổi tên khi ghi xong -> không bao giờ có file dở dang
    return len(times)


def year_ranges(start: date, end: date):
    y = start.year
    while y <= end.year:
        s = max(start, date(y, 1, 1))
        e = min(end, date(y, 12, 31))
        yield y, s, e
        y += 1


def fetch(kind: str, url: str, variables: list, start: date, end: date, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(loc, y, s, e) for loc in config.LOCATIONS for y, s, e in year_ranges(start, end)]
    total_rows, done = 0, 0
    log(f"[{kind}] {len(jobs)} phần cần tải ({start} -> {end})")

    for i, (loc, y, s, e) in enumerate(jobs, 1):
        # Năm hiện tại luôn tải lại vì dữ liệu còn đang tăng thêm
        path = out_dir / f"{loc[0]}_{y}.csv"
        if path.exists() and y != end.year:
            done += 1
            continue
        params = {
            "latitude": loc[3],
            "longitude": loc[4],
            "start_date": s.isoformat(),
            "end_date": e.isoformat(),
            "hourly": ",".join(variables),
            "timezone": config.TIMEZONE,
        }
        data = get_json(url, params)
        n = write_csv(path, loc, data.get("hourly", {}), variables)
        total_rows += n
        log(f"[{kind}] {i}/{len(jobs)}  {loc[1]} {y}: {n:,} dòng")
        time.sleep(PAUSE_SECONDS)

    log(f"[{kind}] XONG. Tải mới {total_rows:,} dòng, bỏ qua {done} phần đã có.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5, help="số năm dữ liệu thời tiết (mặc định 5)")
    ap.add_argument("--only", choices=["weather", "aq"], help="chỉ tải một loại")
    args = ap.parse_args()

    today = date.today()

    if args.only in (None, "weather"):
        # Dữ liệu lưu trữ ERA5 trễ khoảng 5 ngày, nên lấy đến 6 ngày trước cho chắc
        w_end = today - timedelta(days=6)
        w_start = date(today.year - args.years, today.month, 1)
        fetch("Thời tiết", config.ARCHIVE_URL, config.WEATHER_VARS,
              w_start, w_end, config.HISTORY_WEATHER_DIR)

    if args.only in (None, "aq"):
        aq_start = date.fromisoformat(config.AQ_EARLIEST_DATE)
        aq_end = today - timedelta(days=1)
        fetch("Không khí", config.AQ_URL, config.AQ_VARS,
              aq_start, aq_end, config.HISTORY_AQ_DIR)

    log("Hoàn tất tải dữ liệu lịch sử.")


if __name__ == "__main__":
    main()
