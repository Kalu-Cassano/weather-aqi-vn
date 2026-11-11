"""Hàm gọi API Open-Meteo dùng chung, chỉ dùng thư viện chuẩn của Python.

Tự động thử lại khi mạng lỗi, và tự chờ khi bị giới hạn số lần gọi (HTTP 429).
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

USER_AGENT = "student-bigdata-project/1.0"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _wait_for_rate_limit(reason: str) -> int:
    """Chọn thời gian chờ (giây) dựa trên loại giới hạn mà server báo."""
    r = reason.lower()
    if "daily" in r:
        return 3600          # hết hạn mức ngày: chờ 1 tiếng rồi thử lại
    if "hourly" in r:
        return 600           # hết hạn mức giờ: chờ 10 phút
    return 65                # giới hạn theo phút hoặc không rõ: chờ hơn 1 phút


def get_json(url: str, params: dict, max_retries: int = 50):
    query = urllib.parse.urlencode(params, safe=",")
    full_url = f"{url}?{query}"
    attempt = 0
    while True:
        attempt += 1
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                reason = json.loads(body).get("reason", body)
            except json.JSONDecodeError:
                reason = body
            if e.code == 429:
                wait = _wait_for_rate_limit(reason)
                cap = int(os.environ.get("OM_MAX_WAIT", "0") or 0)
                if cap and wait > cap:
                    raise RuntimeError(f"Rate limit hit ({reason.strip()[:80]}), not waiting {wait}s here.")
                log(f"Bị giới hạn số lần gọi ({reason.strip()[:80]}). Chờ {wait // 60} phút...")
                time.sleep(wait)
                continue  # giới hạn tốc độ không tính vào số lần thử
            if e.code == 400:
                # Lỗi tham số: thử lại cũng vô ích
                raise RuntimeError(f"API báo lỗi tham số: {reason}") from e
            err = f"HTTP {e.code}: {reason[:120]}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = f"Lỗi mạng: {e}"

        if attempt >= max_retries:
            raise RuntimeError(f"Thất bại sau {attempt} lần thử. {err}")
        wait = min(300, 5 * attempt)
        log(f"{err}. Thử lại sau {wait} giây (lần {attempt}/{max_retries})")
        time.sleep(wait)
