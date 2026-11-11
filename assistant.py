"""TRỢ LÝ HỎI ĐÁP AI: LLM chạy trên GPU của máy (Ollama) trả lời dựa trên DỮ LIỆU CỦA DỰ ÁN.

Cách hoạt động (kiểu RAG đơn giản trên dữ liệu bảng):
  1. Nhận diện tỉnh/thành trong câu hỏi (kể cả tên tỉnh cũ trước sáp nhập, tên thành phố).
  2. Lấy dữ liệu liên quan: thời tiết + AQI hiện tại, dự báo của MÔ HÌNH NHÓM (XGBoost),
     dự báo của Open-Meteo, bảng tóm tắt 34 tỉnh.
  3. Đưa dữ liệu đó vào prompt, LLM chỉ được trả lời dựa trên dữ liệu này.

Dùng trong app.py:
    import assistant
    ok, msg = assistant.check_ollama()
    for chunk in assistant.ask(question, history, selected_location):  # stream
        ...
    assistant.build_context(question, selected_location)  # để hiển thị "dữ liệu AI đã dùng"
"""
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request

import pandas as pd

import config

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
MAX_PROVINCES = 3

RT = config.REALTIME_DIR
NAMES = {l[0]: l[1] for l in config.LOCATIONS}

# ---------------------------------------------------------------------------
# Nhận diện tỉnh trong câu hỏi
# ---------------------------------------------------------------------------
# Tên tỉnh cũ (trước 01/07/2025) và tên thành phố quen thuộc -> tỉnh mới
ALIASES = {
    "tp_hcm": ["thành phố hồ chí minh", "ho chi minh city", "hồ chí minh", "tp hcm", "tphcm", "hcmc", "hcm",
               "sài gòn", "saigon",
               "bình dương", "bà rịa vũng tàu", "bà rịa", "vũng tàu", "thủ dầu một"],
    "ha_noi": ["hà nội", "hanoi", "thủ đô"],
    "hue": ["thừa thiên huế", "huế", "hue city"],
    "tuyen_quang": ["hà giang"], "lao_cai": ["yên bái", "sa pa", "sapa"],
    "thai_nguyen": ["bắc kạn", "bắc cạn"], "phu_tho": ["vĩnh phúc", "hòa bình", "hoà bình", "việt trì"],
    "bac_ninh": ["bắc giang"], "hung_yen": ["thái bình"], "hai_phong": ["hải dương"],
    "ninh_binh": ["hà nam", "nam định"], "quang_tri": ["quảng bình", "đồng hới"],
    "da_nang": ["quảng nam", "hội an", "hoi an", "danang"], "quang_ngai": ["kon tum"],
    "gia_lai": ["bình định", "quy nhơn", "pleiku"], "khanh_hoa": ["ninh thuận", "nha trang", "nhatrang", "phan rang"],
    "lam_dong": ["đắk nông", "đăk nông", "bình thuận", "đà lạt", "dalat", "phan thiết", "mũi né"],
    "dak_lak": ["đắk lắk", "đăk lăk", "phú yên", "buôn ma thuột", "tuy hòa"],
    "dong_nai": ["bình phước", "biên hòa", "biên hoà"], "tay_ninh": ["long an", "tân an"],
    "can_tho": ["sóc trăng", "hậu giang"], "vinh_long": ["bến tre", "trà vinh"],
    "dong_thap": ["tiền giang", "mỹ tho"], "ca_mau": ["bạc liêu"],
    "an_giang": ["kiên giang", "rạch giá", "phú quốc", "phu quoc island"],
    "quang_ninh": ["hạ long", "halong", "vịnh hạ long", "ha long bay"], "nghe_an": ["vinh"],
    "dien_bien": ["điện biên phủ"],
}


def _norm(text: str) -> str:
    """Bỏ dấu, chữ thường, chỉ giữ chữ và số -> so khớp được cả khi gõ không dấu."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


def _build_patterns():
    pats = []
    for code, province, place, _, _ in config.LOCATIONS:
        for name in {province, place}:
            pats.append((_norm(name), code))
    for code, names in ALIASES.items():
        for name in names:
            pats.append((_norm(name), code))
    # tên dài so khớp trước (để "vĩnh long", "trà vinh" không bị nhầm thành "vinh")
    return sorted(set(pats), key=lambda p: -len(p[0]))


PATTERNS = _build_patterns()


def detect_provinces(question: str) -> list:
    text = _norm(question)
    hits = {}  # mã tỉnh -> vị trí xuất hiện đầu tiên trong câu
    for pat, code in PATTERNS:
        pos = text.find(pat)
        if pos >= 0:
            hits[code] = min(pos, hits.get(code, pos))
            text = text.replace(pat, " " + "|" * (len(pat) - 2) + " ")  # giữ nguyên độ dài
    return sorted(hits, key=hits.get)[:MAX_PROVINCES]


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def _read(name):
    p = RT / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _latest_per_location(df):
    if df is None or df.empty:
        return None
    return df.sort_values("fetched_at").groupby("location").tail(1).set_index("location")


def _aqi_level(aqi):
    if aqi is None or pd.isna(aqi):
        return "unknown"
    for limit, label in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
                         (200, "Unhealthy"), (300, "Very unhealthy")]:
        if aqi <= limit:
            return label
    return "Hazardous"


WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
       51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "moderate rain",
       65: "heavy rain", 80: "light showers", 81: "showers", 82: "violent showers", 95: "thunderstorm",
       96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def _f(v, nd=1):
    return "?" if v is None or pd.isna(v) else f"{float(v):.{nd}f}"


# Fixed sources for the numeric data. News articles get the next numbers.
SOURCES = {
    "weather": (1, "Open-Meteo: current weather (Forecast API)", "https://open-meteo.com/en/docs"),
    "air": (2, "CAMS air quality, via Open-Meteo (Air Quality API)", "https://open-meteo.com/en/docs/air-quality-api"),
    "model": (3, "Group 8: our XGBoost forecast model (test results)",
              "https://github.com/Kalu-Cassano/weather-aqi-vn/blob/main/reports/metrics.csv"),
    "omf": (4, "Open-Meteo: hourly forecast", "https://open-meteo.com/en/docs"),
    "daily": (5, "Open-Meteo: daily forecast for the next days", "https://open-meteo.com/en/docs"),
    "history": (6, "Daily history: Open-Meteo historical weather API (ERA5) and CAMS, processed by our pipeline",
                "https://open-meteo.com/en/docs/historical-weather-api"),
}
FIRST_NEWS_NUMBER = 7
_DAILY_CACHE = {}  # province code -> (time fetched, rows)


def _daily_forecast(code):
    """Next 3 days from Open-Meteo, fetched when a question needs it (cached 30 minutes)."""
    hit = _DAILY_CACHE.get(code)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    lat, lon = next((l[3], l[4]) for l in config.LOCATIONS if l[0] == code)
    url = (f"{config.FORECAST_URL}?latitude={lat}&longitude={lon}&timezone={config.TIMEZONE}&forecast_days=4"
           "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "weather-aqi-vn"})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read()).get("daily", {})
        rows = [dict(date=d["time"][i], tmax=d["temperature_2m_max"][i], tmin=d["temperature_2m_min"][i],
                     rain=d["precipitation_sum"][i], prob=(d.get("precipitation_probability_max") or [None] * 9)[i],
                     code=d["weather_code"][i]) for i in range(len(d.get("time", [])))]
    except Exception:
        rows = []
    _DAILY_CACHE[code] = (time.time(), rows)
    return rows


def _observed_days(code, hist):
    """Daily min/max temperature and total rain from the data we collected (last 3 days)."""
    if hist is None or hist.empty:
        return []
    h = hist[hist["location"] == code].copy()
    if h.empty:
        return []
    h["day"] = h["time"].astype(str).str[:10]
    h = h.drop_duplicates(subset=["time"], keep="last")
    out = []
    for day, g in list(h.groupby("day"))[-3:]:
        out.append(f"{day}: {_f(g['temperature_2m'].min())} to {_f(g['temperature_2m'].max())}°C, "
                   f"rain {_f(g['precipitation'].sum())} mm ({len(g)} hourly values)")
    return out


# ---------------------------------------------------------------------------
# The past: daily history (make_history.py) and weather events from Wikipedia (wiki_events.py)
# ---------------------------------------------------------------------------
HISTORY_FILE = config.DATA_DIR / "processed" / "daily_history.csv"
EVENTS_FILE = config.DATA_DIR / "history" / "events.csv"
_FILE_CACHE = {}
_MONTHS_EN = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                          "august", "september", "october", "november", "december"], 1)}
_MONTHS_EN.update({k[:3]: v for k, v in list(_MONTHS_EN.items())})


def _read_file(path):
    """Read a CSV once and keep it in memory until the file changes."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    hit = _FILE_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        df = pd.read_csv(path, dtype={"years": str})
    except Exception:
        return None
    _FILE_CACHE[path] = (mtime, df)
    return df


def parse_periods(question, now):
    """Find past dates in the question. Returns a list of (kind, value, label):
    kind "day" (YYYY-MM-DD), "month" (YYYY-MM), "year" (YYYY) or "range" ((start, end))."""
    q = question.lower()
    out = []

    def add(kind, value, label):
        if (kind, value) not in [(k, v) for k, v, _ in out]:
            out.append((kind, value, label))

    def year_for(month, day=1):
        y = now.year
        return y - 1 if (month, day) > (now.month, now.day) else y

    for m in re.finditer(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", q):
        add("day", f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}", m[0])
    for m in re.finditer(r"(?<![\d/])(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?(?![\d/])", q):
        d, mo = int(m[1]), int(m[2])
        if 1 <= d <= 31 and 1 <= mo <= 12:
            y = int(m[3]) if m[3] else year_for(mo, d)
            add("day", f"{y:04d}-{mo:02d}-{d:02d}", m[0])
    for m in re.finditer(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+năm\s+(\d{4}))?", q):
        d, mo = int(m[1]), int(m[2])
        y = int(m[3]) if m[3] else year_for(mo, d)
        add("day", f"{y:04d}-{mo:02d}-{d:02d}", m[0])
    for m in re.finditer(r"tháng\s+(\d{1,2})(?:\s*(?:/|-|năm)\s*(\d{4}))?", q):
        mo = int(m[1])
        if 1 <= mo <= 12 and not re.search(r"ngày\s+\d{1,2}\s+$", q[:m.start()]):
            y = int(m[2]) if m[2] else year_for(mo)
            add("month", f"{y:04d}-{mo:02d}", m[0])
    for m in re.finditer(r"\b(\d{1,2})?\s*(january|february|march|april|may|june|july|august|september|october|"
                         r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s*(\d{1,2})?,?\s*(\d{4})?\b", q):
        mo = _MONTHS_EN.get(m[2][:3])
        day = m[1] or m[3]
        if not mo or (not day and not m[4] and m[2] in ("may", "mar")):
            continue  # plain "may" / "mar" are usually not months
        y = int(m[4]) if m[4] else year_for(mo, int(day) if day else 1)
        if day:
            add("day", f"{y:04d}-{mo:02d}-{int(day):02d}", m[0].strip())
        else:
            add("month", f"{y:04d}-{mo:02d}", m[0].strip())
    if not any(k in ("day", "month") for k, _, _ in out):
        for m in re.finditer(r"(?<!\d)(19\d\d|20\d\d)(?!\d)", q):
            add("year", m[1], m[1])
    if re.search(r"(?<!\w)(năm ngoái|năm trước|last year)(?!\w)", q):
        add("year", str(now.year - 1), "last year")
    if re.search(r"(?<!\w)(tháng trước|last month)(?!\w)", q):
        y, mo = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
        add("month", f"{y:04d}-{mo:02d}", "last month")
    if re.search(r"(?<!\w)(tuần trước|last week)(?!\w)", q):
        start, end = (now - pd.Timedelta(days=13)).strftime("%Y-%m-%d"), (now - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        add("range", (start, end), "last week")
    for m in re.finditer(r"(?<!\w)(\d{1,3}) (?:ngày trước|days ago)(?!\w)", q):
        add("day", (now - pd.Timedelta(days=int(m[1]))).strftime("%Y-%m-%d"), m[0])
    if re.search(r"(?<!\w)(hôm kia|day before yesterday)(?!\w)", q):
        add("day", (now - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "2 days ago")
    return out[:3]


_ARCHIVE_CACHE = {}


def _archive_days(code, start, end):
    """Daily values from the Open-Meteo historical API (ERA5, from 1940), fetched only when our own
    history file does not cover the dates. Returns a DataFrame with the same columns (no PM2.5)."""
    key = (code, start, end)
    if key in _ARCHIVE_CACHE:
        return _ARCHIVE_CACHE[key]
    lat, lon = next((l[3], l[4]) for l in config.LOCATIONS if l[0] == code)
    url = (f"{config.ARCHIVE_URL}?latitude={lat}&longitude={lon}&timezone={config.TIMEZONE}"
           f"&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,"
           "temperature_2m_mean,precipitation_sum,wind_speed_10m_max")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "weather-aqi-vn"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read()).get("daily", {})
        df = pd.DataFrame(dict(date=d["time"], tmin=d["temperature_2m_min"], tmax=d["temperature_2m_max"],
                               tmean=d["temperature_2m_mean"], rain=d["precipitation_sum"],
                               wind_max=d["wind_speed_10m_max"])).dropna(subset=["tmax", "tmin"])
    except Exception:
        df = pd.DataFrame(columns=["date", "tmin", "tmax", "tmean", "rain", "wind_max"])
    _ARCHIVE_CACHE[key] = df
    return df


def _period_bounds(kind, value):
    if kind == "day":
        return value, value
    if kind == "month":
        start = pd.Timestamp(value + "-01")
        return start.strftime("%Y-%m-%d"), (start + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    if kind == "year":
        return f"{value}-01-01", f"{value}-12-31"
    return value


def _history_lines(code, periods):
    if not periods:
        return []
    h = _read_file(HISTORY_FILE)
    h = h[h["location"] == code] if h is not None else pd.DataFrame(columns=["date"])
    yesterday = (_vn_now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lines = []
    for kind, value, label in periods:
        start, end = _period_bounds(kind, value)
        end = min(end, yesterday)
        name = value if kind != "range" else f"{value[0]} to {value[1]}"
        if start > end or start < "1940-01-01":
            lines.append(f"- [6] {name} ({label}): NO DATA (this date is in the future or before 1940).")
            continue
        sel = h[(h["date"] >= start) & (h["date"] <= end)] if not h.empty else h
        expected = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        if len(sel) < expected:  # not (fully) in our file: ask the Open-Meteo historical API
            live = _archive_days(code, start, end)
            if not live.empty:
                live = live[~live["date"].isin(sel["date"])] if not sel.empty else live
                sel = pd.concat([sel, live], ignore_index=True).sort_values("date")
        has_pm = "pm25" in sel and sel["pm25"].notna().any()
        if sel.empty:
            lines.append(f"- [6] {name} ({label}): NO DATA for this date.")
            continue
        pm = f", mean PM2.5 {_f(sel['pm25'].mean())} µg/m³" if has_pm else ""
        if kind == "day":
            r = sel.iloc[0]
            lines.append(f"- [6] {name}: lowest {_f(r['tmin'])}°C, highest {_f(r['tmax'])}°C, daily mean {_f(r['tmean'])}°C, "
                         f"rain {_f(r['rain'])} mm, max wind {_f(r['wind_max'])} km/h{pm}")
        else:
            hot = sel.loc[sel["tmax"].idxmax()]
            cold = sel.loc[sel["tmin"].idxmin()]
            wet = sel.loc[sel["rain"].idxmax()]
            lines.append(f"- [6] {name} ({len(sel)} days): {_f(sel['tmin'].min())} to {_f(sel['tmax'].max())}°C, "
                         f"mean {_f(sel['tmean'].mean())}°C, total rain {_f(sel['rain'].sum())} mm, rainy days (>1 mm) "
                         f"{int((sel['rain'] > 1).sum())}{pm}; hottest day {hot['date']} ({_f(hot['tmax'])}°C), "
                         f"coldest {cold['date']} ({_f(cold['tmin'])}°C), wettest {wet['date']} ({_f(wet['rain'])} mm)")
            if kind == "year":
                m = sel.assign(month=sel["date"].str[:7]).groupby("month").agg(
                    tmin=("tmin", "min"), tmax=("tmax", "max"), rain=("rain", "sum"))
                lines.append("- [6] by month: " + "; ".join(
                    f"{i}: {_f(r.tmin)}-{_f(r.tmax)}°C, rain {_f(r.rain, 0)} mm" for i, r in m.iterrows()))
    return lines


_PAST_WORDS = re.compile(r"(?<!\w)(năm|lịch sử|từng|trước đây|kỷ lục|thiệt hại|tàn phá|history|historic|ever|past|"
                         r"deadliest|strongest|worst|damage|record|in the past)(?!\w)", re.IGNORECASE)


def retrieve_events(question, codes, periods, limit=3):
    ev = _read_file(EVENTS_FILE)
    if ev is None or ev.empty:
        return []
    ev = ev.fillna("")
    try:
        from news_scraper import classify
        q_topics, q_regions = map(set, classify(question))
    except Exception:
        q_topics, q_regions = set(), set()
    q_topics.discard("weather")
    q_years = {str(v)[:4] for k, v, _ in periods if k != "range"}
    q_tokens = _tokens(question)
    vi = bool(_VI_CHARS.search(question))
    if not (q_years or q_topics or _PAST_WORDS.search(question)):
        return []
    scored = []
    for _, r in ev.iterrows():
        years = set(filter(None, str(r["years"]).split(";")))
        if q_years and not (years & q_years):
            continue
        score = 3 * len(q_tokens & _tokens(r["title"]))
        score += 3 * len(set(filter(None, str(r["provinces"]).split(";"))) & set(codes))
        score += 2 * len(set(filter(None, str(r["regions"]).split(";"))) & q_regions)
        score += 2 * len(set(filter(None, str(r["topics"]).split(";"))) & q_topics)
        score += min(3, len(q_tokens & _tokens(r["summary"])))
        if score < 3:  # the year alone is not enough: the topic, place or name must match too
            continue
        score += 4 * len(years & q_years) + (1 if (r["lang"] == "vi") == vi else 0)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def _province_block(code, w, aq, preds, omf, used, hist=None, periods=None) -> str:
    """Data for one province. Every line starts with its source number, e.g. [1]."""
    lines = [f"### {NAMES[code]}"]
    past = _history_lines(code, periods or [])
    if past:  # question about past dates: give only the history, so the AI does not mix it with today's data
        used.add("history")
        return "\n".join(lines + past)
    if w is not None and code in w.index:
        r = w.loc[code]
        code_w = None if pd.isna(r.get("weather_code")) else int(r["weather_code"])
        lines.append(
            f"- [1] Observed at {str(r['time']).replace('T', ' ')}: {_f(r['temperature_2m'])}°C "
            f"(feels like {_f(r['apparent_temperature'])}°C), humidity {_f(r['relative_humidity_2m'], 0)}%, "
            f"wind {_f(r['wind_speed_10m'])} km/h, rain {_f(r['precipitation'])} mm, "
            f"cloud cover {_f(r['cloud_cover'], 0)}%, weather: {WMO.get(code_w, 'unknown')}")
        used.add("weather")
    if aq is not None and code in aq.index:
        r = aq.loc[code]
        lines.append(f"- [2] Air quality at {str(r['time']).replace('T', ' ')}: US AQI {_f(r['us_aqi'], 0)} "
                     f"({_aqi_level(r['us_aqi'])}), PM2.5 {_f(r['pm2_5'])} µg/m³, PM10 {_f(r['pm10'])} µg/m³")
        used.add("air")
    if preds is not None and not preds.empty:
        p = preds[preds["location"] == code]
        if not p.empty:
            p = p[p["made_at"] == p["made_at"].max()].sort_values("horizon_h")
            for target, label, unit in [("temperature_2m", "Temperature", "°C"), ("pm2_5", "PM2.5", " µg/m³")]:
                q = p[p["target"] == target]
                if not q.empty:
                    parts = [f"{str(r['target_time']).replace('T', ' ')} ({int(r['horizon_h'])}h ahead): "
                             f"{_f(r['prediction'])}{unit}" for _, r in q.iterrows()]
                    lines.append(f"- [3] {label} forecast from our model: " + "; ".join(parts))
                    used.add("model")
    if omf is not None and not omf.empty:
        o = omf[omf["location"] == code]
        if not o.empty:
            o = o[o["fetched_at"] == o["fetched_at"].max()].sort_values("target_time")
            parts = [f"{str(r['target_time']).replace('T', ' ')}: {_f(r['temperature_2m'])}°C, "
                     f"rain {_f(r['precipitation'])} mm" for _, r in o.iterrows()]
            lines.append("- [4] Open-Meteo hourly forecast: " + "; ".join(parts))
            used.add("omf")
    days = _observed_days(code, hist)
    if days:
        lines.append("- [1] Observed daily values collected by our system: " + "; ".join(days))
        used.add("weather")
    daily = _daily_forecast(code)
    if daily:
        parts = [f"{r['date']}: {_f(r['tmin'])} to {_f(r['tmax'])}°C, rain {_f(r['rain'])} mm, "
                 f"rain chance {_f(r['prob'], 0)}%, {WMO.get(r['code'], 'unknown')}" for r in daily]
        lines.append("- [5] Open-Meteo daily forecast (NOT our model): " + "; ".join(parts))
        used.add("daily")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# News retrieval (articles collected by news_scraper.py)
# ---------------------------------------------------------------------------
NEWS_MAX = 4
NEWS_DAYS = 7
_STOP = set("the a an is are was be to of in on at for and or with how what which when where will can do does "
            "now today right this that there any have has about please tell me us you your "
            "la co khong nao gi the nhu bao gio hom nay bay gio o tai va cua cho toi minh ban nhe a "
            "khi nay duoc khong roi bao nhieu gio hom nay ngay mai sang chieu toi dem "
            "weather thoi tiet nhiet".split())  # too common or ambiguous without accents ("bao": bão / bao nhiêu)
_REGION_WORDS = {
    "North": ["mien bac", "bac bo", "north"], "Central": ["mien trung", "trung bo", "central"],
    "Highlands": ["tay nguyen", "highland"], "South": ["mien nam", "nam bo", "mien tay", "mekong", "south"],
}
_NEWSY = re.compile(r"(?<!\w)(bão|cảnh báo|tin tức|thiên tai|lũ|lụt|ngập|sạt lở|news|warning|alert|disaster)(?!\w)",
                    re.IGNORECASE)


def _tokens(text):
    return {t for t in _norm(text).split() if len(t) >= 3 and t not in _STOP}


def retrieve_news(question, codes, now_vn):
    df = _read("news.csv")
    if df is None or df.empty:
        return []
    df = df.fillna("")
    cutoff = (now_vn - pd.Timedelta(days=NEWS_DAYS)).strftime("%Y-%m-%dT%H:%M")
    df = df[df["published"] >= cutoff]
    if df.empty:
        return []
    try:
        from news_scraper import classify  # accent-aware topic and region matching
        q_topics, q_regions = map(set, classify(question))
    except Exception:
        q_topics, q_regions = set(), set()
    qn = _norm(question)
    q_regions |= {r for r, words in _REGION_WORDS.items() if any(f" {w} " in qn for w in words)}
    q_topics.discard("weather")  # too general to pick articles
    q_tokens = _tokens(question)
    scored = []
    for _, r in df.iterrows():
        provs = set(filter(None, str(r["provinces"]).split(";")))
        regions = set(filter(None, str(r["regions"]).split(";")))
        topics = set(filter(None, str(r["topics"]).split(";")))
        score = 3 * len(provs & set(codes)) + 2 * len(regions & q_regions) + 2 * len(topics & q_topics)
        score += len(q_tokens & _tokens(f"{r['title']} {r['summary']}"))
        if score < 2:
            continue
        try:
            age_h = (now_vn - pd.to_datetime(r["published"])).total_seconds() / 3600
        except Exception:
            age_h = 999
        score += 2 if age_h <= 24 else (1 if age_h <= 72 else 0)
        scored.append((score, r["published"], r))
    if not scored and _NEWSY.search(question):
        # a general news question ("có tin cảnh báo gì không?"): give the newest articles
        return [r for _, r in df.sort_values("published", ascending=False).head(3).iterrows()]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored[:NEWS_MAX]]


def _news_block(items, start):
    if not items:
        return ""
    lines = ["### News articles collected by our web scraper"]
    for i, r in enumerate(items, start):
        lines.append(f"- [{i}] {r['title']} ({r['source']}, published {str(r['published']).replace('T', ' ')}). "
                     f"Summary: {r['summary']}")
    return "\n".join(lines)


def _vn_now():
    # Vietnam time on any server (Streamlit Cloud runs in UTC). Vietnam has no daylight saving.
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)


def build_context_and_sources(question: str, selected_location: str = None):
    """Returns (context text for the AI, list of sources). Each source: dict(n, title, url, date)."""
    w = _latest_per_location(_read("weather_current.csv"))
    aq = _latest_per_location(_read("aq_current.csv"))
    preds = _read("predictions.csv")
    omf = _read("om_forecast.csv")

    codes = detect_provinces(question)
    if not codes and selected_location:
        codes = [selected_location]

    hist = _read("weather_current.csv")
    periods = parse_periods(question, _vn_now())
    used = set()
    blocks = [_province_block(c, w, aq, preds, omf, used, hist, periods) for c in codes]

    # Summary of all 34 provinces (for "which province is the hottest / most polluted")
    rows = []
    for code, name in NAMES.items():
        t = w.loc[code, "temperature_2m"] if w is not None and code in w.index else None
        a_ = aq.loc[code, "us_aqi"] if aq is not None and code in aq.index else None
        rows.append(f"{name}: {_f(t)}°C, AQI {_f(a_, 0)}")
    if w is not None:
        used.add("weather")
    if aq is not None:
        used.add("air")
    summary = "### [1][2] Latest values for all 34 provinces\n" + "; ".join(rows)

    now_vn = _vn_now()
    head = [f"Current time: {now_vn:%H:%M} on {now_vn:%d %b %Y} (Vietnam time, UTC+7)."]
    data_time = ""
    if w is not None and "time" in w:
        try:
            latest = pd.to_datetime(w["time"]).max()
            data_time = f"{latest:%d/%m/%Y %H:%M}"
            age_h = (now_vn - latest).total_seconds() / 3600
            head.append(f"Latest observation time in the data: {latest:%H:%M %d %b %Y} ({age_h:.1f} hours ago).")
            if age_h > 2:
                head.append("WARNING: the data is OLD. Do not call it 'now'. Tell the user the time of the data "
                            "and that it may be out of date. Forecast times that are already in the past are not forecasts any more.")
        except Exception:
            pass
    head.append("Coverage: this DATA has the latest observation [1][2], daily values we observed in the last few days [1], "
                "our model's forecast of temperature and PM2.5 for the next 1, 3 and 6 hours [3], Open-Meteo's hourly [4] "
                "and daily [5] forecasts (daily: today and the next 3 days, only for the provinces listed), and news from the "
                "last 7 days. Daily history [6] (only for the provinces listed and the dates asked) and past weather events from "
                "Wikipedia (if listed) cover older dates. There is NO forecast of AQI and NO data for more than 3 days ahead. "
                "Each day in [5] is shown with its date: match 'tomorrow'/'ngày mai' to the right date.")

    sources = [dict(n=SOURCES[k][0], title=SOURCES[k][1], url=SOURCES[k][2], date=data_time if k in ("weather", "air") else "")
               for k in ("weather", "air", "model", "omf", "daily", "history") if k in used]
    news_items = retrieve_news(question, codes, now_vn)
    for i, r in enumerate(news_items, FIRST_NEWS_NUMBER):
        sources.append(dict(n=i, title=f"{r['title']} ({r['source']})", url=r["link"],
                            date=str(r["published"]).replace("T", " ")))
    news = _news_block(news_items, FIRST_NEWS_NUMBER)
    start = FIRST_NEWS_NUMBER + len(news_items)
    events = retrieve_events(question, codes, periods)
    ev_lines = []
    for i, r in enumerate(events, start):
        sources.append(dict(n=i, title=f"{r['title']} (Wikipedia)", url=r["url"], date=""))
        ev_lines.append(f"- [{i}] {r['title']}: {r['summary']}")
    ev_block = ("### Past weather events in Vietnam (Wikipedia, collected by our scraper)\n" + "\n".join(ev_lines)) if ev_lines else ""
    if "history" in used:  # past-date question: today's values for all provinces would only confuse the AI
        summary = ""
    text = "\n\n".join(["\n".join(head)] + blocks + [b for b in (news, ev_block, summary) if b])
    return text, sources


def build_context(question: str, selected_location: str = None) -> str:
    """Text given to the AI (shown in the app under "Data used by AI"), with the source list."""
    text, sources = build_context_and_sources(question, selected_location)
    listing = "\n".join(f"[{x['n']}] {x['title']}: {x['url']}" for x in sources)
    return text + ("\n\n### Source list\n" + listing if listing else "")


SYSTEM_PROMPT = """You are the weather and air quality assistant of the web app "Weather & AQI VN" (a student Big Data project, Group 8).

STRICT RULES:
1. Use ONLY facts and numbers from the DATA section. Do not use your own knowledge for any number, event, forecast or date.
2. Cite every fact with its source number in square brackets right after the fact, e.g. "Hà Nội đang 19°C [1]" or "Storm No. 15 is moving to Quảng Ngãi [5]". Every line in DATA starts with its source number.
3. If DATA does not contain what the user asks (for example another past day, tomorrow or later, another country, a province or detail that is not in DATA), do NOT guess. Say clearly that you do not have this information, e.g. "Mình không có thông tin về ... trong dữ liệu hiện tại." or "I don't have information about ... in our current data." You may then say what you can answer instead.
4. Do NOT write any links and do NOT write a "Sources" list. The app adds the source list automatically.
5. Answer in the same language as the question (Vietnamese or English). In English use simple, clear English (B1 level). Keep it short (about 120 words). Use short bullet points for numbers when helpful. Give practical advice (umbrella, mask, drink water...) only when it follows from the data.
6. Always say the time of the data. Only call it "now" when it is less than 2 hours old. Only talk about forecasts for times after the current time.
7. Forecasts: for the next 1 to 6 hours prefer "our model" [3] (trained by Group 8) and you can compare with Open-Meteo [4]. For tomorrow and the next days use only Open-Meteo's daily forecast [5] and say it is Open-Meteo's forecast, not our model. There is no AQI forecast: never predict AQI.
8. The past: use daily history [6] for numbers on past dates (say they come from reanalysis/model data, not a weather station) and Wikipedia articles for past storms, floods and disasters. If [6] says NO DATA for a date and no article covers it, say you do not have this information. If the question is only about the past, answer only from [6] and Wikipedia: do not talk about current data or say that current observations do not have that date.
9. US AQI levels (EPA): 0-50 good, 51-100 moderate, 101-150 unhealthy for sensitive groups, 151-200 unhealthy, above 200 very unhealthy.
10. Vietnam has 34 provinces since 1 July 2025. If the user uses an old province name, explain briefly which new province it belongs to.
11. Politely refuse questions that are not about weather, air quality or this web app."""


# ---------------------------------------------------------------------------
# AI backends
#   - On the home PC: Ollama on the local GPU (default).
#   - On the cloud (no GPU): Groq API, used when GROQ_API_KEY is set (Streamlit secrets).
# ---------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _secret(name, default=""):
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _backend():
    return "groq" if _secret("GROQ_API_KEY") else "ollama"


def check_ollama():
    """Status of the AI backend in use. Returns (ok, message)."""
    if _backend() == "groq":
        model = _secret("GROQ_MODEL", "qwen/qwen3.8-27b")
        return True, f"Cloud AI: {model} (Groq API)"
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return False, "Cannot connect to Ollama. Please open the Ollama app on the computer that runs this web app."
    if not any(m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL + ":") for m in models):
        return False, f"Ollama is running but the model {OLLAMA_MODEL} is missing. Run: ollama pull {OLLAMA_MODEL}"
    return True, f"Local AI: {OLLAMA_MODEL} (running on our GPU)"


def _stream_ollama(messages):
    body = json.dumps({
        "model": OLLAMA_MODEL, "messages": messages, "stream": True,
        "think": False,  # skip the "thinking" step for faster answers
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e:
        yield f"Ollama error: {e.read().decode('utf-8', 'replace')[:200]}"
        return
    except Exception as e:
        yield f"Cannot connect to Ollama ({e}). Please check that the Ollama app is open."
        return
    with resp:
        for line in resp:
            if not line.strip():
                continue
            data = json.loads(line)
            yield data.get("message", {}).get("content", "")
            if data.get("done"):
                break


def _stream_groq(messages):
    key = _secret("GROQ_API_KEY")
    model = _secret("GROQ_MODEL", "qwen/qwen3.8-27b")
    payload = {"model": model, "messages": messages, "stream": True,
               "temperature": 0.2, "max_tokens": 700, "reasoning_format": "hidden"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
               "User-Agent": "weather-aqi-vn"}
    resp = None
    for attempt in range(2):
        req = urllib.request.Request(GROQ_URL, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 400 and "reasoning_format" in msg and attempt == 0:
                payload.pop("reasoning_format")  # this model does not support the option
                continue
            if e.code == 429:
                yield "The free AI quota is busy right now. Please wait a minute and ask again."
            else:
                yield f"Groq error {e.code}: {msg[:200]}"
            return
        except Exception as e:
            yield f"Cannot connect to the cloud AI ({e})."
            return
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            yield delta.get("content") or ""


_OUT_OF_RANGE = re.compile(
    r"(?<!\w)(tuần sau|tuần tới|tháng sau|tháng tới|năm sau|năm tới|cuối tuần sau|mùa sau|next week|next month|next year|"
    r"(?:[4-9]|[1-9]\d+) ngày (?:tới|sau|nữa)|in (?:[4-9]|[1-9]\d+) days|(?:[4-9]|[1-9]\d+) days (?:from now|later))(?!\w)",
    re.IGNORECASE)


def out_of_range_answer(question):
    """Fixed answer (no AI) for time ranges our data can never cover, so nothing can be invented."""
    m = _OUT_OF_RANGE.search(question)
    if not m:
        return None
    if _VI_CHARS.search(question):
        return (f"Mình không có thông tin về **{m.group(0)}** trong dữ liệu hiện tại. Hệ thống chỉ có: dữ liệu quan trắc "
                "vài ngày gần đây, dự báo của mô hình nhóm cho 1–6 giờ tới, dự báo theo ngày của Open-Meteo cho 3 ngày tới, "
                "và tin tức trong 7 ngày gần nhất. Bạn có thể hỏi về các khoảng thời gian này.")
    return (f"I don't have information about **{m.group(0)}** in our current data. We only have: observations from the "
            "last few days, our model's forecast for the next 1–6 hours, Open-Meteo's daily forecast for the next 3 days, "
            "and news from the last 7 days. You can ask about these.")


_VI_CHARS = re.compile(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]", re.IGNORECASE)


def _sources_markdown(answer, sources, question):
    vi = bool(_VI_CHARS.search(question))
    if not sources:
        return ("\n\n---\n*Không có nguồn dữ liệu nào cho câu hỏi này.*" if vi
                else "\n\n---\n*No data source was available for this question.*")
    cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
    chosen = [x for x in sources if x["n"] in cited]
    if chosen:
        title = "**Nguồn:**" if vi else "**Sources:**"
    else:  # nothing cited (e.g. "I don't know"): show what was checked
        chosen = [x for x in sources if x["n"] < FIRST_NEWS_NUMBER]
        title = "**Nguồn dữ liệu đã kiểm tra:**" if vi else "**Data sources checked:**"
    lines = []
    for x in chosen:
        date = f" · {x['date']}" if x.get("date") else ""
        lines.append(f"- [{x['n']}] [{x['title']}]({x['url']}){date}")
    return "\n\n---\n" + title + "\n" + "\n".join(lines)


def ask(question: str, history=None, selected_location: str = None):
    """Generator that yields the answer piece by piece (use with st.write_stream).
    history: list of {"role": "user"/"assistant", "content": ...} from earlier turns."""
    fixed = out_of_range_answer(question)
    if fixed:
        yield fixed
        return
    context, sources = build_context_and_sources(question, selected_location)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nDATA:\n" + context}]
    for m in (history or [])[-6:]:  # keep only the last 3 question/answer pairs
        content = m["content"].split("\n\n---\n")[0]  # drop the source list added by the app
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": question})

    stream = _stream_groq(messages) if _backend() == "groq" else _stream_ollama(messages)
    answer = []
    in_think = False
    for chunk in stream:
        # hide any <think>...</think> part if the model still prints it
        if "<think>" in chunk:
            before, _, chunk = chunk.partition("<think>")
            if before:
                yield before
            in_think = True
        if in_think:
            if "</think>" in chunk:
                in_think = False
                chunk = chunk.partition("</think>")[2].lstrip()
            else:
                continue
        if chunk:
            answer.append(chunk)
            yield chunk
    # The source list is added by code (not by the AI), so links can never be invented.
    yield _sources_markdown("".join(answer), sources, question)
