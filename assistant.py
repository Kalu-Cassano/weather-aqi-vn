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
}
FIRST_NEWS_NUMBER = 5


def _province_block(code, w, aq, preds, omf, used) -> str:
    """Data for one province. Every line starts with its source number, e.g. [1]."""
    lines = [f"### {NAMES[code]}"]
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

    used = set()
    blocks = [_province_block(c, w, aq, preds, omf, used) for c in codes]

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
    head.append("Coverage: this DATA only has the latest observation, forecasts for the next 1 to 6 hours, "
                "and news from the last 7 days. It has NO data for other past days or for days after the next 6 hours.")

    sources = [dict(n=SOURCES[k][0], title=SOURCES[k][1], url=SOURCES[k][2], date=data_time if k in ("weather", "air") else "")
               for k in ("weather", "air", "model", "omf") if k in used]
    news_items = retrieve_news(question, codes, now_vn)
    for i, r in enumerate(news_items, FIRST_NEWS_NUMBER):
        sources.append(dict(n=i, title=f"{r['title']} ({r['source']})", url=r["link"],
                            date=str(r["published"]).replace("T", " ")))
    news = _news_block(news_items, FIRST_NEWS_NUMBER)
    text = "\n\n".join(["\n".join(head)] + blocks + ([news] if news else []) + [summary])
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
7. Forecasts: prefer "our model" [3] and say it is the model trained by Group 8; you can compare with Open-Meteo [4].
8. US AQI levels (EPA): 0-50 good, 51-100 moderate, 101-150 unhealthy for sensitive groups, 151-200 unhealthy, above 200 very unhealthy.
9. Vietnam has 34 provinces since 1 July 2025. If the user uses an old province name, explain briefly which new province it belongs to.
10. Politely refuse questions that are not about weather, air quality or this web app."""


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
