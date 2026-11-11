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


def _province_block(code, w, aq, preds, omf) -> str:
    lines = [f"### {NAMES[code]}"]
    if w is not None and code in w.index:
        r = w.loc[code]
        code_w = None if pd.isna(r.get("weather_code")) else int(r["weather_code"])
        lines.append(
            f"- Now ({r['time']}): {_f(r['temperature_2m'])}°C (feels like {_f(r['apparent_temperature'])}°C), "
            f"humidity {_f(r['relative_humidity_2m'], 0)}%, wind {_f(r['wind_speed_10m'])} km/h, "
            f"rain {_f(r['precipitation'])} mm, cloud cover {_f(r['cloud_cover'], 0)}%, "
            f"weather: {WMO.get(code_w, 'unknown')}")
    if aq is not None and code in aq.index:
        r = aq.loc[code]
        lines.append(f"- Air quality ({r['time']}): US AQI {_f(r['us_aqi'], 0)} ({_aqi_level(r['us_aqi'])}), "
                     f"PM2.5 {_f(r['pm2_5'])} µg/m³, PM10 {_f(r['pm10'])} µg/m³")
    if preds is not None and not preds.empty:
        p = preds[preds["location"] == code]
        if not p.empty:
            p = p[p["made_at"] == p["made_at"].max()].sort_values("horizon_h")
            for target, label, unit in [("temperature_2m", "temperature", "°C"), ("pm2_5", "PM2.5", " µg/m³")]:
                q = p[p["target"] == target]
                if not q.empty:
                    parts = [f"{r['target_time'][11:16]} ({int(r['horizon_h'])}h ahead): {_f(r['prediction'])}{unit}"
                             for _, r in q.iterrows()]
                    lines.append(f"- {label.capitalize()} forecast from OUR MODEL (self-trained XGBoost): " + "; ".join(parts))
    if omf is not None and not omf.empty:
        o = omf[omf["location"] == code]
        if not o.empty:
            o = o[o["fetched_at"] == o["fetched_at"].max()].sort_values("target_time")
            parts = [f"{r['target_time'][11:16]}: {_f(r['temperature_2m'])}°C, rain {_f(r['precipitation'])} mm"
                     for _, r in o.iterrows()]
            lines.append("- Open-Meteo forecast for the next hours: " + "; ".join(parts))
    return "\n".join(lines)


def build_context(question: str, selected_location: str = None) -> str:
    w = _latest_per_location(_read("weather_current.csv"))
    aq = _latest_per_location(_read("aq_current.csv"))
    preds = _read("predictions.csv")
    omf = _read("om_forecast.csv")

    codes = detect_provinces(question)
    if not codes and selected_location:
        codes = [selected_location]

    blocks = [_province_block(c, w, aq, preds, omf) for c in codes]

    # Bảng tóm tắt 34 tỉnh (để trả lời câu hỏi kiểu "tỉnh nào nóng nhất / ô nhiễm nhất")
    rows = []
    for code, name in NAMES.items():
        t = w.loc[code, "temperature_2m"] if w is not None and code in w.index else None
        a = aq.loc[code, "us_aqi"] if aq is not None and code in aq.index else None
        rows.append(f"{name}: {_f(t)}°C, AQI {_f(a, 0)}")
    summary = "### Latest summary for all 34 provinces\n" + "; ".join(rows)

    from datetime import datetime
    now = f"Current time: {datetime.now():%H:%M} on {datetime.now():%d %b %Y} (Vietnam time, UTC+7)."
    return "\n\n".join([now] + blocks + [summary])


SYSTEM_PROMPT = """You are the weather and air quality assistant of the web app "Weather & AQI VN" (a student Big Data project, Group 8).
Rules:
- Use ONLY the numbers in the DATA section below. Never invent numbers. If the data does not have the answer, say that the system does not have this data yet.
- Answer in the same language as the user's question (Vietnamese or English). In English, use simple, clear English (easy B1 level). Keep answers short (about 100 words) and friendly. Give practical advice (umbrella, mask, stay inside, drink water...).
- When you use a forecast, prefer "our model" (the self-trained XGBoost model) and say so. You can compare it with Open-Meteo.
- Always say the time of the data you use. US AQI levels (EPA): 0-50 good, 51-100 moderate, 101-150 unhealthy for sensitive groups, 151-200 unhealthy, above 200 very unhealthy.
- Vietnam has had 34 provinces since 1 July 2025. If the user uses an old province name, explain in one short sentence which new province it belongs to.
- Write Vietnamese place names as they appear in the data.
- Politely refuse questions that are not about weather, air quality or this web app."""


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
        "options": {"temperature": 0.3, "num_ctx": 8192},
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
               "temperature": 0.3, "max_tokens": 600, "reasoning_format": "hidden"}
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


def ask(question: str, history=None, selected_location: str = None):
    """Generator that yields the answer piece by piece (use with st.write_stream).
    history: list of {"role": "user"/"assistant", "content": ...} from earlier turns."""
    context = build_context(question, selected_location)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nDATA:\n" + context}]
    for m in (history or [])[-6:]:  # keep only the last 3 question/answer pairs
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    stream = _stream_groq(messages) if _backend() == "groq" else _stream_ollama(messages)
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
            yield chunk
