"""WEB SCRAPER 2: past weather events in Vietnam from Wikipedia (storms, floods, disasters).

Uses the official MediaWiki API (allowed and documented; content is CC BY-SA, so every answer
links back to the article):
  1. List the articles in Wikipedia categories about storms and floods in Vietnam
     (and their sub-categories), in English and Vietnamese.
  2. Download the introduction of every article (plain text).
  3. Split the long overview article "Tropical cyclones in Vietnam" into one record per paragraph.
  4. Tag each record with years, provinces, regions and topics.

Output:
  data/history/events.csv
  live/processed/events.csv   (commit this folder so the online web app has it)

Run once (takes a few minutes):
    python wiki_events.py
"""
import csv
import json
import re
import shutil
import time
import urllib.parse
import urllib.request

import config
from om_client import log

OUT = config.DATA_DIR / "history" / "events.csv"
LIVE = config.ROOT / "live" / "processed" / "events.csv"
USER_AGENT = "WeatherAQIVN-StudentProject/1.0 (Big Data course, Group 8; https://github.com/Kalu-Cassano/weather-aqi-vn)"
COLUMNS = ["title", "lang", "url", "years", "summary", "provinces", "regions", "topics"]

CATEGORIES = {
    "en": ["Typhoons in Vietnam", "Tropical cyclones in Vietnam", "Tropical storms in Vietnam",
           "Floods in Vietnam", "Natural disasters in Vietnam", "Landslides in Vietnam",
           "Droughts in Vietnam", "Heat waves in Vietnam", "Weather events in Vietnam"],
    "vi": ["Bão tại Việt Nam", "Bão ở Việt Nam", "Lũ lụt tại Việt Nam", "Lũ lụt ở Việt Nam",
           "Thiên tai tại Việt Nam", "Thiên tai ở Việt Nam"],
}
OVERVIEW = {"en": ["Tropical cyclones in Vietnam"], "vi": []}
CATEGORY_WORD = {"en": "Category", "vi": "Thể loại"}
MAX_DEPTH = 1          # also read sub-categories, one level down
MAX_PAGES = 600


def api(lang, **params):
    params.update(format="json", formatversion=2, maxlag=5)
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            if "error" in data and data["error"].get("code") == "maxlag":
                time.sleep(5)
                continue
            return data
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    return {}


def category_pages(lang, category, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if category in seen:
        return []
    seen.add(category)
    pages, cont = [], None
    while True:
        params = dict(action="query", list="categorymembers", cmlimit=500, cmtype="page|subcat",
                      cmtitle=f"{CATEGORY_WORD[lang]}:{category}")
        if cont:
            params["cmcontinue"] = cont
        data = api(lang, **params)
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 0:
                pages.append(m["title"])
            elif m.get("ns") == 14 and depth < MAX_DEPTH:
                sub = m["title"].split(":", 1)[1]
                pages += category_pages(lang, sub, depth + 1, seen)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            return pages
        time.sleep(0.5)


def intros(lang, titles):
    """Plain-text introductions, 20 titles per request."""
    out = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        data = api(lang, action="query", prop="extracts|info", exintro=1, explaintext=1, inprop="url",
                   redirects=1, titles="|".join(chunk))
        for p in data.get("query", {}).get("pages", []):
            if p.get("extract"):
                out[p["title"]] = (p["extract"].strip(), p.get("fullurl") or
                                   f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(p['title'].replace(' ', '_'))}")
        time.sleep(0.5)
    return out


def overview_chunks(lang, title):
    """Split a long overview article into paragraphs that mention a year."""
    data = api(lang, action="query", prop="extracts|info", explaintext=1, inprop="url", redirects=1, titles=title)
    pages = data.get("query", {}).get("pages", [])
    if not pages or not pages[0].get("extract"):
        return []
    page = pages[0]
    url = page.get("fullurl", "")
    section, chunks = "", []
    for para in page["extract"].split("\n"):
        para = para.strip()
        m = re.match(r"^=+\s*(.+?)\s*=+$", para)
        if m:
            section = m.group(1)
            continue
        if len(para) < 80 or not re.search(r"\b(1[89]\d\d|20\d\d)\b", para):
            continue
        name = f"{page['title']}: {section}" if section else page["title"]
        anchor = urllib.parse.quote(section.replace(" ", "_")) if section else ""
        chunks.append((name, para, url + (f"#{anchor}" if anchor else "")))
    return chunks


def main():
    from assistant import detect_provinces
    from news_scraper import classify

    records, seen_titles = [], set()
    for lang, cats in CATEGORIES.items():
        titles = []
        for cat in cats:
            try:
                found = category_pages(lang, cat)
            except Exception as e:
                log(f"Wikipedia ({lang}): cannot read category '{cat}' ({type(e).__name__})")
                continue
            if found:
                log(f"Wikipedia ({lang}): category '{cat}': {len(found)} pages")
            titles += found
        titles = [t for t in dict.fromkeys(titles) if t not in seen_titles][:MAX_PAGES]
        seen_titles.update(titles)
        texts = intros(lang, titles)
        for title, (text, url) in texts.items():
            records.append((title, lang, url, text))
        for title in OVERVIEW.get(lang, []):
            for name, para, url in overview_chunks(lang, title):
                records.append((name, lang, url, para))

    rows = []
    for title, lang, url, text in records:
        summary = re.sub(r"\s+", " ", text)[:1500]
        full = f"{title} {summary}"
        years = sorted(set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", full)))
        topics, regions = classify(full)
        rows.append(dict(title=title, lang=lang, url=url, years=";".join(years), summary=summary,
                         provinces=";".join(detect_provinces(full)), regions=";".join(regions),
                         topics=";".join(topics)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, LIVE)
    log(f"Wikipedia events: {len(rows)} records "
        f"({sum(r['lang'] == 'en' for r in rows)} English, {sum(r['lang'] == 'vi' for r in rows)} Vietnamese) -> {LIVE}")


if __name__ == "__main__":
    main()
