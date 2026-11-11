"""WEB SCRAPER: weather and air quality news from Vietnamese news sites.

Every hour (called by collect_realtime.py):
  1. Read the RSS feeds of several news sites (Vietnamese and English).
  2. Keep only articles about weather, natural disasters and air quality.
  3. Open each NEW article page and scrape its summary and publish time from the HTML
     (meta tags), checking robots.txt first.
  4. Tag each article with the provinces and regions it mentions.
  5. Save to data/realtime/news.csv (newest 30 days). The AI assistant uses these
     articles to answer questions and cites them as sources.

Polite scraping: one run per hour, at most MAX_PAGES article pages per run, a pause between
requests, robots.txt respected, only title + short summary + link are stored.

Run by hand:  python news_scraper.py
"""
import csv
import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

import config
from om_client import log

NEWS_FILE = config.REALTIME_DIR / "news.csv"
COLUMNS = ["fetched_at", "published", "source", "lang", "title", "summary", "link", "provinces", "regions", "topics"]
KEEP_DAYS = 30
MAX_PAGES = 15         # article pages opened per run
PAUSE = 1.0            # seconds between article pages
USER_AGENT = "Mozilla/5.0 (compatible; WeatherAQIVN-StudentProject/1.0; Big Data course, Group 8)"
VN_TZ = timezone(timedelta(hours=7))

FEEDS = [
    ("VnExpress", "vi", "https://vnexpress.net/rss/thoi-su.rss"),
    ("VnExpress", "vi", "https://vnexpress.net/rss/tin-moi-nhat.rss"),
    ("Tuổi Trẻ", "vi", "https://tuoitre.vn/rss/tin-moi-nhat.rss"),
    ("Thanh Niên", "vi", "https://thanhnien.vn/rss/home.rss"),
    ("VietnamPlus", "vi", "https://www.vietnamplus.vn/rss/news.rss"),
    ("VnExpress International", "en", "https://e.vnexpress.net/rss/news.rss"),
    ("Tuoi Tre News", "en", "https://news.tuoitre.vn/home.rss"),
    ("Viet Nam News", "en", "https://vietnamnews.vn/rss/society.rss"),
]

# Topic keywords. Vietnamese words keep their accents ("mưa" = rain, "mua" = buy).
TOPICS = {
    "storm": ["bão", "áp thấp nhiệt đới", "siêu bão", "storm", "typhoon", "tropical depression", "cyclone"],
    "rain_flood": ["mưa lớn", "mưa to", "mưa", "lũ", "lũ quét", "lụt", "ngập", "ngập lụt", "sạt lở", "triều cường",
                   "rain", "heavy rain", "downpour", "flood", "flooding", "landslide", "inundation"],
    "heat": ["nắng nóng", "nắng gắt", "nhiệt độ cao", "heatwave", "heat wave", "scorching", "heat"],
    "cold": ["rét", "rét đậm", "rét hại", "không khí lạnh", "gió mùa đông bắc", "băng giá", "sương muối",
             "cold spell", "cold air", "cold snap", "frost"],
    "air": ["ô nhiễm không khí", "chất lượng không khí", "bụi mịn", "pm2.5", "pm 2.5", "aqi", "khói mù",
            "air pollution", "air quality", "fine dust", "smog", "haze"],
    "weather": ["thời tiết", "dự báo thời tiết", "thiên tai", "khí tượng", "dông", "lốc", "sét", "mưa đá",
                "sương mù", "hạn hán", "xâm nhập mặn", "weather", "forecast", "thunderstorm", "hail",
                "drought", "saltwater intrusion", "natural disaster", "meteorolog"],
}
REGIONS = {
    "North": ["miền bắc", "bắc bộ", "tây bắc", "đông bắc bộ", "việt bắc", "northern vietnam", "north vietnam", "the north"],
    "Central": ["miền trung", "trung bộ", "bắc trung bộ", "nam trung bộ", "duyên hải", "central vietnam", "central region",
                "central coast"],
    "Highlands": ["tây nguyên", "central highlands"],
    "South": ["miền nam", "nam bộ", "đông nam bộ", "tây nam bộ", "miền tây", "đồng bằng sông cửu long", "mekong delta",
              "southern vietnam", "south vietnam", "the south"],
}
# A news article must have at least one strong topic word (plain "weather"/"mưa" alone is not enough
# when it is the only match in a long political or sports text).
STRONG = {"storm", "rain_flood", "heat", "cold", "air"}


def _compile(words):
    return [re.compile(r"(?<![\w])" + re.escape(w) + r"(?![\w])", re.IGNORECASE) for w in words]


TOPIC_RE = {k: _compile(v) for k, v in TOPICS.items()}
REGION_RE = {k: _compile(v) for k, v in REGIONS.items()}


def classify(text):
    t = text.lower()
    topics = [k for k, pats in TOPIC_RE.items() if any(p.search(t) for p in pats)]
    regions = [k for k, pats in REGION_RE.items() if any(p.search(t) for p in pats)]
    return topics, regions


def is_relevant(title, summary):
    topics, _ = classify(title)
    if set(topics) & STRONG:
        return True
    topics2, _ = classify(f"{title} {summary}")
    # the summary alone must have a strong topic AND a weather word, to skip off-topic stories
    return bool(set(topics2) & STRONG) and "weather" in topics2


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        charset = r.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


_ROBOTS = {}


def allowed(url):
    """Check robots.txt of the site (cached). If robots.txt cannot be read, allow."""
    parts = urllib.parse.urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in _ROBOTS:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.parse(_get(base + "/robots.txt", timeout=10).splitlines())
        except Exception:
            rp = None
        _ROBOTS[base] = rp
    rp = _ROBOTS[base]
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(value):
    if not value:
        return ""
    value = value.strip()
    try:
        d = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=VN_TZ)
    return d.astimezone(VN_TZ).strftime("%Y-%m-%dT%H:%M")


def read_feed(url):
    root = ET.fromstring(_get(url).encode("utf-8"))
    items = []
    for it in root.iter("item"):
        items.append({
            "title": _clean(it.findtext("title")),
            "link": (it.findtext("link") or "").strip(),
            "summary": _clean(it.findtext("description"))[:500],
            "published": _parse_date(it.findtext("pubDate")),
        })
    return items


class _MetaParser(HTMLParser):
    """Scrape the description and publish time from an article page's <meta> tags."""
    WANT = {"description", "og:description", "twitter:description",
            "article:published_time", "pubdate", "datepublished", "og:updated_time"}

    def __init__(self):
        super().__init__()
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        a = dict(attrs)
        key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
        if key in self.WANT and a.get("content") and key not in self.meta:
            self.meta[key] = a["content"]


def scrape_article(url):
    p = _MetaParser()
    p.feed(_get(url)[:300_000])  # the <head> is near the top; no need to parse the whole page
    m = p.meta
    summary = _clean(m.get("og:description") or m.get("description") or m.get("twitter:description"))
    published = _parse_date(m.get("article:published_time") or m.get("pubdate") or m.get("datepublished"))
    return summary[:500], published


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_existing():
    if not NEWS_FILE.exists():
        return []
    with open(NEWS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run():
    from assistant import detect_provinces  # same province matching as the chatbot

    now = datetime.now(VN_TZ)
    existing = _load_existing()
    known = {r["link"] for r in existing}
    fresh, pages, feed_ok = [], 0, 0

    for source, lang, feed in FEEDS:
        try:
            items = read_feed(feed)
            feed_ok += 1
        except Exception as e:
            log(f"News: cannot read {source} feed ({type(e).__name__}).")
            continue
        for it in items:
            link = it["link"]
            if not link or link in known or not it["title"]:
                continue
            if not is_relevant(it["title"], it["summary"]):
                continue
            known.add(link)
            summary, published = it["summary"], it["published"]
            # Scrape the article page for a better summary / publish time
            if pages < MAX_PAGES and allowed(link):
                try:
                    s2, p2 = scrape_article(link)
                    summary = s2 if len(s2) > len(summary) else summary
                    published = published or p2
                    pages += 1
                    time.sleep(PAUSE)
                except Exception:
                    pass
            text = f"{it['title']} {summary}"
            topics, regions = classify(text)
            fresh.append({
                "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "published": published or now.strftime("%Y-%m-%dT%H:%M"),
                "source": source, "lang": lang, "title": it["title"], "summary": summary, "link": link,
                "provinces": ";".join(detect_provinces(text)), "regions": ";".join(regions),
                "topics": ";".join(topics),
            })

    cutoff = (now - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M")
    rows = [r for r in existing + fresh if (r.get("published") or "") >= cutoff]
    rows.sort(key=lambda r: r.get("published") or "", reverse=True)
    NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = NEWS_FILE.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(NEWS_FILE)
    log(f"News: {feed_ok}/{len(FEEDS)} feeds read, {len(fresh)} new weather articles "
        f"({pages} pages scraped), {len(rows)} articles kept.")


if __name__ == "__main__":
    run()
