"""WEB SCRAPER 3: build a large archive of weather and disaster news articles (past years).

news_scraper.py only reads RSS feeds, which list the newest articles. This script goes back in time:
  1. Read robots.txt of each news site and find its sitemaps (the list of all article links).
  2. Read the sitemaps, newest first, and keep only links whose address (slug) is about weather:
     storms, floods, heavy rain, heat, cold, landslides, air pollution...
  3. Open each kept article and scrape its title, short description and publish time (<meta> tags).
  4. Tag each article with provinces, regions and topics, and save it.

Polite scraping: robots.txt respected, one request per site every PAUSE seconds (sites run in
parallel, one thread each), only title + short summary + link are stored.

Output (the chatbot reads it; the script can be stopped and started again, it continues):
  data/history/news_archive.csv
  live/processed/news_archive.csv   (commit this file so the online web app has it)

Usage:
    python news_archive.py --test          # 3 minutes, to check that it works
    python news_archive.py                 # runs up to 9 hours (stop any time with Ctrl+C)
    python news_archive.py --hours 5
"""
import argparse
import csv
import gzip
import html
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

import config
import news_scraper as ns

OUT = config.DATA_DIR / "history" / "news_archive.csv"
LIVE = config.ROOT / "live" / "processed" / "news_archive.csv"
COLUMNS = ns.COLUMNS
PAUSE = 1.2               # seconds between two requests to the same site
MAX_SITEMAPS = 3000       # sitemap files read per site at most
OLDEST_YEAR = 2015
MAX_TOTAL = 60000         # stop at this many articles (keeps the online web app fast and the file < 50 MB)

SITES = [
    ("VnExpress", "vi", "https://vnexpress.net"),
    ("Tuổi Trẻ", "vi", "https://tuoitre.vn"),
    ("Thanh Niên", "vi", "https://thanhnien.vn"),
    ("Dân Trí", "vi", "https://dantri.com.vn"),
    ("VietNamNet", "vi", "https://vietnamnet.vn"),
    ("Người Lao Động", "vi", "https://nld.com.vn"),
    ("Lao Động", "vi", "https://laodong.vn"),
    ("VietnamPlus", "vi", "https://www.vietnamplus.vn"),
    ("VOV", "vi", "https://vov.vn"),
    ("Báo Tin Tức", "vi", "https://baotintuc.vn"),
    ("Sài Gòn Giải Phóng", "vi", "https://www.sggp.org.vn"),
    ("Nhân Dân", "vi", "https://nhandan.vn"),
    ("VnExpress International", "en", "https://e.vnexpress.net"),
    ("Tuoi Tre News", "en", "https://news.tuoitre.vn"),
    ("Viet Nam News", "en", "https://vietnamnews.vn"),
]

# Words in the article address (without accents, joined by "-"). Plain "bao" is NOT used: it can mean
# bão (storm), báo (newspaper), bảo (protect)... so storm words always come with a second word.
SLUG_WORDS = """
bao-so con-bao sieu-bao bao-manh bao-do-bo bao-vao bao-gay bao-tan-pha tam-bao hoan-luu-bao sau-bao
ap-thap ap-thap-nhiet-doi vung-ap-thap
mua-lon mua-to mua-bao mua-dong mua-da mua-ret mua-lu lu-lut lu-quet lu-lon lu-ong lu-dau-nguon nuoc-lu dinh-lu
ngap-lut ngap-ung ngap-sau ngap-nang bi-ngap sat-lo lo-dat trieu-cuong
nang-nong nang-gat nang-dot nhiet-do-cao nang-nong-gay-gat
ret-dam ret-hai khong-khi-lanh gio-mua-dong-bac bang-gia suong-muoi tuyet-roi
thoi-tiet du-bao-thoi-tiet thien-tai khi-tuong han-han xam-nhap-man dong-loc loc-xoay set-danh suong-mu
bui-min o-nhiem-khong-khi chat-luong-khong-khi
typhoon storm tropical-storm tropical-depression flood floods flooding flash-flood landslide landslides
heatwave heat-wave heavy-rain downpour cold-spell cold-snap drought saltwater-intrusion air-pollution
air-quality smog weather natural-disaster
""".split()
STORM_NAMES = """yagi noru kalmaegi bualoi ragasa wipha kajiki trami noul molave vamco damrey doksuri saola talim
koinu matmo fengshen fung-wong conson chanthu rai sonca mangkhut wutip podul kammuri linfa nangka goni
etau krovanh haiyan rammasun xangsane ketsana sinlaku tembin usagi nesat sarika haima""".split()
SLUG_WORDS += [f"bao-{n}" for n in STORM_NAMES] + [f"typhoon-{n}" for n in STORM_NAMES] + [f"storm-{n}" for n in STORM_NAMES]
SLUG_RE = re.compile(r"(?:^|-)(?:" + "|".join(sorted(map(re.escape, set(SLUG_WORDS)), key=len, reverse=True)) + r")(?:-|$)")
SKIP_SITEMAP = re.compile(r"image|video|photo|tag|categor|chu-de|author|tac-gia|podcast|emagazine|topic", re.I)


class _Meta(ns._MetaParser):
    WANT = ns._MetaParser.WANT | {"og:title", "twitter:title"}


def _fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": ns.USER_AGENT, "Accept-Language": "vi,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        charset = r.headers.get_content_charset() or "utf-8"
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode(charset, errors="replace")


def _slug(url):
    path = urllib.parse.unquote(urllib.parse.urlsplit(url).path).lower()
    last = path.rstrip("/").rsplit("/", 1)[-1]
    last = re.sub(r"\.(html?|htm|vnp|aspx?)$", "", last)
    last = re.sub(r"[-_]?\d{5,}$", "", last)          # article id at the end
    return re.sub(r"[^a-z0-9]+", "-", last).strip("-")


def _sitemap_date(url):
    """Year and month in a sitemap file name, for sorting newest first (0 if none)."""
    m = re.search(r"(20\d\d)[-_/]?(\d{1,2})?(?!\d)", url)
    if not m:
        return 999999  # no date in the name (e.g. "sitemap-news.xml"): read it first
    return int(m[1]) * 100 + int(m[2] or 12)


def _locs(xml, tag):
    out = []
    for block in re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}>", xml, flags=re.S | re.I):
        loc = re.search(r"<loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</loc>", block, flags=re.S | re.I)
        if not loc:
            continue
        title = re.search(r"<(?:news:title|image:title)>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</", block, flags=re.S | re.I)
        date = re.search(r"<(?:news:publication_date|lastmod)>\s*(.*?)\s*</", block, flags=re.S | re.I)
        out.append((html.unescape(loc[1]).strip(), html.unescape(title[1]).strip() if title else "",
                    date[1].strip() if date else ""))
    return out


class Archive:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows, self.links, self.new = [], set(), 0
        if OUT.exists():
            with open(OUT, encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
            self.links = {r["link"] for r in self.rows}
        self.stats = {}

    def has(self, link):
        with self.lock:
            return link in self.links

    def add(self, row):
        with self.lock:
            if row["link"] in self.links:
                return
            self.links.add(row["link"])
            self.rows.append(row)
            self.new += 1
            if self.new % 100 == 0:
                self._save()

    def _save(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(self.rows)
        tmp.replace(OUT)

    def save(self):
        with self.lock:
            self._save()
        LIVE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT, LIVE)


def site_worker(name, lang, base, archive, stop_at):
    from assistant import detect_provinces
    st = archive.stats.setdefault(name, dict(sitemaps=0, links=0, matched=0, saved=0, errors=0, note=""))
    last = [0.0]

    def get(url):
        wait = PAUSE - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        last[0] = time.time()
        return _fetch(url)

    # 1. find the sitemaps
    roots = []
    try:
        robots = get(base + "/robots.txt")
        roots = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots)
    except Exception:
        st["note"] = "no robots.txt"
    roots += [base + "/sitemap.xml", base + "/sitemap_index.xml"]
    roots = list(dict.fromkeys(roots))

    queue, seen_maps = [(r, 0) for r in roots], set()
    while queue and time.time() < stop_at and st["sitemaps"] < MAX_SITEMAPS and len(archive.rows) < MAX_TOTAL:
        queue.sort(key=lambda x: _sitemap_date(x[0]), reverse=True)   # newest first
        url, depth = queue.pop(0)
        if url in seen_maps or not ns.allowed(url):
            continue
        seen_maps.add(url)
        try:
            xml = get(url)
        except Exception:
            st["errors"] += 1
            continue
        st["sitemaps"] += 1
        if re.search(r"<sitemapindex", xml, re.I):
            for child, _, _ in _locs(xml, "sitemap"):
                if depth < 3 and not SKIP_SITEMAP.search(child) and (_sitemap_date(child) // 100) >= OLDEST_YEAR:
                    queue.append((child, depth + 1))
            continue
        # 2. a list of articles: keep the ones about weather
        for link, title, date in _locs(xml, "url"):
            if time.time() >= stop_at or len(archive.rows) >= MAX_TOTAL:
                break
            st["links"] += 1
            if not link.startswith("http") or archive.has(link):
                continue
            words = _slug(link) + "-" + re.sub(r"[^a-z0-9]+", "-", _plain(title))
            if not SLUG_RE.search(words):
                continue
            st["matched"] += 1
            if not ns.allowed(link):
                continue
            # 3. open the article: title, description, publish time
            try:
                p = _Meta()
                p.feed(get(link)[:300_000])
            except Exception:
                st["errors"] += 1
                continue
            m = p.meta
            t = ns._clean(html.unescape(m.get("og:title") or m.get("twitter:title") or title))
            summary = ns._clean(html.unescape(m.get("og:description") or m.get("description") or ""))[:400]
            if not t:
                continue
            topics, regions = ns.classify(f"{t} {summary}")
            if not set(topics) & ns.STRONG and "weather" not in topics:
                continue
            published = ns._parse_date(m.get("article:published_time") or m.get("datepublished")
                                       or m.get("pubdate") or date) or ""
            archive.add(dict(fetched_at=datetime.now().strftime("%Y-%m-%dT%H:%M"), published=published,
                             source=name, lang=lang, title=t, summary=summary, link=link,
                             provinces=";".join(detect_provinces(f"{t} {summary}")),
                             regions=";".join(regions), topics=";".join(topics)))
            st["saved"] += 1


def _plain(text):
    import unicodedata
    text = text.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=9)
    ap.add_argument("--test", action="store_true", help="run 3 minutes only")
    ap.add_argument("--max", type=int, default=MAX_TOTAL, help="stop at this many articles")
    args = ap.parse_args()
    globals()["MAX_TOTAL"] = args.max
    seconds = 180 if args.test else args.hours * 3600
    stop_at = time.time() + seconds

    archive = Archive()
    print(f"Archive has {len(archive.rows)} articles. Scraping {len(SITES)} sites for "
          f"{seconds / 60:.0f} minutes (Ctrl+C to stop, progress is saved)...", flush=True)
    threads = [threading.Thread(target=site_worker, args=(n, l, b, archive, stop_at), daemon=True) for n, l, b in SITES]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads) and time.time() < stop_at + 60:
            time.sleep(60 if not args.test else 30)
            total = len(archive.rows)
            print(f"[{datetime.now():%H:%M}] archive: {total} articles (+{archive.new} this run)", flush=True)
            for n, s in archive.stats.items():
                print(f"    {n:<24} sitemaps {s['sitemaps']:>4}  links {s['links']:>7}  weather {s['matched']:>5}  "
                      f"saved {s['saved']:>5}  errors {s['errors']:>3} {s['note']}", flush=True)
            archive.save()
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    archive.save()
    print(f"Done: {len(archive.rows)} articles in {OUT} and {LIVE} "
          f"({LIVE.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
