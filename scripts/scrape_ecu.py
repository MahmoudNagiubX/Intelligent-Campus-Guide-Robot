"""
Build data/ecu_knowledge.json from public ECU website pages.

Usage:
  python scripts/scrape_ecu.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_PATH = ROOT / "data" / "ecu_knowledge.json"
BASE_URL = "https://ecu.edu.eg"
REQUEST_DELAY_SEC = 1.5
MAX_CONTENT_CHARS = 600
USER_AGENT = "NavigatorCampusGuide/1.0 (+https://ecu.edu.eg)"

PRIORITY_URLS = [
    "https://ecu.edu.eg/",
    "https://ecu.edu.eg/faculties/",
    "https://ecu.edu.eg/faculties/engineering-and-technology/",
    "https://ecu.edu.eg/faculties/computers-and-information-systems/",
    "https://ecu.edu.eg/faculties/economics-and-international-trade/",
    "https://ecu.edu.eg/faculties/pharmacy-drug-technology/",
    "https://ecu.edu.eg/faculties/physical-therapy/",
    "https://ecu.edu.eg/faculties/arts-and-design/",
    "https://ecu.edu.eg/faculties/veterinary-medicine/",
    "https://ecu.edu.eg/faculties/media/",
    "https://ecu.edu.eg/faculties/law/",
    "https://ecu.edu.eg/about-ecu/university-profile/",
    "https://ecu.edu.eg/about-ecu/university-board-and-management/",
    "https://ecu.edu.eg/about-ecu/vision-mission-and-objectives/",
    "https://ecu.edu.eg/about-ecu/centers/",
    "https://ecu.edu.eg/contact-us/",
    "https://ecu.edu.eg/library/",
    "https://ecu.edu.eg/education-and-students-affairs/managements-of-education-and-student-affairs/",
    "https://ecu.edu.eg/about-ecu/ri-center/",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ECU public pages into data/ecu_knowledge.json")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        emit(f"{OUTPUT_PATH} already exists. Use --force to refresh.")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    entries = []
    seen: set[str] = set()

    for url in PRIORITY_URLS:
        normalized_url = _normalize_url(url)
        if normalized_url in seen or not _is_allowed_url(normalized_url):
            continue
        seen.add(normalized_url)
        entry = _scrape_page(session, normalized_url)
        if entry is not None:
            entries.append(entry)
            emit(f"[OK] {entry['title']} <- {normalized_url}")
        time.sleep(REQUEST_DELAY_SEC)

    payload = {
        "scraped_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_url": BASE_URL,
        "entry_count": len(entries),
        "entries": entries,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit(f"Wrote {len(entries)} entries to {OUTPUT_PATH}")


def _scrape_page(session: requests.Session, url: str) -> dict | None:
    try:
        response = session.get(url, timeout=12)
        response.raise_for_status()
    except Exception as exc:
        emit(f"[WARN] failed {url}: {exc}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    for selector in ("script", "style", "nav", "footer", "header", "noscript"):
        for node in soup.select(selector):
            node.decompose()

    title = _extract_title(soup, url)
    headings = _extract_headings(soup)
    content = _clean_text(" ".join(soup.stripped_strings))[:MAX_CONTENT_CHARS].strip()
    if not content:
        return None
    return {
        "title": title,
        "url": url,
        "content": content,
        "keywords": _keywords(title, headings, url),
        "category": _category_for(url),
        "scraped_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _extract_title(soup: BeautifulSoup, url: str) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _clean_text(h1.get_text(" ", strip=True))
    if soup.title and soup.title.get_text(strip=True):
        return _clean_text(soup.title.get_text(" ", strip=True))
    return url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title() or "ECU"


def _extract_headings(soup: BeautifulSoup) -> list[str]:
    values = []
    for tag in soup.find_all(["h1", "h2"]):
        text = _clean_text(tag.get_text(" ", strip=True))
        if text:
            values.append(text)
    return values


def _keywords(title: str, headings: list[str], url: str) -> list[str]:
    bucket: list[str] = []
    for text in [title, *headings]:
        _add_keyword(bucket, text)
        for word in re.findall(r"[A-Za-z]{3,}", text):
            _add_keyword(bucket, word.lower())

    slug = urlparse(url).path.strip("/").replace("-", " ")
    _add_keyword(bucket, slug)
    if "engineering" in slug:
        _add_keyword(bucket, "engineering faculty")
        _add_keyword(bucket, "faculty of engineering")
    if "computers" in slug:
        _add_keyword(bucket, "computer science")
        _add_keyword(bucket, "faculty of computers")
        _add_keyword(bucket, "cis")
    if "pharmacy" in slug:
        _add_keyword(bucket, "faculty of pharmacy")
    if "physical therapy" in slug:
        _add_keyword(bucket, "pt")
    if "media" in slug:
        _add_keyword(bucket, "mass communication")
    return bucket


def _add_keyword(bucket: list[str], value: str) -> None:
    value = _clean_text(value).lower()
    if value and value not in bucket:
        bucket.append(value)


def _category_for(url: str) -> str:
    path = urlparse(url).path.lower()
    if "/faculties/" in path:
        return "faculty"
    if "contact" in path:
        return "contact"
    if "library" in path:
        return "library"
    if "about" in path:
        return "about"
    return "general"


def _normalize_url(url: str) -> str:
    url = urljoin(BASE_URL, url)
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl()


def _is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "ecu.edu.eg":
        return False
    path = parsed.path.lower()
    return not path.endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".zip"))


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


if __name__ == "__main__":
    main()
