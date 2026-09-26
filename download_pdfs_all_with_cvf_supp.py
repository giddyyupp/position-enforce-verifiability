import argparse
import asyncio
import hashlib
import os
import random
import re
from dataclasses import dataclass
from typing import Optional, Iterable
from urllib.parse import urljoin, urlparse, parse_qs

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm


# -----------------------
# Config
# -----------------------
START_YEAR = 2021
END_YEAR = 2025

TIMEOUT = 60
RETRIES = 6

# Domain-specific concurrency caps (prevents rate-limit storms)
DOMAIN_LIMITS = {
    "openaccess.thecvf.com": 5,
    "proceedings.mlr.press": 20,
    "proceedings.neurips.cc": 30,
    "papers.nips.cc": 30,
    "neurips.cc": 25,
    "iclr.cc": 25,
    "openreview.net": 25,
    "arxiv.org": 15,
}

USER_AGENT = "pdf-downloader/1.0 (research use)"


# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name

def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower()

def openreview_pdf_from_any_url(u: str) -> str:
    # forum?id=XXXX -> pdf?id=XXXX
    try:
        q = parse_qs(urlparse(u).query)
        if "id" in q and q["id"]:
            return f"https://openreview.net/pdf?id={q['id'][0]}"
    except Exception:
        pass
    if "openreview.net/forum?id=" in u:
        return u.replace("/forum?", "/pdf?")
    if "openreview.net/pdf?id=" in u:
        return u
    return ""

def extract_arxiv_id(u: str) -> str:
    # https://arxiv.org/abs/1511.05493 -> 1511.05493
    p = urlparse(u).path
    m = re.search(r"/abs/([^/]+)$", p)
    if m:
        return m.group(1)
    m = re.search(r"/pdf/([^/]+)$", p)
    if m:
        return m.group(1).replace(".pdf", "")
    return ""


# -----------------------
# Async HTTP (with retries, per-domain semaphores)
# -----------------------
class Fetcher:
    def __init__(self, concurrency: int):
        self.global_conc = concurrency
        self.domain_sems = {}
        for dom, cap in DOMAIN_LIMITS.items():
            self.domain_sems[dom] = asyncio.Semaphore(min(cap, concurrency))

        limits = httpx.Limits(
            max_connections=concurrency * 2,
            max_keepalive_connections=concurrency
        )
        self.client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=TIMEOUT,
            http2=True,
            limits=limits,
        )

    async def close(self):
        await self.client.aclose()

    async def get_text(self, url: str) -> Optional[str]:
        sem = self.domain_sems.get(domain_of(url), asyncio.Semaphore(self.global_conc))

        for attempt in range(RETRIES):
            try:
                async with sem:
                    r = await self.client.get(url)
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        await asyncio.sleep(min(60, int(ra)))
                    else:
                        await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
                    continue
                if r.status_code in (500, 502, 503, 504):
                    await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
                    continue
                r.raise_for_status()
                return r.text
            except Exception:
                await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
        return None

    async def download(self, url: str, out_path: str, skip_existing: bool = True) -> bool:
        if not url:
            return False
        if skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True

        ensure_dir(os.path.dirname(out_path))
        sem = self.domain_sems.get(domain_of(url), asyncio.Semaphore(self.global_conc))

        for attempt in range(RETRIES):
            try:
                async with sem:
                    r = await self.client.get(url)
                if r.status_code == 404:
                    return False
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        await asyncio.sleep(min(60, int(ra)))
                    else:
                        await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
                    continue
                if r.status_code in (500, 502, 503, 504):
                    await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
                    continue
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
            except Exception:
                await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
        return False

    async def get_json(self, url: str) -> Optional[dict]:
        sem = self.domain_sems.get(domain_of(url), asyncio.Semaphore(self.global_conc))

        for attempt in range(RETRIES):
            try:
                async with sem:
                    r = await self.client.get(url)
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        await asyncio.sleep(min(60, int(ra)))
                    else:
                        await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
                    continue
                if r.status_code in (500, 502, 503, 504):
                    await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception:
                await asyncio.sleep(min(10, (2 ** attempt) * 0.5 + random.random()))
        return None


# -----------------------
# CVPR/ICCV (CVF Open Access)
# -----------------------
def cvf_index_candidates(conf: str, year: int) -> list[str]:
    base = "https://openaccess.thecvf.com/"
    # order matters: ?day=all helps some years
    return [
        f"{base}{conf}{year}.py?day=all",
        f"{base}{conf}{year}?day=all",
        f"{base}{conf}{year}.py",
        f"{base}{conf}{year}",
        f"{base}{conf}{year}/",
    ]

def parse_cvf_paper_links(index_html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.endswith("_paper.html") or h.endswith("_paper.php") or "_paper.html" in h or "_paper.php" in h:
            urls.append(urljoin(base_url, h))
    return list(dict.fromkeys(urls))

def parse_cvf_pdf_from_paper_page(paper_html: str, paper_url: str) -> str:
    soup = BeautifulSoup(paper_html, "html.parser")
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() == "pdf":
            return urljoin(paper_url, a["href"])
    return ""

def parse_cvf_supp_pdf_from_paper_page(paper_html: str, paper_url: str) -> str:
    """Return supplementary *PDF* URL if present on a CVF paper page.

    CVF Open Access typically exposes links as: [pdf] [supp] [bibtex] ...
    We only return the supplementary link if it ends with .pdf.
    """
    soup = BeautifulSoup(paper_html, "html.parser")
    for a in soup.find_all("a", href=True):
        txt = a.get_text(strip=True).lower()
        if txt not in {"supp", "supplementary", "supplement"}:
            continue
        href = (a.get("href") or "").strip()
        if not href:
            continue
        u = urljoin(paper_url, href)
        if u.lower().endswith(".pdf"):
            return u
    return ""


_CVF_MAIN_RE = re.compile(r"^(\d{5})_([0-9a-f]{10})\.pdf$", re.IGNORECASE)

def build_cvf_main_pdf_map(folder: str) -> dict[str, str]:
    """Map sha1(pdf_url)->existing main pdf filename in folder.

    The downloader names main PDFs as: 00012_<sha1(pdf_url)>.pdf
    We use the sha1 part to align supplementary files with the same base name.
    """
    mp: dict[str, str] = {}
    if not os.path.isdir(folder):
        return mp
    for fn in os.listdir(folder):
        if fn.endswith("_supp.pdf"):
            continue
        m = _CVF_MAIN_RE.match(fn)
        if not m:
            continue
        h = m.group(2).lower()
        # keep first occurrence if any collisions
        mp.setdefault(h, fn)
    return mp

async def cvf_day_pages_if_needed(fetcher: Fetcher, idx_html: str, idx_url_used: str, conf: str, year: int) -> list[str]:
    # First try direct paper links
    paper_urls = parse_cvf_paper_links(idx_html, idx_url_used)
    if paper_urls:
        return paper_urls

    soup = BeautifulSoup(idx_html, "html.parser")

    idx_base = idx_url_used.split("?", 1)[0]
    idx_base_alt = idx_base[:-3] if idx_base.endswith(".py") else None  # try without .py

    day_urls = []

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        text = a.get_text(" ", strip=True)

        # If href has day=, use it
        if "day=" in href:
            if href.startswith("?"):
                day_urls.append(idx_base + href)
                if idx_base_alt:
                    day_urls.append(idx_base_alt + href)
            else:
                day_urls.append(urljoin(idx_base, href))
                if idx_base_alt:
                    day_urls.append(urljoin(idx_base_alt, href))
            continue

        # Otherwise, parse date from visible text like "Day 1: 2018-06-19" :contentReference[oaicite:6]{index=6}
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if m:
            day_urls.append(f"{idx_base}?day={m.group(1)}")
            if idx_base_alt:
                day_urls.append(f"{idx_base_alt}?day={m.group(1)}")

    day_urls = list(dict.fromkeys(day_urls))

    # CVPR 2018 hard fallback if needed (3 conference days) :contentReference[oaicite:7]{index=7}
    if not day_urls and conf == "CVPR" and year == 2018:
        bases = [idx_base] + ([idx_base_alt] if idx_base_alt else [])
        for b in bases:
            day_urls.extend([
                f"{b}?day=2018-06-19",
                f"{b}?day=2018-06-20",
                f"{b}?day=2018-06-21",
            ])
        day_urls = list(dict.fromkeys(day_urls))

    # Hard fallback for CVPR day pages if CVF landing page doesn't expose them
    if not day_urls and conf == "CVPR" and year == 2019:
        bases = [idx_base] + ([idx_base_alt] if idx_base_alt else [])
        # CVPR 2019 ran June 16–20, 2019 (main conference days typically 18–20, but day pages exist across the span)
        for b in bases:
            for d in ["2019-06-16","2019-06-17","2019-06-18","2019-06-19","2019-06-20"]:
                day_urls.append(f"{b}?day={d}")
        day_urls = list(dict.fromkeys(day_urls))

    if not day_urls and conf == "CVPR" and year == 2020:
        bases = [idx_base] + ([idx_base_alt] if idx_base_alt else [])
        # CVPR 2020 virtual span; CVF uses day pages around June 14–19, 2020
        for b in bases:
            for d in ["2020-06-14","2020-06-15","2020-06-16","2020-06-17","2020-06-18","2020-06-19"]:
                day_urls.append(f"{b}?day={d}")
        day_urls = list(dict.fromkeys(day_urls))

    if not day_urls:
        return []

    day_pages = await asyncio.gather(*[fetcher.get_text(u) for u in day_urls])

    all_papers = []
    for durl, dhtml in zip(day_urls, day_pages):
        if not dhtml:
            continue
        all_papers.extend(parse_cvf_paper_links(dhtml, durl))

    return list(dict.fromkeys(all_papers))


async def download_cvf_conf_year(fetcher: Fetcher, out_dir: str, conf: str, year: int, skip_existing: bool):
    folder = os.path.join(out_dir, f"{conf}_{year}")
    ensure_dir(folder)

    idx_html = None
    idx_url_used = None
    for cand in cvf_index_candidates(conf, year):
        idx_html = await fetcher.get_text(cand)
        if idx_html:
            idx_url_used = cand
            break
    if not idx_html or not idx_url_used:
        return 0

    paper_pages = await cvf_day_pages_if_needed(fetcher, idx_html, idx_url_used, conf, year)
    if not paper_pages:
        return 0

    # Fetch paper pages concurrently (bounded by domain semaphore inside fetcher)
    paper_htmls = await asyncio.gather(*[fetcher.get_text(u) for u in paper_pages])

    pdf_urls = []
    for purl, phtml in zip(paper_pages, paper_htmls):
        if not phtml:
            continue
        pdf = parse_cvf_pdf_from_paper_page(phtml, purl)
        if pdf:
            pdf_urls.append(pdf)
    pdf_urls = list(dict.fromkeys(pdf_urls))

    # Download PDFs
    tasks = []
    for i, pdf_url in enumerate(pdf_urls):
        fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
        out_path = os.path.join(folder, fname)
        tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))

    results = await asyncio.gather(*tasks)
    return sum(1 for r in results if r)


async def download_cvf_supp_only_conf_year(fetcher: Fetcher, out_dir: str, conf: str, year: int, skip_existing: bool):
    """Download supplementary PDFs for CVPR/ICCV from CVF Open Access.

    Naming: if the main PDF already exists in <out_dir>/<conf>_<year>/ with the
    script's default name (00012_<sha1(main_pdf_url)>.pdf), the supplementary
    will be saved as:
        00012_<sha1(main_pdf_url)>_supp.pdf

    If the main PDF cannot be found, we still download the supplementary (when
    present) using a fallback name:
        missing_<sha1(main_pdf_url)>_supp.pdf
    """
    folder = os.path.join(out_dir, f"{conf}_{year}")
    ensure_dir(folder)

    # Map sha1(main_pdf_url) -> existing main filename, so supp can share base name.
    main_map = build_cvf_main_pdf_map(folder)

    idx_html = None
    idx_url_used = None
    for cand in cvf_index_candidates(conf, year):
        idx_html = await fetcher.get_text(cand)
        if idx_html:
            idx_url_used = cand
            break
    if not idx_html or not idx_url_used:
        return 0

    paper_pages = await cvf_day_pages_if_needed(fetcher, idx_html, idx_url_used, conf, year)
    if not paper_pages:
        return 0

    paper_htmls = await asyncio.gather(*[fetcher.get_text(u) for u in paper_pages])

    supp_jobs: list[tuple[str, str]] = []  # (supp_url, out_path)
    for purl, phtml in zip(paper_pages, paper_htmls):
        if not phtml:
            continue

        main_pdf_url = parse_cvf_pdf_from_paper_page(phtml, purl)
        if not main_pdf_url:
            continue

        supp_url = parse_cvf_supp_pdf_from_paper_page(phtml, purl)
        if not supp_url:
            continue

        h = sha1(main_pdf_url)
        main_fn = main_map.get(h)
        if main_fn:
            base = main_fn[:-4]  # strip .pdf
            supp_fn = f"{base}_supp.pdf"
        else:
            supp_fn = f"missing_{h}_supp.pdf"

        out_path = os.path.join(folder, supp_fn)
        supp_jobs.append((supp_url, out_path))

    # De-duplicate by output path (some pages can repeat in CVF index/day views)
    seen_out = set()
    deduped = []
    for u, p in supp_jobs:
        if p in seen_out:
            continue
        seen_out.add(p)
        deduped.append((u, p))

    tasks = [fetcher.download(u, p, skip_existing=skip_existing) for u, p in deduped]
    if not tasks:
        return 0

    results = await asyncio.gather(*tasks)
    return sum(1 for r in results if r)


# -----------------------
# ICML (PMLR)
# -----------------------
async def build_icml_volume_map(fetcher: Fetcher) -> dict[int, str]:
    html = await fetcher.get_text("https://proceedings.mlr.press/")
    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    mapping: dict[int, str] = {}

    # Collect candidates per year, then pick best match (main proceedings)
    cand: dict[int, list[tuple[str, str]]] = {}

    for li in soup.find_all("li"):
        txt = li.get_text(" ", strip=True)
        m = re.search(r"\bICML\b.*\b(20\d{2})\b", txt)
        if not m:
            continue
        year = int(m.group(1))
        a = li.find("a", href=True)
        if not a:
            continue
        url = urljoin("https://proceedings.mlr.press/", a["href"])
        cand.setdefault(year, []).append((txt, url))

    def score(text: str) -> int:
        t = text.lower()
        s = 0
        # prefer main proceedings wording
        if "proceedings of the" in t and "international conference on machine learning" in t:
            s += 100
        if "proceedings of icml" in t:
            s += 80
        # avoid workshops/tutorials
        if "workshop" in t:
            s -= 50
        if "tutorial" in t:
            s -= 30
        if "symposium" in t:
            s -= 20
        return s

    for year, items in cand.items():
        items_sorted = sorted(items, key=lambda x: score(x[0]), reverse=True)
        mapping[year] = items_sorted[0][1]

    return mapping


async def find_icml_volume_url(fetcher: Fetcher, year: int, icml_map: dict[int, str]) -> Optional[str]:
    return icml_map.get(year)


def parse_pmlr_volume_pdf_links(volume_html: str, vol_url: str) -> list[str]:
    soup = BeautifulSoup(volume_html, "html.parser")
    pdfs = []

    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True).lower()
        href = a["href"].strip()

        # Main paper PDFs appear as "Download PDF" links on many volumes (e.g., v37, v80, v97, v119). :contentReference[oaicite:3]{index=3}
        if txt in {"download pdf", "pdf"}:
            pdfs.append(urljoin(vol_url, href))
            continue

        # Fallback: any direct .pdf link (avoid supplementary/other file bundles)
        if href.lower().endswith(".pdf") and "supp" not in href.lower():
            pdfs.append(urljoin(vol_url, href))

    return list(dict.fromkeys(pdfs))


async def download_icml_year(fetcher: Fetcher, out_dir: str, year: int, skip_existing: bool, icml_map: dict[int, str]):
    folder = os.path.join(out_dir, f"ICML_{year}")
    ensure_dir(folder)

    vol_url = icml_map.get(year)
    if not vol_url:
        return 0

    vol_html = await fetcher.get_text(vol_url)
    if not vol_html:
        return 0

    pdf_urls = parse_pmlr_volume_pdf_links(vol_html, vol_url)
    if not pdf_urls:
        return 0

    tasks = []
    for i, pdf_url in enumerate(pdf_urls):
        fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
        out_path = os.path.join(folder, fname)
        tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))

    results = await asyncio.gather(*tasks)
    return sum(1 for r in results if r)


# -----------------------
# NeurIPS 2015–2024 (Proceedings book pages, exclude D&B)
# -----------------------
_ABS_RE = re.compile(r"-Abstract(?:-[A-Za-z_]+)?\.html$", re.IGNORECASE)

def neurips_book_candidates(year: int) -> list[str]:
    return [
        f"https://proceedings.neurips.cc/paper_files/paper/{year}",
        f"https://proceedings.neurips.cc/paper/{year}",
        f"https://papers.nips.cc/paper/{year}",
    ]

def parse_neurips_book_abs_pages(book_html: str, book_url: str, year: int, exclude_db: bool = True) -> list[str]:
    soup = BeautifulSoup(book_html, "html.parser")
    urls = []

    for li in soup.find_all("li"):
        a = li.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        if not _ABS_RE.search(href):
            continue
        if f"/paper/{year}/" not in href and f"/paper_files/paper/{year}/" not in href:
            continue

        if exclude_db and "Datasets and Benchmarks Track" in li.get_text(" ", strip=True):
            continue

        urls.append(urljoin(book_url, href))

    return list(dict.fromkeys(urls))

def parse_neurips_abs_pdf(abs_html: str, abs_url: str) -> str:
    soup = BeautifulSoup(abs_html, "html.parser")
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() == "paper":
            return urljoin(abs_url, a["href"])
    return ""

async def download_neurips_2015_2024(fetcher: Fetcher, out_dir: str, year: int, skip_existing: bool):
    folder = os.path.join(out_dir, f"NeurIPS_{year}")
    ensure_dir(folder)

    book_html = None
    book_url = None
    for cand in neurips_book_candidates(year):
        book_html = await fetcher.get_text(cand)
        if book_html:
            book_url = cand
            break
    if not book_html or not book_url:
        return 0

    abs_pages = parse_neurips_book_abs_pages(book_html, book_url, year, exclude_db=True)
    if not abs_pages:
        return 0

    abs_htmls = await asyncio.gather(*[fetcher.get_text(u) for u in abs_pages])
    pdf_urls = []
    for aurl, ahtml in zip(abs_pages, abs_htmls):
        if not ahtml:
            continue
        pdf = parse_neurips_abs_pdf(ahtml, aurl)
        if pdf:
            pdf_urls.append(pdf)
    pdf_urls = list(dict.fromkeys(pdf_urls))

    tasks = []
    for i, pdf_url in enumerate(pdf_urls):
        fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
        out_path = os.path.join(folder, fname)
        tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))

    results = await asyncio.gather(*tasks)
    return sum(1 for r in results if r)


# -----------------------
# NeurIPS 2025 (Virtual posters, exclude D&B + Journal Track)
# -----------------------
def neurips_2025_papers_url() -> str:
    return "https://neurips.cc/virtual/2025/papers.html"

def extract_poster_id(u: str) -> Optional[str]:
    m = re.search(r"/poster/(\d+)\b", u)
    return m.group(1) if m else None

def parse_virtual_poster_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/poster/" in href:
            urls.append(urljoin(base_url, href))
    return list(dict.fromkeys(urls))

def parse_virtual_page_poster_ids(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    ids = set()
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"].strip())
        pid = extract_poster_id(u)
        if pid:
            ids.add(pid)
    return ids

def neurips_2025_exclusion_pages() -> list[str]:
    # Exclude D&B + Journal Track
    return [
        "https://neurips.cc/virtual/2025/loc/san-diego/events/datasets-benchmarks-2025",
        "https://neurips.cc/virtual/2025/loc/mexico-city/events/datasets-benchmarks-2025",
        "https://neurips.cc/virtual/2025/loc/san-diego/events/2025-journal-track-papers",
        "https://neurips.cc/virtual/2025/loc/mexico-city/events/2025-journal-track-papers",
    ]

def parse_poster_openreview_pdf(poster_html: str) -> str:
    soup = BeautifulSoup(poster_html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "openreview.net" in href:
            pdf = openreview_pdf_from_any_url(href)
            if pdf:
                return pdf
    return ""

async def download_neurips_2025(fetcher: Fetcher, out_dir: str, skip_existing: bool):
    year = 2025
    folder = os.path.join(out_dir, f"NeurIPS_{year}")
    ensure_dir(folder)

    main_html = await fetcher.get_text(neurips_2025_papers_url())
    if not main_html:
        return 0

    poster_urls = parse_virtual_poster_links(main_html, neurips_2025_papers_url())
    if not poster_urls:
        return 0

    exclude_ids: set[str] = set()
    for ex_url in neurips_2025_exclusion_pages():
        ex_html = await fetcher.get_text(ex_url)
        if ex_html:
            exclude_ids |= parse_virtual_page_poster_ids(ex_html, ex_url)

    filtered_posters = []
    for u in poster_urls:
        pid = extract_poster_id(u)
        if pid and pid in exclude_ids:
            continue
        filtered_posters.append(u)
    filtered_posters = list(dict.fromkeys(filtered_posters))

    poster_htmls = await asyncio.gather(*[fetcher.get_text(u) for u in filtered_posters])
    pdf_urls = []
    for html in poster_htmls:
        if not html:
            continue
        pdf = parse_poster_openreview_pdf(html)
        if pdf:
            pdf_urls.append(pdf)
    pdf_urls = list(dict.fromkeys(pdf_urls))

    tasks = []
    for i, pdf_url in enumerate(pdf_urls):
        fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
        out_path = os.path.join(folder, fname)
        tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))

    results = await asyncio.gather(*tasks)
    return sum(1 for r in results if r)


# -----------------------
# ICLR 2015–2016 (Archive pages, arXiv PDFs)
# -----------------------
ICLR_ARCHIVE_2015 = "https://iclr.cc/archive/www/doku.php?id=iclr2015:main.html"
ICLR_ARCHIVE_2016 = "https://iclr.cc/archive/www/doku.php?id=iclr2016:main.html"

async def download_iclr_2015_2016(fetcher: Fetcher, out_dir: str, year: int, skip_existing: bool):
    folder = os.path.join(out_dir, f"ICLR_{year}")
    ensure_dir(folder)

    url = ICLR_ARCHIVE_2015 if year == 2015 else ICLR_ARCHIVE_2016
    html = await fetcher.get_text(url)
    if not html:
        return 0

    soup = BeautifulSoup(html, "html.parser")
    pdf_urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # ICLR archive pages often link to ar5iv.org/abs/<id> (mirror), not only arxiv.org. :contentReference[oaicite:4]{index=4}
        if ("arxiv.org/abs/" in href) or ("arxiv.org/pdf/" in href) or ("ar5iv.org/abs/" in href):
            aid = extract_arxiv_id(href)
            if aid:
                pdf_urls.append(f"https://arxiv.org/pdf/{aid}.pdf")

    pdf_urls = list(dict.fromkeys(pdf_urls))
    if not pdf_urls:
        return 0

    tasks = []
    for i, pdf_url in enumerate(pdf_urls):
        fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
        out_path = os.path.join(folder, fname)
        tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))

    results = await asyncio.gather(*tasks)
    return sum(1 for r in results if r)


# -----------------------
# ICLR 2017–2025 (Virtual posters -> OpenReview PDFs)
# -----------------------
PAPERCOPILOT_RAW = "https://raw.githubusercontent.com/papercopilot/paperlists/main"

def is_iclr_accepted(status: str) -> bool:
    s = (status or "").strip().lower()
    if not s:
        return False
    if any(x in s for x in ["reject", "withdraw", "desk", "not accept", "not accepted"]):
        return False
    return any(x in s for x in ["accept", "oral", "poster", "spotlight"])

def extract_openreview_pdf_from_record(rec: dict) -> str:
    # try common fields, nested content, and ids
    def getv(d, k):
        v = d.get(k)
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return v

    content = rec.get("content") if isinstance(rec.get("content"), dict) else {}
    candidates = []

    for d in (rec, content):
        for k in ["pdf", "pdf_url", "pdfUrl", "url", "paper_url", "paperUrl", "openreview_url", "forum", "forum_url", "id"]:
            v = getv(d, k)
            if isinstance(v, str) and v:
                candidates.append(v)

    # if a candidate is a forum link, convert to pdf
    for c in candidates:
        if "openreview.net/pdf?id=" in c:
            return c
        if "openreview.net/forum?id=" in c:
            return c.replace("/forum?", "/pdf?")
    # if we have an OpenReview note id
    for c in candidates:
        if re.fullmatch(r"[A-Za-z0-9_-]{8,}", c) and "http" not in c:
            return f"https://openreview.net/pdf?id={c}"

    return ""

async def download_iclr_from_papercopilot(fetcher: Fetcher, out_dir: str, year: int, skip_existing: bool) -> int:
    folder = os.path.join(out_dir, f"ICLR_{year}")
    ensure_dir(folder)

    url = f"{PAPERCOPILOT_RAW}/iclr/iclr{year}.json"
    txt = await fetcher.get_text(url)
    if not txt:
        return 0

    import json
    data = json.loads(txt)
    if isinstance(data, dict):
        data = data.get("papers") or data.get("data") or data
    if not isinstance(data, list):
        return 0

    pdf_urls = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        content = rec.get("content") if isinstance(rec.get("content"), dict) else rec
        status = content.get("status") or content.get("decision") or content.get("final_decision") or ""
        if not is_iclr_accepted(str(status)):
            continue
        pdf = extract_openreview_pdf_from_record(rec)
        if pdf:
            pdf_urls.append(pdf)

    pdf_urls = list(dict.fromkeys(pdf_urls))
    if not pdf_urls:
        return 0

    # batch download (avoid rate-limit storms)
    downloaded = 0
    batch_size = 80
    for start in range(0, len(pdf_urls), batch_size):
        batch = pdf_urls[start:start + batch_size]
        tasks = []
        for i, pdf_url in enumerate(batch, start=start):
            fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
            out_path = os.path.join(folder, fname)
            tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))
        results = await asyncio.gather(*tasks)
        downloaded += sum(1 for r in results if r)
        await asyncio.sleep(0.5)

    return downloaded


import asyncio
import os
import re
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup

def _extract_openreview_forum_id(href: str) -> str:
    # href can be "/forum?id=XXXX" or full url
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://openreview.net" + href
    try:
        q = parse_qs(urlparse(href).query)
        if "id" in q and q["id"]:
            return q["id"][0]
    except Exception:
        return ""
    return ""

def _is_accepted_neurips_2025_block(li_text: str) -> bool:
    t = li_text.lower()
    # Accepted entries show as "Published: ..." and have "NeurIPS 2025 poster/spotlight/oral"
    if "submitted to neurips 2025" in t:
        return False
    if "neurips 2025 poster" in t or "neurips 2025 spotlight" in t or "neurips 2025 oral" in t:
        return True
    if "published:" in t and "neurips 2025" in t:
        return True
    return False

async def download_neurips_2025_openreview(fetcher, out_dir: str, skip_existing: bool) -> int:
    """
    Downloads NeurIPS 2025 main conference accepted PDFs from OpenReview.
    Output folder: <out_dir>/NeurIPS_2025/
    """
    folder = os.path.join(out_dir, "NeurIPS_2025")
    os.makedirs(folder, exist_ok=True)

    base = "https://openreview.net/submissions?venue=NeurIPS.cc%2F2025%2FConference"
    page = 1
    accepted_forum_ids = []
    seen_ids = set()

    # Iterate pages until we see several consecutive pages with zero accepted entries
    empty_pages = 0
    MAX_EMPTY_PAGES = 5

    while True:
        url = f"{base}&page={page}"
        html = await fetcher.get_text(url)
        if not html:
            empty_pages += 1
            if empty_pages >= MAX_EMPTY_PAGES:
                break
            page += 1
            continue

        soup = BeautifulSoup(html, "html.parser")

        # On these pages, each submission appears as a <li> containing a "#### <title>" and status lines.
        items = soup.find_all("li")
        accepted_this_page = 0

        for li in items:
            li_text = li.get_text(" ", strip=True)
            if not li_text:
                continue
            if not _is_accepted_neurips_2025_block(li_text):
                continue

            # Find the forum link inside this list item
            forum_id = ""
            for a in li.find_all("a", href=True):
                href = a["href"]
                if "forum?id=" in href:
                    forum_id = _extract_openreview_forum_id(href)
                    break

            if forum_id and forum_id not in seen_ids:
                seen_ids.add(forum_id)
                accepted_forum_ids.append(forum_id)
                accepted_this_page += 1

        if accepted_this_page == 0:
            empty_pages += 1
            if empty_pages >= MAX_EMPTY_PAGES:
                break
        else:
            empty_pages = 0

        page += 1

    # Convert to PDF URLs
    pdf_urls = [f"https://openreview.net/pdf?id={fid}" for fid in accepted_forum_ids]

    # Download in batches (OpenReview can throttle if you burst too hard)
    downloaded = 0
    batch_size = 80

    for start in range(0, len(pdf_urls), batch_size):
        batch = pdf_urls[start:start + batch_size]
        tasks = []
        for i, pdf_url in enumerate(batch, start=start):
            fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
            out_path = os.path.join(folder, fname)
            tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))
        results = await asyncio.gather(*tasks)
        downloaded += sum(1 for r in results if r)
        await asyncio.sleep(0.5)

    print(f"[NeurIPS 2025] OpenReview accepted found={len(pdf_urls)} downloaded={downloaded}")
    return downloaded


import asyncio
import os
import re
from urllib.parse import quote

def _get_note_content_value(note: dict, key: str) -> str:
    """
    OpenReview v2 sometimes stores values as:
      content[key] = {"value": "..."} or content[key] = "..."
    """
    content = note.get("content", {}) or {}
    v = content.get(key, "")
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    return v if isinstance(v, str) else ""

def _is_neurips_2025_accepted_main(note: dict) -> bool:
    """
    Accepted papers have content.venue like:
      "NeurIPS 2025 poster" / "NeurIPS 2025 spotlight" / "NeurIPS 2025 oral"
    Rejected papers commonly show "Submitted to NeurIPS 2025".
    """
    venue = _get_note_content_value(note, "venue").strip().lower()
    if not venue:
        return False
    if "submitted to neurips 2025" in venue:
        return False
    return any(x in venue for x in ["neurips 2025 poster", "neurips 2025 spotlight", "neurips 2025 oral"])

async def neurips_2025_accepted_note_ids_api(fetcher) -> list[str]:
    """
    Uses OpenReview API v2 to get all notes for the main venueid and filters to accepted.
    This avoids scraping /submissions pages.
    """
    venueid = "NeurIPS.cc/2025/Conference"
    base = "https://api2.openreview.net/notes"

    # API2 query params for content fields use "content.<field>"
    # We'll query venueid and paginate.
    params = f"?content.venueid={quote(venueid, safe='')}&limit=1000&offset="
    offset = 0
    note_ids = []

    while True:
        url = base + params + str(offset)
        data = await fetcher.get_json(url)
        if not data:
            break

        notes = data.get("notes", []) or []
        if not notes:
            break

        for n in notes:
            if _is_neurips_2025_accepted_main(n):
                nid = n.get("id")
                if isinstance(nid, str) and nid:
                    note_ids.append(nid)

        if len(notes) < 1000:
            break
        offset += 1000

    # de-dup preserving order
    seen = set()
    out = []
    for nid in note_ids:
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out

async def download_neurips_2025_main(fetcher, out_dir: str, skip_existing: bool) -> int:
    folder = os.path.join(out_dir, "NeurIPS_2025")
    os.makedirs(folder, exist_ok=True)

    ids = await neurips_2025_accepted_note_ids_api(fetcher)
    print(f"[NeurIPS 2025] accepted_ids_found={len(ids)} (expected ~5290 main-track)")

    pdf_urls = [f"https://openreview.net/pdf?id={nid}" for nid in ids]

    downloaded = 0
    batch_size = 60  # keep modest to avoid 429 storms
    for start in range(0, len(pdf_urls), batch_size):
        batch = pdf_urls[start:start + batch_size]
        tasks = []
        for i, pdf_url in enumerate(batch, start=start):
            fname = f"{i:05d}_{sha1(pdf_url)}.pdf"
            out_path = os.path.join(folder, fname)
            tasks.append(fetcher.download(pdf_url, out_path, skip_existing=skip_existing))
        results = await asyncio.gather(*tasks)
        downloaded += sum(1 for r in results if r)
        await asyncio.sleep(0.5)

    print(f"[NeurIPS 2025] downloaded={downloaded} failures={len(pdf_urls) - downloaded}")
    return downloaded


# -----------------------
# Orchestrator
# -----------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="pdfs_2015_2025")
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--venues", type=str, default="CVPR,ICCV,ICML,ICLR,NeurIPS")
    ap.add_argument(
        "--cvf-supp-only",
        action="store_true",
        help="For CVPR/ICCV, download only supplementary PDFs (named to match existing main PDFs).",
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    fetcher = Fetcher(concurrency=args.concurrency)

    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    years = list(range(args.start_year, args.end_year + 1))

    try:
        for year in years:
            for venue in venues:
                n = 0

                if venue in ("CVPR", "ICCV"):
                    # ICCV is odd years only, but we’ll attempt anyway and just get 0 for even years
                    if args.cvf_supp_only:
                        n = await download_cvf_supp_only_conf_year(fetcher, args.out_dir, venue, year, args.skip_existing)
                    else:
                        n = await download_cvf_conf_year(fetcher, args.out_dir, venue, year, args.skip_existing)
                elif venue == "ICML":
                    icml_map = await build_icml_volume_map(fetcher)
                    n = await download_icml_year(fetcher, args.out_dir, year, args.skip_existing, icml_map)
                elif venue == "NeurIPS":
                    if year <= 2024:
                        n = await download_neurips_2015_2024(fetcher, args.out_dir, year, args.skip_existing)
                    elif year == 2025:
                        n = await download_neurips_2025_main(fetcher, args.out_dir, args.skip_existing)                
                elif venue == "ICLR":
                    if year in (2015, 2016):
                        n = await download_iclr_2015_2016(fetcher, args.out_dir, year, args.skip_existing)
                    else:
                        n = await download_iclr_from_papercopilot(fetcher, args.out_dir, year, args.skip_existing)


                print(f"[{venue} {year}] downloaded={n}")

    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
