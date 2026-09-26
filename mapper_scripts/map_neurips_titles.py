import re, json, hashlib, asyncio
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

def openreview_pdf_from_any_url(u: str) -> str:
    if "openreview.net/forum?id=" in u:
        return u.replace("/forum?", "/pdf?")
    if "openreview.net/pdf?id=" in u:
        return u
    m = re.search(r"[?&]id=([^&]+)", u)
    if m and "openreview.net" in u:
        return f"https://openreview.net/pdf?id={m.group(1)}"
    return ""

def parse_poster_title_and_openreview_pdf(poster_html: str, poster_url: str):
    soup = BeautifulSoup(poster_html, "html.parser")  # no lxml dependency

    # 1) Prefer h1 (on NeurIPS 2025 poster pages, this is the paper title)
    title = ""
    for h1 in soup.find_all("h1"):
        t = h1.get_text(" ", strip=True)
        if t and t.lower() not in ("main navigation",):
            title = t
            break

    # 2) Fallback: choose best heading if h1 not found
    if not title:
        bad = {"main navigation", "abstract", "video", "downloads"}
        heads = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            t = h.get_text(" ", strip=True)
            if not t:
                continue
            if t.lower() in bad:
                continue
            heads.append(t)
        if heads:
            title = max(heads, key=len)

    # clean common suffixes
    title = title.replace("· NeurIPS 2025", "").strip()

    # Extract OpenReview link
    pdf_url = ""
    for a in soup.find_all("a", href=True):
        href = urljoin(poster_url, a["href"])
        if "openreview.net" in href:
            pdf = openreview_pdf_from_any_url(href)
            if pdf:
                pdf_url = pdf
                break

    return title, pdf_url


def sha1_10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

# ---------- NeurIPS 2021–2024 (proceedings.neurips.cc / papers.nips.cc) ----------

ABS_RE = re.compile(r"-Abstract(?:-[A-Za-z_]+)?\.html$", re.IGNORECASE)

def neurips_book_candidates(year: int):
    return [
        f"https://proceedings.neurips.cc/paper_files/paper/{year}",
        f"https://proceedings.neurips.cc/paper/{year}",
        f"https://papers.nips.cc/paper/{year}",
    ]

def parse_book_abs_pages(book_html: str, book_url: str, year: int, exclude_db=True):
    soup = BeautifulSoup(book_html, "lxml")
    out = []
    for li in soup.find_all("li"):
        a = li.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        if not ABS_RE.search(href):
            continue
        if exclude_db and "Datasets and Benchmarks" in li.get_text(" ", strip=True):
            continue
        out.append(urljoin(book_url, href))
    return list(dict.fromkeys(out))

def parse_abs_title_and_pdf(abs_html: str, abs_url: str):
    soup = BeautifulSoup(abs_html, "lxml")

    # title: on these pages it’s the H1 heading (# Title) :contentReference[oaicite:2]{index=2}
    h1 = soup.find(["h1", "h2"])
    title = h1.get_text(" ", strip=True) if h1 else ""

    # pdf link: anchor with visible text "Paper" :contentReference[oaicite:3]{index=3}
    pdf_url = ""
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() == "paper":
            pdf_url = urljoin(abs_url, a["href"])
            break
    return title, pdf_url

# ---------- NeurIPS 2025 (neurips.cc virtual poster pages -> OpenReview PDF) ----------

def neurips_2025_papers_url():
    return "https://neurips.cc/virtual/2025/papers.html"

def extract_poster_id(u: str):
    m = re.search(r"/poster/(\d+)\b", u)
    return m.group(1) if m else None

def parse_virtual_poster_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/poster/" in href:
            urls.append(urljoin(base_url, href))
    return list(dict.fromkeys(urls))

def parse_virtual_page_poster_ids(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    ids = set()
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"].strip())
        pid = extract_poster_id(u)
        if pid:
            ids.add(pid)
    return ids

def neurips_2025_exclusion_pages():
    # same exclusions as your downloader
    return [
        "https://neurips.cc/virtual/2025/loc/san-diego/events/datasets-benchmarks-2025",
        "https://neurips.cc/virtual/2025/loc/mexico-city/events/datasets-benchmarks-2025",
        "https://neurips.cc/virtual/2025/loc/san-diego/events/2025-journal-track-papers",
        "https://neurips.cc/virtual/2025/loc/mexico-city/events/2025-journal-track-papers",
    ]

# def openreview_pdf_from_any_url(u: str) -> str:
#     # forum?id=XXXX -> pdf?id=XXXX
#     if "openreview.net/forum?id=" in u:
#         return u.replace("/forum?", "/pdf?")
#     if "openreview.net/pdf?id=" in u:
#         return u
#     m = re.search(r"[?&]id=([^&]+)", u)
#     if m and "openreview.net" in u:
#         return f"https://openreview.net/pdf?id={m.group(1)}"
#     return ""

# def parse_poster_title_and_openreview_pdf(poster_html: str, poster_url: str):
#     soup = BeautifulSoup(poster_html, "lxml")

#     # title usually in h1/h2; fallback to <title>
#     h = soup.find(["h1", "h2"])
#     title = h.get_text(" ", strip=True) if h else (soup.title.get_text(" ", strip=True) if soup.title else "")
#     title = title.replace("· NeurIPS 2025", "").strip()

#     pdf_url = ""
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if "openreview.net" in href:
#             pdf = openreview_pdf_from_any_url(href)
#             if pdf:
#                 pdf_url = pdf
#                 break
#     return title, pdf_url

# ---------- Main mapping ----------

async def fetch_text(client: httpx.AsyncClient, url: str):
    r = await client.get(url)
    r.raise_for_status()
    return r.text

async def build_hash_map_neurips(year: int, concurrency: int = 25):
    headers = {"User-Agent": "neurips-mapper/1.0 (research use)"}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=headers, timeout=60, follow_redirects=True, limits=limits) as client:

        h2rec = {}

        if year <= 2024:
            book_html = None
            book_url = None
            for cand in neurips_book_candidates(year):
                try:
                    book_html = await fetch_text(client, cand)
                    book_url = cand
                    break
                except Exception:
                    continue
            if not book_html:
                raise RuntimeError(f"Could not load NeurIPS {year} proceedings index")

            abs_pages = parse_book_abs_pages(book_html, book_url, year, exclude_db=True)
            # fetch abstract pages concurrently
            sem = asyncio.Semaphore(concurrency)

            async def fetch_one(u):
                async with sem:
                    try:
                        html = await fetch_text(client, u)
                        t, pdf = parse_abs_title_and_pdf(html, u)
                        if t and pdf:
                            h2rec[sha1_10(pdf)] = {"title": t, "pdf_url": pdf, "source": "proceedings"}
                    except Exception:
                        return

            await asyncio.gather(*[fetch_one(u) for u in abs_pages])
            return h2rec

        if year == 2025:
            main_html = await fetch_text(client, neurips_2025_papers_url())
            poster_urls = parse_virtual_poster_links(main_html, neurips_2025_papers_url())

            # exclusions: D&B + journal track
            exclude_ids = set()
            for ex_url in neurips_2025_exclusion_pages():
                try:
                    ex_html = await fetch_text(client, ex_url)
                    exclude_ids |= parse_virtual_page_poster_ids(ex_html, ex_url)
                except Exception:
                    continue

            filtered = []
            for u in poster_urls:
                pid = extract_poster_id(u)
                if pid and pid in exclude_ids:
                    continue
                filtered.append(u)
            filtered = list(dict.fromkeys(filtered))

            sem = asyncio.Semaphore(concurrency)

            async def fetch_one(u):
                async with sem:
                    try:
                        html = await fetch_text(client, u)
                        t, pdf = parse_poster_title_and_openreview_pdf(html, u)
                        if t and pdf:
                            h2rec[sha1_10(pdf)] = {"title": t, "pdf_url": pdf, "source": "neurips_virtual"}
                    except Exception:
                        return

            await asyncio.gather(*[fetch_one(u) for u in filtered])
            return h2rec

        raise ValueError("This script supports NeurIPS years 2021–2025")

def map_neurips_pdfs(pdf_dir: str, year: int, out_json: str):
    pdf_dir = Path(pdf_dir)
    pdf_re = re.compile(r"^\d+_([0-9a-f]{10})\.pdf$", re.IGNORECASE)

    h2rec = asyncio.run(build_hash_map_neurips(year))

    mapped, unmatched = [], []
    for p in sorted(pdf_dir.glob("*.pdf")):
        m = pdf_re.match(p.name)
        if not m:
            continue
        h = m.group(1).lower()
        rec = h2rec.get(h)
        if rec:
            mapped.append({"pdf": p.name, "hash": h, **rec})
        else:
            unmatched.append(p.name)

    out = {
        "conference": "NeurIPS",
        "year": year,
        "counts": {
            "pdfs": len(list(pdf_dir.glob("*.pdf"))),
            "hash_map": len(h2rec),
            "mapped": len(mapped),
            "unmatched": len(unmatched),
        },
        "mapped": mapped,
        "unmatched": unmatched,
    }
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print("Saved:", out_json, "Counts:", out["counts"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    map_neurips_pdfs(args.pdf_dir, args.year, args.out)
