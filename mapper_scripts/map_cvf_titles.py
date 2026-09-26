import argparse, hashlib, json, re, time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://openaccess.thecvf.com/"

def sha1_10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def fetch(url: str, timeout=60, retries=5, sleep_s=0.2, ua="cvf-fast-mapper/1.0"):
    headers = {"User-Agent": ua}
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            time.sleep(sleep_s)
            return r.text
        except Exception as e:
            last = e
            time.sleep(min(5, 1.6 ** i))
    raise RuntimeError(f"Failed to fetch {url}: {last}")

def build_hash_map_from_dayall(conf: str, year: int):
    url = f"{BASE}{conf}{year}?day=all"
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    # We walk the page in order and pair each [pdf] with the most recent title anchor.
    # Title anchors look like: /content/CVPR2021/html/..._paper.html
    title_href_part = f"/content/{conf}{year}/html/"
    h2rec = {}

    current_title = None

    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True)
        href = a["href"].strip()

        if title_href_part in href and href.endswith("_paper.html") and txt:
            current_title = txt
            continue

        if txt.lower() == "pdf" and current_title:
            pdf_url = urljoin(url, href)
            h = sha1_10(pdf_url)
            h2rec[h] = {
                "title": current_title,
                "pdf_url": pdf_url,
                "source": "cvf_day_all",
            }
            # keep current_title for the rest of that entry, then it will be overwritten by next title
            continue

    return h2rec

def map_local_pdfs(pdf_dir: Path, h2rec: dict, conf: str, year: int, out_json: Path):
    pdf_re = re.compile(r"^\d+_([0-9a-f]{10})\.pdf$", re.IGNORECASE)
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
        "conference": conf,
        "year": year,
        "counts": {
            "pdfs": len(list(pdf_dir.glob("*.pdf"))),
            "hash_map_size": len(h2rec),
            "mapped": len(mapped),
            "unmatched": len(unmatched),
        },
        "mapped": mapped,
        "unmatched": unmatched,
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print("Saved:", out_json)
    print("Counts:", out["counts"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", choices=["CVPR", "ICCV"], required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    h2rec = build_hash_map_from_dayall(args.conf, args.year)
    map_local_pdfs(Path(args.pdf_dir), h2rec, args.conf, args.year, Path(args.out))
