import argparse
import json
import os
from pathlib import Path
from typing import List
import requests
from urllib.parse import quote
import time, random
from tqdm import tqdm
import re

S2_API_KEY=""  # TODO: Your API KEY. 

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

class S2Client:
    def __init__(self, api_key=None, qps=1.0):
        self.api_key = S2_API_KEY #  or os.getenv("S2_API_KEY")
        self.min_interval = 1.0 / float(qps)  # qps=1.0 -> 1.0s
        self._last_call = 0.0

    def _throttle(self):
        now = time.time()
        wait = self.min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def search_title(self, title, year=None, limit=5,
                     fields="title,year,venue,citationCount,externalIds",
                     max_retries=8):
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {"query": title, "limit": limit, "fields": fields}
        if year:
            params["year"] = str(year)

        backoff = 1.0
        for _ in range(max_retries):
            self._throttle()
            r = requests.get(S2_SEARCH, headers=headers, params=params, timeout=30)

            if r.status_code == 200:
                return r.json()

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                sleep_s = float(ra) if ra else (backoff + random.random())
                time.sleep(sleep_s)
                backoff = min(backoff * 2, 60)
                continue

            if 500 <= r.status_code < 600:
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60)
                continue

            r.raise_for_status()

        raise RuntimeError("Too many 429/5xx responses; increase spacing or retries.")



OPENALEX = "https://api.openalex.org/works"

def openalex_citations_for_title(title, year=None):
    # search is fuzzy across title/abstract/fulltext
    url = f"{OPENALEX}?search={quote(title)}&per-page=5"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])

    # pick a reasonable best match
    best = None
    for w in results:
        if year is not None and w.get("publication_year") not in (year, None):
            # keep it simple; you can refine with similarity scoring
            pass
        if best is None:
            best = w

    if not best:
        return None

    return {
        "openalex_id": best.get("id"),
        "matched_title": best.get("display_name"),
        "year": best.get("publication_year"),
        "cited_by_count": best.get("cited_by_count"),
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched_dir", required=False, default='./matched_json', help="Path to directory containing Matched jsons with pdf and title info")
    ap.add_argument("--out", required=False, default='./citation_counts', help="Output JSON path")
    args = ap.parse_args()

    matched_dir = str(Path(args.matched_dir).expanduser().resolve())
    out_path = str(Path(args.out).expanduser().resolve())

    venues = os.listdir(matched_dir)

    client = S2Client(qps=1.0)  # exactly 1 per second

    for venue in venues:
        mapped = json.load(open(os.path.join(matched_dir, venue)))['mapped']

        venue_only = os.path.splitext(venue)[0]

        out_path_venu = os.path.join(out_path, venue_only)
        os.makedirs(out_path_venu, exist_ok=True)

        year = venue_only.split('_')[1]
        conf = venue_only.split('_')[0]

        for mapped_pdf in tqdm(mapped):
            out_file_name = os.path.splitext(mapped_pdf['pdf'])[0] + '.json'

            if os.path.exists(os.path.join(out_path_venu, out_file_name)):
                print("already done, continue!")
                continue

            print(out_file_name)

            # Example:
            title = mapped_pdf['title']
            print(title)

            # citation_count = openalex_citations_for_title(title, year=year)
            citation_count_s2 = client.search_title(title)
            try: 
                print(citation_count_s2['data'][0]['citationCount'])
                Path(os.path.join(out_path_venu, out_file_name)).write_text(json.dumps(citation_count_s2['data'][0], ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"\nSaved: {os.path.join(out_path_venu, out_file_name)}")
            except Exception as e:
                print(citation_count_s2)
                print(e)
                pass


if __name__ == "__main__":
    main()
