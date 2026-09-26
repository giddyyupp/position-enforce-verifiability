import os
import json
from pathlib import Path

TITLES_JSON = '../paper_titles/accepted_papers_ML.json'

def map_icml(year, pdf_dir, out_json):
    paper_titles = json.load(open(TITLES_JSON, 'r'))['ICML']

    mapped = []

    year = year
    conf = 'ICML'

    papers = sorted(os.listdir(pdf_dir))

    titles = paper_titles[year]

    # assert len(titles) == len(papers)

    for ind, title in enumerate(titles):
        mapped.append({"pdf": papers[ind], "title": title})

    out = {
        "conference": conf,
        "year": year,
        "counts": {
            "pdfs": len(papers),
            "mapped": len(mapped),
            "unmatched": 0,
        },
        "mapped": mapped,
        "unmatched": [],
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

    map_icml(args.year, Path(args.pdf_dir), Path(args.out))
