import os, re, json, hashlib, requests
from pathlib import Path


PAPERCOPILOT_RAW = "https://raw.githubusercontent.com/papercopilot/paperlists/main"

def slugify(title: str) -> str:
    # mimic common "title -> filename" rules
    s = title.lower().strip()
    s = re.sub(r"['’]", "", s)                 # remove apostrophes
    s = re.sub(r"[^a-z0-9]+", "_", s)          # non-alnum -> underscore
    s = re.sub(r"_+", "_", s).strip("_")       # collapse underscores
    return s

def filename_to_slug(fname: str) -> str:
    # 0009_some_slug.pdf -> some_slug
    stem = Path(fname).stem
    stem = re.sub(r"^\d+_", "", stem)          # drop leading index_
    return stem

def map_slugs(pdf_dir: str, titles_json: str, out_json: str,
              conf="ICLR", year="2023"):
    pdf_dir = Path(pdf_dir)
    data = json.loads(Path(titles_json).read_text("utf-8"))

    titles = data[conf][str(year)]
    slug_to_titles = {}
    for t in titles:
        slug_to_titles.setdefault(slugify(t), []).append(t)

    mapped = []
    unmatched_pdfs = []
    collisions = {k:v for k,v in slug_to_titles.items() if len(v) > 1}

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        fslug = filename_to_slug(pdf.name)
        hits = slug_to_titles.get(fslug)
        if not hits:
            unmatched_pdfs.append(pdf.name)
        else:
            mapped.append({
                "pdf": pdf.name,
                "slug": fslug,
                "title": hits[0],
                "all_title_candidates": hits if len(hits) > 1 else None
            })

    out = {
        "conference": conf,
        "year": str(year),
        "counts": {
            "titles_in_list": len(titles),
            "pdfs_found": len(list(pdf_dir.glob("*.pdf"))),
            "mapped": len(mapped),
            "unmatched_pdfs": len(unmatched_pdfs),
            "slug_collisions": len(collisions),
        },
        "mapped": mapped,
        "unmatched_pdfs": unmatched_pdfs,
        "slug_collisions": collisions,
    }

    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print("Saved:", out_json)
    print("Counts:", out["counts"])


def sha1_10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def is_accepted(status: str) -> bool:
    s = (status or "").strip().lower()
    if not s:
        return False
    if any(x in s for x in ["reject", "withdraw", "desk", "not accept", "not accepted"]):
        return False
    return any(x in s for x in ["accept", "oral", "poster", "spotlight"])

def get_forum_id(rec: dict) -> str | None:
    # PaperCopilot entries vary; try common fields
    # Often: rec["id"] is list like ["xxxx"]; or rec["openreview_id"] / rec["forum"]
    if isinstance(rec.get("id"), list) and rec["id"]:
        return rec["id"][0]
    for k in ["openreview_id", "forum", "forum_id", "paper_id", "id"]:
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    # Try openreview url field
    for k in ["url", "openreview_url", "forum_url"]:
        u = rec.get(k)
        if isinstance(u, str) and "openreview.net" in u and "id=" in u:
            return u.split("id=", 1)[1].split("&", 1)[0]
    return None

def get_title(rec: dict) -> str | None:
    for k in ["title", "paper_title", "name"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # sometimes nested in content
    c = rec.get("content")
    if isinstance(c, dict):
        v = c.get("title")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def map_iclr(year: int, pdf_dir: Path, out_json: Path):
    url = f"{PAPERCOPILOT_RAW}/iclr/iclr{year}.json"
    data = requests.get(url, timeout=60).json()
    if isinstance(data, dict):
        data = data.get("papers") or data.get("data") or data
    assert isinstance(data, list), "Unexpected PaperCopilot JSON format"

    # Build hash -> title mapping
    h2title = {}
    collisions = {}
    for rec in data:
        content = rec.get("content") if isinstance(rec.get("content"), dict) else rec
        status = (content.get("status") or content.get("decision") or content.get("final_decision") or "")
        if not is_accepted(status):
            continue

        fid = get_forum_id(rec)
        title = get_title(rec)
        if not fid or not title:
            continue

        pdf_url = f"https://openreview.net/pdf?id={fid}"
        h = sha1_10(pdf_url)

        if h in h2title and h2title[h] != title:
            collisions.setdefault(h, set()).update([h2title[h], title])
        else:
            h2title[h] = title

    # Match to local PDFs
    pdf_re = re.compile(r"^\d+_([0-9a-f]{10})\.pdf$", re.IGNORECASE)
    mapped, unmatched = [], []

    for p in sorted(pdf_dir.glob("*.pdf")):
        m = pdf_re.match(p.name)
        if not m:
            continue
        h = m.group(1).lower()
        title = h2title.get(h)
        if title:
            mapped.append({"pdf": p.name, "hash": h, "title": title})
        else:
            unmatched.append(p.name)

    out = {
        "conference": "ICLR",
        "year": year,
        "mapped": mapped,
        "unmatched": unmatched,
        "collisions": {k: sorted(v) for k, v in collisions.items()},
        "counts": {"mapped": len(mapped), "unmatched": len(unmatched), "hash_map_size": len(h2title)},
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print("Saved:", out_json)
    print("Mapped:", len(mapped), "Unmatched:", len(unmatched))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--pdf_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.year == 2023:
        # Year 2023 is different somehwo
        map_slugs(args.pdf_dir,
                  "../paper_titles/accepted_papers_ML.json",
                  args.out,
                  conf="ICLR", year=f"{args.year}") 
    else:
        map_iclr(args.year, Path(args.pdf_dir), Path(args.out))
