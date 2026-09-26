"""
Extract candidate open-source CODE URLs from a PDF, using:
1) Local PDF text extraction + URL regex collection (ground truth candidates)
2) lmdeploy LLM pass to classify which URLs are likely "open-sourced code" links

Requirements:
- pip install pypdf lmdeploy
  (optional fallback: pip install pdfplumber)

Usage:
  python extract_code_urls_from_pdf.py \
      --pdf /path/to/paper.pdf \
      --model internlm/internlm3-8b-instruct \
      --session-len 32768 \
      --max-pages 50
"""

import os
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from string import Template
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

from helpers import *

# -----------------------------
# PDF extraction
# -----------------------------

gen_cfg = GenerationConfig(
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    max_new_tokens=12000,
)


def strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()

def extract_first_braced_block(s: str) -> str:
    """Extract a best-effort first JSON object-like block { ... }."""
    s = strip_fences(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON-like object found.")
    return s[start:end + 1]


@dataclass
class UrlHit:
    url: str
    page: int
    snippet: str


def extract_pdf_text_pypdf(pdf_path: str, max_pages: Optional[int] = None) -> List[str]:
    """Return list of page texts (1-indexed by our convention)."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf not available. Install with: pip install pypdf") from e

    reader = PdfReader(pdf_path)
    n_pages = len(reader.pages)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    pages = []
    for i in range(n_pages):
        page = reader.pages[i]
        txt = page.extract_text() or ""
        pages.append(txt)
    return pages


def extract_pdf_text(pdf_path: str, max_pages: Optional[int] = None) -> List[str]:
    """Try pypdf, fall back to pdfplumber if needed."""
    try:
        return extract_pdf_text_pypdf(pdf_path, max_pages=max_pages)
    except Exception:
        # fallback
        try:
            import pdfplumber
        except Exception as e:
            raise RuntimeError(
                "Failed to extract with pypdf and pdfplumber is not installed. "
                "Install one of: pip install pypdf  OR  pip install pdfplumber"
            ) from e

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            if max_pages is not None:
                n_pages = min(n_pages, max_pages)
            for i in range(n_pages):
                txt = pdf.pages[i].extract_text() or ""
                pages.append(txt)
        return pages


# -----------------------------
# URL finding + normalization
# -----------------------------

URL_REGEX = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s<>()\[\]{}"']+ |
        www\.[^\s<>()\[\]{}"']+
    )
    """
)

# Common “naked” repo domains found in PDFs (no scheme).
NAKED_DOMAIN_REGEX = re.compile(
    r"""(?ix)
    \b(
        github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)? |
        gitlab\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)? |
        bitbucket\.org/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?
    )
    """
)

TRAILING_PUNCT = ".,;:)]}>\"'"


def normalize_url(u: str) -> str:
    u = u.strip()
    # remove trailing punctuation common in PDFs
    u = u.rstrip(TRAILING_PUNCT)
    # fix line-break hyphenation artifacts if any
    u = u.replace("\n", "")
    if u.startswith("www."):
        u = "https://" + u
    if re.match(r"^(github|gitlab|bitbucket)\.com/", u, re.I):
        u = "https://" + u
    return u


def make_snippet(text: str, start: int, end: int, window: int = 90) -> str:
    a = max(0, start - window)
    b = min(len(text), end + window)
    snippet = text[a:b]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet[:240]


def find_urls_in_pages(pages: List[str]) -> List[UrlHit]:
    hits: List[UrlHit] = []
    for idx, txt in enumerate(pages, start=1):
        if not txt:
            continue

        # 1) URLs with scheme or www
        for m in URL_REGEX.finditer(txt):
            raw = m.group(1)
            url = normalize_url(raw)
            snippet = make_snippet(txt, m.start(1), m.end(1))
            hits.append(UrlHit(url=url, page=idx, snippet=snippet))

        # 2) Naked repo domains (github.com/user/repo)
        for m in NAKED_DOMAIN_REGEX.finditer(txt):
            raw = m.group(1)
            url = normalize_url(raw)
            snippet = make_snippet(txt, m.start(1), m.end(1))
            hits.append(UrlHit(url=url, page=idx, snippet=snippet))

    # Deduplicate by (url, page, snippet)
    uniq = {}
    for h in hits:
        key = (h.url, h.page, h.snippet)
        uniq[key] = h
    return list(uniq.values())


def group_hits_by_url(hits: List[UrlHit], max_evidence_per_url: int = 3) -> Dict[str, List[UrlHit]]:
    by_url: Dict[str, List[UrlHit]] = {}
    for h in hits:
        by_url.setdefault(h.url, []).append(h)

    # Keep top N snippets per url (prefer ones that mention "code/github/repo")
    def score_snip(s: str) -> int:
        s2 = s.lower()
        score = 0
        for k in ["code", "github", "gitlab", "repository", "repo", "source", "implementation", "released", "available"]:
            if k in s2:
                score += 1
        return score

    for u, lst in by_url.items():
        lst.sort(key=lambda x: score_snip(x.snippet), reverse=True)
        by_url[u] = lst[:max_evidence_per_url]

    return by_url


# -----------------------------
# lmdeploy: robust call + JSON parsing
# -----------------------------

def extract_text(resp) -> str:
    """Robustly extract generated text from lmdeploy outputs (str/list/object/dict/None)."""
    if resp is None:
        return ""
    if isinstance(resp, list):
        if not resp:
            return ""
        resp = resp[0]
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return (resp.get("text") or resp.get("response") or resp.get("output") or "")
    for attr in ("text", "response", "output"):
        v = getattr(resp, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    s = str(resp)
    return s if s.strip() else ""


import json
import re
from typing import Any

_ILLEGAL_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def fast_json_loads_repair(raw: str, max_fixes: int = 200) -> Any:
    """
    Fast repair: uses JSONDecodeError.pos to patch only the offending character(s),
    instead of scanning the entire string.

    Fixes:
      - Invalid control character (common from PDF/OCR) -> removed or \n escaped
      - Invalid \escape -> doubles the offending backslash
      - Stray ')' where '}' should be -> replaces
      - Extra data -> parses only first JSON value via raw_decode
    """
    s = raw

    # Normalize line endings and remove illegal control chars (FAST regex)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _ILLEGAL_CTRL.sub("", s)

    dec = json.JSONDecoder()

    # Always parse first JSON value (avoids "Extra data")
    def raw_decode_first(text: str):
        return dec.raw_decode(text.lstrip())

    fixes = 0
    while True:
        try:
            obj, _end = raw_decode_first(s)
            return obj
        except json.JSONDecodeError as e:
            fixes += 1
            if fixes > max_fixes:
                raise

            pos = e.pos
            if pos is None:
                raise

            # Guard against weird positions
            if pos < 0 or pos >= len(s):
                raise

            msg = (e.msg or "").lower()
            ch = s[pos]

            # 1) Your earlier typo: ')' used instead of '}'
            if ch == ")":
                s = s[:pos] + "}" + s[pos + 1:]
                continue

            # 2) Invalid backslash escape inside a string: make it literal by doubling it
            # JSON complains "Invalid \\escape"
            if "invalid \\escape" in msg and ch == "\\":
                s = s[:pos] + "\\\\" + s[pos + 1:]
                continue

            # 3) Invalid control character (often a literal newline inside a quoted string)
            if "invalid control character" in msg:
                if ch == "\n":
                    s = s[:pos] + "\\n" + s[pos + 1:]
                elif ch == "\t":
                    s = s[:pos] + "\\t" + s[pos + 1:]
                else:
                    # safest fast move: drop it
                    s = s[:pos] + s[pos + 1:]
                continue

            # 4) If we hit "Extra data" anyway, raw_decode_first should prevent it,
            # but keep a fallback
            if "extra data" in msg:
                obj, _end = raw_decode_first(s)
                return obj

            # If we don't recognize the error, raise it (don’t silently mangle)
            raise


def parse_first_json_object(s: str) -> dict:
    s = s.strip()
    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I)
        s = re.sub(r"\s*```$", "", s.strip())

    # Try direct
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        #print("json parse fails! Running replace commands!")
        pass

    # Fallback: extract first {...} region
    # start = s.find("{")
    # end = s.rfind("}")
    # if start == -1 or end == -1 or end <= start:
    #     raise ValueError("No complete JSON object found (output likely truncated).")

    # s = s.replace(')', '}')
    s = s.replace('\\', '')
    s = s.replace("“", "").replace("”", "")
    s = s.replace('".', '')
    s = s.replace('jtjme" ', 'jtjme')
    s = s.replace('nhmokmfhw5" ', 'nhmokmfhw5')
    s = s.replace('hypersphere",', 'hypersphere,')
    s = s.replace("mimic',", 'mimic",')
    
    # s = s.replace(', and"', ', and"}')

    # s, repaired = safe_json_loads_repair_first(s)
    
    # s = fast_json_loads_repair(s)
    print(s)
    
    try:
        return json.loads(s)
    except:
        raise Exception("Json parse fails!") 


PROMPT_TMPL = Template(
    """You will be given URL candidates extracted from a paper PDF (with page snippets).
Your job: identify which URL is likely links to the paper's open-sourced CODE (e.g., GitHub/GitLab repo, code release page). Remember there should be only single repo for the paper, rest should go to other_urls. Single evidence is enough.
Be strict: include URLs only if the snippet suggests code or implementation, or the URL itself is clearly a repository.
NOTE: do not include any double quotes in the evidence field! This is very important as it breaks the json parsing. This is a must!!
NOTE: only include top-4 relevant other_urls in the final json.
NOTE: try to remove unparsable characters like mathematical etc. DO NOT PUT INTO THE EVIDENCE FIELD!!

Return ONLY valid JSON (no markdown, no commentary) in this exact schema:
{
  "code_urls": [
    {
      "url": "string",
      "confidence": 0.0,
      "reason": "short string",
      "evidence": [
        {"page": 0, "snippet": "string"}
      ]
    }
  ],
  "other_urls": [
    {
      "url": "string",
      "type": "dataset|project_page|paper|supplement|other",
      "reason": "short string",
      "evidence": [
        {"page": 0, "snippet": "string"}
      ]
    }
  ]
}

Candidates:
$candidates
"""
)


def build_candidates_block(by_url: Dict[str, List[UrlHit]]) -> str:
    lines = []
    for url, evs in by_url.items():
        if "doi.org" in url:
            continue
        if "arxiv.org" in url:
            continue
        if "amromosavstore" in url:
            continue
        if ".readthedocs.io/en/latest" in url:
            continue
        #if "www.cs.toronto.edu" in url:
        #    continue
        lines.append(f"- URL: {url}")
        for e in evs:
            lines.append(f"  - page {e.page}: {e.snippet}")
    return "\n".join(lines)


def classify_urls_with_lmdeploy(
    pipe,
    by_url: Dict[str, List[UrlHit]],
    session_len: int = 32768,
    max_new_tokens: int = 800,
) -> dict:

    candidates = build_candidates_block(by_url)
    prompt = PROMPT_TMPL.substitute(candidates=candidates)

    resp = pipe([prompt], gen_config=gen_cfg)
    text = extract_text(resp).strip()
    if not text:
        raise RuntimeError("lmdeploy returned empty output. Try reducing candidates or increasing session_len.")
    return parse_first_json_object(text)


# -----------------------------
# Orchestration
# -----------------------------

def extract_code_urls_from_pdf(
    pdf_path: str,
    model: str,
    max_pages: Optional[int] = None,
    session_len: int = 32768,
) -> dict:
    pages = extract_pdf_text(pdf_path, max_pages=max_pages)
    hits = find_urls_in_pages(pages)
    by_url = group_hits_by_url(hits, max_evidence_per_url=3)

    # If there are no URL hits, return empty quickly.
    if not by_url:
        return {
            "pdf": {"path": pdf_path},
            "code_urls": [],
            "note": "No URLs detected in extracted PDF text."
        }

    # LLM classification pass
    print(by_url)
    if len(by_url) == 0:
        return {
            "pdf": {"path": pdf_path},
            "code_urls": [],
            "note": "No URLs detected in extracted PDF text."
        }

    result = classify_urls_with_lmdeploy(
        pipe=model,
        by_url=by_url,
        session_len=session_len,
    )

    # Fill pdf metadata and also return raw candidates for debugging
    result["pdf"] = {"path": pdf_path}
    result["raw_candidates_count"] = len(by_url)

    print(result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Path to PDF files")
    ap.add_argument("--model", required=False, default="internlm/internlm3-8b-instruct", help="lmdeploy model name/path (e.g., internlm/internlm3-8b-instruct)")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit number of pages to read")
    ap.add_argument("--session-len", type=int, default=32768, help="lmdeploy Turbomind session length")
    ap.add_argument("--out", required=True, help="Required output JSON path")
    args = ap.parse_args()

    pdf_path = str(Path(args.pdf_dir).expanduser().resolve())
    out_path = str(Path(args.out).expanduser().resolve())

    out_dir = os.path.join(out_path, os.path.basename(pdf_path))
    os.makedirs(out_dir, exist_ok=True)

    session_len = 32768 * 4
    pipe = pipeline(args.model, backend_config=TurbomindEngineConfig(session_len=session_len))

    pdf_files = os.listdir(pdf_path)
    skip_count = 0
    for pdf_file in pdf_files:
        out_file_name = os.path.splitext(pdf_file)[0] + '.json'

        if os.path.exists(os.path.join(out_dir,out_file_name)):
            print("already done, continue!")
            continue

        skip_count += 1
        # if skip_count < 2:
        #    continue
        print(out_file_name)
        try:
            result = extract_code_urls_from_pdf(
                pdf_path=os.path.join(pdf_path, pdf_file),
                model=pipe,
                max_pages=args.max_pages,
                session_len=args.session_len,
            )

            out_text = json.dumps(result, indent=2, ensure_ascii=False)
            print(out_text)

            Path(os.path.join(out_dir, out_file_name)).write_text(out_text, encoding="utf-8")
            print(f"\nSaved: {os.path.join(out_dir, out_file_name )}")
        except:
            print("ERROR!!!")
            continue


if __name__ == "__main__":
    main()
