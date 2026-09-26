"""\
Classify research paper PDFs into ONE of 10 ML topics with evidence.

This script:
1) Extracts text from each PDF (pypdf; optional fallback: pdfplumber)
2) Builds a compact evidence bundle (page snippets)
3) Uses an lmdeploy pipeline (TurboMind) to assign exactly one topic per PDF
4) Saves parseable JSON outputs

Key fixes vs older drafts:
- Always passes a *single string prompt* into lmdeploy (prevents SentencePiece "not a string" errors).
- Smaller default session length and output length (prevents 60-90 minute runs on a single paper).
- No huge debug printing during JSON parsing.
- Optional per-PDF JSON outputs + an aggregate summary JSON.

Install:
  pip install pypdf lmdeploy requests
  (optional fallback) pip install pdfplumber

Example:
  python classify_pdfs_by_topic.py \
    --pdf_dir /path/to/pdfs \
    --out_json /path/to/summary.json \
    --per_pdf_dir /path/to/per_paper_json \
    --model internlm/internlm3-8b-instruct \
    --max_pages 6 \
    --max_snippets 12

Notes:
- Evidence snippets are taken ONLY from extracted PDF text.
- Output uses topic_id in [1..10] or null if unknown.
- Page indices in evidence are 0-based (first page = 0).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

# -----------------------------
# LLM generation config
# -----------------------------

# Keep generation short; JSON should be small.
GEN_CFG = GenerationConfig(
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    max_new_tokens=1024,
)

# -----------------------------
# Prompt template
# -----------------------------

TOPIC_CLASSIFY_PROMPT = Template(
    """You are classifying research paper PDFs into exactly ONE of the predefined topics below.

You will be given a bundle of papers. For each paper, you MUST choose the single best topic and justify it using evidence snippets from the provided PDF text only.
Do NOT guess. If the paper cannot be confidently assigned based on the provided snippets, set topic_id to null and use this rationale exactly:
unknown (insufficient evidence in provided PDF snippets)

Return ONLY valid JSON. No markdown. No commentary.

Strict rules:
- Use ONLY the provided PDF snippets. Do not use outside knowledge.
- Each paper MUST have exactly one selected topic, or null if unknown.
- Evidence:
  - Every paper entry must include an evidence list.
  - Evidence items must contain:
    - page: integer page index as provided
    - snippet: exact snippet copied from the provided materials (max 25 words)
  - Do NOT include any double quotes in the snippet field.
  - Remove or avoid unparsable characters (math symbols, weird glyphs). Do NOT put them in snippet.
- Keep rationale short (1 to 2 sentences).

Topics (choose exactly one):
1) 3D Vision and Neural Rendering
   Reconstructing, synthesizing, or understanding 3D environments from images, videos, multi-sensor data; neural fields like NeRF; Gaussian splatting.

2) Image and Video Synthesis
   Generative models producing realistic images or videos; diffusion, GANs, transformers.

3) Multimodal Learning (Vision + Language + Reasoning)
   Joint vision and text modeling; VLMs, VQA, captioning with reasoning, aligned multimodal outputs.

4) Large Language Models and Foundation Models
   Analysis, training, scaling, interpretability, reasoning; LLMs and VL-LLMs.

5) Generative AI and Diffusion Models
   New generative architectures, efficient sampling, controllability, cross-modal generation (vision, language, audio).

6) Reinforcement Learning and Decision Making
   RL algorithms, planning, hierarchical RL, robotics and control.

7) Trustworthy, Fair and Robust Machine Learning
   Interpretability, robustness, distribution shift, adversarial, fairness, privacy, causal reasoning.

8) Optimization and Theory of Learning
   Statistical learning theory, optimization methods, generalization bounds, theory insights.

9) Scalability, Systems and Distributed ML
   Hardware-aware training, distributed algorithms, large-scale pipelines, efficient training systems.

10) Application-Driven ML
   ML use cases in science, healthcare, climate, social domains; real-world evaluation, benchmarks.

Output JSON schema (exactly this):
{
  "results": [
    {
      "paper": { "paper_id": "string", "filename": "string" },
      "topic": { "topic_id": 0, "topic_name": "string" },
      "confidence": 0.0,
      "rationale": "string",
      "evidence": [
        { "page": 0, "snippet": "string" }
      ]
    }
  ]
}

Paper bundle:
$paper_bundle
"""
)

TOPICS: Dict[int, str] = {
    1: "3D Vision and Neural Rendering",
    2: "Image and Video Synthesis",
    3: "Multimodal Learning (Vision + Language + Reasoning)",
    4: "Large Language Models and Foundation Models",
    5: "Generative AI and Diffusion Models",
    6: "Reinforcement Learning and Decision Making",
    7: "Trustworthy, Fair and Robust Machine Learning",
    8: "Optimization and Theory of Learning",
    9: "Scalability, Systems and Distributed ML",
    10: "Application-Driven ML",
}

# Keyword bank used ONLY to select informative snippets for the input bundle.
TOPIC_KEYWORDS: Dict[int, List[str]] = {
    1: [
        "nerf",
        "neural radiance",
        "radiance field",
        "neural field",
        "gaussian splatting",
        "3d reconstruction",
        "multi-view",
        "multiview",
        "point cloud",
        "mesh",
        "rendering",
        "slam",
        "depth",
    ],
    2: [
        "image synthesis",
        "video synthesis",
        "text-to-image",
        "text to image",
        "text-to-video",
        "text to video",
        "gan",
        "diffusion",
        "generative",
    ],
    3: [
        "vision-language",
        "vision language",
        "vlm",
        "vqa",
        "caption",
        "multimodal",
        "grounding",
        "instruction",
        "reasoning",
    ],
    4: [
        "large language model",
        "llm",
        "foundation model",
        "pretrain",
        "fine-tune",
        "finetune",
        "scaling law",
        "alignment",
        "rlhf",
        "interpretability",
    ],
    5: [
        "diffusion",
        "denoising",
        "sampling",
        "score-based",
        "score based",
        "guidance",
        "controllable",
        "controlnet",
        "inversion",
        "cross-modal",
        "audio generation",
    ],
    6: [
        "reinforcement learning",
        "policy",
        "actor-critic",
        "actor critic",
        "q-learning",
        "q learning",
        "planning",
        "model-based",
        "model based",
        "control",
        "robot",
    ],
    7: [
        "robust",
        "robustness",
        "distribution shift",
        "out-of-distribution",
        "ood",
        "adversarial",
        "fairness",
        "privacy",
        "interpretability",
        "causal",
    ],
    8: [
        "optimization",
        "generalization",
        "convergence",
        "theorem",
        "proof",
        "convex",
        "non-convex",
        "nonconvex",
        "bound",
        "sample complexity",
    ],
    9: [
        "distributed",
        "scalability",
        "systems",
        "hardware-aware",
        "gpu",
        "pipeline parallel",
        "data parallel",
        "throughput",
        "latency",
        "checkpointing",
    ],
    10: [
        "healthcare",
        "clinical",
        "medical",
        "biology",
        "protein",
        "genomics",
        "climate",
        "weather",
        "remote sensing",
        "materials",
        "social",
    ],
}

# Flatten keywords for faster scoring.
ALL_KEYWORDS: List[str] = sorted({kw for kws in TOPIC_KEYWORDS.values() for kw in kws}, key=len, reverse=True)

# -----------------------------
# PDF extraction
# -----------------------------


def extract_pdf_text_pypdf(pdf_path: str, max_pages: Optional[int] = None) -> List[str]:
    """Return list of page texts (0-indexed list)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    n_pages = len(reader.pages)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    pages: List[str] = []
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
        try:
            import pdfplumber
        except Exception as e:
            raise RuntimeError(
                "Failed to extract with pypdf and pdfplumber is not installed. "
                "Install one of: pip install pypdf  OR  pip install pdfplumber"
            ) from e

        pages: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            if max_pages is not None:
                n_pages = min(n_pages, max_pages)
            for i in range(n_pages):
                txt = pdf.pages[i].extract_text() or ""
                pages.append(txt)
        return pages


# -----------------------------
# Snippet selection
# -----------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean_text_for_bundle(s: str) -> str:
    # Remove control chars, normalize whitespace, and remove quotes.
    s = s.replace("\x00", " ")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    s = s.replace('"', "").replace("“", "").replace("”", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _score_sentence(sent_lower: str) -> int:
    # Count keyword hits; simple and fast.
    score = 0
    for kw in ALL_KEYWORDS:
        if kw in sent_lower:
            score += 1
    return score


@dataclass
class Snip:
    page: int
    text: str
    score: int


def build_snippets_from_pages(
    pages: List[str],
    max_snippets: int = 12,
    max_chars_per_page: int = 8000,
) -> List[Dict[str, Any]]:
    """Select a compact set of informative snippets across the document."""
    candidates: List[Snip] = []

    for page_idx, raw in enumerate(pages):
        txt = _clean_text_for_bundle(raw)
        if not txt:
            continue

        # Trim per-page to avoid pathological long pages.
        txt = txt[:max_chars_per_page]

        sents = _SENT_SPLIT.split(txt)
        for sent in sents:
            sent = sent.strip()
            if len(sent) < 40:
                continue
            sc = _score_sentence(sent.lower())
            if sc <= 0:
                continue
            candidates.append(Snip(page=page_idx, text=sent[:360], score=sc))

    # Fallback if no keyword hits: take a few early sentences from the first pages.
    if not candidates:
        fallback: List[Dict[str, Any]] = []
        for page_idx, raw in enumerate(pages[: min(3, len(pages))]):
            txt = _clean_text_for_bundle(raw)[:max_chars_per_page]
            if not txt:
                continue
            sents = [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]
            for sent in sents[:3]:
                if len(sent) < 25:
                    continue
                fallback.append({"page": page_idx, "text": sent[:360]})
                if len(fallback) >= max_snippets:
                    return fallback
        return fallback

    candidates.sort(key=lambda x: x.score, reverse=True)

    chosen: List[Dict[str, Any]] = []
    seen = set()
    for c in candidates:
        key = (c.page, c.text[:120])
        if key in seen:
            continue
        seen.add(key)
        chosen.append({"page": c.page, "text": c.text})
        if len(chosen) >= max_snippets:
            break

    return chosen[:max_snippets]


def build_paper_bundle(paper_id: str, filename: str, pages: List[str], max_snippets: int) -> str:
    snippets = build_snippets_from_pages(pages, max_snippets=max_snippets)
    bundle_obj = [
        {
            "paper_id": paper_id,
            "filename": filename,
            "snippets": snippets,
        }
    ]
    return json.dumps(bundle_obj, ensure_ascii=False, indent=2)


# -----------------------------
# lmdeploy call + JSON parsing
# -----------------------------


def extract_text(resp: Any) -> str:
    """Extract generated text from lmdeploy outputs."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list) and resp:
        # lmdeploy sometimes returns list of responses
        return extract_text(resp[0])
    if isinstance(resp, dict):
        for k in ("text", "response", "output"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        return ""
    for attr in ("text", "response", "output"):
        v = getattr(resp, attr, None)
        if isinstance(v, str):
            return v
    return str(resp)


def strip_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def extract_first_braced_block(s: str) -> str:
    s = strip_fences(s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return s[start : end + 1]


def parse_first_json_object(text: str) -> Dict[str, Any]:
    raw = extract_first_braced_block(text)
    # Remove illegal control chars that can break json.loads
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    return json.loads(raw)


def ensure_str(x, name="prompt") -> str:
    if x is None:
        raise TypeError(f"{name} is None (expected str)")
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if not isinstance(x, str):
        raise TypeError(f"{name} is {type(x)} (expected str). repr={repr(x)[:200]}")
    return x

def classify_one_pdf(pipe, pdf_path: str, max_pages: Optional[int], max_snippets: int) -> Dict[str, Any]:
    pages = extract_pdf_text(pdf_path, max_pages=max_pages)
    paper_id = Path(pdf_path).stem
    filename = Path(pdf_path).name

    paper_bundle = build_paper_bundle(paper_id=paper_id, filename=filename, pages=pages, max_snippets=max_snippets)
    prompt = TOPIC_CLASSIFY_PROMPT.substitute(paper_bundle=paper_bundle)

    # Critical: ensure prompt is a *plain string*.
    if not isinstance(prompt, str):
        prompt = str(prompt)

    if not isinstance(prompt, str):
        raise RuntimeError("Prompt is not String.")
    
    prompt = ensure_str(prompt, "prompt")

    # Critical: pass a single string to avoid ambiguity with message-list formats.
    try:
        resp = pipe(prompt, gen_config=GEN_CFG)   # IMPORTANT: pass a string, not [prompt]
    except Exception as e:
        if "not a string" in str(e).lower():
            raise TypeError(f"Tokenizer got non-string input. type={type(prompt)} repr={repr(prompt)[:200]}") from e
        raise

    out_text = extract_text(resp).strip()
    if not out_text:
        raise RuntimeError("lmdeploy returned empty output.")

    out = parse_first_json_object(out_text)
    return out


def _coerce_topic_fields(result_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Light post-processing to keep schema consistent."""
    if not isinstance(result_obj, dict) or "results" not in result_obj:
        return result_obj

    results = result_obj.get("results")
    if not isinstance(results, list):
        return result_obj

    for r in results:
        if not isinstance(r, dict):
            continue
        topic = r.get("topic")
        if not isinstance(topic, dict):
            continue
        tid = topic.get("topic_id")
        if isinstance(tid, int) and tid in TOPICS:
            topic["topic_name"] = TOPICS[tid]
        elif tid is None:
            topic["topic_name"] = None
        else:
            topic["topic_id"] = None
            topic["topic_name"] = None

    return result_obj


# -----------------------------
# CLI / orchestration
# -----------------------------


def list_pdfs(pdf_dir: str) -> List[str]:
    p = Path(pdf_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"pdf_dir not found or not a directory: {pdf_dir}")
    return sorted(str(x) for x in p.glob("*.pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    ap.add_argument("--out", required=True, help="Path to aggregate output JSON file")
    ap.add_argument(
        "--model",
        default="internlm/internlm3-8b-instruct",
        help="lmdeploy model name/path",
    )
    ap.add_argument("--max_pages", type=int, default=2, help="Max pages to read per PDF")
    ap.add_argument("--max_snippets", type=int, default=4, help="Max snippets included in the evidence bundle")
    ap.add_argument("--session_len", type=int, default=8192*4, help="TurboMind session length")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error")
    args = ap.parse_args()

    pdf_dir = str(Path(args.pdf_dir).expanduser().resolve())
    out_dir = str(Path(args.out).expanduser().resolve())

    venue = os.path.basename(pdf_dir)

    pdf_files = list_pdfs(pdf_dir)
    if not pdf_files:
        raise RuntimeError(f"No PDFs found in: {pdf_dir}")
    
    os.makedirs(os.path.join(out_dir, venue), exist_ok=True)

    pass_count = 0

    # Initialize once. First call can still be slow due to model load/engine init.
    pipe = pipeline(args.model, backend_config=TurbomindEngineConfig(session_len=args.session_len))

    # t0_all = time.time()
    for idx, pdf_path in enumerate(pdf_files, start=1):

        out_file_name = os.path.splitext(os.path.basename(pdf_path))[0] + '.json'

        if os.path.exists(os.path.join(out_dir, venue, out_file_name)):
            print("already done, continue!")
            continue

        pass_count += 1

        if pass_count < 8:
            print(f"Pass count: {pass_count}, continue!")
            continue

        t0 = time.time()
        try:
            result_obj = classify_one_pdf(pipe, pdf_path, max_pages=args.max_pages, max_snippets=args.max_snippets)
            result_obj = _coerce_topic_fields(result_obj)

            result_obj.setdefault("meta", {})
            result_obj["meta"].update(
                {
                    "pdf_path": pdf_path,
                    "max_pages": args.max_pages,
                    "max_snippets": args.max_snippets,
                    "model": args.model,
                }
            )

            out_json = os.path.join(out_dir, venue, out_file_name)
            # Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved aggregate JSON: {out_json}")

            dt = time.time() - t0
            print(f"[{idx}/{len(pdf_files)}] OK {Path(pdf_path).name} ({dt:.1f}s)")

        except Exception as e:
            dt = time.time() - t0
            # err = {"pdf": pdf_path, "error": str(e)}
            # errors.append(err)
            print(f"[{idx}/{len(pdf_files)}] ERROR {Path(pdf_path).name} ({dt:.1f}s): {e}")
            if args.fail_fast:
                break
            continue


if __name__ == "__main__":
    main()
