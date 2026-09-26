
import os
import argparse
import re
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from string import Template
from urllib.parse import urlparse, urljoin

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

# -----------------------------
# 1) Local repo evidence extractor
# -----------------------------

KEYWORDS = [
    "train", "trainer", "fit", "finetune",
    "infer", "inference", "predict", "test", "eval", "evaluate",
    "checkpoint", "weights", "pretrained", "resume",
    "nnunet", "nnunetv2", "dataloader", "dataset", "metrics",
    "requirements", "environment.yml", "setup.py", "pyproject.toml",
]

TEXT_FILE_EXTS = {".md", ".txt", ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sh", ".bash"}

def run(cmd: List[str], cwd: str) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, errors="ignore")


def run_anywhere(cmd: List[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> str:
    """Run a command and return stdout as text (non-interactive)."""
    return subprocess.check_output(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        errors="ignore",
        stderr=subprocess.STDOUT,
    )


def infer_github_owner_repo(repo_url: str) -> Optional[Tuple[str, str]]:
    """Infer (owner, repo) from a GitHub URL, else return None."""
    try:
        u = urlparse(repo_url)
        if u.netloc.lower() not in {"github.com", "www.github.com"}:
            return None
        parts = [p for p in u.path.split("/") if p]
        if len(parts) < 2:
            return None
        owner = parts[0]
        repo = parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo
    except Exception:
        return None



# -----------------------------
# 2.5) Resolve GitHub Pages (github.io) project pages to GitHub repos
# -----------------------------

REPO_LINK_HINT_WORDS = {
    "code", "source", "github", "repo", "repository", "implementation", "project", "official"
}


def is_github_pages_url(url: str) -> bool:
    """Return True if url is a GitHub Pages site (github.io)."""
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        return host.endswith("github.io")
    except Exception:
        return False


def canonical_github_repo_url(url: str) -> Optional[str]:
    """Return canonical https://github.com/<owner>/<repo> if possible."""
    owner_repo = infer_github_owner_repo(url)
    if not owner_repo:
        return None
    owner, repo = owner_repo
    return f"https://github.com/{owner}/{repo}"


def _extract_links_bs4(html: str, base_url: str) -> List[Tuple[str, str]]:
    """Return list of (absolute_url, anchor_text). Requires bs4."""
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html, "html.parser")
    out: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        if not href:
            continue
        absu = urljoin(base_url, href)
        txt = (a.get_text() or "").strip()
        out.append((absu, txt))
    return out


def _extract_links_regex(html: str, base_url: str) -> List[Tuple[str, str]]:
    """Fallback link extractor when bs4 is not installed."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
        href = m.group(1).strip()
        if not href:
            continue
        out.append((urljoin(base_url, href), ""))
    return out


def resolve_repo_from_github_pages(project_url: str, timeout: int = 25) -> Optional[str]:
    """
    If project_url is a github.io page, try to find a GitHub repo link on that page.
    Returns canonical repo URL or None.

    Heuristics:
      - prefer links whose anchor text includes words like "code", "source", "repo"
      - prefer links that look like a repository root rather than a specific subpage
    """
    if not is_github_pages_url(project_url):
        return None

    try:
        import requests
    except ImportError:
        # If requests isn't available, we can't fetch the page.
        return None

    try:
        resp = requests.get(project_url, timeout=timeout, headers={"User-Agent": "repo-audit/1.0"})
        resp.raise_for_status()
        html = resp.text
        final_url = resp.url  # handle redirects
    except Exception:
        return None

    # Extract links
    links: List[Tuple[str, str]] = []
    try:
        links = _extract_links_bs4(html, final_url)
    except Exception:
        links = _extract_links_regex(html, final_url)

    candidates: List[Tuple[int, str]] = []

    for href, anchor_text in links:
        can = canonical_github_repo_url(href)
        if not can:
            continue

        score = 0
        text_l = (anchor_text or "").strip().lower()
        url_l = href.lower()

        # Strong signal: explicit "code" button/link
        for w in REPO_LINK_HINT_WORDS:
            if w in text_l:
                score += 5

        # Mild signal: URL path contains common repo root hints
        if any(tok in url_l for tok in ["github.com", "/tree/", "/blob/", "/releases", "/issues"]):
            score += 1

        # Prefer links that are closer to repo root (fewer path segments after owner/repo)
        try:
            u = urlparse(can)
            parts = [p for p in u.path.split("/") if p]
            # parts should be [owner, repo]
            if len(parts) == 2:
                score += 4
        except Exception:
            pass

        candidates.append((score, can))

    if not candidates:
        return None

    # Pick highest score, deterministic tie-breaker by URL
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1]

def derive_repo_dir_name(repo_url: str) -> str:
    """Best-effort folder name derived from URL."""
    try:
        u = urlparse(repo_url)
        parts = [p for p in u.path.split("/") if p]
        name = parts[-1] if parts else "repo"
        if name.endswith(".git"):
            name = name[:-4]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return name or "repo"
    except Exception:
        return "repo"


def clone_repo_to_local(
    repo_url: str,
    clone_root: Optional[str] = None,
    depth: int = 1,
    branch: Optional[str] = None,
    recurse_submodules: bool = False,
    force_reclone: bool = False,
) -> str:
    """Clone repo_url to a local directory and return the local path."""
    if clone_root is None:
        clone_root = tempfile.mkdtemp(prefix="repo_audit_")
        target_dir = os.path.join(clone_root, "repo")
    else:
        os.makedirs(clone_root, exist_ok=True)
        target_dir = os.path.join(clone_root, derive_repo_dir_name(repo_url))

    if os.path.exists(target_dir) and force_reclone:
        shutil.rmtree(target_dir, ignore_errors=True)

    if os.path.exists(target_dir) and (Path(target_dir) / ".git").exists():
        # Reuse existing clone by fetching.
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        run_anywhere(["git", "fetch", "--all", "--prune"], cwd=target_dir, env=env)
        # Reset to origin default or specified branch.
        if branch:
            run_anywhere(["git", "checkout", branch], cwd=target_dir, env=env)
            run_anywhere(["git", "reset", "--hard", f"origin/{branch}"], cwd=target_dir, env=env)
        else:
            # Determine default branch from origin/HEAD.
            head = run_anywhere(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], cwd=target_dir, env=env).strip()
            default_branch = head.split("/")[-1] if head else "main"
            run_anywhere(["git", "checkout", default_branch], cwd=target_dir, env=env)
            run_anywhere(["git", "reset", "--hard", f"origin/{default_branch}"], cwd=target_dir, env=env)
        if recurse_submodules:
            run_anywhere(["git", "submodule", "update", "--init", "--recursive"], cwd=target_dir, env=env)
        return target_dir

    # Fresh clone.
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    cmd = ["git", "clone"]
    if depth and depth > 0:
        cmd += ["--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    if recurse_submodules:
        cmd += ["--recurse-submodules"]
    cmd += [repo_url, target_dir]
    run_anywhere(cmd, cwd=None, env=env)
    return target_dir

def is_text_file(p: Path) -> bool:
    if p.suffix.lower() in TEXT_FILE_EXTS:
        return True
    # also allow files with no extension but common names
    return p.name in {"README", "LICENSE", "Makefile", "Dockerfile"}

def safe_read(path: Path, max_chars: int = 12000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt

def pick_relevant_files(repo_dir: str, file_list: List[str], max_files: int = 20) -> List[str]:
    scored: List[Tuple[int, str]] = []
    for f in file_list:
        lf = f.lower()
        score = sum(1 for k in KEYWORDS if k in lf)
        if score > 0:
            scored.append((score, f))
    scored.sort(reverse=True)
    picked = [f for _, f in scored[:max_files]]
    return picked

def build_audit_bundle(repo_dir: str, max_files_to_embed: int = 16) -> str:
    repo_path = Path(repo_dir).resolve()

    # file list (more reliable than os.walk, respects gitignore if tracked)
    try:
        files = run(["git", "ls-files"], cwd=str(repo_path)).splitlines()
    except Exception:
        # fallback: walk
        files = []
        for p in repo_path.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(repo_path)))

    # README
    readme_candidates = ["README.md", "README.MD", "README", "readme.md"]
    readme_text = ""
    for c in readme_candidates:
        rp = repo_path / c
        if rp.exists():
            readme_text = safe_read(rp, max_chars=20000)
            break

    # file tree (compact)
    tree = "\n".join(files[:4000])
    if len(files) > 4000:
        tree += "\n[TRUNCATED FILE LIST]"

    # select and embed relevant files
    picked = pick_relevant_files(str(repo_path), files, max_files=max_files_to_embed)
    embedded_parts = []
    for f in picked:
        p = repo_path / f
        if not p.exists():
            continue
        if not is_text_file(p):
            continue
        content = safe_read(p, max_chars=12000)
        if content.strip():
            embedded_parts.append(f"\n\n--- FILE: {f} ---\n{content}")

    bundle = f"""\
[REPO_DIR]
{repo_path}

[README]
{readme_text if readme_text.strip() else "[NO README FOUND]"}

[FILE_TREE_GIT_LS_FILES]
{tree}

[SELECTED_FILE_CONTENTS]
{''.join(embedded_parts) if embedded_parts else "[NO RELEVANT FILE CONTENTS EMBEDDED]"}
"""
    return bundle


# -----------------------------
# 2) URL-only issues (optional)
#   Local clone does NOT contain issues; fetch if you want that field grounded.
#   (Works for public repos; unauthenticated GitHub API has low rate limits.)
# -----------------------------

def fetch_open_issues_via_github_api(owner: str, repo: str, max_issues: int = 30) -> str:
    """
    Optional: requires 'requests' and internet access on YOUR machine.
    If you don't want web calls, skip this and keep issues as unknown.
    """
    try:
        import requests
    except ImportError:
        return "[ISSUES] requests not installed; skipping issues fetch."

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "open", "per_page": max_issues}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return f"[ISSUES] Failed to fetch issues: {r.status_code} {r.text[:200]}"

    items = r.json()
    # PRs also appear in this endpoint; filter out entries with 'pull_request'
    issues = [it for it in items if "pull_request" not in it]

    lines = ["[OPEN_ISSUES_EXCERPTS]"]
    for it in issues:
        num = it.get("number")
        title = it.get("title", "")
        body = (it.get("body") or "").strip()
        body = re.sub(r"\s+", " ", body)
        body = body[:500] + ("..." if len(body) > 500 else "")
        lines.append(f"- Issue #{num}: {title}\n  Body: {body}")
    return "\n".join(lines)


# -----------------------------
# 3) The strict JSON-only prompt
# -----------------------------

AUDIT_PROMPT = Template("""\
You are auditing a GitHub repository for training and inference availability.
You MUST base every answer only on the repo evidence bundle below (README, file tree, embedded file contents, and optional issue excerpts).
Do NOT guess. If evidence is missing, set "value" to null and say: "unknown (not found in provided repo materials)".

Return ONLY one valid JSON object matching the schema. No markdown, no commentary.

Evidence rule:
- Every field must include an "evidence" list.
- Each evidence item must contain:
  - "source_type": one of ["README","FILE_TREE","FILE_CONTENT","ISSUE"]
  - "location": a precise pointer (e.g., "README > Training", "path/to/file.py", "Issue #12")
  - "quote": an exact quote (max 25 words) copied from the provided materials

Conservative interpretation:
- training_code_available = true only if repo contains training-related code such as train.py, main.py OR explicit instructions to train using provided code, NOT JUST INSTALLATION INSTRUCTIONS!.
- inference_or_test_code_available = true only if repo contains inference/eval code OR explicit instructions to run inference/testing using provided code, NOT JUST INSTALLATION INSTRUCTIONS!.
- weights_available_for_this_method = true only if checkpoints are provided or linked for THIS method (ignore other methods).
- presents_dataset = true only if the repo presents a dataset (ignore other methods).
- open_issues_about_installation_or_reproducibility = true only if open issues explicitly mention install problems, missing steps, reproducibility, missing data conversion, etc.
- If issues are not provided, set the issues field to null with unknown rationale. Ignore any issues if the author of the issue is NielsRogge.
- acknowledges = true only if the repository acknowledges any previous codebase. set null otherwise.  

Dont forget to fill the full repo url. 

JSON schema:
{
  "repo": {
    "url": "",
    "name": ""
  },
  "assessment": {
    "is_training_based": { "value": null, "rationale": "", "evidence": [] },
    "requires_specific_training": { "value": null, "rationale": "", "evidence": [] },
    "training_code_available": { "value": null, "rationale": "", "evidence": [] },
    "inference_or_test_code_available": { "value": null, "rationale": "", "evidence": [] },
    "training_instructions_available": { "value": null, "rationale": "", "evidence": [] },
    "inference_or_test_instructions_available": { "value": null, "rationale": "", "evidence": [] },
    "weights_available_for_this_method": { "value": null, "rationale": "", "evidence": [] },
    "presents_dataset": { "value": null, "rationale": "", "evidence": [] },
    "open_issues_about_installation_or_reproducibility": { "value": null, "rationale": "", "evidence": [] },
    "acknowledgement": { "value": null, "rationale": "", "evidence": [] }
  },
  "open_issues_lowerbound_notes": { "value": "", "evidence": [] }
}

[REPO_EVIDENCE_BUNDLE]
$bundle
""")


def try_parse_json(text: str):
    text = text.strip()
    if not text:
        return None, "empty"

    # If it starts with ```json, strip fences
    if text.startswith("```"):
        text = text.strip("`")
        # crude but usually enough
        text = text.replace("json", "", 1).strip()

    # Must contain at least one opening brace
    if "{" not in text:
        return None, "no_brace"

    # Try to extract first full JSON object if possible
    start = text.find("{")
    end = text.rfind("}")
    if end <= start:
        return None, "truncated_no_closing_brace"

    candidate = text[start:end+1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        return None, "invalid_or_truncated"
    
# -----------------------------
# 4) lmdeploy call + JSON validation
# -----------------------------

def extract_text(resp) -> str:
    # lmdeploy pipeline can return str, list[str], or an object with .text
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        # list of strings or list of response objects
        parts = []
        for x in resp:
            if isinstance(x, str):
                parts.append(x)
            else:
                parts.append(getattr(x, "text", str(x)))
        return "\n".join(parts)
    return getattr(resp, "text", str(resp))


def call_llm_with_lmdeploy(pipeline, prompt: str):
    """
    Adjust this to your lmdeploy setup. Two common patterns:
      - pipeline(model_name_or_path)
      - pipeline(model_name_or_path, backend_config=...)
    """
    pipe = pipeline
    # pipe = pipeline(
    #     model_name_or_path,
    #     backend_config=TurbomindEngineConfig(session_len=32768)  # try 8192 or higher)
    # )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=8000,
        # stop_words=["}"]  # optional, if you want to stop after JSON closes
    )

    # Many deployments accept generation config in the call; if not, set globally.
    # Keep temperature at 0 to reduce hallucinations.
    resp = pipe(prompt, gen_config=gen_cfg)
    text = extract_text(resp).strip()
    print(text)
    # resp may be a string or an object depending on lmdeploy version
    return text if isinstance(text, str) else getattr(text, "text", str(text))

def ensure_valid_json(text: str) -> dict:
    # Sometimes models add leading/trailing whitespace; that's fine.
    text = text.strip()
    return json.loads(text)

def audit_repo_local(
    repo_dir: str,
    repo_url: str,
    pipeline,
    issues_owner_repo: Optional[Tuple[str, str]] = None,
) -> dict:
    bundle = build_audit_bundle(repo_dir)

    if issues_owner_repo is not None:
        owner, repo = issues_owner_repo
        issues_excerpt = fetch_open_issues_via_github_api(owner, repo, max_issues=25)
        bundle = bundle + "\n\n" + issues_excerpt

    prompt = AUDIT_PROMPT.substitute(bundle=bundle)

    out = call_llm_with_lmdeploy(pipeline, prompt)

    try:
        data, _ = try_parse_json(out)
        print(data)
        # data = ensure_valid_json(out)
        # Fill repo metadata if model left blank (optional)
        data.setdefault("repo", {})
        data["repo"].setdefault("url", repo_url)
        data["repo"].setdefault("name", repo_url.rstrip("/").split("/")[-1])
        return data
    except Exception:
        raise RuntimeError("LLM output was not valid JSON (or could not be parsed).")


def audit_repo_from_url(
    repo_url: str,
    pipeline,
    issues_owner_repo: Optional[Tuple[str, str]] = None,
    clone_root: Optional[str] = None,
    keep_clone: bool = False,
    depth: int = 1,
    branch: Optional[str] = None,
    recurse_submodules: bool = False,
    force_reclone: bool = False,
) -> dict:
    """Clone a repo URL locally, then run the same checker on the cloned path."""
    # If it's a GitHub URL and the caller didn't provide (owner, repo), infer it.
    if issues_owner_repo is None:
        issues_owner_repo = infer_github_owner_repo(repo_url)

    # If clone_root is None, we create a fresh temp root.
    created_temp_root = False
    tmp_root = clone_root
    if tmp_root is None:
        tmp_root = tempfile.mkdtemp(prefix="repo_audit_")
        created_temp_root = True

    repo_dir = clone_repo_to_local(
        repo_url=repo_url,
        clone_root=tmp_root,
        depth=depth,
        branch=branch,
        recurse_submodules=recurse_submodules,
        force_reclone=force_reclone,
    )

    try:
        return audit_repo_local(
            repo_dir=repo_dir,
            repo_url=repo_url,
            pipeline=pipeline,
            issues_owner_repo=issues_owner_repo,
        )
    finally:
        # Clean up the temp clone unless asked to keep it.
        if not keep_clone and created_temp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="Path to read json files. E.g., ./url_checker/CVPR_2020")
    ap.add_argument("--repo_dir", required=True, help="where to clone the repos. E.g., ./repos_to_clone")
    ap.add_argument("--out", required=True, help="Where to save output JSONs. E.g., ./repos_checked/")
    ap.add_argument("--model", required=False, default="internlm/internlm3-8b-instruct", help="lmdeploy model name/path (e.g., internlm/internlm3-8b-instruct)")
    args = ap.parse_args()

    print(args)

    json_path = str(Path(args.json_dir).expanduser().resolve())
    repo_clone_path = str(Path(args.repo_dir).expanduser().resolve())
    output_path = str(Path(args.out).expanduser().resolve())

    out_dir = os.path.join(output_path, os.path.basename(json_path))
    repo_clone_path = os.path.join(repo_clone_path, os.path.basename(json_path))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(repo_clone_path, exist_ok=True)

    # repo_url = "https://github.com/Yanfeng-Zhou/nnWNet"

    # Choose one:
    # model = "meta-llama/Llama-3.1-8B"
    # model = "internlm/internlm3-8b-instruct"

    session_len = 32768 * 16
    pipe = pipeline(args.model, backend_config=TurbomindEngineConfig(session_len=session_len))

    # Optional: pass ("owner", "repo") explicitly, else it will be inferred for GitHub URLs.
    issues_owner_repo = None

    json_files = os.listdir(json_path)
    json_files = sorted(json_files)

    done_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json"))

    latest_idx = -1
    if done_files:
        last_file = done_files[-1]
        m = re.match(r"(\d+)_", last_file)
        if m:
            latest_idx = int(m.group(1))

    print(f"Latest completed index: {latest_idx}")

    for json_file in json_files[latest_idx:]:
        out_file_name = json_file  # os.path.splitext(json_file)[0] + '.json'
        
        if os.path.exists(os.path.join(out_dir, out_file_name)):
            print("already done, continue!")
            continue

        data = json.load(open(os.path.join(json_path, json_file), 'r'))

        if len(data['url_data']) and len(data['url_data']['code_urls']):
            repo_url = data['url_data']['code_urls'][0]['url']

            # If the URL points to a GitHub Pages (github.io) project page, try to resolve the underlying code repo.
            if is_github_pages_url(repo_url):
                resolved = resolve_repo_from_github_pages(repo_url)
                if resolved:
                    print(f"Resolved GitHub Pages URL to repo: {repo_url} -> {resolved}")
                    repo_url = resolved
                else:
                    print(f"GitHub Pages URL provided but no repo link found on page: {repo_url}")

            try:
                # Clones to a temp directory by default. Set keep_clone=True to inspect the clone afterwards.
                result = audit_repo_from_url(
                    repo_url=repo_url,
                    pipeline=pipe,
                    issues_owner_repo=issues_owner_repo,
                    clone_root=repo_clone_path,
                    keep_clone=False,
                    depth=1,
                    recurse_submodules=False,
                )
            except:
                print(f"Repo {repo_url} cant be cloned. Most likely not a code repo.")
                continue

            out_text = json.dumps(result, indent=2, ensure_ascii=False)
            print(out_text)

            Path(os.path.join(out_dir, out_file_name)).write_text(out_text, encoding="utf-8")
            print(f"\nSaved: {os.path.join(out_dir, out_file_name )}")
        else:
            print(f"There is no code_url in the json. Skipping.")

