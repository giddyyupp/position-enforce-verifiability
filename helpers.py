import json
import re
from typing import Any, Tuple

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))

def _remove_control_chars(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL_CHARS.sub("", s)

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _extract_first_json_blob(s: str) -> str:
    # Extract the first top-level {...} or [...] blob if there’s extra junk around it.
    m = re.search(r"[\{\[]", s)
    if not m:
        return s
    start = m.start()
    opening = s[start]
    closing = "}" if opening == "{" else "]"

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]  # fallback

def safe_json_loads_repair_first(raw: str, max_passes: int = 2) -> Tuple[Any, str]:
    """
    Returns (obj, repaired_json_string_used_for_parsing).

    Key behavior:
      - Repairs common typos (notably ')' where '}' should be)
      - Then parses ONLY the first JSON value (ignores trailing junk that would cause 'Extra data')
    """
    s = _strip_code_fences(raw)
    s = _normalize_quotes(s)
    s = _remove_control_chars(s)
    s = _extract_first_json_blob(s)
    s = _remove_trailing_commas(s)

    decoder = json.JSONDecoder()

    for _ in range(max_passes):
        try:
            obj, end = decoder.raw_decode(s)  # parse first value only
            used = s[:end]
            return obj, used
        except json.JSONDecodeError as e:
            pos = e.pos if e.pos is not None else 0

            # Fix your exact common typo: ')' instead of '}'
            if 0 <= pos < len(s) and s[pos] == ")":
                s = s[:pos] + "}" + s[pos + 1:]
                continue

            # Also fix cases where ')' is right before a comma/]/} but parser points slightly later
            win_start = max(0, pos - 120)
            win = s[win_start: min(len(s), pos + 1)]
            j = win.rfind(")")
            if j != -1:
                abs_j = win_start + j
                # Replace that ')' with '}' if it looks like an object is ending
                k = abs_j + 1
                while k < len(s) and s[k].isspace():
                    k += 1
                if k == len(s) or s[k] in ",]}":
                    s = s[:abs_j] + "}" + s[abs_j + 1:]
                    continue

            # Trailing commas are another frequent source
            s2 = _remove_trailing_commas(s)
            if s2 != s:
                s = s2
                continue

            # If we still can’t parse, break and raise
            raise

    # fallback (should usually never reach)
    obj, end = decoder.raw_decode(s)
    print(obj)
    return obj, s[:end]



import os
from collections import defaultdict

def print_subfolder_file_counts(root_dir: str, recursive: bool = True) -> None:
    """
    Prints: <relative_folder_path>\t<count_of_files_in_that_folder>
    If recursive=True, includes all subfolders (and the root itself).
    Counts files only (not directories).
    """
    root_dir = os.path.abspath(root_dir)
    counts = defaultdict(int)

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            rel = os.path.relpath(dirpath, root_dir)
            counts[rel] += sum(1 for f in filenames if os.path.isfile(os.path.join(dirpath, f)))
    else:
        # only immediate children folders (plus root)
        counts["."] = sum(
            1 for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        )
        for name in os.listdir(root_dir):
            p = os.path.join(root_dir, name)
            if os.path.isdir(p):
                counts[name] = sum(
                    1 for f in os.listdir(p)
                    if os.path.isfile(os.path.join(p, f))
                )

    for folder in sorted(counts.keys()):
        print(f"{folder}\t{counts[folder]}")


# if __name__ == '__main__':
#     print_subfolder_file_counts('./all_pdfs')
    