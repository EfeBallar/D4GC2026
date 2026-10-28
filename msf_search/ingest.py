"""Load the MSF collection: Markdown text + CSV metadata, clean it, flag quality issues.

Pure standard library. Runs off-premises (at HQ) as part of the preprocessing step,
but is cheap enough to run on a laptop too.
"""
import csv
import hashlib
import html
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .digitise import IMAGE_EXT, inbox_id

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)(]|\([^)]*\))*\)")
_URL = re.compile(r"https?://\S+")
_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Doc:
    id: str
    genre: str = "Unknown"
    language: str = "unknown"
    title: str = ""
    text: str = ""
    md_path: str = ""
    original_path: str = ""
    md5: str = ""
    duplicate_of: str = ""          # id of the canonical copy, if this is an exact duplicate
    also_filed_as: list = field(default_factory=list)
    quality: str = "ok"             # ok | boilerplate | very_short
    year: int = 0                   # best guess from the opening text, 0 = unknown
    transcribed: bool = False       # text comes from OCR / a vision model (scripts/digitise.py)


def clean_markdown(md: str) -> str:
    md = _IMG.sub(" ", md)
    md = _LINK.sub(r"\1", md)
    md = _URL.sub(" ", md)
    md = _TAG.sub(" ", md)
    md = html.unescape(md)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in md.splitlines()]
    out, blank = [], False
    for ln in lines:
        if ln in ("-", "*", "|", "---"):
            continue
        if not ln:
            if not blank:
                out.append("")
            blank = True
            continue
        out.append(ln)
        blank = False
    return "\n".join(out).strip()


_JUNK_TITLE = re.compile(
    r"(doi|issn|vol\.?\s*\d|www\.|http|©|copyright|accepted manuscript|permalink|journal|"
    r"elsevier|lancet|bull world|int j |page \d|^\W*\d|et al\.|licen[cs]e|downloaded|received:|"
    r"msfaccess|skip to|contents|table of)", re.I)


def _good_title(ln: str, min_words: int = 4) -> bool:
    ln = ln.strip()
    if not (8 <= len(ln) <= 200) or _JUNK_TITLE.search(ln):
        return False
    letters = sum(ch.isalpha() for ch in ln)
    return letters / max(1, len(ln)) > 0.7 and len(ln.split()) >= min_words


def guess_title(text: str, doc_id: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:80]:
        m = _HEADING.match(ln)
        if m and _good_title(m.group(2), min_words=2):
            return m.group(2).strip()[:160]
    for ln in lines[:80]:
        ln = ln.strip("-*#> ").strip()
        if _good_title(ln, min_words=5):
            return (ln[:157] + "...") if len(ln) > 160 else ln
    for ln in lines:
        ln = ln.strip("-*#> ").strip()
        if len(ln) > 15:
            return (ln[:157] + "...") if len(ln) > 160 else ln
    return doc_id


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def guess_year(text: str, head_chars: int = 2000) -> int:
    """Most frequent plausible year in the opening text (dates, copyright lines). 0 if none."""
    latest = time.localtime().tm_year
    years = [int(y) for y in _YEAR.findall(text[:head_chars]) if 1950 <= int(y) <= latest]
    return Counter(years).most_common(1)[0][0] if years else 0


def load_collection(data_dir: str, csv_path: str | None = None, boilerplate_min_docs: int = 8,
                    transcriptions_dir: str | None = None, inbox_dir: str | None = None) -> list[Doc]:
    """transcriptions_dir: output of scripts/digitise.py. A transcription replaces the thin parsed
    text of a scanned PDF, and a transcribed inbox photo becomes a new "Field Note" document."""
    data_dir = Path(data_dir)
    trans = Path(transcriptions_dir) if transcriptions_dir else None
    parsed = data_dir / "Parsed"
    original = data_dir / "Original"
    meta = {}
    if csv_path:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                meta[row["id"].strip()] = row

    originals = {}
    if original.exists():
        for p in original.iterdir():
            originals[p.stem] = str(p)

    docs: list[Doc] = []
    for p in sorted(parsed.glob("*.md")):
        t = trans / f"{p.stem}.md" if trans else None
        src = t if t and t.exists() else p
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if src is not p:
            # Keep the thin original extraction too: OCR can miss text a PDF's own text layer has
            # (names over slide photos), so a transcription only ever adds searchable text.
            raw = src.read_text(encoding="utf-8", errors="ignore") + "\n\n## Embedded text layer\n\n" + raw
        m = meta.get(p.stem, {})
        docs.append(Doc(
            id=p.stem,
            genre=(m.get("genre") or "Unknown").strip(),
            language=(m.get("language") or "unknown").strip(),
            text=clean_markdown(raw),
            md_path=str(src),
            original_path=originals.get(p.stem, ""),
            md5=hashlib.md5(raw.encode("utf-8")).hexdigest(),
            transcribed=src is not p,
        ))
    if trans and inbox_dir and Path(inbox_dir).exists():
        for img in sorted(Path(inbox_dir).iterdir()):
            t = trans / f"{inbox_id(img)}.md"
            if img.suffix.lower() not in IMAGE_EXT or not t.exists():
                continue
            raw = t.read_text(encoding="utf-8", errors="ignore")
            docs.append(Doc(id=inbox_id(img), genre="Field Note (handwritten)", language="unknown",
                            text=clean_markdown(raw), md_path=str(t), original_path=str(img),
                            md5=hashlib.md5(raw.encode("utf-8")).hexdigest(), transcribed=True))

    # 1) Corpus-level boilerplate removal: short lines that recur in many different docs
    #    (navigation menus, cookie banners, "Sign up to stay updated"...).
    line_df = Counter()
    for d in docs:
        line_df.update({ln for ln in d.text.splitlines() if 0 < len(ln) < 200})
    common = {ln for ln, c in line_df.items() if c >= boilerplate_min_docs}
    for d in docs:
        d.text = "\n".join(ln for ln in d.text.splitlines() if ln not in common).strip()
        d.title = guess_title(d.text, d.id)
        d.year = guess_year(d.text)

    # 2) Exact duplicates (same file content under two ids) -> keep one canonical copy.
    by_hash: dict[str, list[Doc]] = {}
    for d in docs:
        by_hash.setdefault(d.md5, []).append(d)
    for group in by_hash.values():
        if len(group) < 2:
            continue
        # Prefer D-series ids as canonical; everything else points at it.
        group.sort(key=lambda d: (not d.id.startswith("D"), d.id))
        canon = group[0]
        if len(group) >= 5:
            # Many different ids with identical content = failed scrape of a landing page.
            for d in group:
                d.quality = "boilerplate"
            continue
        for d in group[1:]:
            d.duplicate_of = canon.id
            canon.also_filed_as.append(d.id)

    for d in docs:
        if d.quality == "ok" and len(d.text) < 400:
            d.quality = "very_short"
    return docs


def quality_report(docs: list[Doc]) -> dict:
    return {
        "documents": len(docs),
        "exact_duplicates": sum(1 for d in docs if d.duplicate_of),
        "boilerplate_scrapes": sum(1 for d in docs if d.quality == "boilerplate"),
        "very_short": sum(1 for d in docs if d.quality == "very_short"),
        "languages": dict(Counter(d.language for d in docs).most_common()),
        "genres": dict(Counter(d.genre for d in docs).most_common()),
        "total_chars": sum(len(d.text) for d in docs),
    }
