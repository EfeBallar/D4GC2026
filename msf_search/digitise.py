"""HQ "digitise anything": turn scans, slide decks and photos of handwritten notes into searchable text.

Preprocessing only (runs off-premises). The field device never sees an OCR engine or a model: it
gets the transcribed text inside index.db, with per-page provenance so the UI can say "check the
original". Per page, local tesseract runs first (free). A page escalates to a vision LLM (any
OpenAI-compatible endpoint: env VLM_BASE_URL, VLM_API_KEY, VLM_MODEL) when tesseract is unsure
(mean confidence < 60 or > 20% unreadable words); inbox images (handwriting) always go to the VLM.
Without a VLM configured, tesseract's text is kept. Both mark unreadable words as [?], so a doctor
never sees a guessed dose presented as fact.

Outputs, per document: data/transcriptions/<id>.md (one "## Page N" section per page) and a
<id>.json sidecar. Resumable: finished documents are skipped, finished pages are cached.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = ("Transcribe this page exactly. Keep all numbers, doses and units verbatim. "
          "Mark unreadable words as [?]. Output only the transcription, as Markdown.")
PHOTO_ONLY_BELOW = 0.45   # below: show the photo only, queue for HQ; 0.45-0.60: "low-confidence" badge
IMAGE_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_UNREADABLE = re.compile(r"\[\?\]")
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def inbox_id(path: Path) -> str:
    return "INBOX-" + _SAFE.sub("_", path.stem).strip("_")


def pdf_pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


def find_low_text_pdfs(data_dir: str, threshold: int = 300) -> list[dict]:
    """PDFs whose parsed Markdown holds fewer than `threshold` characters per page:
    scans and image-only slide decks where the provided text extraction failed."""
    data_dir = Path(data_dir)
    found = []
    for pdf in sorted((data_dir / "Original").glob("*.pdf")):
        md = data_dir / "Parsed" / f"{pdf.stem}.md"
        chars = len(md.read_text(encoding="utf-8", errors="ignore")) if md.exists() else 0
        pages = pdf_pages(pdf)
        if pages and chars / pages < threshold:
            found.append({"id": pdf.stem, "path": pdf, "pages": pages, "chars_per_page": round(chars / pages)})
    return found


def find_inbox_images(inbox: str) -> list[dict]:
    p = Path(inbox)
    imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.exists() else []
    return [{"id": inbox_id(f), "path": f, "pages": 1} for f in imgs]


# ---------- engines ----------
class Tesseract:
    source = "tesseract"

    def __init__(self, langs: str = "eng+fra", min_conf: float = 30):
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
        have = {ln.strip() for ln in out.splitlines()[1:]}  # first line is a header
        wanted = langs.split("+")
        self.langs = "+".join(l for l in wanted if l in have) or "eng"
        self.missing = [l for l in wanted if l not in have]
        ver = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
        self.model = f"{ver} ({self.langs})"
        self.min_conf = min_conf
        self.tokens = {"prompt": 0, "completion": 0}

    def transcribe(self, image: Path, *, psm: int | None = None, timeout: int | None = None) -> dict:
        """-> {text, mean_conf (0-100), words, unreadable}"""
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}  # we parallelise across pages instead
        command = ["tesseract", str(image), "-", "-l", self.langs]
        if psm is not None:
            command.extend(["--psm", str(psm)])
        tsv = subprocess.run(command + ["tsv"], capture_output=True, text=True,
                             env=env, check=True, timeout=timeout).stdout
        # Rebuild lines from word boxes; words tesseract is unsure of become [?], like the VLM prompt asks.
        lines, last_par, confs, reliable_chars = [], None, [], 0
        for row in tsv.splitlines()[1:]:
            f = row.split("\t")
            if len(f) < 12 or f[0] != "5" or not f[11].strip():
                continue
            par, line = (f[2], f[3]), (f[2], f[3], f[4])
            if par != last_par and lines:
                lines.append(("", ""))
            if not lines or lines[-1][0] != line:
                lines.append((line, ""))
            confs.append(float(f[10]))
            if confs[-1] >= 60:
                reliable_chars += sum(c.isalpha() for c in f[11])
            word = f[11] if confs[-1] >= self.min_conf else "[?]"
            lines[-1] = (line, (lines[-1][1] + " " + word).strip())
            last_par = par
        return {"text": "\n".join(t for _, t in lines).strip(),
                "mean_conf": round(sum(confs) / len(confs), 1) if confs else 0.0,
                "words": len(confs), "unreadable": sum(1 for c in confs if c < self.min_conf),
                "reliable_chars": reliable_chars}


class VisionLLM:
    source = "vlm"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key, self.model, self.timeout = api_key, model, timeout
        self.tokens = {"prompt": 0, "completion": 0}
        self.missing = []

    def transcribe(self, image: Path) -> str:
        mime = IMAGE_EXT.get(image.suffix.lower(), "image/png")
        data_uri = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}}]}]}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt * 3)
        usage = out.get("usage") or {}
        self.tokens["prompt"] += usage.get("prompt_tokens", 0)
        self.tokens["completion"] += usage.get("completion_tokens", 0)
        return out["choices"][0]["message"]["content"].strip()


def make_engines(engine: str = "auto", langs: str = "eng+fra"):
    """-> (tesseract or None, vlm or None). auto: tesseract first, VLM for unsure pages if configured."""
    # Same local-first default as enrich.py (Ollama's OpenAI-format endpoint); VLM_MODEL must name a
    # vision model (e.g. a pulled llava / qwen2.5vl), otherwise pages stay with tesseract.
    vlm = (VisionLLM(os.environ.get("VLM_BASE_URL") or "http://localhost:11434/v1", os.environ.get("VLM_API_KEY", ""),
                     os.environ["VLM_MODEL"]) if os.environ.get("VLM_MODEL") else None)
    if engine == "vlm" and not vlm:
        raise SystemExit("engine vlm needs VLM_MODEL (and VLM_BASE_URL / VLM_API_KEY for a non-local provider)")
    tess = Tesseract(langs) if engine != "vlm" and shutil.which("tesseract") else None
    if engine == "tesseract":
        vlm = None
    if not tess and not vlm:
        raise SystemExit("no VLM configured and tesseract is not installed")
    return tess, vlm


# ---------- per document ----------
def _render(pdf: Path, page: int, dpi: int, tmp: Path) -> Path:
    prefix = tmp / f"p{page:04d}"
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                    str(pdf), str(prefix)], check=True, capture_output=True)
    return prefix.with_suffix(".png")


def digitise(item: dict, tess, vlm, out_dir: str, dpi: int = 150, workers: int = 4,
             min_conf: float = 60, max_unreadable: float = 0.2) -> dict | None:
    """Transcribe one PDF or inbox image. Returns its sidecar, or None if it was already done."""
    out = Path(out_dir)
    md_path, meta_path = out / f"{item['id']}.md", out / f"{item['id']}.json"
    if md_path.exists() and meta_path.exists() and "page_info" in json.loads(meta_path.read_text()):
        return None
    cache = out / ".pages" / item["id"]  # page-level resume: a crash on page 80 doesn't re-bill 79 pages
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    is_pdf = item["path"].suffix.lower() == ".pdf"

    def page(n: int) -> dict:
        done = cache / f"{n:04d}.json"
        if done.exists():
            return json.loads(done.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(item["path"], n, dpi, Path(tmp)) if is_pdf else item["path"]
            rec = {"page": n, "engine": None, "model": None, "mean_conf": None, "unreadable_ratio": None}
            if tess:
                t = tess.transcribe(img)
                rec.update(engine="tesseract", model=tess.model, mean_conf=t["mean_conf"], text=t["text"],
                           unreadable_ratio=round(t["unreadable"] / max(1, t["words"]), 3))
            unsure = not tess or rec["mean_conf"] < min_conf or rec["unreadable_ratio"] > max_unreadable
            if vlm and (unsure or not is_pdf):  # handwriting / photos always go to the VLM
                rec.update(engine="vlm", model=vlm.model, text=vlm.transcribe(img))
        done.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(page, range(1, item["pages"] + 1)))
    md = ("\n\n".join(f"## Page {r['page']}\n\n{r['text']}" for r in recs) if is_pdf else recs[0]["text"])
    n_vlm = sum(1 for r in recs if r["engine"] == "vlm")
    tess_confs = [r["mean_conf"] for r in recs if r["engine"] == "tesseract"]
    meta = {"source": "vlm" if n_vlm else "tesseract",
            "model": " + ".join(sorted({r["model"] for r in recs})), "pages": item["pages"],
            "unreadable_count": len(_UNREADABLE.findall(md)), "original": item["path"].name,
            "seconds": round(time.time() - t0, 1), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text_source": "ocr_vlm" if n_vlm else "ocr_tesseract",
            # mean tesseract confidence (0-1) over pages kept from tesseract; null when the VLM did them all
            "ocr_confidence": round(sum(tess_confs) / len(tess_confs) / 100, 3) if tess_confs else None,
            "engines": {"tesseract": item["pages"] - n_vlm, "vlm": n_vlm},
            "page_info": [{k: r[k] for k in ("page", "engine", "model", "mean_conf", "unreadable_ratio")}
                          for r in recs]}
    if meta["ocr_confidence"] is not None and meta["ocr_confidence"] < PHOTO_ONLY_BELOW:
        # Too unreliable to show as text: the field gets the photo only, HQ queues it for a VLM / human.
        meta["status"] = "photo_only_queued"
        with (out / "hq_queue.jsonl").open("a", encoding="utf-8") as q:
            q.write(json.dumps({"id": item["id"], "original": item["path"].name, "ocr_confidence": meta["ocr_confidence"],
                                "reason": "tesseract confidence below %.2f" % PHOTO_ONLY_BELOW,
                                "queued_at": meta["created_at"]}) + "\n")
    md_path.write_text(md + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.rmtree(cache, ignore_errors=True)
    if not any(cache.parent.iterdir()):
        cache.parent.rmdir()
    return meta
