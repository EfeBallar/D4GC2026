"""HQ "digitise anything": turn scans, slide decks and photos of handwritten notes into searchable text.

Preprocessing only (runs off-premises). The field device never sees an OCR engine or a model: it
gets the transcribed text inside index.db, flagged as transcribed so the UI can say "check the
original". Two engines:
  * vlm:       any OpenAI-compatible vision endpoint (env VLM_BASE_URL, VLM_API_KEY, VLM_MODEL).
               Best for handwriting, tables and doses. Swappable, so no vendor lock-in.
  * tesseract: local open-source OCR, the zero-cost fallback when no API is configured.
Both mark unreadable words as [?], so a doctor never sees a guessed dose presented as fact.

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

    def transcribe(self, image: Path) -> str:
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}  # we parallelise across pages instead
        tsv = subprocess.run(["tesseract", str(image), "-", "-l", self.langs, "tsv"],
                             capture_output=True, text=True, env=env, check=True).stdout
        # Rebuild lines from word boxes; words tesseract is unsure of become [?], like the VLM prompt asks.
        lines, last_par = [], None
        for row in tsv.splitlines()[1:]:
            f = row.split("\t")
            if len(f) < 12 or f[0] != "5" or not f[11].strip():
                continue
            par, line = (f[2], f[3]), (f[2], f[3], f[4])
            if par != last_par and lines:
                lines.append(("", ""))
            if not lines or lines[-1][0] != line:
                lines.append((line, ""))
            word = f[11] if float(f[10]) >= self.min_conf else "[?]"
            lines[-1] = (line, (lines[-1][1] + " " + word).strip())
            last_par = par
        return "\n".join(t for _, t in lines).strip()


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
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
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


def make_engine(engine: str = "auto", langs: str = "eng+fra"):
    env = {k: os.environ.get(k, "") for k in ("VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL")}
    if engine == "vlm" or (engine == "auto" and all(env.values())):
        if not all(env.values()):
            raise SystemExit("engine vlm needs VLM_BASE_URL, VLM_API_KEY and VLM_MODEL")
        return VisionLLM(env["VLM_BASE_URL"], env["VLM_API_KEY"], env["VLM_MODEL"])
    if not shutil.which("tesseract"):
        raise SystemExit("no VLM configured and tesseract is not installed")
    return Tesseract(langs)


# ---------- per document ----------
def _render(pdf: Path, page: int, dpi: int, tmp: Path) -> Path:
    prefix = tmp / f"p{page:04d}"
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                    str(pdf), str(prefix)], check=True, capture_output=True)
    return prefix.with_suffix(".png")


def digitise(item: dict, engine, out_dir: str, dpi: int = 150, workers: int = 4) -> dict | None:
    """Transcribe one PDF or inbox image. Returns its sidecar, or None if it was already done."""
    out = Path(out_dir)
    md_path, meta_path = out / f"{item['id']}.md", out / f"{item['id']}.json"
    if md_path.exists() and meta_path.exists():
        return None
    cache = out / ".pages" / item["id"]  # page-level resume: a crash on page 80 doesn't re-bill 79 pages
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    is_pdf = item["path"].suffix.lower() == ".pdf"

    def page_text(n: int) -> str:
        done = cache / f"{n:04d}.md"
        if done.exists():
            return done.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            img = _render(item["path"], n, dpi, Path(tmp)) if is_pdf else item["path"]
            text = engine.transcribe(img)
        done.write_text(text, encoding="utf-8")
        return text

    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(page_text, range(1, item["pages"] + 1)))
    md = "\n\n".join(f"## Page {n}\n\n{t}" for n, t in enumerate(texts, 1)) if is_pdf else texts[0]
    meta = {"source": engine.source, "model": engine.model, "pages": item["pages"],
            "unreadable_count": len(_UNREADABLE.findall(md)), "original": item["path"].name,
            "seconds": round(time.time() - t0, 1), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    md_path.write_text(md + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.rmtree(cache, ignore_errors=True)
    if not any(cache.parent.iterdir()):
        cache.parent.rmdir()
    return meta
