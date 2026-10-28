"""Point the phone's camera at a label, get the documents that cover it.

The phone stays a camera and a browser: it takes a photo and posts it. The site server does the
reading, reusing digitise.py exactly as HQ uses it for inbox photos — tesseract first, escalating
to a vision model when VLM_MODEL is configured, words tesseract is unsure of marked [?], and the
same confidence tiers (below 0.45 the text is not trustworthy, so we show the photo and say so).

Why the OCR text is not simply used as the query: a medicine box is mostly noise — batch numbers,
"Store below 25°C", a manufacturer's address. Searching all of it buries the one word that
matters. Instead the text is matched against the glossary and abbreviation list the search already
uses, which turns "PARACETAMOL 500mg" into the concept the library indexes, in every language.

No camera stream is involved: the page uses a capture input, which hands off to the phone's own
camera app. getUserMedia needs a secure origin, and a field server on a local hotspot is plain
HTTP, so a live viewfinder would be blocked on every phone.
"""
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..textnorm import STOPWORDS, tokenize

MAX_BYTES = 10 * 1024 * 1024
SUFFIX = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
          "image/heic": ".heic", "image/heif": ".heif", "image/webp": ".webp"}
PHOTO_ONLY_BELOW = 45.0     # mean confidence, 0-100; below this the text is not worth searching
LOW_CONFIDENCE_BELOW = 60.0
MAX_TERMS = 6
WORD_MAX_DF = 40        # a word in more of the library than this cannot identify a label

# Packaging vocabulary. These are rare in a medical corpus, so rarity alone keeps them, and every
# one of them that reaches the query costs coverage: the ranking asks how much of the query a
# document accounts for, and no guideline accounts for "batch" or "mumbai". That is what made a
# paracetamol box and an artesunate vial return nothing. Clinical words stay out of this list:
# injection, solution, oral and suspension all mean something to the library.
LABEL_NOISE = set("""
batch lot expiry expires manufactured manufacturer manufacture pharma pharmaceutical pharmaceuticals
tablet tablets capsule capsules sachet sachets bottle bottles blister ampoule vials
store stored storage below above keep dry place protect light moisture refrigerate
contents content weight volume approx each contains containing dissolve litre liter
ready formula product licence license registration reference serial
limited private company works plant india china germany france mumbai delhi
ip bp usp ph eur gmp iso
""".split())


def _engines(langs: str):
    """(tesseract or None, vlm or None, note). Never raises: a field box may have neither."""
    try:
        from ..digitise import make_engines
        tess, vlm = make_engines("auto", langs)
        return tess, vlm, ""
    except SystemExit as e:
        # make_engines() calls sys.exit() when nothing is installed. SystemExit derives from
        # BaseException, so a bare "except Exception" lets it through and kills the request
        # mid-flight, leaving the phone with a closed connection and no error to show.
        return None, None, str(e) or "no OCR engine on this server"
    except FileNotFoundError:
        return None, None, "tesseract is not installed on this server"
    except Exception as e:                                  # a missing language pack, a bad VLM env
        return None, None, f"no OCR engine available ({type(e).__name__}: {e})"


def _prepare_images(image: Path, folder: Path) -> list[Path]:
    """Decode phone formats, honour EXIF orientation, and improve small/low-contrast print."""
    try:
        from PIL import Image, ImageOps, ImageFilter
    except ImportError:
        return [image]
    try:
        opened = Image.open(image)
    except OSError:
        # macOS's native decoder supports iPhone HEIC even without pillow-heif.
        if not shutil.which("sips"):
            raise ValueError("This photo format could not be opened. Choose a JPEG or PNG photo.")
        converted = folder / "decoded.png"
        subprocess.run(["sips", "-s", "format", "png", str(image), "--out", str(converted)],
                       capture_output=True, check=True, timeout=15)
        opened = Image.open(converted)
    with opened:
        photo = ImageOps.exif_transpose(opened)
        photo.thumbnail((2600, 2600))
        if photo.mode in ("RGBA", "LA") or "transparency" in photo.info:
            rgba = photo.convert("RGBA")
            photo = Image.new("RGB", rgba.size, "white")
            photo.paste(rgba, mask=rgba.getchannel("A"))
        else:
            photo = photo.convert("RGB")
        if max(photo.size) < 1400:
            scale = 1400 / max(photo.size)
            photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.Resampling.LANCZOS)
        original = folder / "upright.png"
        photo.save(original)
        enhanced = ImageOps.autocontrast(ImageOps.grayscale(photo), cutoff=1)
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        paths = [original]
        for angle in (0, 90, 180, 270):
            path = folder / f"contrast-{angle}.png"
            enhanced.rotate(angle, expand=True, fillcolor=255).save(path)
            paths.append(path)
        return paths


def _read_score(out: dict) -> float:
    # Mean confidence alone favours a single readable logo over a whole drug name.
    chars = out.get("reliable_chars")
    if chars is None:
        chars = len(re.sub(r"\W", "", out.get("text", "").replace("[?]", "")))
    return max(0, out.get("mean_conf", 0) - 25) * min(chars, 600) ** 0.5


def read_label(image: Path, langs: str = "eng+fra+spa+ara") -> dict:
    with tempfile.TemporaryDirectory(prefix="label-read-") as folder:
        try:
            images = _prepare_images(image, Path(folder))
        except Exception:
            return {"ok": False, "reason": "could not open this photo; choose a JPEG or PNG image",
                    "text": "", "mean_conf": 0.0}
        return _read_label(images, langs)


def _read_label(images: list[Path], langs: str) -> dict:
    image = images[0]
    tess, vlm, note = _engines(langs)
    if tess is None and vlm is None:
        return {"ok": False, "reason": note, "text": "", "mean_conf": 0.0}

    out, source = None, ""
    if tess is not None:
        def attempt(path):
            try:
                return tess.transcribe(path, psm=11, timeout=12)
            except Exception:
                return None
        # Test rotations even when the first attempt reports high confidence:
        # sideways photos can produce a confident but incomplete read.
        with ThreadPoolExecutor(max_workers=2) as pool:
            reads = [r for r in pool.map(attempt, images) if r and r.get("text", "").strip()]
        if reads:
            out = max(reads, key=_read_score)
            source = "tesseract"
        else:
            note = "no text could be read from the photo"

        if out is None or out.get("mean_conf", 0) < LOW_CONFIDENCE_BELOW:
            try:
                retry = tess.transcribe(image, psm=6, timeout=25)
                if out is None or _read_score(retry) > _read_score(out):
                    out = retry
                    source = "tesseract"
            except Exception:
                pass  # keep the first read if the alternate layout cannot be read

    # Escalate an unsure read to the vision model, exactly as the inbox pipeline does.
    if vlm is not None and (out is None or out.get("mean_conf", 0) < LOW_CONFIDENCE_BELOW):
        try:
            text = vlm.transcribe(image)
            if text:
                return {"ok": True, "text": text, "mean_conf": None, "source": "vlm",
                        "model": getattr(vlm, "model", ""), "tier": "vlm", "note": note}
        except Exception as e:
            note = (note + "; " if note else "") + f"vision model failed ({type(e).__name__})"

    if out is None:
        return {"ok": False, "reason": note or "no text read", "text": "", "mean_conf": 0.0}

    conf = out.get("mean_conf", 0.0)
    tier = "photo_only" if conf < PHOTO_ONLY_BELOW else ("low" if conf < LOW_CONFIDENCE_BELOW else "ok")
    return {"ok": True, "text": out.get("text", ""), "mean_conf": conf, "source": source,
            "words": out.get("words", 0), "unreadable": out.get("unreadable", 0),
            "tier": tier, "note": note,
            "languages": getattr(tess, "langs", ""), "missing_languages": getattr(tess, "missing", [])}


def _label_text(searcher, text: str) -> str:
    # Labels often print O.R.S. / O R S. Only join letters if the library knows
    # the abbreviation; never guess substitutions such as 0 -> O in a medicine name.
    def join(match):
        compact = re.sub(r"[.\s]", "", match.group()).lower()
        return compact if (compact,) in searcher.glossary.phrase_to_concept else match.group()
    return re.sub(r"\b[A-Za-z](?:[. \t]+[A-Za-z]){1,5}\b\.?", join, text)


def terms_from(searcher, text: str, limit: int = MAX_TERMS) -> list[dict]:
    """The words on the label the library actually knows, in the order they are worth searching.

    Glossary concepts and abbreviations first (a drug or disease the index is built around), or
    rare-enough words when no concept is found. Anything the library has never heard of is dropped rather than
    searched, so a batch number cannot drag the results somewhere arbitrary.
    """
    toks = tokenize(_label_text(searcher, text))
    found, seen = [], set()
    for start, end, concept in searcher.glossary.find(toks):
        if concept in seen or all(t in LABEL_NOISE for t in toks[start:end]):
            continue
        seen.add(concept)
        found.append({"term": concept, "kind": "concept", "as_read": " ".join(toks[start:end])})

    # A recognized name is the query. Adding ingredients to ORS (e.g. citrate,
    # glucose) makes the coverage gate reject every relevant document. Likewise,
    # a rare manufacturer's name is not evidence of the product's identity.
    if found:
        return found[:limit]

    if len(found) < limit:
        # Score every remaining candidate, then keep the RAREST. Taking them in the order they
        # appear lets "Batch", "Store below 25 C" and a manufacturer's city crowd out the drug
        # name further down the box, which is how a paracetamol label returned nothing.
        cand: dict[str, int] = {}
        for t in toks:
            if (len(t) < 5 or t in STOPWORDS or t in LABEL_NOISE or t in seen or t in cand
                    or t.isdigit() or not t.isalpha()):
                continue
            n = searcher.con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT c.doc_id FROM chunks_fts f "
                "JOIN chunks c ON c.id=f.rowid WHERE chunks_fts MATCH ? LIMIT ?)",
                (f'"{t}"', WORD_MAX_DF + 1)).fetchone()[0]
            if 0 < n <= WORD_MAX_DF:
                cand[t] = n
        room = min(limit, 3)
        for t, n in sorted(cand.items(), key=lambda kv: kv[1])[:room]:
            found.append({"term": t, "kind": "word", "docs": n})
    return found[:limit]


def _reference_only(heading: str, text: str) -> bool:
    """Recognise abbreviation lists / contents pages, not treatment guidance."""
    opening = " ".join(tokenize((heading or "") + " " + text[:160]))
    if re.search(r"\b(abbreviations|acronyms|abreviations|abreviaturas|sigles)\b", opening):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    abbreviations = sum(bool(re.match(r"^\|?\s*[A-Z][A-Z0-9 /+°-]{1,12}(?:\s*\||$)", line)) for line in lines)
    table_abbreviations = len(re.findall(r"\|\s*[A-Z][A-Z0-9 /+°-]{1,12}\s*\|", text))
    contents = len(re.findall(r"\b(?:appendix|annexe|anexo|chapter|chapitre|capitulo)\s+\d", text, re.I))
    return abbreviations >= 4 or table_abbreviations >= 4 or contents >= 3


def match_label(searcher, text: str, k: int = 5) -> dict:
    """Search a read or user-corrected label, preserving the normal confidence gate."""
    terms = terms_from(searcher, text)
    if not terms:
        return {"ok": False, "reason": "no recognised name on this label was found in the library",
                "terms": [], "results": [], "abstained": True}

    query = " ".join(t["term"] for t in terms)
    r = searcher.search(query, k=max(k, 15))
    concepts = searcher.parse(query)
    results = []
    for result in r.get("results", []):
        # A generated summary of an abbreviation list can look like a protocol.
        # Show the actual source passage and its provenance instead.
        x = dict(result)
        cid = x.get("text_chunk_id") or (x.get("evidence") or {}).get("chunk_id")
        if cid is None:
            continue
        x.update(chunk_id=cid, matched_via="text")
        searcher._decorate(x, concepts)
        heading, passage = searcher._chunk(cid)
        source_text = searcher.get_document(x["doc_id"])["text"]
        x["reference_only"] = _reference_only(heading, passage) or (
            len(source_text) < 6000 and _reference_only("", source_text))
        results.append(x)

    # Keep the core score (which includes evidence tier and coverage), with lists
    # below substantive passages. Aboutness alone promoted short glossary pages.
    results.sort(key=lambda x: x["reference_only"])
    results = results[:k]
    return {"ok": True, "terms": terms, "query": query,
            "abstained": r.get("abstained", False) or not results, "results": results,
            "ms": r.get("ms", 0.0)}


def scan(searcher, data: bytes, content_type: str = "image/jpeg", k: int = 5) -> dict:
    if not data:
        return {"ok": False, "reason": "no image received"}
    if len(data) > MAX_BYTES:
        return {"ok": False, "reason": f"image too large ({len(data) // 1024} KB, limit "
                                       f"{MAX_BYTES // 1024} KB)"}
    suffix = SUFFIX.get((content_type or "").split(";")[0].strip().lower(), ".jpg")
    with tempfile.TemporaryDirectory(prefix="label-scan-") as folder:
        tmp = Path(folder) / ("label" + suffix)
        tmp.write_bytes(data)
        ocr = read_label(tmp)

    if not ocr.get("ok"):
        return {"ok": False, "reason": ocr.get("reason", "could not read the label"), "ocr": ocr,
                "bytes": len(data)}
    if ocr.get("tier") == "photo_only":
        return {"ok": False, "reason": "the photo is too unclear to read reliably", "ocr": ocr,
                "terms": [], "results": [], "bytes": len(data)}

    return {**match_label(searcher, ocr["text"], k), "ocr": ocr, "bytes": len(data)}
