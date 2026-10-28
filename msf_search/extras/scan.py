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
import tempfile
from pathlib import Path

from ..textnorm import STOPWORDS, tokenize

MAX_BYTES = 10 * 1024 * 1024
SUFFIX = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/heic": ".heic"}
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


def read_label(image: Path, langs: str = "eng+fra+spa+ara") -> dict:
    tess, vlm, note = _engines(langs)
    if tess is None and vlm is None:
        return {"ok": False, "reason": note, "text": "", "mean_conf": 0.0}

    out, source = None, ""
    if tess is not None:
        try:
            out = tess.transcribe(image)
            source = "tesseract"
        except Exception as e:
            out, note = None, f"tesseract failed ({type(e).__name__})"

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
            "tier": tier, "note": note}


def terms_from(searcher, text: str, limit: int = MAX_TERMS) -> list[dict]:
    """The words on the label the library actually knows, in the order they are worth searching.

    Glossary concepts and abbreviations first (a drug or disease the index is built around), then
    rare-enough leftover words. Anything the library has never heard of is dropped rather than
    searched, so a batch number cannot drag the results somewhere arbitrary.
    """
    toks = tokenize(text)
    found, seen = [], set()
    for start, end, concept in searcher.glossary.find(toks):
        if concept in seen:
            continue
        seen.add(concept)
        found.append({"term": concept, "kind": "concept", "as_read": " ".join(toks[start:end])})

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
        # A glossary concept is a term the index is built around, so it is worth far more than a
        # leftover word. When concepts were found, add at most two words beside them; only a label
        # that matched no concept at all leans on words.
        room = min(limit - len(found), 2 if found else 3)
        for t, n in sorted(cand.items(), key=lambda kv: kv[1])[:room]:
            found.append({"term": t, "kind": "word", "docs": n})
    return found[:limit]


def scan(searcher, data: bytes, content_type: str = "image/jpeg", k: int = 5) -> dict:
    if not data:
        return {"ok": False, "reason": "no image received"}
    if len(data) > MAX_BYTES:
        return {"ok": False, "reason": f"image too large ({len(data) // 1024} KB, limit "
                                       f"{MAX_BYTES // 1024} KB)"}
    suffix = SUFFIX.get((content_type or "").split(";")[0].strip().lower(), ".jpg")
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    try:
        tmp.write_bytes(data)
        ocr = read_label(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    if not ocr.get("ok"):
        return {"ok": False, "reason": ocr.get("reason", "could not read the label"), "ocr": ocr,
                "bytes": len(data)}
    if ocr.get("tier") == "photo_only":
        return {"ok": False, "reason": "the photo is too unclear to read reliably", "ocr": ocr,
                "terms": [], "results": [], "bytes": len(data)}

    terms = terms_from(searcher, ocr["text"])
    if not terms:
        return {"ok": False, "reason": "nothing on this label matches anything in the library",
                "ocr": ocr, "terms": [], "results": [], "bytes": len(data)}

    query = " ".join(t["term"] for t in terms)
    r = searcher.search(query, k=max(k, 10))

    # A label yields one or two words, and a one-word query satisfies the coverage gate in any
    # document that happens to contain the word. "Aboutness" - the share of a document's passages
    # that match - is what separates the guideline on a drug from the report that mentions it
    # once, so order by that and drop the passing mentions when something better exists.
    results = sorted(r.get("results", []), key=lambda x: -x.get("aboutness", 0))
    if results and results[0].get("aboutness", 0) >= 0.30:
        results = [x for x in results if x.get("aboutness", 0) >= 0.10]
    results = results[:k]

    return {"ok": True, "ocr": ocr, "terms": terms, "query": query,
            "abstained": r.get("abstained", False) or not results, "results": results,
            "ms": r.get("ms", 0.0), "bytes": len(data)}
