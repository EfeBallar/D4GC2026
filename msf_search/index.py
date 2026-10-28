"""Build the single-file search index (SQLite + FTS5).

Preprocessing runs once at HQ (or on any laptop). The output is ONE file, index.db,
that is copied to the field. SQLite ships with Python on every OS, and FTS5 gives
BM25 ranking for free: no server, no licence, no vendor.
"""
import json
import re
import sqlite3
import time
import zlib
from pathlib import Path

from .chunker import chunk_document
from .ingest import load_collection, quality_report
from .textnorm import normalize_for_index

SCHEMA = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE docs(
  id TEXT PRIMARY KEY, genre TEXT, language TEXT, title TEXT, quality TEXT,
  duplicate_of TEXT, also_filed_as TEXT, original_name TEXT,
  n_chars INTEGER, n_chunks INTEGER, year INTEGER, transcribed INTEGER, text BLOB,
  text_source TEXT, ocr_confidence REAL, machine_generated INTEGER, generator TEXT, model TEXT, generated_at TEXT,
  enriched INTEGER, importance INTEGER, importance_reason TEXT, audience TEXT, actionable INTEGER, summary TEXT,
  topics TEXT, countries TEXT, situation TEXT, doc_type TEXT, superseded_hint TEXT, enrich_terms TEXT,
  enrich_model TEXT, enrich_at TEXT);
CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id TEXT, ord INTEGER, heading TEXT, text BLOB,
  kind TEXT, text_source TEXT, ocr_confidence REAL, machine_generated INTEGER, generator TEXT, model TEXT,
  generated_at TEXT, source_ord INTEGER);
CREATE INDEX chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  title, heading, body, enrich, content='', tokenize='porter unicode61 remove_diacritics 2');
"""


def fts_row(title: str, heading: str, text: str, enrich: str = "") -> tuple[str, str, str, str]:
    """The chunks_fts values for one passage. Shared with delta updates (delta.py): the FTS table is
    contentless, so deleting a row means re-supplying exactly the values that were inserted.
    `enrich` = the doc's HQ keywords/abbreviations/topics, repeated on each of its passages."""
    return (normalize_for_index(title), normalize_for_index(heading), normalize_for_index(text),
            normalize_for_index(enrich))


def load_enrichment(path: str | None) -> dict:
    """data/enrichment.jsonl (scripts/hq_enrich.py) -> {doc_id: record}; later lines win."""
    out = {}
    if path and Path(path).exists():
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                if rec.get("status") == "ok":
                    out[rec["id"]] = rec
    return out


def enrich_terms(e: dict, doc_text: str) -> str:
    """Keywords (4 languages) + topics + grounded abbreviations, for the `enrich` FTS column."""
    words = set(re.findall(r"\w+", doc_text))
    abbr = [f"{a['abbr']} {a['long']}" for a in e.get("abbreviations", []) if a["abbr"] in words]
    return " ".join([*(k for l in ("en", "fr", "es", "ar") for k in e.get(f"keywords_{l}", [])),
                     *e.get("topics", []), *abbr])


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def synthetic_chunks(e: dict, text_chunks: list) -> list[tuple]:
    """HQ-made passages: (kind, heading, text, source_ord). A Q&A passage points at the original passage
    that contains its evidence quote, so the UI can always show the source."""
    out = [("summary", "Summary", e["summary"], None)] if e.get("summary") else []
    norm = [_norm_ws(c.text) for c in text_chunks]
    for qa in e.get("qa", []):
        q = _norm_ws(qa["evidence_quote"])
        src = next((c.ord for c, n in zip(text_chunks, norm) if q in n), None)
        if src is None:  # quote spans two passages: take the one holding its start
            src = next((c.ord for c, n in zip(text_chunks, norm) if q[:60] in n), None)
        out.append(("qa", qa["question"], f"{qa['answer']}\n\nEvidence: \"{qa['evidence_quote']}\"", src))
    return out


_PAGE = re.compile(r"^Page (\d+)$")


def chunk_provenance(d, heading: str) -> tuple:
    """(text_source, ocr_confidence, machine_generated, generator, model, generated_at) for a text passage.
    OCR passages carry their own page's engine and confidence; the kept embedded text layer is "parsed"."""
    m = _PAGE.match(heading or "")
    if d.text_source == "parsed" or (d.page_info and not m):
        return ("parsed", None, 0, None, None, None)
    page = d.page_info.get(int(m.group(1)), {}) if m else {}
    if page:
        src = "ocr_vlm" if page.get("engine") == "vlm" else "ocr_tesseract"
        conf = round(page["mean_conf"] / 100, 3) if src == "ocr_tesseract" and page.get("mean_conf") is not None else None
        return (src, conf, 0, None, page.get("model"), d.ocr_at)
    return (d.text_source, d.ocr_confidence, 0, None, d.ocr_model, d.ocr_at)


def build_index(data_dir: str, csv_path: str | None, out_path: str,
                max_words: int = 180, overlap: int = 30, verbose: bool = True,
                exclude_ids: set | None = None, transcriptions_dir: str | None = None,
                inbox_dir: str | None = None, enrichment_path: str | None = None) -> dict:
    t0 = time.time()
    docs = load_collection(data_dir, csv_path, transcriptions_dir=transcriptions_dir, inbox_dir=inbox_dir)
    enrichment = load_enrichment(enrichment_path)
    llm_abbr = {}
    if exclude_ids:  # e.g. a base library that a delta update completes later
        docs = [d for d in docs if d.id not in exclude_ids]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)

    n_chunks = n_synthetic = 0
    for d in docs:
        indexable = d.quality not in ("boilerplate",) and not d.duplicate_of
        chunks = chunk_document(d.id, d.text, max_words, overlap) if indexable else []
        e = enrichment.get(d.id) if indexable else None
        terms = enrich_terms(e, d.text) if e else ""
        con.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d.id, d.genre, d.language, d.title, d.quality, d.duplicate_of,
             json.dumps(d.also_filed_as), Path(d.original_path).name if d.original_path else "",
             len(d.text), len(chunks), (e or {}).get("year") or d.year, int(d.transcribed),
             zlib.compress(d.text.encode("utf-8"), 9),
             d.text_source, d.ocr_confidence, 0, None, d.ocr_model, d.ocr_at,
             int(bool(e)), (e or {}).get("importance"), (e or {}).get("importance_reason"), (e or {}).get("audience"),
             int(bool((e or {}).get("actionable"))) if e else None, (e or {}).get("summary") or None,
             json.dumps((e or {}).get("topics", [])), json.dumps((e or {}).get("countries", [])),
             (e or {}).get("situation"), (e or {}).get("doc_type"), (e or {}).get("superseded_hint"), terms,
             (e or {}).get("model"), (e or {}).get("generated_at")))
        for c in chunks:
            cur = con.execute(
                "INSERT INTO chunks(doc_id, ord, heading, text, kind, text_source, ocr_confidence, machine_generated, "
                "generator, model, generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (c.doc_id, c.ord, c.heading, zlib.compress(c.text.encode("utf-8"), 9), "text",
                 *chunk_provenance(d, c.heading)))
            con.execute("INSERT INTO chunks_fts(rowid, title, heading, body, enrich) VALUES (?,?,?,?,?)",
                        (cur.lastrowid, *fts_row(d.title, c.heading, c.text, terms)))
        n_chunks += len(chunks)
        if e:
            for a in e.get("abbreviations", []):
                if re.search(rf"\b{re.escape(a['abbr'])}\b", d.text):  # only abbreviations the doc really uses
                    llm_abbr.setdefault(a["abbr"], set()).add(a["long"].strip().lower())
            for i, (kind, heading, text, src) in enumerate(synthetic_chunks(e, chunks)):
                cur = con.execute(
                    "INSERT INTO chunks(doc_id, ord, heading, text, kind, text_source, ocr_confidence, "
                    "machine_generated, generator, model, generated_at, source_ord) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (d.id, len(chunks) + i, heading, zlib.compress(text.encode("utf-8"), 9), kind,
                     "parsed", None, 1, kind, e.get("model"), e.get("generated_at"), src))
                con.execute("INSERT INTO chunks_fts(rowid, title, heading, body, enrich) VALUES (?,?,?,?,?)",
                            (cur.lastrowid, *fts_row(d.title, heading, text, terms)))
                n_synthetic += 1

    report = quality_report(docs)
    report.update({"chunks": n_chunks, "synthetic_chunks": n_synthetic,
                   "enriched_docs": sum(1 for d in docs if d.id in enrichment),
                   "build_seconds": round(time.time() - t0, 1),
                   "chunk_max_words": max_words, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    con.execute("INSERT INTO meta VALUES ('report', ?)", (json.dumps(report),))
    con.execute("INSERT INTO meta VALUES ('library_version', ?)", (time.strftime("%Y-%m-%d-%H%M"),))
    con.execute("INSERT INTO meta VALUES ('applied_deltas', '[]')")
    con.execute("INSERT INTO meta VALUES ('llm_abbreviations', ?)",
                (json.dumps({k: sorted(v) for k, v in llm_abbr.items()}, ensure_ascii=False),))
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    con.commit()
    con.execute("VACUUM")
    con.close()
    report["index_mb"] = round(out.stat().st_size / 1e6, 1)
    con = sqlite3.connect(out)
    con.execute("UPDATE meta SET value=? WHERE key='report'", (json.dumps(report),))
    con.commit()
    con.close()
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report
