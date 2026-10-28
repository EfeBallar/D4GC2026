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
  text_source TEXT, ocr_confidence REAL, machine_generated INTEGER, generator TEXT, model TEXT, generated_at TEXT);
CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id TEXT, ord INTEGER, heading TEXT, text BLOB,
  kind TEXT, text_source TEXT, ocr_confidence REAL, machine_generated INTEGER, generator TEXT, model TEXT,
  generated_at TEXT);
CREATE INDEX chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  title, heading, body, content='', tokenize='porter unicode61 remove_diacritics 2');
"""


def fts_row(title: str, heading: str, text: str) -> tuple[str, str, str]:
    """The chunks_fts values for one passage. Shared with delta updates (delta.py): the FTS table is
    contentless, so deleting a row means re-supplying exactly the values that were inserted."""
    return normalize_for_index(title), normalize_for_index(heading), normalize_for_index(text)


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
                inbox_dir: str | None = None) -> dict:
    t0 = time.time()
    docs = load_collection(data_dir, csv_path, transcriptions_dir=transcriptions_dir, inbox_dir=inbox_dir)
    if exclude_ids:  # e.g. a base library that a delta update completes later
        docs = [d for d in docs if d.id not in exclude_ids]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)

    n_chunks = 0
    for d in docs:
        indexable = d.quality not in ("boilerplate",) and not d.duplicate_of
        chunks = chunk_document(d.id, d.text, max_words, overlap) if indexable else []
        con.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d.id, d.genre, d.language, d.title, d.quality, d.duplicate_of,
             json.dumps(d.also_filed_as), Path(d.original_path).name if d.original_path else "",
             len(d.text), len(chunks), d.year, int(d.transcribed), zlib.compress(d.text.encode("utf-8"), 9),
             d.text_source, d.ocr_confidence, 0, None, d.ocr_model, d.ocr_at))
        for c in chunks:
            cur = con.execute(
                "INSERT INTO chunks(doc_id, ord, heading, text, kind, text_source, ocr_confidence, machine_generated, "
                "generator, model, generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (c.doc_id, c.ord, c.heading, zlib.compress(c.text.encode("utf-8"), 9), "text",
                 *chunk_provenance(d, c.heading)))
            con.execute("INSERT INTO chunks_fts(rowid, title, heading, body) VALUES (?,?,?,?)",
                        (cur.lastrowid, *fts_row(d.title, c.heading, c.text)))
        n_chunks += len(chunks)

    report = quality_report(docs)
    report.update({"chunks": n_chunks, "build_seconds": round(time.time() - t0, 1),
                   "chunk_max_words": max_words, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    con.execute("INSERT INTO meta VALUES ('report', ?)", (json.dumps(report),))
    con.execute("INSERT INTO meta VALUES ('library_version', ?)", (time.strftime("%Y-%m-%d-%H%M"),))
    con.execute("INSERT INTO meta VALUES ('applied_deltas', '[]')")
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
