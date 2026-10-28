"""Delta updates: send a few new or revised documents to the field instead of a whole index.db.

    HQ:    python scripts/make_delta.py --ids P23,Dc95675e997         -> build/update_<version>.db (KBs)
    Field: python scripts/apply_delta.py build/update_<version>.db    -> merged into build/index.db

A delta is a small SQLite file holding the docs rows, their passages and the passages' FTS rows.
Applying it is one transaction (a power cut leaves the old library intact) and idempotent: a
version that was already applied is skipped, and a document already in the library is replaced,
not duplicated. Updates travel like medicines: by satellite, a phone hotspot or a USB stick.
"""
import json
import sqlite3
import time
import zlib
from collections import Counter
from pathlib import Path

from .index import fts_row


def _connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + ("?mode=ro" if readonly else "")
    return sqlite3.connect(uri, uri=True)


def _attach_uri(path: str) -> str:
    return Path(path).resolve().as_uri() + "?mode=ro"


def make_delta(index_path: str, doc_ids: list[str], out_path: str, version: str | None = None) -> dict:
    version = version or time.strftime("%Y-%m-%d-%H%M")
    src = _connect(index_path, readonly=True)
    found = {r[0] for r in src.execute(
        f"SELECT id FROM docs WHERE id IN ({','.join('?' * len(doc_ids))})", doc_ids)}
    missing = [d for d in doc_ids if d not in found]
    if missing:
        raise SystemExit(f"not in {index_path}: {', '.join(missing)}")
    src_version = (src.execute("SELECT value FROM meta WHERE key='library_version'").fetchone() or [""])[0]
    src.close()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = _connect(str(out))
    con.execute("ATTACH ? AS src", (_attach_uri(index_path),))
    marks = ",".join("?" * len(doc_ids))
    # Copy rows with the source's own column layout, so the delta always matches the index schema.
    con.execute(f"CREATE TABLE docs AS SELECT * FROM src.docs WHERE id IN ({marks})", doc_ids)
    ccols = ",".join(r[1] for r in con.execute("PRAGMA src.table_info(chunks)") if r[1] != "id")
    con.execute(f"CREATE TABLE chunks AS SELECT {ccols} FROM src.chunks "
                f"WHERE doc_id IN ({marks}) ORDER BY doc_id, ord", doc_ids)
    con.execute("CREATE TABLE chunks_fts_rows(doc_id TEXT, ord INTEGER, title TEXT, heading TEXT, body TEXT, enrich TEXT)")
    titles = {d: (t, e or "") for d, t, e in con.execute("SELECT id, title, enrich_terms FROM docs")}
    con.executemany("INSERT INTO chunks_fts_rows VALUES (?,?,?,?,?,?)",
                    [(d, o, *fts_row(titles[d][0], h, zlib.decompress(t).decode("utf-8"), titles[d][1]))
                     for d, o, h, t in con.execute("SELECT doc_id, ord, heading, text FROM chunks").fetchall()])
    n_chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("version", version), ("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("doc_ids", json.dumps(sorted(found))), ("source_library_version", src_version)])
    con.commit()
    con.execute("DETACH src")
    con.execute("VACUUM")
    con.close()
    return {"version": version, "path": str(out), "docs": len(found), "chunks": n_chunks,
            "kb": round(out.stat().st_size / 1024, 1)}


def _remove_doc(con: sqlite3.Connection, doc_id: str) -> bool:
    row = con.execute("SELECT title, enrich_terms FROM docs WHERE id=?", (doc_id,)).fetchone()
    if not row:
        return False
    for cid, heading, text in con.execute(
            "SELECT id, heading, text FROM chunks WHERE doc_id=?", (doc_id,)).fetchall():
        con.execute("INSERT INTO chunks_fts(chunks_fts, rowid, title, heading, body, enrich) VALUES ('delete',?,?,?,?,?)",
                    (cid, *fts_row(row[0], heading, zlib.decompress(text).decode("utf-8"), row[1] or "")))
    con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    con.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    return True


def _refresh_report(con: sqlite3.Connection) -> dict:
    """Keep the meta report (shown in the UI: counts, languages, genres) in line with the docs table."""
    report = json.loads(con.execute("SELECT value FROM meta WHERE key='report'").fetchone()[0])
    rows = con.execute("SELECT genre, language, quality, duplicate_of, n_chars FROM docs").fetchall()
    report.update({
        "documents": len(rows),
        "exact_duplicates": sum(1 for r in rows if r[3]),
        "boilerplate_scrapes": sum(1 for r in rows if r[2] == "boilerplate"),
        "very_short": sum(1 for r in rows if r[2] == "very_short"),
        "languages": dict(Counter(r[1] for r in rows).most_common()),
        "genres": dict(Counter(r[0] for r in rows).most_common()),
        "total_chars": sum(r[4] for r in rows),
        "chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
    })
    return report


def apply_delta(index_path: str, delta_path: str) -> dict:
    con = _connect(index_path)
    con.execute("ATTACH ? AS u", (_attach_uri(delta_path),))
    version = con.execute("SELECT value FROM u.meta WHERE key='version'").fetchone()[0]
    applied = json.loads((con.execute("SELECT value FROM meta WHERE key='applied_deltas'").fetchone() or ["[]"])[0])
    if version in applied:
        con.close()
        return {"version": version, "status": "already applied", "added": 0, "replaced": 0}

    cols = [r[1] for r in con.execute("PRAGMA main.table_info(docs)")]
    delta_cols = {r[1] for r in con.execute("PRAGMA u.table_info(docs)")}
    if set(cols) != delta_cols:
        raise SystemExit(f"schema mismatch: index docs columns {cols} vs delta {sorted(delta_cols)}. "
                         "Rebuild the field index with the same code version.")
    added = replaced = n_chunks = 0
    with con:  # one transaction: all of the delta, or none of it
        for (doc_id,) in con.execute("SELECT id FROM u.docs").fetchall():
            if _remove_doc(con, doc_id):
                replaced += 1
            else:
                added += 1
        con.execute(f"INSERT INTO docs({','.join(cols)}) SELECT {','.join(cols)} FROM u.docs")
        ccols = [r[1] for r in con.execute("PRAGMA u.table_info(chunks)")]
        for row in con.execute(
                f"SELECT {','.join('c.' + c for c in ccols)}, f.title, f.heading, f.body, f.enrich FROM u.chunks c "
                "JOIN u.chunks_fts_rows f ON f.doc_id=c.doc_id AND f.ord=c.ord ORDER BY c.doc_id, c.ord").fetchall():
            cur = con.execute(f"INSERT INTO chunks({','.join(ccols)}) VALUES ({','.join('?' * len(ccols))})",
                              row[:len(ccols)])
            con.execute("INSERT INTO chunks_fts(rowid, title, heading, body, enrich) VALUES (?,?,?,?,?)",
                        (cur.lastrowid, *row[len(ccols):]))
            n_chunks += 1
        applied.append(version)
        con.execute("INSERT OR REPLACE INTO meta VALUES ('library_version', ?)", (version,))
        con.execute("INSERT OR REPLACE INTO meta VALUES ('applied_deltas', ?)", (json.dumps(applied),))
        report = _refresh_report(con)
        report["index_mb"] = round(Path(index_path).stat().st_size / 1e6, 1)
        con.execute("UPDATE meta SET value=? WHERE key='report'", (json.dumps(report),))
    con.close()
    return {"version": version, "status": "applied", "added": added, "replaced": replaced,
            "chunks": n_chunks, "applied_deltas": applied}
