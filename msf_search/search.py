"""Query-time retrieval. Runs entirely on-premises from the single index.db file.

Pipeline per query (typically a few milliseconds on a laptop CPU):
  1. Parse the query into concepts; expand each via the multilingual glossary.
  2. BM25 over passages (SQLite FTS5), title/heading weighted higher.
  3. Concept coverage: which share of the query's (IDF-weighted) concepts does each
     passage / document actually contain?  This is the false-positive guard.
  4. Optional: fuse with dense multilingual embeddings if they were built.
  5. Aggregate passages -> documents, attach the best snippet, assign confidence,
     and ABSTAIN ("no reliable match") instead of returning weak guesses.
"""
import html
import json
import math
import re
import sqlite3
import time
import zlib
from pathlib import Path

from .glossary import Glossary
from .textnorm import STOPWORDS, TOKEN_RE, normalize_token, tokenize

_SENT = re.compile(r"(?<=[.!?؟;])\s+|\n+")

# Evidence tiers: each document's score is multiplied by its genre's weight, so a clinical
# guideline outranks a press release or financial report that matches the query equally well.
# Only reorders within a confidence level: a weak match never becomes strong because of its genre.
# Edit freely. Tier order = badge rank in the UI; genres not listed get DEFAULT_TIER.
EVIDENCE_TIERS = {
    # tier label: (score multiplier, genres)
    "Guideline": (1.25, ["Clinical Guideline Chapter"]),
    "Research": (1.10, ["Peer-reviewed Article", "Study Protocol", "Research Policy",
                        "Research Governance Document"]),
    "Report": (1.00, ["Activity Report Chapter", "Evaluation Report", "Evaluation Methodology Guide", "Report"]),
    "Reference": (1.00, ["Legal Encyclopedia Entry", "Historical Case Study", "Advocacy Brief"]),
    "News & testimony": (0.90, ["Press Release", "Statement", "Project Update", "Voices from the Field",
                                "Interview", "First-person Field Account", "Speech", "Scientific News",
                                "Scientific Publication Notice", "Field Note (handwritten)"]),
    "Background": (0.80, ["Financial Report", "Timeline", "Summary", "Opinion", "Op-Ed"]),
}
DEFAULT_TIER = ("Other", 1.00)
GENRE_TIER = {g: (label, rank, w) for rank, (label, (w, gs)) in enumerate(EVIDENCE_TIERS.items(), 1) for g in gs}


PROVENANCE_COLS = "text_source, ocr_confidence, machine_generated, generator, model, generated_at"


def provenance(text_source, ocr_confidence, machine_generated, generator, model, generated_at) -> dict:
    """The provenance object on every doc, chunk and result (API.md). machine_generated text must be labelled."""
    return {"text_source": text_source or "parsed", "ocr_confidence": ocr_confidence,
            "machine_generated": bool(machine_generated), "generator": generator, "model": model,
            "generated_at": generated_at}


class Concept:
    def __init__(self, label: str, variants: list[tuple]):
        self.label = label
        self.variants = variants
        self.weight = 1.0

    def fts(self) -> str:
        parts = []
        for v in self.variants:
            parts.append('"' + " ".join(v) + '"')
        return " OR ".join(parts)


class Searcher:
    def __init__(self, index_path: str, glossary_path: str | None = None, dense=None,
                 abbreviations_path: str | None = None):
        self.index_path = index_path
        self.con = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, check_same_thread=False)
        self.glossary = Glossary(glossary_path)
        if abbreviations_path and Path(abbreviations_path).exists():
            self.glossary.load_abbreviations(abbreviations_path)
        self.dense = dense
        self._data_version = None
        self.refresh()

    def refresh(self):
        """Reload document metadata if another process changed the file (apply_delta.py), so a
        running server serves the update without a restart. One cheap PRAGMA per request."""
        v = self.con.execute("PRAGMA data_version").fetchone()[0]
        if v == self._data_version:
            return
        self._data_version = v
        self.library_version = (self.con.execute(
            "SELECT value FROM meta WHERE key='library_version'").fetchone() or [""])[0]
        self.report = json.loads(self.con.execute("SELECT value FROM meta WHERE key='report'").fetchone()[0])
        self.n_docs = self.con.execute("SELECT count(*) FROM docs WHERE n_chunks>0").fetchone()[0]
        self.docs = {r[0]: {"id": r[0], "genre": r[1], "language": r[2], "title": r[3],
                             "quality": r[4], "also_filed_as": json.loads(r[5] or "[]"),
                             "original_name": r[6], "n_chunks": r[7], "year": r[8] or None,
                             "transcribed": bool(r[9]), "provenance": provenance(*r[10:16])}
                     for r in self.con.execute(
                         "SELECT id, genre, language, title, quality, also_filed_as, original_name, n_chunks, year, transcribed, "
                         f"{PROVENANCE_COLS} FROM docs")}
        for meta in self.docs.values():
            label, rank, w = GENRE_TIER.get(meta["genre"], (DEFAULT_TIER[0], len(EVIDENCE_TIERS) + 1, DEFAULT_TIER[1]))
            meta.update({"tier": label, "tier_rank": rank, "prior": w})

    # ---------- query understanding ----------
    def parse(self, query: str) -> list[Concept]:
        toks = tokenize(query)
        concepts, covered = [], set()
        for s, e, c in self.glossary.find(toks):
            concepts.append(Concept(c, list(self.glossary.variants[c])))
            covered.update(range(s, e))
        for i, t in enumerate(toks):
            if i in covered or t in STOPWORDS or len(t) < 2:
                continue
            if any(c.label == t for c in concepts):
                continue
            concepts.append(Concept(t, [(t,)]))
        return concepts

    def _concept_rows(self, c: Concept) -> tuple[set, dict]:
        rows = self.con.execute(
            "SELECT f.rowid, c.doc_id FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
            "WHERE chunks_fts MATCH ?", (c.fts(),)).fetchall()
        per_doc: dict[str, int] = {}
        for _, d in rows:
            per_doc[d] = per_doc.get(d, 0) + 1
        return {r[0] for r in rows}, per_doc

    # ---------- main entry ----------
    def search(self, query: str, k: int = 10, languages: list | None = None, genres: list | None = None,
               strict: bool = True, candidates: int = 400) -> dict:
        t0 = time.perf_counter()
        self.refresh()
        concepts = self.parse(query)
        if not concepts:
            return {"query": query, "results": [], "abstained": True, "reason": "empty query", "ms": 0}

        # Concept coverage sets + IDF weights (rare concepts matter more than "patients").
        chunk_sets, doc_sets = [], []
        for c in concepts:
            cs, ds = self._concept_rows(c)
            chunk_sets.append(cs)
            doc_sets.append(ds)
            # A term that appears nowhere in the library gets the MAXIMUM weight: the
            # library cannot answer that part of the question, so coverage must drop.
            c.weight = math.log(1 + self.n_docs / (1 + len(ds))) if ds else math.log(1 + self.n_docs)
            c.found = bool(ds)
        total_w = sum(c.weight for c in concepts) or 1.0

        # BM25 candidates over passages, with optional metadata filters pushed into SQL.
        where, params = ["chunks_fts MATCH ?"], [" OR ".join(f"({c.fts()})" for c in concepts)]
        if languages:
            where.append(f"d.language IN ({','.join('?' * len(languages))})")
            params += languages
        if genres:
            where.append(f"d.genre IN ({','.join('?' * len(genres))})")
            params += genres
        rows = self.con.execute(
            "SELECT f.rowid, c.doc_id, bm25(chunks_fts, 4.0, 2.0, 1.0) AS s "
            "FROM chunks_fts f JOIN chunks c ON c.id=f.rowid JOIN docs d ON d.id=c.doc_id "
            f"WHERE {' AND '.join(where)} ORDER BY s LIMIT ?", params + [candidates]).fetchall()

        chunk_doc = {r[0]: r[1] for r in rows}
        lex = {r[0]: -r[2] for r in rows}
        dense_scores = {}
        if self.dense is not None:
            dense_scores = self.dense.search(query, top=200, allowed=None)
            for cid in dense_scores:
                if cid not in chunk_doc:
                    d = self.con.execute("SELECT doc_id FROM chunks WHERE id=?", (cid,)).fetchone()
                    if d:
                        chunk_doc[cid] = d[0]
        if not chunk_doc:
            return {"query": query, "concepts": [c.label for c in concepts], "results": [],
                    "unknown_terms": [c.label for c in concepts if not c.found],  # not filter misses
                    "abstained": True, "reason": "no passage contains these terms",
                    "ms": round((time.perf_counter() - t0) * 1000, 1)}

        max_lex = max(lex.values()) if lex else 1.0
        unknown_w = sum(c.weight for c in concepts if not getattr(c, "found", True))
        docs: dict[str, dict] = {}
        for cid, did in chunk_doc.items():
            meta = self.docs.get(did)
            if meta is None:
                continue
            if languages and meta["language"] not in languages:
                continue
            if genres and meta["genre"] not in genres:
                continue
            ccov = sum(c.weight for c, cs in zip(concepts, chunk_sets) if cid in cs) / total_w
            s_lex = lex.get(cid, 0.0) / max_lex
            s = s_lex * (0.2 + 0.8 * ccov)
            if dense_scores:
                s = 0.6 * s + 0.4 * dense_scores.get(cid, 0.0)
            d = docs.setdefault(did, {"best": -1, "best_chunk": None, "best_cov": 0.0, "hits": 0, "dense": 0.0})
            d["hits"] += 1
            d["dense"] = max(d["dense"], getattr(self.dense, "last_abs", {}).get(cid, 0.0))
            if s > d["best"]:
                d["best"], d["best_chunk"], d["best_cov"] = s, cid, ccov

        results = []
        for did, d in docs.items():
            dcov = sum(c.weight for c, ds in zip(concepts, doc_sets) if did in ds) / total_w
            # "Aboutness": is the document ABOUT these concepts, or does a 900-page report
            # merely mention them in passing? Share of the document's passages that match.
            n = max(1, self.docs[did]["n_chunks"])
            about = sum(c.weight * min(1.0, 6.0 * ds.get(did, 0) / n)
                        for c, ds in zip(concepts, doc_sets)) / total_w
            score = d["best"] * (0.55 + 0.45 * about) * self.docs[did]["prior"]
            semantic_ok = (dense_scores and d["dense"] >= self.dense.high_threshold
                           and unknown_w < 0.5 * total_w)  # never let embeddings override missing key terms
            if d["best_cov"] >= 0.9 or semantic_ok:
                conf = "high"
            elif d["best_cov"] >= 0.65:
                conf = "medium"
            else:
                conf = "low"
            results.append({"doc_id": did, "score": round(score, 4), "confidence": conf,
                            "passage_coverage": round(d["best_cov"], 2), "doc_coverage": round(dcov, 2),
                            "aboutness": round(about, 2), "chunk_id": d["best_chunk"]})
        tier = {"high": 0, "medium": 1, "low": 2}
        results.sort(key=lambda r: (tier[r["confidence"]], -r["score"]))

        # False-positive guard: relative score cut-off + confidence filter.
        top = max((r["score"] for r in results), default=0)
        results = [r for r in results if r["score"] >= 0.2 * top]
        weaker = [r for r in results if r["confidence"] == "low"]
        if strict:
            results = [r for r in results if r["confidence"] != "low"]
        results = results[:k]
        for r in results + weaker[:3]:
            self._decorate(r, concepts)

        ms = round((time.perf_counter() - t0) * 1000, 1)
        unknown = [c.label for c in concepts if not getattr(c, "found", True)]
        return {"query": query, "concepts": [{"label": c.label, "weight": round(c.weight, 2),
                                              "variants": [" ".join(v) for v in c.variants]} for c in concepts],
                "unknown_terms": unknown, "results": results,
                "weaker_matches": weaker[:3] if strict else [],
                "abstained": not results,
                "reason": "" if results else "no document covers enough of the query",
                "ms": ms}

    # ---------- presentation ----------
    def _chunk(self, cid: int) -> tuple[str, str]:
        h, t = self.con.execute("SELECT heading, text FROM chunks WHERE id=?", (cid,)).fetchone()
        return h, zlib.decompress(t).decode("utf-8")

    def chunk_provenance(self, cid: int) -> dict:
        return provenance(*self.con.execute(f"SELECT {PROVENANCE_COLS} FROM chunks WHERE id=?", (cid,)).fetchone())

    def _decorate(self, r: dict, concepts: list[Concept]):
        meta = self.docs[r["doc_id"]]
        r.update({k: meta[k] for k in ("title", "genre", "language", "also_filed_as", "original_name",
                                        "year", "tier", "tier_rank", "transcribed")})
        heading, text = self._chunk(r["chunk_id"])
        r["heading"] = heading
        r["snippet"] = make_snippet(text, concepts)
        r["provenance"] = self.chunk_provenance(r["chunk_id"])   # of the passage shown
        r["doc_provenance"] = meta["provenance"]
        m = re.match(r"^Page (\d+)$", heading or "")
        r["page"] = int(m.group(1)) if m else None

    def get_document(self, doc_id: str) -> dict | None:
        self.refresh()
        row =self.con.execute("SELECT text FROM docs WHERE id=?", (doc_id,)).fetchone()
        if not row or doc_id not in self.docs:
            return None
        return {**self.docs[doc_id], "text": zlib.decompress(row[0]).decode("utf-8")}

    def passages(self, result: dict, n: int = 1) -> list[str]:
        return [self._chunk(result["chunk_id"])[1]]


def _matches(tok_norm: str, variant_tokens: set) -> bool:
    for v in variant_tokens:
        if tok_norm == v:
            return True
        if len(v) >= 5 and tok_norm.startswith(v[: max(5, len(v) - 2)]):
            return True
    return False


def make_snippet(text: str, concepts: list[Concept], max_words: int = 55) -> str:
    vtoks = {t for c in concepts for v in c.variants for t in v if t not in STOPWORDS}
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if not sents:
        return ""
    best_i, best_s = 0, -1
    for i, s in enumerate(sents):
        found = {normalize_token(t) for t in TOKEN_RE.findall(s)}
        score = sum(1 for v in vtoks if any(_matches(f, {v}) for f in found))
        if score > best_s:
            best_i, best_s = i, score
    out = sents[best_i]
    j = best_i + 1
    while j < len(sents) and len(out.split()) < max_words * 0.6:
        out += " " + sents[j]
        j += 1
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]) + " ..."

    def hl(m):
        w = m.group(0)
        return f"<mark>{w}</mark>" if _matches(normalize_token(w), vtoks) else w
    return TOKEN_RE.sub(hl, html.escape(out))
