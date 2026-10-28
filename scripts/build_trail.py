#!/usr/bin/env python3
"""HQ step: link each guideline chapter to the MSF research sitting behind it.

This collection holds both sides of the evidence chain — 70 clinical guideline chapters and 215
peer-reviewed articles — but nothing connects them. A clinician reading a recommendation cannot
see the study that produced it, and a researcher cannot see whether their trial reached practice.

The link is built from shared RARE vocabulary, not shared topic words. A term appearing in 300
documents ("patients", "treatment") says nothing; a term in five ("practecal", "bedaquiline",
"artesunate") is a strong signal that two documents concern the same clinical question. Each link
carries the terms that produced it, so a human can judge it rather than trust a score.

    python scripts/build_trail.py --index build/index.db --out data/trail.json

Honest about what this is: lexical co-occurrence, not a citation graph. The documents do not cite
each other in machine-readable form. Treat a link as "these concern the same question", which is
why the UI shows the shared terms and never claims "this study caused this recommendation".
"""
import argparse
import json
import math
import re
import sqlite3
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msf_search.textnorm import STOPWORDS, tokenize  # noqa: E402

GUIDELINE = {"Clinical Guideline Chapter", "Study Protocol"}

# Some PDFs lose their title to front matter (HANDOFF: "PDF title extraction is heuristic").
# A document whose title is a licence sentence cannot be shown as evidence for anything.
BAD_TITLE = ("this is an open access", "the author ", "the authors ", "for the french translation",
             "editorial committee", "all rights reserved", "this article is distributed",
             "open access this article", "copyright ")
# ...and some lose it to a URL fragment, e.g. "p://www.elsevier.com/locate/inhe".
BAD_TITLE_RE = re.compile(r"^\s*(?:h?t?t?p?s?://|p://|www\.|doi:|\d{1,3}\s*$)", re.I)


def unusable_title(t: str) -> bool:
    t = (t or "").strip()
    return len(t) < 12 or t.lower().startswith(BAD_TITLE) or bool(BAD_TITLE_RE.match(t))
RESEARCH = {"Peer-reviewed Article", "Scientific Publication Notice", "Research Policy",
            "Scientific News", "Research Governance Document"}

DF_MIN = 2        # a term in one document links nothing
DF_MAX = 60       # ...and a term in 60+ of 561 is background vocabulary
TOP_TERMS = 220   # strongest terms kept per document
MAX_LINKS = 3     # ranks 4-5 were consistently weaker than rank 1-3
MIN_SCORE = 0.055   # below this the matches are author names and page furniture, not clinical terms
MIN_SHARED = 3      # one coincidental rare word is not an evidence link

# Rare words that are rare for the wrong reason: front matter, author blocks, licence boilerplate.
# Without this, two papers "match" because they share a copy-editor's name.
NON_CLINICAL = set("""
editorial committee contributors contributor layout acknowledgements acknowledgments declared
conflict interest interests correspondence corresponding affiliation affiliations author authors
reviewer reviewers copyright licence license creative commons distributed reproduction medium
provided cited citation permissions email orcid appendix annex figure figures table tables
version versions updates updated latest publication published publisher journal volume issue
manuscript preprint supplementary supplemental attribution noncommercial redistribution
typeset proofs guest downloaded downloads endpoint subscriber subscription reprints
institutional repository metadata
""".split())


def doc_terms(con):
    meta, counts = {}, {}
    for r in con.execute("SELECT id, genre, language, title, year FROM docs WHERE n_chunks>0"):
        meta[r[0]] = {"id": r[0], "genre": r[1] or "Other", "language": r[2] or "??",
                      "title": (r[3] or r[0])[:140], "year": r[4]}
    memo: dict[str, str] = {}
    for doc_id, blob in con.execute("SELECT doc_id, text FROM chunks WHERE kind='text'"):
        if doc_id not in meta:
            continue
        c = counts.setdefault(doc_id, Counter())
        for t in tokenize(zlib.decompress(blob).decode("utf-8", "replace")):
            if len(t) < 4 or t in STOPWORDS or t in NON_CLINICAL or t.isdigit():
                continue
            c[memo.setdefault(t, t)] += 1
    return meta, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="build/index.db")
    ap.add_argument("--out", default="data/trail.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    con = sqlite3.connect(f"file:{a.index}?mode=ro", uri=True)
    meta, counts = doc_terms(con)
    n = len(counts)

    df = Counter()
    for c in counts.values():
        df.update(c.keys())

    # Keep each document's rarest, most specific vocabulary.
    profile: dict[str, dict] = {}
    for doc_id, c in counts.items():
        scored = {t: (1 + math.log(f)) * math.log(n / df[t])
                  for t, f in c.items() if DF_MIN <= df[t] <= DF_MAX}
        top = sorted(scored.items(), key=lambda kv: -kv[1])[:TOP_TERMS]
        norm = math.sqrt(sum(v * v for _, v in top)) or 1.0
        profile[doc_id] = {t: v / norm for t, v in top}

    guides = [d for d in profile if meta[d]["genre"] in GUIDELINE]
    research = [d for d in profile if meta[d]["genre"] in RESEARCH]

    # Inverted index over the research side so this stays fast without numpy.
    posting: dict[str, list] = {}
    for d in research:
        for t in profile[d]:
            posting.setdefault(t, []).append(d)

    links = {}
    for g in guides:
        acc: Counter = Counter()
        shared: dict[str, list] = {}
        for t, w in profile[g].items():
            for d in posting.get(t, ()):
                contrib = w * profile[d][t]
                acc[d] += contrib
                shared.setdefault(d, []).append((t, contrib))
        out = []
        for d, score in acc.most_common(MAX_LINKS * 6):
            if score < MIN_SCORE or len(shared[d]) < MIN_SHARED:
                continue
            if unusable_title(meta[d]["title"]):
                continue
            ranked = sorted(shared[d], key=lambda kv: -kv[1])
            # One dominant term carrying the whole score is a coincidence, not a shared subject.
            if ranked[0][1] > 0.6 * score:
                continue
            terms = [t for t, _ in ranked[:6]]
            out.append({"doc_id": d, "score": round(score, 4), "shared_terms": terms,
                        "title": meta[d]["title"], "genre": meta[d]["genre"],
                        "year": meta[d]["year"], "language": meta[d]["language"]})
            if len(out) >= MAX_LINKS:
                break
        if out:
            links[g] = out

    guide_list = [{"id": g, "title": meta[g]["title"], "genre": meta[g]["genre"],
                   "language": meta[g]["language"], "year": meta[g]["year"],
                   "n_links": len(links.get(g, []))}
                  for g in sorted(guides, key=lambda d: -len(links.get(d, [])))]

    out = {"built_at": time.strftime("%Y-%m-%d %H:%M"),
           "n_guidelines": len(guides), "n_research": len(research),
           "n_linked": len(links),
           "method": "shared rare vocabulary (document frequency 2-60 of %d), cosine over "
                     "IDF-weighted term profiles. Lexical co-occurrence, NOT a citation graph." % n,
           "guidelines": guide_list, "links": links}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(guides)} guideline chapters, {len(research)} research documents, "
          f"{len(links)} linked, {time.perf_counter() - t0:.1f}s -> {p} "
          f"({p.stat().st_size / 1024:.0f} KB)")
    for g in guide_list[:6]:
        if not g["n_links"]:
            continue
        print(f"\n  {g['id']} [{g['language']}] {g['title'][:62]}")
        for l in links[g["id"]][:3]:
            print(f"      {l['score']:.3f}  {l['doc_id']} ({l['year'] or '?'}) {l['title'][:48]}")
            print(f"              shared: {', '.join(l['shared_terms'])}")


if __name__ == "__main__":
    main()
