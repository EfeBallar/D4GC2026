#!/usr/bin/env python3
"""HQ step: lay all documents out in 2D so the field UI can draw the library as a map.

Runs off-premises, like every other heavy step, and writes one small file (data/galaxy.json)
that ships inside the bundle. No model download, no network: tf-idf over the same tokeniser the
index uses, k-means for topic clusters, and a two-component PCA by power iteration for the
coordinates. All standard library.

    python scripts/build_galaxy.py --index build/index.db --out data/galaxy.json

The point of the picture is not decoration: clusters are labelled by their own top terms, so a
sparse or missing region is a visible gap in what MSF has written down.
"""
import argparse
import json
import math
import random
import sqlite3
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msf_search.textnorm import STOPWORDS, tokenize  # noqa: E402

MIN_DF = 5            # a term must appear in at least this many documents
MAX_DF_RATIO = 0.40   # ...and in no more than this share of them
VOCAB = 1200
K = 6                 # topic clusters
ITERS = 30


def load_docs(con):
    """Real text only: machine-made summary/Q&A passages would bend the map toward HQ's words."""
    meta = {r[0]: {"id": r[0], "genre": r[1] or "Other", "language": r[2] or "??",
                   "title": r[3] or r[0], "year": r[4], "n_chunks": r[5]}
            for r in con.execute("SELECT id, genre, language, title, year, n_chunks "
                                 "FROM docs WHERE n_chunks>0")}
    memo: dict[str, str] = {}
    counts: dict[str, Counter] = {}
    for doc_id, blob in con.execute("SELECT doc_id, text FROM chunks WHERE kind='text'"):
        if doc_id not in meta:
            continue
        text = zlib.decompress(blob).decode("utf-8", "replace")
        c = counts.setdefault(doc_id, Counter())
        for t in tokenize(text):
            if len(t) < 3 or t in STOPWORDS or t.isdigit():
                continue
            c[memo.setdefault(t, t)] += 1
    return meta, counts


def build_vectors(counts, n_docs, meta):
    """Vocabulary is chosen per language, then unioned.

    A single corpus-wide document-frequency cut is biased by the language mix: this collection is
    80% English, so an English threshold keeps almost no Arabic term, every Arabic document ends up
    with an empty vector, and all 51 of them land on one identical point. Choosing terms within each
    language and taking the union gives every language its own axes to spread along.
    """
    df = Counter()
    for c in counts.values():
        df.update(c.keys())

    by_lang: dict[str, list[str]] = {}
    for doc_id in counts:
        by_lang.setdefault(meta[doc_id]["language"], []).append(doc_id)

    vocab_set: set[str] = set()
    for lang, docs in by_lang.items():
        dfl = Counter()
        for d in docs:
            dfl.update(counts[d].keys())
        n_l = len(docs)
        lo = MIN_DF if n_l > 60 else 2          # small language, lower bar or nothing survives
        hi = MAX_DF_RATIO * n_l
        share = max(250, int(VOCAB * n_l / n_docs))
        # Take this language's own top terms. Ranking the merged set by GLOBAL frequency
        # afterwards would simply delete them again, which is how Arabic collapsed before.
        vocab_set.update([t for t, n in dfl.most_common() if lo <= n <= hi][:share])

    vocab = sorted(vocab_set, key=lambda t: -df[t])
    idx = {t: i for i, t in enumerate(vocab)}
    idf = [math.log(n_docs / df[t]) for t in vocab]

    vecs = {}
    for doc_id, c in counts.items():
        v = {}
        for t, n in c.items():
            i = idx.get(t)
            if i is not None:
                v[i] = (1.0 + math.log(n)) * idf[i]
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[doc_id] = {i: x / norm for i, x in v.items()}
    return vocab, vecs


def kmeans(ids, vecs, dim, k=K, iters=ITERS, seed=7):
    rng = random.Random(seed)
    # k-means++ so the clusters are stable between builds
    centroids = [dict(vecs[rng.choice(ids)])]
    for _ in range(k - 1):
        d2 = []
        for i in ids:
            best = max((sum(vecs[i].get(j, 0.0) * c.get(j, 0.0) for j in vecs[i]) for c in centroids),
                       default=0.0)
            d2.append(max(0.0, 1.0 - best) ** 2)
        total = sum(d2) or 1.0
        r, acc = rng.random() * total, 0.0
        pick = ids[-1]
        for i, w in zip(ids, d2):
            acc += w
            if acc >= r:
                pick = i
                break
        centroids.append(dict(vecs[pick]))

    assign = {i: 0 for i in ids}
    for _ in range(iters):
        moved = 0
        for i in ids:
            v = vecs[i]
            best, bi = -2.0, 0
            for ci, c in enumerate(centroids):
                s = sum(x * c.get(j, 0.0) for j, x in v.items())
                if s > best:
                    best, bi = s, ci
            if assign[i] != bi:
                moved += 1
            assign[i] = bi
        new = [Counter() for _ in range(k)]
        sizes = [0] * k
        for i in ids:
            new[assign[i]].update(vecs[i])
            sizes[assign[i]] += 1
        for ci in range(k):
            if sizes[ci]:
                n = sizes[ci]
                c = {j: x / n for j, x in new[ci].items()}
                norm = math.sqrt(sum(x * x for x in c.values())) or 1.0
                centroids[ci] = {j: x / norm for j, x in c.items()}
        if not moved:
            break
    return assign, centroids


def pca2(ids, vecs, dim, seed=11):
    """Top two principal components of the centred tf-idf matrix, by power iteration.

    Centring would destroy sparsity, so the mean is kept separate and applied as a correction:
    (X - m) v  ==  X v - (m . v).
    """
    m = [0.0] * dim
    for i in ids:
        for j, x in vecs[i].items():
            m[j] += x
    n = len(ids)
    m = [x / n for x in m]

    def project(v):
        mv = sum(m[j] * v[j] for j in range(dim))
        return [sum(x * v[j] for j, x in vecs[i].items()) - mv for i in ids]

    def transpose(u):
        w = [0.0] * dim
        for i, ui in zip(ids, u):
            if ui:
                for j, x in vecs[i].items():
                    w[j] += ui * x
        su = sum(u)
        return [w[j] - su * m[j] for j in range(dim)]

    rng = random.Random(seed)
    comps = []
    for _ in range(2):
        v = [rng.gauss(0, 1) for _ in range(dim)]
        for _ in range(60):
            for c in comps:                                  # keep components orthogonal
                d = sum(a * b for a, b in zip(v, c))
                v = [a - d * b for a, b in zip(v, c)]
            w = transpose(project(v))
            for c in comps:
                d = sum(a * b for a, b in zip(w, c))
                w = [a - d * b for a, b in zip(w, c)]
            nw = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / nw for x in w]
        comps.append(v)
    return project(comps[0]), project(comps[1])


def spread(values, lo=2.0, hi=98.0):
    """Scale to [-1, 1] on robust percentiles, so a few outliers cannot squash the picture."""
    s = sorted(values)
    n = len(s) - 1
    a, b = s[int(n * lo / 100)], s[int(n * hi / 100)]
    span = (b - a) or 1.0
    return [max(-1.0, min(1.0, 2.0 * (v - a) / span - 1.0)) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="build/index.db")
    ap.add_argument("--out", default="data/galaxy.json")
    ap.add_argument("--clusters", type=int, default=K)
    a = ap.parse_args()

    t0 = time.perf_counter()
    con = sqlite3.connect(f"file:{a.index}?mode=ro", uri=True)
    meta, counts = load_docs(con)
    ids = sorted(counts)
    if not ids:
        sys.exit("no text passages found: build the index first")
    vocab, vecs = build_vectors(counts, len(ids), meta)
    dim = len(vocab)
    assign, centroids = kmeans(ids, vecs, dim, k=a.clusters)
    xs, ys = pca2(ids, vecs, dim)
    xs, ys = spread(xs), spread(ys)

    kept = {ci for ci in range(a.clusters) if sum(1 for i in ids if assign[i] == ci) >= 5}
    if kept:
        biggest = max(kept, key=lambda ci: sum(1 for i in ids if assign[i] == ci))
        for i in ids:
            if assign[i] not in kept:
                assign[i] = biggest

    clusters = []
    for ci, c in enumerate(centroids):
        if ci not in kept:
            continue
        top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
        members = [i for i in ids if assign[i] == ci]
        langs = Counter(meta[i]["language"] for i in members)
        clusters.append({"id": ci, "terms": [vocab[j] for j, _ in top],
                         "label": ", ".join(vocab[j] for j, _ in top[:3]),
                         "size": len(members), "languages": dict(langs)})

    docs = [{"id": i, "x": round(x, 4), "y": round(y, 4), "cluster": assign[i],
             "language": meta[i]["language"], "genre": meta[i]["genre"],
             "year": meta[i]["year"], "chunks": meta[i]["n_chunks"],
             "title": (meta[i]["title"] or i)[:120]}
            for i, x, y in zip(ids, xs, ys)]

    out = {"built_at": time.strftime("%Y-%m-%d %H:%M"), "n_docs": len(docs),
           "vocab_size": dim, "clusters": clusters, "docs": docs,
           "method": "tf-idf over the index tokeniser, k-means topic clusters, "
                     "2-component PCA by power iteration (standard library only)"}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"{len(docs)} documents, vocab {dim}, {a.clusters} clusters, "
          f"{time.perf_counter() - t0:.1f}s -> {p} ({p.stat().st_size / 1024:.0f} KB)")
    for c in clusters:
        print(f"  cluster {c['id']}: {c['size']:>3} docs  {c['label']}")


if __name__ == "__main__":
    main()
