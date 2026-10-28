"""Quick, label-free evaluation (no relevance judgments were provided).

1) Known-item retrieval: for each document, sample a passage, take 3 distinctive words
   from it as the query, and check whether the source document is returned.
   -> Recall@1/5/10, MRR, latency.  (Optimistic for lexical search: query words are
   copied from the document. Treat as a smoke test, not a final score.)
2) False-positive guard: off-topic queries should return NOTHING. -> abstention rate.
"""
import argparse, json, math, random, re, sqlite3, statistics, sys, zlib
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.search import Searcher
from msf_search.textnorm import STOPWORDS, tokenize

OFF_TOPIC = ["bitcoin price prediction", "football champions league final", "best pizza recipe",
             "kubernetes autoscaling", "iphone battery replacement", "stock market dividend tax",
             "wedding dress shopping", "formula one race calendar", "taylor swift tour tickets",
             "how to train a puppy", "mortgage interest rates", "playstation game release",
             "javascript react hooks tutorial", "cheap flights to paris", "lasagna recipe oven",
             "netflix series recommendations", "car engine oil change", "real estate investment",
             "guitar chords for beginners", "skincare routine for oily skin"]

ap = argparse.ArgumentParser()
ap.add_argument("--index", default=str(ROOT / "build/index.db"))
ap.add_argument("--glossary", default=str(ROOT / "data/glossary.csv"))
ap.add_argument("--abbrev", default=str(ROOT / "data/abbreviations.csv"), help='"" to disable')
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--terms", type=int, default=3)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--eval", default=None, help="hand-made eval set, e.g. data/eval_queries.csv")
a = ap.parse_args()
random.seed(a.seed)

s = Searcher(a.index, a.glossary, abbreviations_path=a.abbrev)
con = sqlite3.connect(a.index)
chunks = con.execute("SELECT c.doc_id, c.text FROM chunks c").fetchall()
by_doc = {}
for d, t in chunks:
    by_doc.setdefault(d, []).append(t)
df = Counter()
for d, ts in by_doc.items():
    df.update(set(tok for t in ts for tok in tokenize(zlib.decompress(t).decode())))
N = len(by_doc)

items = []
for d in random.sample(sorted(by_doc), min(a.n, len(by_doc))):
    cands = [zlib.decompress(t).decode() for t in by_doc[d]]
    cands = [c for c in cands if len(c.split()) >= 60] or cands
    text = random.choice(cands)
    toks = [t for t in tokenize(text) if t not in STOPWORDS and len(t) >= 4 and t.isalpha()]
    tf = Counter(toks)
    ranked = sorted(tf, key=lambda t: -tf[t] * math.log(N / (1 + df[t])))[:15]
    if len(ranked) < a.terms:
        continue
    items.append((d, " ".join(random.sample(ranked, a.terms))))

ranks, lat, abst = [], [], 0
for d, q in items:
    r = s.search(q, k=10, strict=True)
    lat.append(r["ms"])
    ids = [x["doc_id"] for x in r["results"]]
    abst += r["abstained"]
    ranks.append(ids.index(d) + 1 if d in ids else None)
found = [x for x in ranks if x]
res = {
    "known_item_queries": len(items),
    "recall@1": round(sum(1 for x in found if x <= 1) / len(items), 3),
    "recall@5": round(sum(1 for x in found if x <= 5) / len(items), 3),
    "recall@10": round(len(found) / len(items), 3),
    "mrr@10": round(sum(1 / x for x in found) / len(items), 3),
    "abstained_on_answerable": round(abst / len(items), 3),
    "latency_ms_median": round(statistics.median(lat), 1),
    "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
}
off = [s.search(q, k=10, strict=True) for q in OFF_TOPIC]
res["off_topic_queries"] = len(OFF_TOPIC)
res["off_topic_correctly_abstained"] = round(sum(r["abstained"] for r in off) / len(OFF_TOPIC), 3)
res["off_topic_leaks"] = {r["query"]: [x["doc_id"] for x in r["results"][:3]] for r in off if not r["abstained"]}
res["index_mb"] = s.report.get("index_mb")
if a.eval:
    from msf_search import evalset
    res["eval_set"] = evalset.score(s, evalset.load(a.eval))
print(json.dumps(res, indent=2, ensure_ascii=False))
