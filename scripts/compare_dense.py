"""Measured trade-off: keyword core vs keyword + embedding model(s).

Step 1 (run):    python scripts/compare_dense.py --dense e5 m2v
   -> writes build/judge_sheet.csv : one row per (query, document) returned by ANY system
      (pooled, so each pair is judged once), plus latency / RAM / model-size stats.
Step 2 (judge):  open the CSV, put 1 (relevant) or 0 in the 'relevant' column. ~15 min as a team.
Step 3 (score):  python scripts/compare_dense.py --score build/judge_sheet.csv
   -> Precision@5 per system, overall and for cross-lingual / paraphrase queries -> pitch slide.
"""
import argparse, csv, json, re, statistics, sys, time, tracemalloc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [  # (query, type)
    ("treatment of severe acute malnutrition in children", "keyword"),
    ("cholera outbreak response", "keyword"),
    ("snakebite antivenom", "keyword"),
    ("tuberculosis diagnosis Xpert", "keyword"),
    ("measles vaccination campaign", "keyword"),
    ("care for survivors of sexual violence", "keyword"),
    ("mental health support for refugees", "keyword"),
    ("malaria in pregnant women", "keyword"),
    ("burn wound management", "keyword"),
    ("kids not getting enough food", "paraphrase"),
    ("disease spread by mosquitoes treatment", "paraphrase"),
    ("doctors killed in airstrikes", "paraphrase"),
    ("how to stop watery diarrhoea epidemic in a camp", "paraphrase"),
    ("prise en charge du choléra", "cross-lingual"),
    ("paludisme traitement enfant", "cross-lingual"),
    ("malnutrition aiguë sévère", "cross-lingual"),
    ("desnutrición infantil", "cross-lingual"),
    ("violencia sexual atención", "cross-lingual"),
    ("الكوليرا", "cross-lingual"),
    ("سوء التغذية عند الأطفال", "cross-lingual"),
    ("الهجمات على المستشفيات", "cross-lingual"),
]
OFF_TOPIC = ["bitcoin price prediction", "football champions league final", "best pizza recipe",
             "iphone battery replacement", "mortgage interest rates", "taylor swift tour tickets",
             "how to train a puppy", "javascript react hooks tutorial", "cheap flights to paris",
             "skincare routine for oily skin"]
K = 5
_TAG = re.compile(r"</?mark>")


def run(a):
    from msf_search.search import Searcher
    systems = {"keyword": None}
    sizes = {"keyword": 0.0}
    for tag in a.dense:
        from msf_search.dense import DenseIndex
        systems[tag] = DenseIndex.load(a.index, tag)
        sizes[tag] = round(Path(a.index).with_name(tag + ".npy").stat().st_size / 1e6, 1)
    pool, stats = {}, {}
    for name, dense in systems.items():
        s = Searcher(a.index, a.glossary, dense=dense, abbreviations_path=a.abbrev)
        s.search("warm up")
        lat, top_cos, off_cos, leaks = [], [], [], []
        for q, _ in QUERIES:  # timing pass (no tracing overhead)
            t0 = time.perf_counter()
            s.search(q, k=K, strict=True)
            lat.append((time.perf_counter() - t0) * 1000)
        for q in OFF_TOPIC:   # false-positive pass: these should return NOTHING
            r = s.search(q, k=K, strict=True)
            if not r["abstained"]:
                leaks.append(q)
            if dense is not None and dense.last_abs:
                off_cos.append(max(dense.last_abs.values()))
        tracemalloc.start()
        for q, qtype in QUERIES:
            r = s.search(q, k=K, strict=True)
            if dense is not None and dense.last_abs:
                top_cos.append(max(dense.last_abs.values()))
            for rank, x in enumerate(r["results"][:K], 1):
                row = pool.setdefault((q, x["doc_id"]), {
                    "query": q, "type": qtype, "doc_id": x["doc_id"], "language": x["language"],
                    "genre": x["genre"], "title": x["title"], "snippet": _TAG.sub("", x["snippet"])[:300],
                    "relevant": ""})
                row[f"rank_{name}"] = rank
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats[name] = {"latency_ms_median": round(statistics.median(lat), 1),
                       "latency_ms_max": round(max(lat), 1),
                       "python_heap_peak_mb": round(peak / 1e6, 1),
                       "vector_file_mb": sizes[name],
                       "off_topic_leaked": f"{len(leaks)}/{len(OFF_TOPIC)}", "leaked_queries": leaks,
                       "model": getattr(dense, "model_name", "none (SQLite FTS5 only)")}
        if top_cos:
            # Calibration: pick --dense-high between the off-topic max and the on-topic median.
            stats[name]["top1_cosine_on_topic_min/median/max"] = [round(min(top_cos), 3),
                                                                  round(statistics.median(top_cos), 3),
                                                                  round(max(top_cos), 3)]
            stats[name]["top1_cosine_off_topic_median/max"] = [round(statistics.median(off_cos), 3),
                                                               round(max(off_cos), 3)]
    out = Path(a.out)
    cols = ["query", "type", "doc_id", "language", "genre", "title", "snippet", "relevant"] + \
           [f"rank_{n}" for n in systems]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sorted(pool.values(), key=lambda r: (r["query"], r["doc_id"])):
            w.writerow({c: row.get(c, "") for c in cols})
    Path(out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n{len(pool)} (query, doc) pairs to judge -> {out}\nFill 'relevant' with 1/0, then: "
          f"python scripts/compare_dense.py --score {out}")


def score(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    systems = [c[5:] for c in rows[0] if c.startswith("rank_")]
    types = sorted({r["type"] for r in rows}) + ["ALL"]
    print(f"Precision@{K} (unjudged rows count as not relevant)")
    print(f"{'system':12}" + "".join(f"{t:>15}" for t in types))
    for s in systems:
        line = f"{s:12}"
        for t in types:
            qs = [q for q, qt in QUERIES if t == "ALL" or qt == t]
            p = []
            for q in qs:
                hits = [r for r in rows if r["query"] == q and r.get(f"rank_{s}")]
                p.append(sum(1 for r in hits if r["relevant"].strip() == "1") / K)
            line += f"{statistics.mean(p):>15.2f}"
        print(line)
    for s in systems:
        empty = sum(1 for q, _ in QUERIES if not any(r["query"] == q and r.get(f"rank_{s}") for r in rows))
        print(f"{s}: returned nothing for {empty}/{len(QUERIES)} queries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "build/index.db"))
    ap.add_argument("--glossary", default=str(ROOT / "data/glossary.csv"))
    ap.add_argument("--abbrev", default=str(ROOT / "data/abbreviations.csv"), help='"" to disable')
    ap.add_argument("--dense", nargs="*", default=[], help="vector tags built with build_dense.py")
    ap.add_argument("--out", default=str(ROOT / "build/judge_sheet.csv"))
    ap.add_argument("--score", default=None)
    a = ap.parse_args()
    score(a.score) if a.score else run(a)
