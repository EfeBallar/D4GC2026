"""Ablation table on the hand-made eval set (data/eval_queries.csv).
   python scripts/ablation.py --data "D4GC data"          (run with .venv-hq/bin/python to include the vector row)
Rows: keyword (glossary + abbreviations + gate) / + enrichment / + Q&A / + question vectors / + importance.
The eval queries target the enriched DEMO SAMPLE, so the gains apply to those docs only."""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search import evalset
from msf_search.search import Searcher

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--eval", default=str(ROOT / "data/eval_queries.csv"))
a = ap.parse_args()
rows = evalset.load(a.eval)
B = ROOT / "build"


def build(out, *flags):
    subprocess.run([sys.executable, "scripts/build_index.py", "--data", a.data, "--csv", "data/documentation.csv",
                    "--out", str(out), *flags], cwd=ROOT, check=True, capture_output=True)
    return round(out.stat().st_size / 1e6, 1)


mb = {"kw": build(B / "abl_keyword.db", "--enrichment", ""),
      "enr": build(B / "abl_enrich.db", "--no-synthetic"),
      "full": build(B / "index.db")}
qvec = None
try:
    from msf_search.qvec import QVecIndex
    qvec = QVecIndex.load(str(B), str(ROOT / "models/potion-multilingual-128M"))
except ImportError:
    pass
g, ab = str(ROOT / "data/glossary.csv"), str(ROOT / "data/abbreviations.csv")
configs = [("keyword (glossary+abbrev+gate)", B / "abl_keyword.db", None, False, mb["kw"]),
           ("+ enrichment (keywords, topics)", B / "abl_enrich.db", None, False, mb["enr"]),
           ("+ Q&A / summary passages", B / "index.db", None, False, mb["full"]),
           ("+ question vectors", B / "index.db", qvec, False, mb["full"]),
           ("+ importance prior", B / "index.db", qvec, True, mb["full"])]
print(f"{len(rows)} hand-written queries ({sum(1 for r in rows if r['relevant'])} answerable, all about the "
      f"enriched demo sample; {sum(1 for r in rows if not r['relevant'])} unanswerable)\n")
print(f"| {'configuration':34} | R@5   | MRR   | FP rate | latency ms | bundle MB |")
print(f"|{'-' * 36}|-------|-------|---------|------------|-----------|")
out = []
for name, idx, qv, imp, size in configs:
    if "vectors" in name and qv is None:
        print(f"| {name:34} | SKIPPED (model2vec or build/qvec.npy missing)                |")
        continue
    s = Searcher(str(idx), g, abbreviations_path=ab, qvec=qv, use_importance=imp)
    r = evalset.score(s, rows)
    extra = f" +{(B / 'qvec.npy').stat().st_size / 1024:.0f} KB vectors, +506 MB model" if qv else ""
    print(f"| {name:34} | {r['recall@5']:.3f} | {r['mrr']:.3f} | {r['false_positive_rate']:7.2f} | "
          f"{r['latency_ms_median']:10.1f} | {size:5.1f}{extra} |")
    out.append({"config": name, **r, "bundle_mb": size})
(B / "ablation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
for f in ("abl_keyword.db", "abl_enrich.db"):
    (B / f).unlink(missing_ok=True)
print("\nmisses (best config):", out[-1]["misses"])
