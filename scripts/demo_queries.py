"""Print top results for a handful of realistic + off-topic queries (sanity check / demo)."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.search import Searcher

QUERIES = [
    "treatment of severe acute malnutrition in children",
    "cholera outbreak response",
    "snakebite antivenom",
    "paludisme traitement enfant",
    "attacks on medical workers in Lebanon",
    "financial report 2024 expenditure",
    "tuberculosis diagnosis Xpert",
    "measles vaccination campaign",
    "sexual violence care",
    "كوليرا",
    "genocide definition",
    # off-topic: should abstain
    "bitcoin price prediction",
    "football champions league final",
    "best pizza recipe",
    "kubernetes autoscaling",
]

ap = argparse.ArgumentParser()
ap.add_argument("--index", default=str(ROOT / "build/index.db"))
ap.add_argument("--glossary", default=str(ROOT / "data/glossary.csv"))
ap.add_argument("--abbrev", default=str(ROOT / "data/abbreviations.csv"), help='"" to disable')
ap.add_argument("q", nargs="*")
a = ap.parse_args()
s = Searcher(a.index, a.glossary, abbreviations_path=a.abbrev)
for q in (a.q or QUERIES):
    r = s.search(q, k=5)
    print(f"\n### {q}   [{r['ms']} ms]  {'ABSTAINED: ' + r['reason'] if r['abstained'] else ''}"
          f"  unknown={r.get('unknown_terms')}")
    for x in r["results"]:
        print(f"  {x['confidence']:6} {x['score']:.3f} cov={x['passage_coverage']} about={x['aboutness']} "
              f"{x['doc_id']} [{x['language']}|{x['tier']}|{x['genre']}|{x['year'] or '?'}] {x['title'][:60]}")
    for x in r.get("weaker_matches", [])[:2]:
        print(f"  (weaker, hidden) {x['doc_id']} {x['title'][:60]}")
