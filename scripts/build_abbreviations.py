"""Preprocessing: mine "long form (ABBR)" definitions from the collection -> data/abbreviations.csv
   python scripts/build_abbreviations.py --data "D4GC data" --csv data/documentation.csv"""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msf_search.abbrev import mine_abbreviations
from msf_search.ingest import load_collection

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder containing Parsed/ and Original/")
ap.add_argument("--csv", default=None, help="documentation.csv with id,genre,language")
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/abbreviations.csv"))
ap.add_argument("--min-count", type=int, default=2)
a = ap.parse_args()

docs = [d for d in load_collection(a.data, a.csv) if d.quality != "boilerplate" and not d.duplicate_of]
rows = mine_abbreviations((d.text for d in docs), min_count=a.min_count)
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["abbr", "expansions", "count", "docs"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} abbreviations from {len(docs)} documents -> {a.out}")
for r in rows[:25]:
    print(f"  {r['abbr']:8} {r['count']:4}x in {r['docs']:3} docs  {r['expansions']}")
