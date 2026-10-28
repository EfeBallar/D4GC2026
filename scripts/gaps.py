"""HQ: what did field staff search for and not find? Merge the unanswered logs synced back from sites.
   python scripts/gaps.py logs/unanswered.jsonl [site2/unanswered.jsonl ...]"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.gaps import summarise

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="*", default=[str(ROOT / "logs/unanswered.jsonl")])
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
s = summarise(a.logs, a.top)
print(f"{s['unanswered']} unanswered searches, {s['distinct_queries']} distinct queries")
print("\nTerms the library has never seen (add documents or glossary rows for these):")
for t, n in s["top_unknown_terms"]:
    print(f"  {n:4}x  {t}")
print("\nMost repeated unanswered queries:")
for q in s["top_queries"]:
    print(f"  {q['count']:4}x  {q['query']}   (last {q['last_seen']})")
