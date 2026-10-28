"""Preprocessing (run once, at HQ or on any laptop):  python scripts/build_index.py --data "D4GC data" --csv documentation.csv"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.index import build_index

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder containing Parsed/ and Original/")
ap.add_argument("--csv", default=None, help="documentation.csv with id,genre,language")
ap.add_argument("--out", default="build/index.db")
ap.add_argument("--max-words", type=int, default=180)
ap.add_argument("--exclude-ids", default="", help="comma-separated doc ids to leave out (base library for delta tests)")
ap.add_argument("--transcriptions", default=str(ROOT / "data/transcriptions"), help="digitise.py output")
ap.add_argument("--inbox", default=str(ROOT / "data/inbox"), help="photos that become Field Note docs once transcribed")
a = ap.parse_args()
build_index(a.data, a.csv, a.out, max_words=a.max_words,
            exclude_ids={x.strip() for x in a.exclude_ids.split(",") if x.strip()},
            transcriptions_dir=a.transcriptions, inbox_dir=a.inbox)
