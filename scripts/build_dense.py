"""OPTIONAL preprocessing step. Needs: pip install model2vec  (or sentence-transformers for e5)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msf_search.dense import build, DEFAULT_MODEL
ap = argparse.ArgumentParser()
ap.add_argument("--index", default="build/index.db")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--out-name", default="dense", help="tag for the vector files, e.g. e5 or m2v")
ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
build(a.index, a.model, a.out_name, a.batch)
