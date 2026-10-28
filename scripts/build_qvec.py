"""HQ: embed the HQ-made questions + summaries (HyDE) with Model2Vec -> build/qvec.npy (int8).
   .venv-hq/bin/python scripts/build_qvec.py [--model models/potion-multilingual-128M]"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--index", default=str(ROOT / "build/index.db"))
ap.add_argument("--model", default=str(ROOT / "models/potion-multilingual-128M"))
ap.add_argument("--out", default=str(ROOT / "build"))
a = ap.parse_args()
try:
    from msf_search.qvec import build
except ImportError as e:
    sys.exit(f"SKIPPED: {e} (pip install model2vec in the HQ environment)")
if not Path(a.model).exists():
    sys.exit(f"SKIPPED: model folder {a.model} not found")
t0 = time.time()
r = build(a.index, a.model, a.out)
print(json.dumps({**r, "seconds": round(time.time() - t0, 1)}))
