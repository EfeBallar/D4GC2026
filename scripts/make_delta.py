"""HQ: export chosen documents from a full index into a small update file for the field.
   python scripts/make_delta.py --ids P23,Dc95675e997 [--index build/index.db] [--version 2026-10-28-1500]"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.delta import make_delta

ap = argparse.ArgumentParser()
ap.add_argument("--index", default=str(ROOT / "build/index.db"), help="full index to export from")
ap.add_argument("--ids", required=True, help="comma-separated doc ids")
ap.add_argument("--version", default=time.strftime("%Y-%m-%d-%H%M"))
ap.add_argument("--out", default=None, help="default: build/update_<version>.db")
a = ap.parse_args()
ids = [x.strip() for x in a.ids.split(",") if x.strip()]
r = make_delta(a.index, ids, a.out or str(ROOT / f"build/update_{a.version}.db"), a.version)
print(f"delta {r['version']}: {r['docs']} docs, {r['chunks']} passages -> {r['path']} ({r['kb']} KB)")
