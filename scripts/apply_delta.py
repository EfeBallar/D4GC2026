"""Field: merge an update file into the local library (idempotent; safe to run twice).
   python scripts/apply_delta.py build/update_2026-10-28-1500.db [--index build/index.db]
A running server picks the change up on its next request; no restart needed."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.delta import apply_delta

ap = argparse.ArgumentParser()
ap.add_argument("delta", help="update_<version>.db made by make_delta.py")
ap.add_argument("--index", default=str(ROOT / "build/index.db"), help="field index to update")
a = ap.parse_args()
kb = Path(a.delta).stat().st_size / 1024
r = apply_delta(a.index, a.delta)
print(f"{a.delta} ({kb:.1f} KB): {r['status']}. Added {r['added']}, replaced {r['replaced']} documents.")
print(json.dumps(r, ensure_ascii=False))
