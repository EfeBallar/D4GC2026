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
ap.add_argument("--site-profile", default="", help="profile id in data/profiles.json: its topics raise urgency")
a = ap.parse_args()
ids = [x.strip() for x in a.ids.split(",") if x.strip()]
site = set()
if a.site_profile:
    import json
    site = set(next(p for p in json.loads((ROOT / "data/profiles.json").read_text())["profiles"]
                    if p["id"] == a.site_profile)["topics"])
r = make_delta(a.index, ids, a.out or str(ROOT / f"build/update_{a.version}.db"), a.version, site_topics=site)
print(f"delta {r['version']}: {r['docs']} docs, {r['chunks']} passages -> {r['path']} ({r['kb']} KB) · urgency {r['urgency']}")
