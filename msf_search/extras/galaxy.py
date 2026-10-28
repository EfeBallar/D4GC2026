"""The library as a map: every document a point, clusters labelled by their own top terms.

Coordinates are precomputed at HQ by scripts/build_galaxy.py and ship inside the bundle, so the
field device only reads a small JSON file. Nothing is computed on site.

What it is for: a search box shows you what the library HAS. A map also shows what it LACKS,
because a thin or empty region is a subject MSF has not written down in that language.
"""
import json
from pathlib import Path

_cache: dict = {}


def payload(path: Path) -> dict:
    """Read data/galaxy.json once, then serve from memory (reloaded if the file changes)."""
    if not path.is_file():
        return {"error": "galaxy.json missing: run scripts/build_galaxy.py", "docs": [], "clusters": []}
    stamp = path.stat().st_mtime_ns
    if _cache.get("stamp") != stamp:
        _cache.clear()
        _cache["stamp"] = stamp
        _cache["data"] = json.loads(path.read_text(encoding="utf-8"))
    return _cache["data"]
