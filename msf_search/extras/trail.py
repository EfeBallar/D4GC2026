"""Evidence trail: the research behind a recommendation.

The collection holds both sides of the chain — guideline chapters and the peer-reviewed studies
MSF produced — but nothing connects them. This serves precomputed links so a clinician can tap a
recommendation and see the field research that concerns the same question.

Built at HQ by scripts/build_trail.py from shared rare vocabulary. Deliberately conservative:
a link needs at least three shared specialist terms, no single term may carry the score, and
front-matter vocabulary (author names, licence boilerplate) is excluded. 30 of 73 chapters end
up with no link at all, and the UI shows nothing rather than a weak suggestion — the same
principle as abstaining on a search it cannot answer.
"""
import json
from pathlib import Path

_cache: dict = {}


def payload(path: Path) -> dict:
    if not path.is_file():
        return {"error": "trail.json missing: run scripts/build_trail.py",
                "guidelines": [], "links": {}}
    stamp = path.stat().st_mtime_ns
    if _cache.get("stamp") != stamp:
        _cache.clear()
        _cache["stamp"] = stamp
        _cache["data"] = json.loads(path.read_text(encoding="utf-8"))
    return _cache["data"]
