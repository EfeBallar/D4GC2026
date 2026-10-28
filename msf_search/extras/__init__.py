"""Extra field features, deliberately kept out of the core.

Everything here is additive, so backend and UI work can land without merge conflicts:
no edits to search.py, index.py or the existing interface. server.py delegates with two lines

    if (r := extras.handle(u.path, qs, searcher)) is not None:
        return self._send(*r)

and each module below takes the ordinary Searcher and returns plain data. Standard library
only, no LLM, no network: these run on the same field device as the rest.
"""
import json
from pathlib import Path

from . import galaxy, privacy, radar, sms, trail

PKG = Path(__file__).resolve().parent
STATIC = PKG.parent / "static"
ROOT = PKG.parents[1]
DATA = ROOT / "data"

JSON = "application/json; charset=utf-8"
HTML = "text/html; charset=utf-8"


def _json(obj, code: int = 200):
    return code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), JSON


def _page(name: str):
    f = STATIC / name
    if not f.is_file():
        return _json({"error": f"{name} is missing"}, 404)
    return 200, f.read_bytes(), HTML


def handle(path: str, qs: dict, searcher):
    """Return (code, body, content_type) for an extras route, or None to fall through."""
    g = lambda k, d="": qs.get(k, [d])[0]
    if path == "/sms":
        return _page("sms.html")
    if path == "/api/sms":
        return _json(sms.reply(searcher, g("q")))
    if path == "/galaxy":
        return _page("galaxy.html")
    if path == "/api/galaxy":
        return _json(galaxy.payload(DATA / "galaxy.json"))
    if path == "/radar":
        return _page("radar.html")
    if path == "/api/radar":
        return _json(radar.payload(DATA / "radar_sim.json", ROOT / "logs" / "unanswered.jsonl"))
    if path == "/hq":
        return _page("hq.html")
    if path == "/trail":
        return _page("trail.html")
    if path == "/api/trail":
        return _json(trail.payload(DATA / "trail.json"))
    if path == "/api/checkpoint":
        log = ROOT / "logs" / "unanswered.jsonl"
        return _json(privacy.clear(log) if g("action") == "clear" else privacy.status(log))
    return None
