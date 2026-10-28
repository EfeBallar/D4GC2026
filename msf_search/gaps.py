"""No-result feedback loop: every search the library could not answer is logged on the device.

When the site is online again, logs/unanswered.jsonl travels back to HQ, which sees what field staff
needed and did not find ("atony" 7x this week), and adds documents or glossary terms to the next
build or delta update. Field searches that fail tell HQ what to send next.
The log stays on the device; it holds the query text only (no user, no IP).
"""
import json
import threading
import time
from collections import Counter
from pathlib import Path

_lock = threading.Lock()


def log_unanswered(log_path: str, result: dict, library_version: str = "", filters: dict | None = None) -> None:
    if not result.get("abstained") or not result.get("query", "").strip():
        return
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "query": result["query"].strip(),
             "unknown_terms": result.get("unknown_terms") or [], "library_version": library_version}
    if filters:  # "cholera" abstaining only because of an Arabic-only filter is a different gap
        entry["filters"] = filters
    p = Path(log_path)
    with _lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summarise(paths, top: int = 25) -> dict:
    """Merge one or more unanswered.jsonl logs (one per site) into the HQ view."""
    entries = []
    for path in paths:
        p = Path(path)
        if p.exists():
            entries += [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    queries = Counter(e["query"].lower() for e in entries)
    terms = Counter(t for e in entries for t in e.get("unknown_terms", []))
    last = {}
    for e in entries:
        last[e["query"].lower()] = e["ts"]
    return {"unanswered": len(entries), "distinct_queries": len(queries),
            "top_unknown_terms": terms.most_common(top),
            "top_queries": [{"query": q, "count": n, "last_seen": last[q]} for q, n in queries.most_common(top)],
            "recent": entries[-top:][::-1]}
