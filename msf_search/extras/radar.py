"""Every question the field asks is a signal.

The device already logs searches it could not answer (gaps.py). Those logs travel back to HQ with
the next sync. Aggregated across sites, a sudden rise in one subject is visible before any formal
report arrives: three sites searching "cholera" in the same week is worth an epidemiologist's
attention.

Two honest limits, both stated on the page itself:
  * Google Flu Trends overestimated influenza for years by reading search volume as incidence.
    This is a hint for a human epidemiologist, never a diagnosis or a case count.
  * The multi-site feed in this demo is SIMULATED, because the hackathon has one site. Every
    simulated record carries simulated=true and the UI marks it. Real device logs are shown
    separately and never mixed into the same series.
"""
import json
from collections import defaultdict
from pathlib import Path

MIN_COUNT = 5        # below this, a "spike" is noise
RATIO = 2.5          # ...and it must be this many times the trailing average
TRAIL = 4            # weeks of history used for the baseline


def _load_sim(path: Path) -> dict:
    if not path.is_file():
        return {"rows": [], "sites": [], "weeks": 0, "note": "no simulated feed"}
    return json.loads(path.read_text(encoding="utf-8"))


def _real(log_path: Path) -> dict:
    """What THIS device actually failed to answer. Real, local, and kept separate."""
    if not log_path.is_file():
        return {"entries": 0, "terms": []}
    terms: defaultdict = defaultdict(int)
    queries: defaultdict = defaultdict(int)
    n = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        n += 1
        q = (e.get("query") or "").strip().lower()
        if q:
            queries[q] += 1
        for t in e.get("unknown_terms") or []:
            terms[t] += 1
    # A query can abstain because a word is missing from the library, or because no single
    # document covers enough of it. Only the first kind yields "unknown terms", so fall back
    # to the query text or the panel looks empty for the more common case.
    top = sorted(terms.items(), key=lambda kv: -kv[1])[:10]
    if not top:
        top = sorted(queries.items(), key=lambda kv: -kv[1])[:10]
    return {"entries": n, "terms": [{"term": t, "count": c} for t, c in top]}


def _spikes(rows, weeks):
    """A term rises sharply at a site, measured against its own trailing average."""
    series = defaultdict(lambda: [0] * weeks)
    for r in rows:
        if 0 <= r["week"] < weeks:
            series[(r["site"], r["term"])][r["week"]] += r["count"]

    alerts = []
    for (site, term), counts in series.items():
        for w in range(1, weeks):
            window = counts[max(0, w - TRAIL):w]
            base = sum(window) / len(window) if window else 0.0
            c = counts[w]
            if c >= MIN_COUNT and c >= RATIO * max(base, 1.0):
                alerts.append({"site": site, "term": term, "week": w, "count": c,
                               "baseline": round(base, 1),
                               "ratio": round(c / max(base, 1.0), 1), "simulated": True})
    alerts.sort(key=lambda a: (-a["week"], -a["count"]))

    # The interesting case is the same subject rising at several sites in the same week.
    cross = defaultdict(list)
    for a in alerts:
        cross[(a["term"], a["week"])].append(a["site"])
    multi = [{"term": t, "week": w, "sites": sorted(s), "n_sites": len(s), "simulated": True}
             for (t, w), s in cross.items() if len(s) >= 2]
    multi.sort(key=lambda m: (-m["n_sites"], -m["week"]))
    return alerts, multi


def payload(sim_path: Path, log_path: Path) -> dict:
    sim = _load_sim(sim_path)
    rows, weeks = sim.get("rows", []), sim.get("weeks", 0)
    alerts, multi = _spikes(rows, weeks) if weeks else ([], [])

    totals: defaultdict = defaultdict(int)
    per_week: defaultdict = defaultdict(int)
    for r in rows:
        totals[(r["site"], r["week"])] += r["count"]
        per_week[r["week"]] += r["count"]

    return {
        "simulated": True,
        "note": sim.get("note", ""),
        "caveat": "A hint for an epidemiologist, not a diagnosis. Search volume is not incidence: "
                  "Google Flu Trends overestimated influenza for years by confusing the two.",
        "generated": sim.get("generated"),
        "weeks": weeks,
        "sites": sim.get("sites", []),
        "volume": [{"site": s, "week": w, "count": c} for (s, w), c in totals.items()],
        "per_week": [per_week.get(w, 0) for w in range(weeks)],
        "alerts": alerts[:40],
        "cross_site": multi[:10],
        "real_device_log": _real(log_path),
    }
