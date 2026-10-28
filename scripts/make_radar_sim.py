#!/usr/bin/env python3
"""Generate the SIMULATED multi-site query feed used by the radar demo.

There is only one site in this hackathon (your laptop), so cross-site early warning cannot be
shown from real data. This writes an obviously-labelled fixture: deterministic, seeded, and
marked simulated in every record, so nothing on screen can be mistaken for MSF data.

    python scripts/make_radar_sim.py --out data/radar_sim.json
"""
import argparse
import json
import random
import time
from pathlib import Path

SITES = [  # name, country, lat, lon  (MSF works in all of these; the numbers are invented)
    ("Adre", "Chad", 13.47, 22.20),
    ("Bentiu", "South Sudan", 9.26, 29.80),
    ("Goma", "DR Congo", -1.68, 29.22),
    ("Cox's Bazar", "Bangladesh", 21.43, 92.00),
    ("Port-au-Prince", "Haiti", 18.54, -72.34),
    ("Maiduguri", "Nigeria", 11.83, 13.15),
]
TOPICS = ["malaria", "measles", "malnutrition", "tuberculosis", "wound care",
          "post-partum haemorrhage", "snakebite", "cholera", "hepatitis E", "meningitis"]
WEEKS = 8
SPIKE_TERM = "cholera"
SPIKE_SITES = ("Adre", "Bentiu", "Goma")   # the cross-site signal the demo points at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/radar_sim.json")
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = []
    for site, country, lat, lon in SITES:
        for w in range(WEEKS):
            for topic in TOPICS:
                base = rng.randint(0, 4)
                if topic == SPIKE_TERM and site in SPIKE_SITES and w >= WEEKS - 2:
                    base += rng.randint(9, 16) * (w - (WEEKS - 3))
                if base:
                    rows.append({"site": site, "country": country, "lat": lat, "lon": lon,
                                 "week": w, "term": topic, "count": base, "simulated": True})

    out = {"simulated": True, "generated": time.strftime("%Y-%m-%d %H:%M"), "seed": a.seed,
           "weeks": WEEKS, "sites": [{"site": s, "country": c, "lat": la, "lon": lo}
                                     for s, c, la, lo in SITES],
           "note": "SIMULATED DATA. Invented query counts for a demonstration of cross-site "
                   "signal detection. Not MSF data and not an epidemiological claim.",
           "rows": rows}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} simulated rows, {len(SITES)} sites, {WEEKS} weeks -> {p}")


if __name__ == "__main__":
    main()
