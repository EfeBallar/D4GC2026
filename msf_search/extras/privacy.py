"""Checkpoint mode: built for the places MSF actually works.

Devices get searched at checkpoints and borders. A local history of what clinicians looked up —
sexual violence, a named condition, a specific patient's symptoms — can endanger staff and
patients. One action clears everything the device kept.

Be honest about the limit, in the UI and in Q&A: this removes the local log. It is not
anti-forensics and it is not a substitute for full-disk encryption, which is what actually
protects a seized laptop. What it does guarantee is that the feedback loop to HQ never carries
anything but anonymous counts: gaps.py logs the query text only — no user, no IP, no patient.
"""
from pathlib import Path


def status(log_path: Path) -> dict:
    exists = log_path.is_file()
    n = 0
    if exists:
        n = sum(1 for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    return {"log": str(log_path), "exists": exists, "entries": n,
            "note": "Clears the local search log. Full-disk encryption is what protects a "
                    "seized device; this is not anti-forensics."}


def clear(log_path: Path) -> dict:
    before = status(log_path)["entries"]
    if log_path.is_file():
        log_path.write_text("", encoding="utf-8")   # truncate, keep the append target valid
    return {"cleared": before, "entries": 0, "ok": True}
