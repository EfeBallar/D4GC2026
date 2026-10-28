"""HQ pipeline in one command: quality -> digitise -> abbreviations -> enrich -> index -> embed -> package.
   python scripts/hq_build.py --data "D4GC data"            [--enrich-all] [--skip-digitise] [--skip-enrich]
Every stage that lacks its tool, key or model prints "SKIPPED (reason)" and the build carries on:
the field library always builds, with or without the LLM layers."""
import argparse, json, shutil, statistics, subprocess, sys, time, zipfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.enrich import LLM
from msf_search.ingest import load_collection, quality_report

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--csv", default=str(ROOT / "data/documentation.csv"))
ap.add_argument("--enrich-all", action="store_true", help="enrich the whole collection, not the demo sample")
ap.add_argument("--skip-digitise", action="store_true")
ap.add_argument("--skip-enrich", action="store_true")
ap.add_argument("--api-seconds-per-doc", type=float, default=20.0, help="assumed hosted-API latency, for the estimate")
ap.add_argument("--api-workers", type=int, default=16)
a = ap.parse_args()
PY, HQPY = sys.executable, str(ROOT / ".venv-hq/bin/python")
report = []


def stage(name):
    def wrap(fn):
        t0 = time.time()
        try:
            out = fn()
        except Exception as e:  # a failing optional stage must not stop the build
            out = f"FAILED ({type(e).__name__}: {e})"
        report.append((name, round(time.time() - t0, 1), out))
        print(f"[{name:13}] {time.time() - t0:6.1f} s  {out}", flush=True)
        return out
    return wrap


def run(*args, py=PY):
    r = subprocess.run([py, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "error")
    return r.stdout


@stage("quality")
def _():
    q = quality_report(load_collection(a.data, a.csv))
    return (f"{q['documents']} files · {q['exact_duplicates']} exact duplicates · {q['boilerplate_scrapes']} failed "
            f"scrapes · languages {q['languages']}")


@stage("digitise")
def _():
    if a.skip_digitise:
        return "SKIPPED (--skip-digitise)"
    if not shutil.which("tesseract") and not shutil.which("pdftoppm"):
        return "SKIPPED (no tesseract / pdftoppm installed)"
    run("scripts/digitise.py", "--data", a.data)
    metas = [json.loads(p.read_text()) for p in (ROOT / "data/transcriptions").glob("*.json")]
    engines = Counter()
    for m in metas:
        engines.update(m.get("engines", {}))
    confs = {m["original"]: m.get("ocr_confidence") for m in metas}
    return (f"{len(metas)} items · {sum(m['pages'] for m in metas)} pages · engines {dict(engines)} · "
            f"confidence {confs} · {sum(m['unreadable_count'] for m in metas)} [?]")


@stage("abbreviations")
def _():
    return run("scripts/build_abbreviations.py", "--data", a.data, "--csv", a.csv).splitlines()[0]


@stage("enrich")
def _():
    path = ROOT / "data/enrichment.jsonl"
    llm = LLM.from_env()
    if a.skip_enrich:
        note = "SKIPPED (--skip-enrich), using existing data/enrichment.jsonl"
    elif not llm.available():
        note = f"SKIPPED (no LLM reachable at {llm.base}), using existing data/enrichment.jsonl"
    else:
        run("scripts/hq_enrich.py", "--data", a.data, *(["--all"] if a.enrich_all else []), "--workers", "4", "--long-doc", "3000")
        note = f"model {llm.model}"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    n_docs = sum(1 for d in load_collection(a.data, a.csv) if d.quality != "boilerplate" and not d.duplicate_of)
    secs = statistics.mean(r["seconds"] for r in recs) if recs else 0
    kept, gen, dropped = sum(len(r["qa"]) for r in recs), sum(r["qa_generated"] for r in recs), Counter()
    for r in recs:
        dropped.update(r["qa_dropped"])
    local_min = (n_docs - len(recs)) * secs / 2 / 60  # measured s/doc, 2 parallel requests on this laptop
    api_min = n_docs * a.api_seconds_per_doc / a.api_workers / 60
    return (f"Enrichment: {len(recs)}/{n_docs} docs ({'demo sample' if not a.enrich_all else 'all'}) · {note} · "
            f"Q&A kept {kept}/{gen}, dropped {dict(dropped)} · importance {dict(Counter(r['importance'] for r in recs))} · "
            f"full run = same command with --enrich-all, est. {local_min:.0f} min locally "
            f"(measured {secs:.0f} s/doc) / {api_min:.0f} min via API (assumed {a.api_seconds_per_doc:.0f} s/doc, "
            f"{a.api_workers} parallel) · cost USD 0 (local open-weight model)")


@stage("index")
def _():
    out = run("scripts/build_index.py", "--data", a.data, "--csv", a.csv)
    r = json.loads(out[out.index("{"):])
    return (f"{r['chunks']} passages + {r.get('synthetic_chunks', 0)} HQ summary/Q&A passages · "
            f"{r.get('enriched_docs', 0)} enriched docs · {r['index_mb']} MB · {r['build_seconds']} s")


@stage("embed")
def _():
    if not Path(HQPY).exists():
        return "SKIPPED (no .venv-hq with model2vec)"
    if not (ROOT / "models/potion-multilingual-128M").exists():
        return "SKIPPED (model folder models/potion-multilingual-128M missing)"
    r = json.loads(run("scripts/build_qvec.py", py=HQPY).strip().splitlines()[-1])
    return f"{r.get('vectors', 0)} question/summary vectors · {r.get('kb', 0)} KB"


@stage("package")
def _():
    info = json.loads(subprocess.run([PY, "-c", "import sqlite3,sys;c=sqlite3.connect('build/index.db');"
                                      "print(c.execute(\"select value from meta where key='library_version'\").fetchone()[0])"],
                                     cwd=ROOT, capture_output=True, text=True).stdout.strip().join('""'))
    pkg = ROOT / f"build/msf_library_{info}.zip"
    files = [ROOT / "build/index.db", ROOT / "build/qvec.npy", ROOT / "build/qvec_ids.json",
             ROOT / "data/glossary.csv", ROOT / "data/abbreviations.csv", ROOT / "data/profiles.json"]
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                z.write(f, f.relative_to(ROOT))
    return f"{pkg.name} · {pkg.stat().st_size / 1e6:.1f} MB (index + vectors + glossary; model folder ships separately)"


print("\nHQ BUILD REPORT")
for name, secs, out in report:
    print(f"  {name:13} {secs:6.1f} s  {out}")
