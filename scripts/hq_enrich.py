"""HQ: enrich every document with one LLM call (summary, multilingual keywords, importance, grounded Q&A).
   Default: local Ollama, mistral:7b-instruct (open weights). Switch provider with env LLM_BASE_URL/LLM_API_KEY/LLM_MODEL.
   python scripts/hq_enrich.py --data "D4GC data" --csv data/documentation.csv --limit 5     # test
   python scripts/hq_enrich.py --data "D4GC data"            # default: the 17-doc demo sample (resumable)
   python scripts/hq_enrich.py --data "D4GC data" --all      # whole collection, same code
Order: guidelines first, then by evidence tier and length, so a partial run covers what matters most."""
import argparse, json, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.enrich import LLM, enrich_doc
from msf_search.ingest import load_collection
from msf_search.search import GENRE_TIER

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--csv", default=str(ROOT / "data/documentation.csv"))
ap.add_argument("--transcriptions", default=str(ROOT / "data/transcriptions"))
ap.add_argument("--inbox", default=str(ROOT / "data/inbox"))
ap.add_argument("--out", default=str(ROOT / "data/enrichment.jsonl"))
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="only the first N documents in priority order")
ap.add_argument("--ids", default="", help="comma-separated ids to enrich")
ap.add_argument("--sample", action="store_true", help="the demo sample (DEMO_SAMPLE) instead of the whole collection")
ap.add_argument("--all", action="store_true", help="the whole collection (hours locally, minutes via an API)")
a = ap.parse_args()
# Demo sample: the documents behind the pitch queries (PPH, cholera, malnutrition, snakebite, TB, measles...).
DEMO_SAMPLE = ("P23,D3a9dd210b9,D3f77693628,D0fdc1d0ef5,D5da0aed2fd,Dce112b1459,D580a93c4d4,D506fa6468a,"
               "D8bc477871b,Da8a79bb208,De081bb3a2d,D2ad107ffa2,P10,Dbb3e40c69c,D7dbb77c873,D160b2d96cd,D4ebb3e2295")
if not (a.ids or a.all or a.limit):
    a.sample = True
if a.sample:
    a.ids = DEMO_SAMPLE

llm = LLM.from_env()
if not llm.available():
    sys.exit(f"SKIPPED: no LLM reachable at {llm.url} (start `ollama serve`, or set LLM_BASE_URL/LLM_API_KEY/LLM_MODEL)")
done = set()
if Path(a.out).exists():
    done = {json.loads(ln)["id"] for ln in Path(a.out).read_text(encoding="utf-8").splitlines() if ln.strip()}
docs = [d for d in load_collection(a.data, a.csv, transcriptions_dir=a.transcriptions, inbox_dir=a.inbox)
        if d.quality != "boilerplate" and not d.duplicate_of]  # skip failed scrapes and duplicates
ids = {x.strip() for x in a.ids.split(",") if x.strip()}
todo = [d for d in docs if d.id not in done and (not ids or d.id in ids)]
todo.sort(key=lambda d: (GENRE_TIER.get(d.genre, ("", 9, 1))[1], len(d.text)))
if a.limit:
    todo = todo[:a.limit]
print(f"{len(docs)} searchable docs, {len(done)} already enriched, {len(todo)} to do · model {llm.model} · {a.workers} workers")

lock, stats, t0 = threading.Lock(), Counter(), time.time()
out = open(a.out, "a", encoding="utf-8")
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    futs = {pool.submit(enrich_doc, llm, d): d for d in todo}
    for f in as_completed(futs):
        d = futs[f]
        try:
            rec = f.result()
        except Exception as e:  # keep going; the doc is retried on the next run
            stats["failed"] += 1
            print(f"  FAILED {d.id}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            stats["ok"] += 1
            stats["qa_kept"] += len(rec["qa"])
            stats["qa_generated"] += rec["qa_generated"]
            for k, v in rec["qa_dropped"].items():
                stats["dropped_" + k] += v
        print(f"  {d.id:14} [{d.genre[:22]:22}] imp={rec['importance']} {rec['audience'] or '?':11} "
              f"qa {len(rec['qa'])}/{rec['qa_generated']} kept  {rec['seconds']:5.0f} s", flush=True)
out.close()
print(f"done in {time.time() - t0:.0f} s: {dict(stats)} · tokens {dict(llm.tokens)}")
