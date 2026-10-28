"""HQ preprocessing: transcribe low-text PDFs (scans, slide decks) and inbox photos (handwritten notes).
   python scripts/digitise.py --data "D4GC data"              # auto: VLM if VLM_* env vars are set, else tesseract
   python scripts/digitise.py --data "D4GC data" --list       # only show what would be transcribed
Then rebuild the index: build_index.py picks up data/transcriptions/ and data/inbox/ automatically.
Optional cost report for the VLM: env VLM_PRICE_IN / VLM_PRICE_OUT in USD per 1M tokens."""
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.digitise import digitise, find_inbox_images, find_low_text_pdfs, make_engine, pdf_pages

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder containing Parsed/ and Original/")
ap.add_argument("--inbox", default=str(ROOT / "data/inbox"), help="photos / scans (jpg, png) to add as new docs")
ap.add_argument("--out", default=str(ROOT / "data/transcriptions"))
ap.add_argument("--threshold", type=int, default=300, help="parsed chars per page below which a PDF is re-transcribed")
ap.add_argument("--ids", default="", help="comma-separated PDF ids to transcribe regardless of the threshold")
ap.add_argument("--engine", choices=["auto", "vlm", "tesseract"], default="auto")
ap.add_argument("--langs", default="eng+fra", help="tesseract languages")
ap.add_argument("--dpi", type=int, default=150)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

items = find_low_text_pdfs(a.data, a.threshold)
for i in [x.strip() for x in a.ids.split(",") if x.strip()]:
    pdf = Path(a.data) / "Original" / f"{i}.pdf"
    if i not in {x["id"] for x in items} and pdf.exists():
        items.append({"id": i, "path": pdf, "pages": pdf_pages(pdf), "chars_per_page": None})
items += find_inbox_images(a.inbox)
print(f"{len(items)} items, {sum(x['pages'] for x in items)} pages:")
for x in items:
    cpp = f"{x['chars_per_page']} chars/page parsed" if x.get("chars_per_page") is not None else "inbox / forced"
    print(f"  {x['id']:24} {x['pages']:4} pages  ({cpp})")
if a.list or not items:
    sys.exit(0)

engine = make_engine(a.engine, a.langs)
print(f"engine: {engine.source} · {engine.model}" + (f" · missing languages: {engine.missing}" if engine.missing else ""))
Path(a.out).mkdir(parents=True, exist_ok=True)
t0, pages, unreadable = time.time(), 0, 0
for x in items:
    meta = digitise(x, engine, a.out, dpi=a.dpi, workers=a.workers)
    if meta is None:
        print(f"  {x['id']:24} skipped (already transcribed)")
        continue
    pages += meta["pages"]
    unreadable += meta["unreadable_count"]
    print(f"  {x['id']:24} {meta['pages']:4} pages  {meta['seconds']:6.1f} s  {meta['unreadable_count']} [?]")
secs = time.time() - t0
cost = "USD 0 (local OCR)" if engine.source == "tesseract" else f"tokens {engine.tokens}"
if engine.source == "vlm" and os.environ.get("VLM_PRICE_IN") and os.environ.get("VLM_PRICE_OUT"):
    usd = (engine.tokens["prompt"] * float(os.environ["VLM_PRICE_IN"])
           + engine.tokens["completion"] * float(os.environ["VLM_PRICE_OUT"])) / 1e6
    cost += f" ≈ USD {usd:.2f}"
print(f"done: {pages} pages in {secs:.0f} s ({secs / max(pages, 1):.1f} s/page), {unreadable} unreadable words, cost: {cost}")
