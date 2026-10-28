"""HQ preprocessing: transcribe low-text PDFs (scans, slide decks) and inbox photos (handwritten notes).
   python scripts/digitise.py --data "D4GC data"              # auto: VLM if VLM_* env vars are set, else tesseract
   python scripts/digitise.py --data "D4GC data" --list       # only show what would be transcribed
Then rebuild the index: build_index.py picks up data/transcriptions/ and data/inbox/ automatically.
Optional cost report for the VLM: env VLM_PRICE_IN / VLM_PRICE_OUT in USD per 1M tokens."""
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msf_search.digitise import digitise, find_inbox_images, find_low_text_pdfs, make_engines, pdf_pages

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

tess, vlm = make_engines(a.engine, a.langs)
print(f"engines: tesseract={tess.model if tess else 'off'}" + (f" (missing languages: {tess.missing})" if tess and tess.missing else "")
      + f" · vlm={vlm.model if vlm else 'not configured (VLM_BASE_URL/VLM_API_KEY/VLM_MODEL)'}")
Path(a.out).mkdir(parents=True, exist_ok=True)
t0, pages, unreadable = time.time(), 0, 0
for x in items:
    meta = digitise(x, tess, vlm, a.out, dpi=a.dpi, workers=a.workers)
    if meta is None:
        print(f"  {x['id']:24} skipped (already transcribed)")
        continue
    pages += meta["pages"]
    unreadable += meta["unreadable_count"]
    conf = f"conf {meta['ocr_confidence']:.2f}" if meta["ocr_confidence"] is not None else "conf n/a"
    print(f"  {x['id']:24} {meta['pages']:4} pages  {meta['seconds']:6.1f} s  {meta['unreadable_count']} [?]  {conf}  engines {meta['engines']}")
secs = time.time() - t0
cost = f"VLM tokens {vlm.tokens}" if vlm else "USD 0 (local OCR only)"
if vlm and os.environ.get("VLM_PRICE_IN") and os.environ.get("VLM_PRICE_OUT"):
    usd = (vlm.tokens["prompt"] * float(os.environ["VLM_PRICE_IN"])
           + vlm.tokens["completion"] * float(os.environ["VLM_PRICE_OUT"])) / 1e6
    cost += f" ≈ USD {usd:.2f}"
print(f"done: {pages} pages in {secs:.0f} s ({secs / max(pages, 1):.1f} s/page), {unreadable} unreadable words, cost: {cost}")
