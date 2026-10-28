## Team quickstart

```
git clone git@github.com:lilyehl/D4GC2026.git        # or https://github.com/lilyehl/D4GC2026.git
# put the "D4GC data" folder NEXT TO the repo (it is not in git): ../D4GC data/{Original,Parsed}
cd D4GC2026
python scripts/build_index.py --data "../D4GC data" --csv data/documentation.csv
python -m msf_search.server --originals "../D4GC data/Original"
# open http://localhost:8000
```
Read **HANDOFF.md** and **API.md** before changing anything, and work on your own branch
(`git checkout -b <you>-<feature>`). Never commit the dataset, `build/`, `*.db`, models or keys.

# MSF Field Library: offline document retrieval

> **Team: read [HANDOFF.md](HANDOFF.md) for the current state and the task list.**

Data 4 Good Challenge 2026 (Doctors Without Borders). This is a document retrieval tool that runs fully on-premises. It is
**one SQLite file plus standard-library Python**: no internet, no GPU, no vendor, no licence cost.

```
python scripts/build_index.py --data "D4GC data" --csv data/documentation.csv   # preprocessing, ~5 s
python -m msf_search.server --originals "D4GC data/Original"                     # field app on :8000
python scripts/demo_queries.py "cholera outbreak children"                       # CLI
python scripts/evaluate.py                                                        # quick metrics
python scripts/build_abbreviations.py --data "D4GC data" --csv data/documentation.csv   # mine "long form (ABBR)" -> data/abbreviations.csv
```

**Digitise anything** (HQ only; the field never runs OCR or a model):
```
python scripts/digitise.py --data "D4GC data" --list   # PDFs with < 300 parsed chars/page + photos in data/inbox/
python scripts/digitise.py --data "D4GC data"          # VLM if VLM_BASE_URL/VLM_API_KEY/VLM_MODEL are set, else tesseract
```
Writes `data/transcriptions/<id>.md` + `.json` (source, model, pages, unreadable `[?]` count); resumable per page.
`build_index.py` then uses them: a scanned PDF keeps its original text layer *plus* the transcription, and an inbox
photo becomes a "Field Note (handwritten)" document. The UI flags both "Transcribed from scan / handwriting: check
the original", with a link that opens the PDF at the matched page. Measured: 101 pages in 105 s with tesseract, USD 0.
Handwriting needs the VLM engine; tesseract only reads print reliably.

**Gap log**: searches with no reliable match are appended to `logs/unanswered.jsonl` (query, unknown terms,
filters, library version; no user or IP). See them at `/gaps`; at HQ merge site logs with `python scripts/gaps.py site1.jsonl site2.jsonl`.

**Delta updates** (ship a few documents instead of a new `index.db`):
```
python scripts/make_delta.py --ids P23,P24                    # HQ: -> build/update_<version>.db, ~40 KB for 2 guideline chapters
python scripts/apply_delta.py build/update_<version>.db       # field: one transaction, idempotent, running server picks it up
python scripts/build_index.py ... --exclude-ids P23,P24 --out build/field_base.db   # demo: a base library missing those docs
```
Demo: on the base library "uterine atony" says *No reliable match (not found anywhere: atony)*; after the
delta it finds the PPH guideline P23. Results after a delta are identical to a full rebuild (checked on 40 queries).
Requirements: **Python 3.10+ and nothing else** (SQLite FTS5 ships with Python). Optional extras are listed below.

## Numbers (full collection, 2-core laptop CPU)

| | |
|---|---|
| Documents in the collection | 582 (470 en, 51 ar, 44 fr, 17 es) |
| Searchable after cleaning | 561 (3 exact duplicates merged, 18 failed scrapes flagged) |
| Passages indexed | ~18,700 |
| Index size (text + search index) | **32 MB**, a single file (the original PDFs/HTML are 386 MB and optional) |
| Build time | ~5 s |
| Query latency | **2.6 ms median, 9 ms p95** |
| Known-item test (300 queries) | Recall@1 0.97, Recall@5 1.00, MRR 0.98 * |
| Off-topic queries correctly answered with "no reliable match" | 18/20 |

\* Label-free smoke test: the query words are sampled from the target document, so this is optimistic for keyword
search. Real relevance judgments from MSF staff would be the next step (see `scripts/evaluate.py`).

## How it works

```
             PREPROCESSING (off-premises, once)                    RETRIEVAL (on-premises, every query)
 Parsed/*.md + CSV ─► clean ─► dedupe/quality ─► passages ─► index.db ─► query ─► glossary expansion ─► BM25 over passages
                      (drop nav     (md5 dupes,     (~180 words,  (SQLite        (en/fr/es/ar terms)      (title > heading > body)
                       boilerplate)  failed scrapes) with heading) FTS5)                │
                                                                                         ▼
                                       documents + snippet + confidence  ◄─ aggregate ◄─ concept coverage + "aboutness"
                                       (or "no reliable match")              to docs      (false-positive guard)
```

1. **Cleaning** (`ingest.py`): strips images, links and markup. Lines that recur in 8 or more documents (menus, cookie
   banners) are removed as boilerplate. Exact duplicates are merged (P16 = D5da0aed2fd, P42 = Da68d33c442,
   P44 = Dbd6e2ef149). **18 "Statement" documents are the same scraped MSF Access landing page** (their originals too).
   They are flagged and kept out of the index, and should be re-harvested.
2. **Passages** (`chunker.py`): documents are split into passages of about 180 words that keep their section heading. A
   900-page financial report can't win just by being long, and the user sees the exact paragraph that matched.
3. **Normalisation** (`textnorm.py`): accent-insensitive (`epidémie` = `epidemie`). Arabic gets light normalisation:
   tashkeel removed, alef/ya/ta-marbuta unified, `ال/وال/بال` prefixes stripped. Porter stemming for English via FTS5.
4. **Multilingual glossary** (`data/glossary.csv`): "cholera" also searches *choléra, cólera, الكوليرا*. So an English
   query finds French, Spanish and Arabic documents without any ML model. **It's a CSV a medical coordinator can edit in
   Excel**: adapting to new terms needs no retraining.
5. **Ranking** (`search.py`): BM25 on passages (title ×4, heading ×2, body ×1) gives candidates. These are rescored by
   * **concept coverage**: the IDF-weighted share of the query's concepts found *in the same passage*;
   * **aboutness**: the share of the document's passages that match, so a document *about* malnutrition beats one
     that mentions it in passing.
6. **False-positive guard**: every result gets a confidence (strong / partial / weak). Weak matches are hidden by
   default. If a query term appears **nowhere** in the library, it gets the maximum weight, so "bitcoin price"
   returns *"No reliable match. Not found anywhere: bitcoin"* instead of 10 random documents.

## Optional layers (never required)

| Layer | What | How |
|---|---|---|
| Dense, cross-lingual | `intfloat/multilingual-e5-small` (MIT licence, CPU), float16 vectors next to `index.db` | `pip install sentence-transformers` then `python scripts/build_dense.py`, then run the server with `--dense` |
| Local AI answer | Small quantized LLM (e.g. Qwen3 1.7B Q4, ~1.1 GB) **only rephrases retrieved passages, with citations**, and says "Not found in the library" otherwise | Install Ollama, `ollama pull qwen3:1.7b`, then run the server with `--llm-model qwen3:1.7b` |

These degrade gracefully. Search-only works on any device. The kit laptop adds a local LLM. Nothing breaks when the
connection drops.

## Code map

```
msf_search/
  textnorm.py   tokenisation + accent/Arabic normalisation + stopwords (en/fr/es/ar)
  ingest.py     load md + CSV, clean, boilerplate removal, duplicate & quality flags
  chunker.py    heading-aware passages
  glossary.py   multilingual query expansion from data/glossary.csv (+ data/abbreviations.csv)
  abbrev.py     mine abbreviation definitions from the corpus (Schwartz-Hearst, no LLM)
  index.py      build the single-file SQLite/FTS5 index (zlib-compressed text)
  delta.py      make / apply small update files (new or revised documents)
  search.py     retrieval, coverage/aboutness scoring, confidence, abstention, snippets
  dense.py      OPTIONAL embeddings layer
  llm.py        OPTIONAL grounded answer via a local model (Ollama API)
  server.py     stdlib HTTP server: UI + JSON API (/api/search, /api/doc/<id>, /api/answer)
  static/index.html   the field app (no CDN, works on phones, RTL-aware)
scripts/ build_index.py · build_dense.py · demo_queries.py · evaluate.py
```

## Fit with the scoring criteria

* **Effectiveness**: passage-level BM25, multilingual expansion, coverage-based abstention (fewer false positives).
* **Creativity**: data-quality findings (duplicates, failed scrapes); an editable multilingual glossary instead of a
  black box; "no reliable match" as a feature.
* **Implementability**: one 32 MB file plus Python's standard library. Runs on a Raspberry Pi, an old laptop, or a USB
  stick. Updates = copy a new `index.db`.
* **Cost-effectiveness**: €0 per query, no cloud, no licences; rebuilding takes 5 s.
* **Vendor independence**: SQLite (public domain), Python, open models only as optional add-ons.

## Known limitations / next steps

* Without the dense layer, cross-lingual recall depends on glossary coverage. Extend the CSV, or build dense vectors.
* The French/Spanish "Clinical Guideline Chapter" files are likely translations of the English chapters. Grouping them
  into one result ("also available in FR/ES") is an easy next feature.
* Tune the confidence thresholds with real queries from field staff; log anonymised queries to grow the glossary.
