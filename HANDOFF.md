# HANDOFF: MSF offline retrieval (D4GC 2026). Read this first.

**Deadline:** upload the pitch PDF and the code ZIP by 18:00, aiming for 17:30. Pitch at 18:30, 5 minutes max.
**Scoring (brief p.7):** effectiveness (find relevant documents, few false positives), creativity, implementability
(low storage and compute, poor internet, no vendor lock-in), cost-effectiveness, presentation.

## 0. Ground rules for whoever continues (human or Claude Code)
- **Don't re-scaffold.** The core in this repo works end to end on the full dataset. Extend it, don't replace it.
- **No LLM on site.** LLMs are used only in preprocessing (allowed off-premises). The field device runs Python's
  standard library plus SQLite. Pitch line: *"We moved the G in RAG to before deployment."*
- Every change is checked with `python scripts/evaluate.py` and `python scripts/demo_queries.py` before moving on.
- Feature freeze at 17:15.

## 1. Current state (measured on the full collection)
```
python scripts/build_index.py --data "../D4GC data" --csv data/documentation.csv   # 5 s -> build/index.db (32 MB)
python -m msf_search.server --originals "../D4GC data/Original"                     # UI on :8000
python scripts/demo_queries.py ["query" ...]   ·   python scripts/evaluate.py
```
| What | Value |
|---|---|
| Documents in the collection | 582. 561 searchable: 3 exact duplicates merged (P16, P42, P44), 18 failed scrapes excluded (identical MSF Access landing page) |
| Passages | ~18.7k (about 180 words each; most files have no headings, so paragraphs are packed) |
| Index | one SQLite file, 32 MB (text zlib-compressed + FTS5) |
| RAM / latency | ~20 MB RSS, ~3 ms median per query on 1 CPU core (natural queries 5–30 ms) |
| Known-item test (300) | Recall@1 0.97, Recall@5 1.00 (optimistic: query words come from the document) |
| Off-topic queries abstained | 18/20 (the leaks were "mortgage interest rates" and "real estate investment", which hit the financial reports) |
| Cross-lingual | "hémorragie du post-partum" / "hemorragia posparto" → P23 (English PPH guideline) #1. "كوليرا" finds Arabic and Spanish documents |

**How ranking works (`msf_search/search.py`):**
1. The query is split into concepts. `data/glossary.csv` expands each one into EN/FR/ES/AR variants.
2. FTS5 BM25 runs over passages (title ×4, heading ×2, body ×1).
3. Each passage gets an IDF-weighted **concept coverage** score. A query term found **nowhere** in the library gets
   the maximum weight, so "bitcoin price" abstains.
4. Each document gets an **aboutness** score: the share of its passages that match. This stops 900-page reports
   that only mention a topic.
5. Confidence is strong / partial / weak. Weak results are hidden, and with none left the app shows "No reliable match".

**Optional layers, written but not verified with real models (my environment had no model downloads):**
- `dense.py`: Model2Vec or sentence-transformers. It was only tested with a fake encoder, so the code paths run.
- `compare_dense.py`: produces a pooled judge sheet, then Precision@5 per system plus an off-topic leak count.
- `llm.py`: Ollama answer endpoint. **We don't plan to demo it.**

## 2. Tasks, in priority order (owner suggestions for 4 people)

> **Status 14:50:** tasks 1 and 4 done, plus the live online/offline pill and delta updates (see README).
> Rebuild the index (`build_index.py`) before running anything: the `docs` table has a new `year` column.
> Task 4 finding: the "pph treatment oxytocin" gap is **not** an abbreviation problem. "pph" already
> expands, but P23 never mentions oxytocin. Demo with "pph treatment" or "uterine atony" instead.
>
> **Status 15:15:** task 5 (gap log, `/gaps`, `scripts/gaps.py`) done. New HQ step `scripts/digitise.py`
> (see README): D0eebba4c05 and De35232409a transcribed with tesseract. Demo query "Orissa cyclone" finds
> D0eebba4c05 page 59, and only does so after digitising. Still missing: `data/inbox/field_note.jpg`. Handwriting
> needs `VLM_*` env vars (same API-key owner as task 3).

### P0: by 15:45
1. **[A] Evidence-tier ranking** (`search.py` + UI badge). Multiply each document's score by a genre prior:
   - Clinical Guideline Chapter 1.25
   - Peer-reviewed / Study Protocol / Research Policy 1.10
   - Activity Report / Evaluation 1.0
   - Press Release / Statement / Project Update / Voices / Interview 0.9
   - Financial Report / Timeline / Summary / Opinion / Op-Ed 0.8

   Show a tier badge and, if possible, the year (regex `\b(19|20)\d{2}\b`, taking the most frequent year in the
   first 2,000 characters). Make the weights an editable dict at the top of `search.py`. It fixes
   financial reports ranking high for clinical queries.
2. **[C] A real eval set: `data/eval_queries.csv`** with columns `query,lang,type,relevant_ids` (ids separated by `|`).
   - 25 hand-written doctor-style queries in EN/FR/ES/AR, with abbreviations and a typo or two. Judge the relevant
     ids with the UI.
   - 10 unanswerable but plausible queries, e.g. "dengue vaccine for adults in Brazil" (check first that the
     library really doesn't cover it).
   - Optional: about 100 LLM-generated questions from random passages. Use a **different prompt or model** than the
     enrichment step to avoid leakage.

   Then extend `evaluate.py` with `--eval data/eval_queries.csv`, printing Recall@5, MRR, false-positive rate
   (unanswerable queries that returned results), and latency.
3. **[B] LLM per-document enrichment** (online preprocessing). Create `scripts/enrich_llm.py`:
   - Calls any OpenAI-compatible endpoint (base URL and key from env vars), one call per searchable document
     (~560 calls). Input is the first ~3k words plus the title, genre and language.
   - Writes JSON to `data/enrichment.jsonl` with fields: id, summary (2 sentences), keywords_en/fr/es/ar (10 each),
     abbreviations, questions (5), countries, diseases, year.
   - Resumable: skip ids already in the file. Run it concurrently, 8–16 workers.

   Then in `index.py`, add an `enrich` column to `chunks_fts` (keywords and questions, repeated on every passage of
   the document, BM25 weight ~1.5) and a `summary` column in `docs`. Show the summary in the UI.
   **Go / no-go at 16:15:** if it isn't finished, ship with the glossary only.
4. **[A] Abbreviation dictionary, no LLM needed:**
   - Regex the corpus for `Long Form (ABBR)` patterns, e.g. `([A-Z][A-Za-z\- ]{3,60}?) \(([A-Z]{2,8})\)`. Keep pairs
     seen at least twice and check that the initials match.
   - Write them to `data/abbreviations.csv` and load them into the glossary so "SAM" expands to "severe acute
     malnutrition".
   - Known gap: "pph treatment oxytocin" currently hides P23 as a weak match, because "pph" isn't in P23's
     text. Abbreviations plus enrichment should fix that, so check it.
5. **[D] No-result feedback log:** when a query abstains, append `{ts, query, unknown_terms}` to
   `logs/unanswered.jsonl` in `server.py`. Add a small "Gaps" view at `/gaps` or a CLI. Pitch: *"field searches that
   fail tell HQ what to add in the next build."*

### P1: by 16:45
6. **[B] Dense test**, only if P0 is done:
   1. `pip install model2vec sentence-transformers`
   2. `python scripts/build_dense.py --model minishlab/potion-multilingual-128M --out-name m2v`
      (check the current model name on huggingface.co/minishlab)
   3. `python scripts/build_dense.py --model intfloat/multilingual-e5-small --out-name e5`
   4. `python scripts/compare_dense.py --dense m2v e5`, then judge the sheet and run `--score`.

   Use the printed on-topic and off-topic cosine statistics to set `--dense-high`. Keep a model only if it improves
   cross-lingual or paraphrase precision **without** extra off-topic leaks; otherwise it goes on a slide as a
   "measured and rejected" trade-off, which is also a good story.
7. **[C] Constrained-hardware run:**
   `docker run --rm --cpus=1 --memory=512m -v "$PWD":/app -w /app python:3.12-slim python scripts/evaluate.py`
   Record latency and RAM for the slide: "runs in 512 MB, one core, no internet."
8. **[C] Ablation table** with eval.py. Rows:
   1. BM25
   2. + glossary
   3. + coverage gate
   4. + evidence tiers
   5. + LLM enrichment
   6. + dense

   Columns: Recall@5, MRR, false-positive rate, latency, RAM, bundle MB. To disable a feature for a row, add flags
   to the Searcher.

### P2: from 16:45
9. **[D] Slides.** 5 minutes, see the outline below. Export to PDF.
10. **All:** rehearse the demo twice with Wi-Fi OFF. Zip the code (without `build/`) and upload both files by 17:30.

## 3. Five-minute pitch outline
1. **30 s, the problem:** a midwife in a clinic with no connection needs the PPH protocol now.
2. **90 s, live demo with Wi-Fi off:**
   - "hémorragie du post-partum" → English guideline P23, passage highlighted;
   - an abbreviation query;
   - "bitcoin price" → "No reliable match";
   - point at the 3 ms latency and the 32 MB size.
3. **60 s, architecture:** heavy work once at HQ (LLM keywords and summaries in 4 languages, dedupe, quality checks);
   the field gets one SQLite file plus the standard library. Data-quality findings: 18 failed scrapes, 3 duplicates.
4. **60 s, evidence:** the ablation table, plus "512 MB, 1 core, no internet".
5. **60 s, scale and cost:**
   - per-site bundles and delta updates ("updates travel like medicines");
   - the no-result feedback loop;
   - €0 per query;
   - SQLite is public domain, so there's no vendor lock-in.

## 4. Known limitations (be honest in Q&A)
- The known-item eval is optimistic. The hand-made eval set (task 2) is the real number.
- PDF title extraction is heuristic. Enrichment summaries improve it.
- Arabic gets light normalisation only, no full stemming.
- French and Spanish guideline chapters are probably translations of the English ones. They aren't grouped yet.
