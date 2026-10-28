# MSF Field Library: HTTP API

Served by `python -m msf_search.server` (standard library, no internet). All responses are JSON (UTF-8)
unless noted. **The contract is additive only: fields are never renamed or removed.** New fields are marked
with the step that added them.

## Label scan

`POST /api/scan` accepts raw image bytes (maximum 10 MB). `POST /api/scan/text` accepts
`{"text": "ORS"}` (maximum 2000 characters / 8192 request bytes) to search a user-confirmed
or corrected printed name without another photo. The latter returns `input_source: "user"`.

Both return `ok`, `terms`, `results` and, on a search, `query`, `abstained`, `ms`.
Image responses also contain `ocr` and `bytes`. OCR below the existing confidence gate
does not trigger a search. Recognised concepts are searched without incidental ingredient
or packaging words; dotted/spaced known abbreviations such as `O.R.S.` are supported.
The phone converts decodable photos to bounded JPEGs before uploading. With Pillow installed,
the server honours EXIF orientation, normalises contrast and size, and checks four rotations
in parallel. On macOS, the native decoder also handles HEIC uploads. Otherwise unsupported
formats return a request to choose JPEG/PNG. OCR responses include available/missing languages.

Scan result excerpts come from original indexed passages, with their provenance, rather
than generated summaries. `reference_only: true` marks detected abbreviation/contents lists;
the UI labels these as reference material, not treatment instructions. Related documents
do not establish the photographed product's identity, strength or preparation instructions.

## Provenance object (on every doc, passage and result)

```json
{"text_source": "parsed" | "ocr_tesseract" | "ocr_vlm",
 "ocr_confidence": 0.93,          // 0-1, tesseract mean word confidence; null if not OCR or VLM-only
 "machine_generated": false,      // true for HQ-made summaries / Q&A: the UI MUST label these
 "generator": null,               // "summary" | "qa" | null
 "model": "tesseract 5.5.1 (eng)",// OCR engine or LLM that produced the text; null for parsed text
 "generated_at": "2026-10-28 15:20:49"}
```
UI rule: never show machine-made text without its label (see `static/features.js`):
- summary: "AI summary (made at HQ) · check the source"
- Q&A: "AI-prepared answer (made at HQ from this document) · check the source"
- OCR: "Transcribed from scan/handwriting (<engine>, confidence X) · check the original" + link to the
  original at `#page=N`.
- OCR with `ocr_confidence` < 0.6: also "Low-confidence transcription: check the original" + photo/PDF link.

## GET /api/search

Query params: `q`, `k` (default 10), `lang` (comma list, e.g. `fr,es`), `genre` (`|` list),
`strict` (`1` default: hide weak matches; `0` show them), `topic` (v2 step 4: restrict to a topic; with an
empty `q` it BROWSES the topic, ordered importance -> evidence tier -> share of matching passages; browse
results add `topic_share`). Responses echo `"topic"`.

```json
{"query": "Orissa cyclone",
 "concepts": [{"label": "orissa", "weight": 5.6, "variants": ["orissa"]}],
 "unknown_terms": [],               // query terms found nowhere in the library
 "results": [{
   "doc_id": "D0eebba4c05", "score": 0.453, "confidence": "high",   // high | medium | low
   "passage_coverage": 1.0, "doc_coverage": 1.0, "aboutness": 0.04, "chunk_id": 18210,
   "title": "Published in October 2000 by Médecins Sans Frontiéres", "genre": "Financial Report",
   "language": "en", "also_filed_as": [], "original_name": "D0eebba4c05.pdf",
   "heading": "Page 59", "snippet": "A <mark>cyclone</mark> that struck India's <mark>Orissa</mark> ...",
   "year": 2000,                     // step: evidence tiers (null = unknown)
   "tier": "Background", "tier_rank": 6,   // evidence tier label and rank (1 = Guideline)
   "transcribed": true,              // text came from OCR
   "provenance": {...},              // v2 step 0: provenance of the passage shown
   "doc_provenance": {...},          // v2 step 0: provenance of the document
   "page": 59,                       // v2 step 0: page of the passage (OCR'd PDFs), else null
   "matched_via": "text",            // v2 step 2: text | summary | qa | qvec (step 5) = kind of the best passage
   "text_chunk_id": 18210,           // v2 step 2: best ORIGINAL passage of this doc
   "evidence": null,                 // v2 step 2: when matched_via != text, the original passage behind it:
                                     //   {"chunk_id", "heading", "page", "snippet", "provenance"}
   "enriched": true,                 // v2 step 2: HQ enrichment exists for this doc (fields below may be null)
   "importance": 1,                  //   1 essential field reference, 2 useful, 3 background, 4 archive
   "importance_reason": "...", "audience": "clinical", "actionable": true,
   "summary": "...",                 //   MACHINE-MADE: always show with summary_provenance / label
   "summary_provenance": {"text_source": "parsed", "machine_generated": true, "generator": "summary",
                          "model": "mistral:7b-instruct", ...},
   "topics": ["measles", "vaccination"], "countries": ["Chad"], "situation": "outbreak"
 }],
 "quick_answer": null,               // v2 step 2: set when the top hit is an HQ Q&A pair:
                                     //   {"doc_id", "title", "question", "answer", "evidence_quote",
                                     //    "evidence": {...original passage...}, "original_name",
                                     //    "provenance": {...generator "qa"...}, "label": "AI-prepared answer ..."}
 "weaker_matches": [...],            // same shape; hidden by default in the UI
 "abstained": false,                 // true = "No reliable match"
 "reason": "", "ms": 2.6}
```

## GET /api/profiles (v2 step 6)
Field profiles from `data/profiles.json` (situation presets; country and topics editable). A profile BOOSTS,
never filters: score × (1 + 0.15 per matching topic (max 3) + 0.10 for a preferred language), after the gate.
```json
{"profiles": [{"id": "nurse_cholera_outbreak", "label": "Nurse · cholera outbreak", "situation": "outbreak",
               "country": "", "topics": ["cholera", "diarrhoea", ...], "languages": ["en", "fr"]}, ...]}
```
Use with `/api/search?...&profile=<id>`: results get `profile_boost` (e.g. 0.4) and the response gets
`"profile": {"id", "label", "boosted": 10}`. UI: "Boosted for your profile (turn off)".
Delta updates: `make_delta.py --site-profile <id>` raises the urgency of docs on that site's topics
(normal -> high -> critical; clinical guidelines are critical); `apply_delta.py` applies critical first.

## GET /api/topics (v2 step 4)
Topics = HQ enrichment topics ∪ curated glossary concepts a doc mentions ≥ 3 times (works without any LLM).
```json
{"topics": [{"topic": "malaria", "count": 54, "enriched": 1, "from_glossary": true}, ...]}
```
Each result also carries `topics` (merged) and `glossary_topics`.

## GET /api/doc/<id>
Document metadata plus full `text`, including `provenance`.

## GET /api/info
```json
{"report": {"documents": 582, "languages": {...}, "genres": {...}, "index_mb": 33.1, ...},
 "library_version": "2026-10-28-1530", "n_docs": 561, "index_mb": 33.1,
 "importance_distribution": {"1": 17},   // enriched docs per importance level; the UI hides the badge
                                         // while only one level occurs
 "llm": false, "originals": true}
```

## GET /original/<file>
The original PDF / HTML / inbox image. Append `#page=N` to open a PDF at a page.

## GET /gaps, GET /api/gaps
Searches with no reliable match (logged to `logs/unanswered.jsonl`):
```json
{"unanswered": 6, "distinct_queries": 4, "top_unknown_terms": [["kubernetes", 1]],
 "top_queries": [{"query": "dengue vaccine for adults in brazil", "count": 3, "last_seen": "2026-10-28T15:12:30"}],
 "recent": [{"ts": "...", "query": "...", "unknown_terms": [], "library_version": "...", "filters": {"lang": ["ar"]}}]}
```

## GET /static/<file>
Static assets. `/static/features.js` holds the UI rendering helpers (pure functions returning HTML):
`renderQuickAnswer(r)`, `renderProvenance(x)`, `renderImportance(x)`, `renderSummary(x)`, `renderEvidence(x)`,
`renderTopicChips(topics)`, `renderProfilePicker(profiles)`, also on `window.MSFFeatures`.

Ranking notes: confidence (coverage) first; inside a confidence level, score × evidence tier × importance
prior (importance applied after the gate). Every doc appears once; summary/Q&A passages count for their doc.

## POST /api/chat (on-device RAG, Server-Sent Events)
Body `{"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}`. Runs the strict search
first; if it abstains the LLM is NOT called. Stream of events:
- `meta`: `{"sources": [card...], "model": "mistral-field", "model_label": "mistral 7B, 4-bit", "llm_called": true,
  "retrieval_ms": 44}`. A source card: `{"n", "doc_id", "chunk_id", "title", "display_title", "type", "genre",
  "language", "year", "heading", "page", "location" ("p. 59" | section), "quote" (one line of the passage used),
  "original_name", "tier", "confidence", "provenance"}`
- `token`: `{"text": "..."}` (repeated; answer cites sources as [1], [2])
- `error`: `{"error": "..."}` if the local model is unreachable (search keeps working)
- `done`: `{"llm_called", "ttft_ms", "total_ms", "tokens", "model", "unsupported_terms"}`. unsupported_terms =
  numbers / rare terms in the answer that are not in the passages (show as a warning)
Refusal (abstain): one token "Not found in the MSF library. Not found anywhere: <terms>." and `llm_called: false`.
UI label (required): "AI answer generated on this device by a small offline model (<model_label>) from MSF
documents · check the sources". features.js: `renderSourceCards(sources, id)`, `linkCitations(text, sources, id)`.
