# MSF Field Library

**Medical knowledge, available offline.**

A local document search and retrieval prototype built for the **Data 4 Good Challenge 2026** with a Médecins Sans Frontières document collection. Search across languages, open the matching source passage, and ask questions with citations—all from a laptop serving a phone or browser over local Wi-Fi.

The core is a SQLite index and Python's standard library. Local AI, photo scanning, and document enrichment are optional layers.

[Get started](#get-started) · [Features](#features) · [How it works](#how-it-works) · [Evaluation](#evaluation) · [Demo guide](DEMO.md) · [API](API.md)

## Features

- **Multilingual search.** English, French, Spanish, and Arabic query expansion through an editable glossary and abbreviation dictionary. Results include a source passage, document metadata, and match confidence.
- **Local AI chat.** An optional model served by Ollama answers from retrieved passages with citations. When retrieval finds no reliable match, the app declines without calling the model.
- **Scan a label.** Photograph printed medicine packaging to find related documents. Phone-photo conversion, orientation checks, contrast enhancement, and an editable name help recover difficult reads. Source excerpts and reference lists are identified separately.
- **SMS interface.** Preview short, source-linked replies with GSM-7/UCS-2 segment counts. An optional AI overview describes the source in the question's language. The demo does not connect to a live mobile network.
- **Library exploration.** Browse a document map, inspect vocabulary-based links between guidelines and research, and review unanswered searches in the HQ dashboard.
- **Portable updates.** Build the library once, then distribute the SQLite file or smaller document updates. Search stays local after deployment.

The interface works on phones and desktops, with a shared light/dark theme switch that remembers your choice.

## Get started

### 1. Clone and add the collection

Requires **Python 3.10+** with SQLite FTS5. Core search needs no Python packages.

```bash
git clone https://github.com/lilyehl/D4GC2026.git
cd D4GC2026
```

The source document collection is **not included**. Place it next to the repository:

```text
workspace/
├── D4GC2026/
└── D4GC data/
    ├── Original/    # original PDFs and HTML files
    └── Parsed/      # Markdown files, named by document ID
```

### 2. Build and run search

```bash
python3 scripts/build_index.py --data "../D4GC data" --csv data/documentation.csv
python3 -m msf_search.server --originals "../D4GC data/Original" --port 8767 --no-chat-warm
```

Open **http://localhost:8767**. A phone on the same Wi-Fi can use `http://<laptop-ip>:8767`.

Try `hémorragie du post-partum`, `pph treatment`, or `Orissa cyclone`. The query `bitcoin price` demonstrates the no-match response.

### 3. Add local AI (optional)

Install Ollama and download a model once while online:

```bash
ollama pull mistral:7b-instruct
```

From a fresh terminal session, with the search-only server above stopped, run:

```bash
./start.sh --port 8767 --model mistral:7b-instruct --check-chat
```

The launcher starts or reuses Ollama, loads the installed model, and starts the app. `--check-chat` checks a complete chat request for a cited answer. Keep the terminal open while using the app. After model installation and library preparation, this configuration works without internet access.

For the existing demo laptop with the `mistral-field` model installed, use `./start.sh --port 8767 --check-chat`.

### Photo scanning (optional)

Install Tesseract on the site server, including language packs for the labels you expect to read. Pillow enables image preprocessing:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install Pillow
```

Start the server from that environment and open `/scan`. The browser prepares supported phone photos for upload; the server reads the text. A configured vision model can assist difficult reads. Photos are processed in temporary files, and uncertain words are marked `[?]`.

Other optional dependencies are listed in [requirements-optional.txt](requirements-optional.txt).

## How it works

```text
Document collection
    → clean and deduplicate
    → split into passages
    → build SQLite / FTS5 index
    → expand the query across languages
    → rank matching passages
    → return sources, or decline when coverage is too weak
```

Ranking combines BM25 with query-concept coverage, how much of a document concerns the query, and a document-type prior. Unknown query terms count against coverage, helping the system avoid confident results for questions the collection cannot support.

Heavy preprocessing happens before distributing the library. Optional enrichment adds labelled summaries, multilingual keywords, and prepared Q&A. The field server can run local chat and OCR; the phone only needs a browser.

| Component | Technology |
|---|---|
| Retrieval | Python standard library, SQLite FTS5, BM25 |
| Languages | Editable glossary, mined abbreviations, text normalisation |
| Interface | HTML, CSS, JavaScript; no CDN required |
| Local AI | Ollama; Mistral 7B in the demo |
| Printed-label OCR | Tesseract, optional Pillow preprocessing |
| Optional extensions | Dense retrieval, vision models, HQ enrichment |

The default AI endpoints are local. `CHAT_*`, `LLM_*`, and `VLM_*` environment variables configure chat, enrichment, and vision providers respectively; using a remote provider changes where those requests are processed.

## Evaluation

Measured on the saved demo index, with **565 searchable documents** and a **33.3 MB index**. These figures describe this collection and test setup.

| Check | Result |
|---|---:|
| Known-item Recall@5, 300 sampled queries | 1.00 |
| Known-item MRR@10 | 0.968 |
| Hand-written evaluation Recall@5, 16 answerable queries | 0.875 |
| False positives, 10 labelled unanswerable queries | 0/10 |
| Off-topic queries correctly declined | 18/20 |

Known-item queries sample words from their target documents, making that check optimistic for keyword retrieval. The hand-written evaluation is small; neither establishes clinical validity.

Reproduce the checks after building the index:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/evaluate.py --eval data/eval_queries.csv
python3 scripts/demo_queries.py
```

## Scope and limitations

This is a hackathon prototype. Results depend on the source collection, glossary coverage, and document extraction quality. Some guideline records contain only abbreviation lists or contents pages. OCR depends on photo quality and installed language packs; a label match does not verify a medicine's identity, strength, or preparation instructions.

The map is precomputed and may differ from the latest index. Evidence-trail links represent shared vocabulary, not verified citations. The radar's multi-site feed is simulated and labelled in the interface. Enrichment covers a subset of the collection, and generated text remains labelled for source checking.

## Project structure

```text
msf_search/       Search, indexing, ingestion, local AI, and HTTP server
  extras/        Label scanning, SMS, map, radar, evidence trail, and HQ tools
  static/        Browser interface and shared theme
scripts/         Index builds, enrichment, evaluation, updates, and demo launcher
data/            Metadata, glossary, evaluation queries, and prepared demo assets
tests/           Label scanning and retrieval regression tests
```

See [DEMO.md](DEMO.md) for the demo walkthrough and [API.md](API.md) for the HTTP contract. Contributor handoff notes are in [HANDOFF.md](HANDOFF.md).

Source documents, inbox photos, built databases, model weights, logs, and credentials stay out of version control.
