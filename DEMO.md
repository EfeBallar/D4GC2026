# Quick demo

Start everything from this repo:

```bash
./start.sh --check-chat
```

The launcher checks/builds the library if missing, starts or reuses local Ollama,
warms an installed model, starts or reuses the backend, and prints all demo URLs.
`--check-chat` sends a complete RAG request and checks for a cited model answer
without a source-check warning. If that check needs attention, the backend stays
available and the launcher prints the problem.

Keep the terminal open. Ctrl+C stops the backend started by this launcher and
leaves Ollama available. The script does not stop existing services or download
models. If an existing server on 8000 lacks the AI-summary option, use:

```bash
./start.sh --port 8767 --check-chat
```

## Checked search sequence

Checked directly against the current index on 28 October 2026, with all languages,
all document types, no field profile and confident matches enabled. These are
retrieval checks, not a guarantee about a future generated answer. Clear active
filters before starting.

| Enter in the search box | Expected result | What to show |
|---|---|---|
| `hémorragie du post-partum` | P23 first, strong match, English guideline. Extracted title: `Obstetric trauma` | A French query retrieves the English passage on postpartum haemorrhage. Open the source |
| `pph treatment` | P23 first, strong match | The abbreviation resolves to the same guideline |
| `uterine atony` | P23 first, strong match, matched through its AI summary | Point to the labelled HQ summary and its original evidence passage |
| `Orissa cyclone` | D0eebba4c05 first, strong match, OCR text from page 59 | Open the scanned PDF at page 59 and show the transcription label |
| `bitcoin price` | No reliable match, unknown term `bitcoin` | Honest abstention when the library cannot support the query |

Suggested 60-90 second sequence: French query, PPH abbreviation, Orissa scan,
then the bitcoin refusal. The malaria topic also works: 54 documents on the
checked index.

## Local AI chat

Use the **Ask (offline AI)** panel and start with:

```text
What is early postpartum haemorrhage?
```

Its retrieved context was checked: all source cards point to P23 and the first
passage contains the definition. The launcher can test actual generation with
`--check-chat`. Only demonstrate the generated reply after that live check passes
and you inspect it. Source-context checks alone do not validate a model answer.

`bitcoin price` also takes the refusal path in chat and does not call the model.

Avoid `oral rehydration for cholera` in chat: its selected passage currently
contains an abbreviations list. Avoid `kids not getting enough food` as a
first-result showcase: an unrelated-looking Gaza document currently ranks above
the intended Nigeria malnutrition result.

## Extra pages

All are served by the same backend. Substitute 8767 if using the alternative port.
The page files and their data handlers were checked locally. Browser rendering
has not been re-tested in this audit.

- `http://localhost:8000/galaxy`: precomputed document map, 561 documents in its current saved data. This count differs from the 565-record search index because the map was built separately.
- `http://localhost:8000/trail`: 43 of 73 guideline/protocol records have vocabulary-based links to research. These are lexical similarities, not a citation graph.
- `http://localhost:8000/sms`: enter `pph treatment`. Checked response uses P23 and fits one GSM-7 segment (155 characters). This is the SMS interface demo, not a connected live modem.
- `http://localhost:8000/radar`: labelled simulation of search-volume changes across sites. The multi-site data is invented demo data.
- `http://localhost:8000/gaps`: locally logged unsuccessful searches.

For a short pitch, prioritise the checked search flow and one extra page rather
than trying every feature.
