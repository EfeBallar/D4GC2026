"""HQ enrichment: one LLM call per document (more for long ones), done once, off-premises.

Any OpenAI-compatible endpoint (env LLM_BASE_URL, LLM_API_KEY, LLM_MODEL): a hosted API, or a local
open model through Ollama (LLM_BASE_URL=http://127.0.0.1:11434/v1), so no vendor lock-in. The field
never runs a model: it gets the results inside index.db, always labelled as machine-made.

Grounding is enforced, not hoped for: a Q&A pair is dropped when its evidence_quote is not found
verbatim in the document, or its answer contains a number the document does not contain (doses!).
"""
import json
import os
import re
import time
import urllib.request
from collections import Counter

AUDIENCES = {"clinical", "operational", "advocacy", "research", "historical"}
SITUATIONS = {"conflict", "outbreak", "malnutrition", "displacement", "maternal", "disaster", "regular"}
QA_LANGS = {"en": 5, "fr": 1, "es": 1, "ar": 1}

SYSTEM = ("You prepare a humanitarian medical library (Doctors Without Borders) for offline use in the field. "
          "Use ONLY the document text given. Never add outside facts. Copy doses, numbers and units exactly. "
          "Reply with one JSON object only.")

# Text FIRST, instructions LAST: if a provider truncates a long prompt, it loses text, not the instructions.
PROMPT = """Document id: {id} | catalogue genre: {genre} | language: {language} | title: {title}{window}

TEXT:
{text}

END OF TEXT. Using only the text above, return a JSON object with exactly these keys:
"doc_type": corrected genre, 2-4 words
"audience": one of clinical | operational | advocacy | research | historical
"actionable": true if it tells a field worker what to do (protocol, dose, procedure), else false
"importance": 1 = essential field reference, 2 = useful, 3 = background, 4 = archive
"importance_reason": one sentence
"year": publication year as a number, or null
"superseded_hint": a sentence if the text says it replaces or is replaced by another version, else null
"summary": 3-4 sentences
"keywords_en", "keywords_fr", "keywords_es", "keywords_ar": 10 keywords each, written in that language
"abbreviations": list of {{"abbr": "...", "long": "..."}} used in the text
"topics": up to 6 short English topic labels, e.g. "cholera", "surgery", "mental health"
"countries": countries where the content takes place
"situation": one of conflict | outbreak | malnutrition | displacement | maternal | disaster | regular
"qa": 8 objects {{"question", "answer", "evidence_quote", "lang"}}: 5 with lang "en", 1 "fr", 1 "es", 1 "ar".
  Questions a field worker would ask, written in that language. answer: short, only from the text.
  evidence_quote: one sentence copied EXACTLY, character for character, from the text above."""

# Default provider: a local open-weight model via Ollama's OpenAI-format endpoint (no key, no vendor).
# Override with env LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (e.g. the Mistral API) without code changes.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "mistral:7b-instruct"

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def _norm(s: str) -> str:
    return _WS.sub(" ", s.translate(_QUOTES)).strip().lower()


def windows(text: str, size: int = 3000, max_windows: int = 5, long_doc: int = 6000) -> list[str]:
    """Whole text for normal docs; for long ones (> long_doc words) up to max_windows evenly spread windows."""
    words = text.split()
    if len(words) <= long_doc:
        return [" ".join(words[:long_doc])]
    n = min(max_windows, -(-len(words) // size))
    step = (len(words) - size) / max(1, n - 1)
    return [" ".join(words[int(i * step):int(i * step) + size]) for i in range(n)]


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2, timeout: int = 900):
        self.base = base_url.rstrip("/")
        self.url = self.base + "/chat/completions"
        self.key, self.model, self.temperature, self.timeout = api_key, model, temperature, timeout
        self.tokens = Counter()

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL, os.environ.get("LLM_API_KEY", ""),
                   os.environ.get("LLM_MODEL") or DEFAULT_MODEL)

    def available(self) -> bool:
        try:
            urllib.request.urlopen(urllib.request.Request(self.base + "/models",
                                                          headers=self._headers()), timeout=5)
            return True
        except OSError:
            return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:  # local Ollama needs no key; hosted APIs get it from the environment only
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def json(self, prompt: str, retries: int = 4) -> dict:
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": prompt}]}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=self._headers())
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                usage = out.get("usage") or {}
                self.tokens["prompt"] += usage.get("prompt_tokens", 0)
                self.tokens["completion"] += usage.get("completion_tokens", 0)
                return json.loads(out["choices"][0]["message"]["content"])
            except (OSError, ValueError, KeyError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)  # backoff: 2, 4, 8 s


def validate_qa(qa: list, doc_text: str) -> tuple[list, Counter]:
    """Keep only grounded pairs: verbatim evidence quote, and no number the document doesn't contain."""
    norm_doc, kept, dropped = _norm(doc_text), [], Counter()
    doc_numbers = set(_NUM.findall(doc_text))
    for x in qa if isinstance(qa, list) else []:
        if not isinstance(x, dict) or not all(isinstance(x.get(k), str) and x[k].strip()
                                               for k in ("question", "answer", "evidence_quote")):
            dropped["malformed"] += 1
        elif len(x["evidence_quote"]) < 20 or _norm(x["evidence_quote"]) not in norm_doc:
            dropped["quote_not_verbatim"] += 1
        elif any(n not in doc_numbers for n in _NUM.findall(x["answer"])):
            dropped["number_not_in_doc"] += 1
        else:
            kept.append({"question": x["question"].strip(), "answer": x["answer"].strip(),
                         "evidence_quote": x["evidence_quote"].strip(), "lang": str(x.get("lang", "en"))[:2]})
    return kept, dropped


def _strs(v, n=10) -> list[str]:
    return [s.strip() for s in v if isinstance(s, str) and s.strip()][:n] if isinstance(v, list) else []


def clean(raw: dict) -> dict:
    """Coerce one window's JSON into the schema (models drift); never trust types."""
    imp = raw.get("importance")
    year = raw.get("year")
    return {
        "doc_type": str(raw.get("doc_type") or "")[:60],
        "audience": raw.get("audience") if raw.get("audience") in AUDIENCES else None,
        "actionable": bool(raw.get("actionable")),
        "importance": min(4, max(1, int(imp))) if isinstance(imp, (int, float)) or str(imp).isdigit() else 3,
        "importance_reason": str(raw.get("importance_reason") or "")[:300],
        "year": int(year) if str(year).isdigit() and 1950 <= int(year) <= 2030 else None,
        "superseded_hint": str(raw["superseded_hint"])[:300] if raw.get("superseded_hint") else None,
        "summary": str(raw.get("summary") or "").strip()[:1200],
        **{f"keywords_{l}": _strs(raw.get(f"keywords_{l}")) for l in ("en", "fr", "es", "ar")},
        "abbreviations": [{"abbr": str(a.get("abbr"))[:12], "long": str(a.get("long"))[:100]}
                          for a in raw.get("abbreviations") or [] if isinstance(a, dict) and a.get("abbr")][:30],
        "topics": [t.lower() for t in _strs(raw.get("topics"), 6)],
        "countries": _strs(raw.get("countries"), 15),
        "situation": raw.get("situation") if raw.get("situation") in SITUATIONS else "regular",
        "qa": raw.get("qa") or [],
    }


def merge(parts: list[dict]) -> dict:
    """Combine windows of a long document: first window leads, lists are unioned, most important wins."""
    out = dict(parts[0])
    for key in ("keywords_en", "keywords_fr", "keywords_es", "keywords_ar", "topics", "countries"):
        seen = []
        for p in parts:
            seen += [v for v in p[key] if v.lower() not in {s.lower() for s in seen}]
        out[key] = seen[:10 if key.startswith("keywords") else 15]
    out["abbreviations"] = list({a["abbr"]: a for p in parts for a in p["abbreviations"]}.values())[:30]
    out["importance"] = min(p["importance"] for p in parts)
    out["actionable"] = any(p["actionable"] for p in parts)
    out["situation"] = Counter(p["situation"] for p in parts).most_common(1)[0][0]
    out["superseded_hint"] = next((p["superseded_hint"] for p in parts if p["superseded_hint"]), None)
    out["qa"] = [q for p in parts for q in p["qa"]]
    return out


def _quota(qa: list) -> list:
    """At most 8 pairs, keeping the language mix (5 en, 1 fr, 1 es, 1 ar) where available."""
    keep, used = [], Counter()
    for q in qa:
        if used[q["lang"]] < QA_LANGS.get(q["lang"], 0):
            keep.append(q)
            used[q["lang"]] += 1
    return keep + [q for q in qa if q not in keep][:max(0, 8 - len(keep))]


def enrich_doc(llm: LLM, doc) -> dict:
    t0 = time.time()
    wins = windows(doc.text)
    parts = []
    for i, w in enumerate(wins):
        tag = f" | part {i + 1} of {len(wins)} of a long document" if len(wins) > 1 else ""
        parts.append(clean(llm.json(PROMPT.format(id=doc.id, genre=doc.genre, language=doc.language,
                                                  title=doc.title, window=tag, text=w))))
    rec = merge(parts)
    qa, dropped = validate_qa(rec.pop("qa"), doc.text)
    rec.update({"id": doc.id, "status": "ok", "model": llm.model, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "windows": len(wins), "qa": _quota(qa), "qa_generated": len(qa) + sum(dropped.values()),
                "qa_dropped": dict(dropped), "seconds": round(time.time() - t0, 1)})
    return rec
