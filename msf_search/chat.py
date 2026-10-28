"""On-device RAG chat: the index is the package HQ sent; a small 4-bit model on the device answers
ONLY from the retrieved passages, with citations.

Safety: the strict search runs first. If it abstains, the LLM is never called and the reply is
"Not found in the MSF library". Original passages come first; HQ summaries / Q&A are extra context,
marked as machine-made. Provider = configuration: env CHAT_BASE_URL / CHAT_API_KEY / CHAT_MODEL
(default: local Ollama, model "mistral-field" = mistral:7b-instruct Q4 with num_ctx 4096, see
models/Modelfile.field). Optional: without a reachable model, /api/chat says so and search still works.
"""
import html
import json
import os
import time
import urllib.request
import re
import zlib

from .textnorm import STOPWORDS, normalize_token

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "mistral-field"
SYSTEM = ("You answer questions from medical staff of Doctors Without Borders in the field. Answer ONLY from the "
          "numbered passages below. Cite the passages you use like [1] or [2]. Copy doses, numbers and units exactly "
          "as written. If the passages do not contain the answer, reply exactly: \"Not found in the library.\" "
          "Be brief (at most 5 sentences). Reply in the language of the question.")
REFUSAL = "Not found in the MSF library."
MODEL_LABEL = os.environ.get("CHAT_MODEL_LABEL") or "mistral 7B, 4-bit"  # shown in the UI label


def config() -> dict:
    return {"base": (os.environ.get("CHAT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            "key": os.environ.get("CHAT_API_KEY", ""), "model": os.environ.get("CHAT_MODEL") or DEFAULT_MODEL}


def warm(cfg: dict | None = None) -> str:
    """Load the model into memory at server start and keep it 30 min (Ollama native API; ignored elsewhere)."""
    cfg = cfg or config()
    root = cfg["base"][:-3] if cfg["base"].endswith("/v1") else cfg["base"]
    body = json.dumps({"model": cfg["model"], "prompt": "ok", "stream": False, "keep_alive": "30m",
                       "options": {"num_predict": 1}}).encode()
    t0 = time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(root + "/api/generate", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=120).read()
        return f"warm ({cfg['model']}, {time.time() - t0:.1f} s)"
    except OSError as e:
        return f"not warmed ({cfg['model']}: {e})"


def _text(searcher, cid: int) -> tuple[str, str]:
    h, t = searcher.con.execute("SELECT heading, text FROM chunks WHERE id=?", (cid,)).fetchone()
    return h or "", zlib.decompress(t).decode("utf-8")


_WEAK_TITLE = re.compile(r"(permalink|^\W|http|www\.|\.org|editorial committee|copyright|©|^for the french)", re.I)
_TAG = re.compile(r"<[^>]+>")


def display_title(meta: dict, heading: str) -> str:
    """Short readable title; for scraped junk titles use the AI summary's first sentence or the section heading."""
    title = (meta.get("title") or "").strip()
    if title and len(title) >= 12 and not _WEAK_TITLE.search(title) and not title[0].islower():
        return title if len(title) <= 80 else title[:77].rstrip() + "..."
    if meta.get("summary"):
        first = re.split(r"(?<=[.!?])\s", meta["summary"].strip())[0]
        return first if len(first) <= 90 else first[:87].rstrip() + "..."
    return heading or title[:80] or meta["id"]


def _quote(text: str, concepts) -> str:
    from .search import make_snippet
    q = html.unescape(_TAG.sub("", make_snippet(text, concepts, max_words=28)))
    return " ".join(q.split())


def build_context(searcher, question: str, k: int = 4) -> dict:
    r = searcher.search(question, k=k, strict=True)
    if r["abstained"]:
        return {"abstained": True, "unknown_terms": r.get("unknown_terms") or [], "search": r}
    passages, sources, extra = [], [], []
    for n, x in enumerate(r["results"][:k], 1):
        cid = x.get("text_chunk_id") or (x.get("evidence") or {}).get("chunk_id") or x["chunk_id"]
        heading, text = _text(searcher, cid)
        words = text.split()
        passages.append(f"[{n}] {x['title']} ({x['genre']}{', ' + heading if heading else ''}):\n"
                        + " ".join(words[:220]))
        page = (x.get("evidence") or {}).get("page") or x.get("page")
        sources.append(_card(searcher, n, x["doc_id"], cid, heading, text, page, r, x["confidence"]))
        if x.get("matched_via") in ("qa", "summary"):
            _, mt = _text(searcher, x["chunk_id"])
            extra.append(f"[{n}] HQ-made {x['matched_via']} (machine-generated, lower trust): " + " ".join(mt.split()[:80]))
    if len(passages) < k:  # few docs matched: add their next-best passages so the model has enough source text
        used = {s["doc_id"] for s in sources}
        for doc_id in list(used):
            rows = searcher.con.execute(
                "SELECT c.id, c.heading, c.text FROM chunks c WHERE c.doc_id=? AND c.kind='text' ORDER BY c.ord",
                (doc_id,)).fetchall()
            terms = {t for c_ in r["concepts"] for v in c_["variants"] for t in v.split()}
            scored = sorted(rows, key=lambda row: -sum(t in zlib.decompress(row[2]).decode("utf-8").lower() for t in terms))
            have = {s.get("chunk_id") for s in sources}
            for cid, heading, text in scored:
                if len(passages) >= k:
                    break
                if cid in have or any(cid == x.get("text_chunk_id") for x in r["results"]):
                    continue
                n = len(passages) + 1
                base = next(s for s in sources if s["doc_id"] == doc_id)
                passages.append(f"[{n}] {base['title']} ({heading}):\n" + " ".join(zlib.decompress(text).decode("utf-8").split()[:220]))
                ptxt = zlib.decompress(text).decode("utf-8")
                m = re.match(r"^Page (\d+)$", heading or "")
                sources.append(_card(searcher, n, doc_id, cid, heading or "", ptxt, int(m.group(1)) if m else None,
                                     r, base["confidence"]))
                have.add(cid)
    ctx = "\n\n".join(passages)
    if extra:
        ctx += "\n\nExtra context (machine-made at HQ; prefer the original passages):\n" + "\n".join(extra)
    return {"abstained": False, "context": ctx, "sources": sources, "search": r}


def _card(searcher, n, doc_id, cid, heading, text, page, r, confidence) -> dict:
    """One source card (API: /api/chat meta.sources). Fields are additive to the earlier contract."""
    from .search import Concept
    meta = searcher.docs[doc_id]
    concepts = [Concept(c["label"], [tuple(v.split()) for v in c["variants"]]) for c in r["concepts"]]
    return {"n": n, "doc_id": doc_id, "chunk_id": cid, "title": meta["title"], "heading": heading, "page": page,
            "original_name": meta.get("original_name"), "tier": meta.get("tier"), "confidence": confidence,
            "display_title": display_title(meta, heading), "type": meta.get("tier"), "genre": meta.get("genre"),
            "language": meta.get("language"), "year": meta.get("year"),
            "location": f"p. {page}" if page else (heading or None), "quote": _quote(text, concepts),
            "provenance": searcher.chunk_provenance(cid)}


def stream_answer(question: str, context: str, history: list | None = None, cfg: dict | None = None,
                  max_tokens: int = 150):
    """Yield text tokens from the OpenAI-format streaming endpoint."""
    cfg = cfg or config()
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in (history or [])[-4:]:
        if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
            msgs.append({"role": h["role"], "content": h["content"][:600]})
    msgs.append({"role": "user", "content": f"PASSAGES:\n{context}\n\nQUESTION: {question}"})
    body = json.dumps({"model": cfg["model"], "messages": msgs, "stream": True, "temperature": 0.1,
                       "max_tokens": max_tokens}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = f"Bearer {cfg['key']}"
    with urllib.request.urlopen(urllib.request.Request(cfg["base"] + "/chat/completions", data=body,
                                                       headers=headers), timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta


def unsupported_terms(answer: str, context: str, question: str, searcher=None, max_df: int = 12) -> list[str]:
    """Specific terms in the answer that occur neither in the passages nor in the question (e.g. a drug the
    model added from its own memory), plus numbers. With a searcher, only terms that are RARE in the library
    (in <= max_df of ~560 docs: drug names, doses) are flagged, so everyday words ("however") are not."""
    seen = {normalize_token(w) for w in re.findall(r"\w+", context + " " + question)}
    stems = {w[:6] for w in seen if len(w) >= 6}
    out = []
    for w in re.findall(r"\w+", answer):
        t = normalize_token(w)
        if t in seen or t in STOPWORDS or (len(t) >= 6 and t[:6] in stems):
            continue
        if t.isdigit() and len(t) >= 2:
            out.append(t)
        elif t.isalpha() and len(t) >= 7:
            if searcher is not None:
                df = searcher.con.execute("SELECT count(DISTINCT c.doc_id) FROM chunks_fts f JOIN chunks c ON "
                                          "c.id=f.rowid WHERE chunks_fts MATCH ?", (f'"{t}"',)).fetchone()[0]
                if df > max_df:
                    continue
            out.append(t)
    return list(dict.fromkeys(out))
