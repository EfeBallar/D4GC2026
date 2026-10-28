"""Works on a 10-euro phone: one short text message, from the same index.

A health worker with a basic handset texts a question to the site server's GSM modem. The server
runs the ordinary search and texts back a single SMS: the answer, then the document id so the
reply can be checked against the printed guideline. Same engine as the web UI, no internet, no
LLM, and the handset needs nothing but the ability to send a text.

Segment arithmetic is honest, because it decides how much answer fits:
  * GSM-7 carries 160 characters per message, but only its own alphabet.
  * One character outside that alphabet switches the whole message to UCS-2, where the limit is 70.
So typographic punctuation is folded to ASCII first (a curly apostrophe would otherwise cost 90
characters of answer), and Arabic replies, which cannot avoid UCS-2, are trimmed to 70.
"""
import html
import re

GSM7 = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ"
    " !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§"
    "¿abcdefghijklmnopqrstuvwxyzäöñüà"
    "€[]{}\\~^|"
)
GSM7_LIMIT = 160
UCS2_LIMIT = 70
NO_MATCH = "No reliable match in the MSF library. Reply with other words, or ask HQ to add it."
EMPTY = "Send a question, for example: PPH treatment"

# Typographic characters that would silently force UCS-2 and halve the answer.
FOLD = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-",
        "—": "-", "…": "...", " ": " ", "•": "-", "·": "-",
        "′": "'", "″": '"', "−": "-", "ﬁ": "fi", "ﬂ": "fl"}

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_PIPES = re.compile(r"\s*\|[\s|]*")
_LEAD = re.compile(r"^(?:[-*#>•·\s]+|\(\s*[a-z0-9]{1,2}\s*\)\s*|\d+[.)]\s+)+")


def _plain(s: str) -> str:
    """Strip markup, decode entities, flatten table pipes, fold typography, collapse whitespace.

    Passages come from markdown converted out of HTML and PDF, so a snippet can arrive as a
    fragment of an abbreviation table ("| PAS | acide para-aminosalicylique |") or carry an
    undecoded entity (&#x27;). Neither is readable in a text message.
    """
    s = _TAG.sub("", s or "").replace("**", "").replace("__", "")
    s = html.unescape(s)
    s = _PIPES.sub(" - ", s)
    s = "".join(FOLD.get(ch, ch) for ch in s)
    return _WS.sub(" ", s).strip(" -\u2013")


def _lede(s: str) -> str:
    """Drop list markers and (a)-style enumerations so the message opens on words."""
    return _LEAD.sub("", s).lstrip() or s


def encoding_of(text: str) -> tuple[str, int]:
    if any(ch not in GSM7 for ch in text):
        return "UCS-2", UCS2_LIMIT
    return "GSM-7", GSM7_LIMIT


def segments(text: str) -> dict:
    enc, per = encoding_of(text)
    return {"encoding": enc, "chars": len(text), "per_segment": per,
            "segments": max(1, -(-len(text) // per))}


def _fit(body: str, tail: str, limit: int) -> str:
    """Trim body on a word boundary so body + tail is one message."""
    room = limit - len(tail)
    if room <= 0:
        return tail.strip()[:limit]
    if len(body) <= room:
        return body + tail
    cut = body[:room - 3]
    sp = cut.rstrip().rfind(" ")
    if sp > room * 0.6:
        cut = cut[:sp]
    return cut.rstrip(" ,;:.-") + "..." + tail


def _one_message(body: str, tail: str) -> str:
    """Fit for GSM-7 first; fall back to the UCS-2 limit only if the result really needs it."""
    text = _fit(body, tail, GSM7_LIMIT)
    if encoding_of(text)[0] == "UCS-2":
        text = _fit(body, tail, UCS2_LIMIT)
    return text


def reply(searcher, query: str) -> dict:
    """The exact text the modem would send back, plus what it costs to send."""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "query": q, "sms": EMPTY, "cost": segments(EMPTY), "ms": 0.0}

    r = searcher.search(q, k=1)
    if r.get("abstained") or not r.get("results"):
        unknown = [t for t in (r.get("unknown_terms") or []) if t][:2]
        text = NO_MATCH
        if unknown:
            hint = "Not in library: " + ", ".join(unknown) + ". Ask HQ to add it."
            if len(hint) <= encoding_of(hint)[1]:
                text = hint
        return {"ok": False, "query": q, "sms": text, "cost": segments(text),
                "abstained": True, "unknown_terms": unknown, "ms": r.get("ms", 0.0)}

    top = r["results"][0]
    meta = searcher.docs.get(top["doc_id"], {})
    body = _lede(_plain(top.get("snippet") or ""))
    if not body:
        try:
            body = _lede(_plain(searcher.passages(top)[0]))[:400]
        except Exception:
            body = _plain(meta.get("title") or top["doc_id"])

    # The language belongs in the reply: this library has no Arabic clinical guidance at all, so
    # an Arabic question is often answered from a French or English document. Saying so costs
    # three characters and lets the reader judge whether they can actually use the answer.
    lang = (meta.get("language") or "").upper()
    tail = f" [{top['doc_id']}{' ' + lang if lang else ''} {top['confidence'][0].upper()}]"
    text = _one_message(body, tail)
    return {"ok": True, "query": q, "sms": text, "cost": segments(text),
            "doc_id": top["doc_id"], "title": meta.get("title"), "tier": meta.get("tier"),
            "language": meta.get("language"), "confidence": top["confidence"],
            "ms": r.get("ms", 0.0)}
