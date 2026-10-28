"""Split documents into passages that carry their section heading.

Passages (not whole documents) are what gets scored, so a 300-page financial report
cannot outrank a 2-page guideline just by being long, and the UI can show the exact
paragraph that matched.
"""
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENT = re.compile(r"(?<=[.!?؟])\s+")


@dataclass
class Chunk:
    doc_id: str
    ord: int
    heading: str
    text: str


def _split_long(par: str, max_words: int, overlap: int) -> list[str]:
    words = par.split()
    if len(words) <= max_words:
        return [par]
    out, step = [], max(1, max_words - overlap)
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return out


def chunk_document(doc_id: str, text: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        first = block.splitlines()[0]
        m = _HEADING.match(first)
        if m:
            sections.append((m.group(2).strip(), []))
            rest = "\n".join(block.splitlines()[1:]).strip()
            if rest:
                sections[-1][1].append(rest)
        else:
            sections[-1][1].append(block)

    chunks: list[Chunk] = []
    for heading, pars in sections:
        buf: list[str] = []
        n = 0
        for par in pars:
            for piece in _split_long(par, max_words, overlap):
                w = len(piece.split())
                if buf and n + w > max_words:
                    chunks.append(Chunk(doc_id, len(chunks), heading, "\n\n".join(buf)))
                    buf, n = [], 0
                buf.append(piece)
                n += w
        if buf:
            chunks.append(Chunk(doc_id, len(chunks), heading, "\n\n".join(buf)))
    if not chunks and text.strip():
        chunks.append(Chunk(doc_id, 0, "", text.strip()[:2000]))
    return chunks
