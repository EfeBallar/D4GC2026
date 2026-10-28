"""OPTIONAL: grounded answer from a small local LLM (Ollama or llama.cpp server).

Retrieval never depends on this. If no model is running, the app simply shows
documents and snippets. The model only rephrases the retrieved passages and must
cite them; it is told to say so when the passages do not contain the answer.
"""
import json
import urllib.request

PROMPT = """You are helping a Doctors Without Borders field worker. Answer the question ONLY using the
numbered passages below. Cite passages like [1]. If the passages do not contain the answer, reply exactly:
"Not found in the library." Keep the answer under 120 words. Answer in the language of the question.

Question: {q}

Passages:
{passages}
"""


def answer(question: str, passages: list[tuple[str, str]], model: str = "qwen3:1.7b",
           url: str = "http://localhost:11434/api/generate", timeout: int = 120) -> dict:
    """passages: list of (doc_id, text). Works with Ollama's /api/generate."""
    ctx = "\n\n".join(f"[{i + 1}] ({d}) {t[:1500]}" for i, (d, t) in enumerate(passages))
    body = json.dumps({"model": model, "prompt": PROMPT.format(q=question, passages=ctx) + " /no_think",
                       "stream": False, "options": {"temperature": 0.1, "num_ctx": 4096}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        text = out.get("response", "").strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return {"ok": True, "answer": text, "model": model}
    except Exception as e:  # model not running / not installed -> retrieval still works
        return {"ok": False, "error": f"local model unavailable ({e.__class__.__name__})"}
