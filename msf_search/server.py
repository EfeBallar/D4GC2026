"""Tiny local web server (standard library only). Serves the UI + a JSON API on the field LAN.

    python -m msf_search.server --index build/index.db --originals "D4GC data/Original"

Then open http://<laptop-ip>:8000 from any phone or PC on the same Wi-Fi. No internet needed.
"""
import argparse
import json
import mimetypes
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .search import Searcher
from . import chat, extras, gaps, llm

STATIC = Path(__file__).parent / "static"


def make_handler(searcher: Searcher, originals: list[Path], llm_model: str | None, gap_log: str | None = None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _sse(self, event: str, data: dict):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/api/chat":
                return self._json({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._json({"error": "invalid JSON"}, 400)
            question = str(req.get("question") or "").strip()
            if not question:
                return self._json({"error": "question is required"}, 400)
            t0 = time.perf_counter()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            cfg = chat.config()
            ctx = chat.build_context(searcher, question)
            if ctx["abstained"]:  # never call the LLM when the library has no reliable match
                if gap_log:
                    gaps.log_unanswered(gap_log, ctx["search"], searcher.library_version, {"via": ["chat"]})
                self._sse("meta", {"sources": [], "model": None, "llm_called": False})
                self._sse("token", {"text": chat.REFUSAL + (f" Not found anywhere: {', '.join(ctx['unknown_terms'])}."
                                                           if ctx["unknown_terms"] else "")})
                return self._sse("done", {"llm_called": False, "ttft_ms": None,
                                          "total_ms": round((time.perf_counter() - t0) * 1000), "model": None})
            self._sse("meta", {"sources": ctx["sources"], "model": cfg["model"], "llm_called": True,
                               "model_label": chat.MODEL_LABEL,
                               "retrieval_ms": round((time.perf_counter() - t0) * 1000)})
            ttft, n, answer = None, 0, ""
            try:
                for tok in chat.stream_answer(question, ctx["context"], req.get("history"), cfg):
                    if ttft is None:
                        ttft = round((time.perf_counter() - t0) * 1000)
                    n += 1
                    answer += tok
                    self._sse("token", {"text": tok})
            except OSError as e:
                self._sse("error", {"error": f"local model not reachable ({e}); search still works"})
            self._sse("done", {"llm_called": True, "ttft_ms": ttft, "tokens": n, "model": cfg["model"],
                               "unsupported_terms": chat.unsupported_terms(answer, ctx["context"], question, searcher),
                               "total_ms": round((time.perf_counter() - t0) * 1000)})

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            g = lambda k, d="": qs.get(k, [d])[0]
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/static/"):
                f = STATIC / Path(urllib.parse.unquote(u.path[len("/static/"):])).name
                if f.is_file():
                    return self._send(200, f.read_bytes(), (mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                                      + ("; charset=utf-8" if f.suffix in (".js", ".css", ".html") else ""))
                return self._json({"error": "not found"}, 404)
            if u.path == "/gaps":
                return self._send(200, (STATIC / "gaps.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/gaps":
                return self._json(gaps.summarise([gap_log] if gap_log else []))
            if u.path == "/api/search":
                langs = [x for x in g("lang").split(",") if x]
                genres = [x for x in g("genre").split("|") if x]
                r = searcher.search(g("q"), k=int(g("k", "10")), languages=langs or None,
                                    genres=genres or None, strict=g("strict", "1") != "0", topic=g("topic") or None,
                                    profile=g("profile") or None)
                if gap_log:
                    gaps.log_unanswered(gap_log, r, searcher.library_version,
                                        {k: v for k, v in (("lang", langs), ("genre", genres)) if v})
                return self._json(r)
            if u.path == "/api/profiles":
                return self._json({"profiles": list(searcher.profiles.values())})
            if u.path == "/api/topics":
                return self._json({"topics": searcher.topics()})
            if u.path == "/api/info":
                searcher.refresh()
                return self._json({"report": searcher.report, "library_version": searcher.library_version,
                                   "n_docs": searcher.n_docs,
                                   "index_mb": round(Path(searcher.index_path).stat().st_size / 1e6, 1),
                                   "importance_distribution": searcher.importance_distribution(),
                                   "question_vectors": searcher.qvec is not None,
                                   "llm": bool(llm_model), "originals": bool(originals)})
            if u.path == "/api/answer":
                if not llm_model:
                    return self._json({"ok": False, "error": "no local model configured"})
                r = searcher.search(g("q"), k=3)
                ps = [(x["doc_id"], searcher.passages(x)[0]) for x in r["results"][:3]]
                if not ps:
                    return self._json({"ok": True, "answer": "Not found in the library.", "sources": []})
                out = llm.answer(g("q"), ps, model=llm_model)
                out["sources"] = [p[0] for p in ps]
                return self._json(out)
            if u.path.startswith("/api/doc/"):
                d = searcher.get_document(urllib.parse.unquote(u.path.rsplit("/", 1)[1]))
                return self._json(d) if d else self._json({"error": "not found"}, 404)
            if u.path.startswith("/original/"):
                name = Path(urllib.parse.unquote(u.path.rsplit("/", 1)[1])).name
                for p in (folder / name for folder in originals):
                    if p.exists():
                        return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            if (r := extras.handle(u.path, qs, searcher)) is not None:
                return self._send(*r)
            self._json({"error": "not found"}, 404)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="build/index.db")
    ap.add_argument("--glossary", default="data/glossary.csv")
    ap.add_argument("--abbrev", default="data/abbreviations.csv", help='"" to disable')
    ap.add_argument("--profiles", default="data/profiles.json")
    ap.add_argument("--originals", default=None, help="optional folder with original PDF/HTML files")
    ap.add_argument("--inbox", default="data/inbox", help="photos digitised into Field Note docs (served as originals)")
    ap.add_argument("--qvec-model", default="models/potion-multilingual-128M",
                    help="HyDE question vectors (needs model2vec + build/qvec.npy); off if missing")
    ap.add_argument("--qvec-threshold", type=float, default=0.55)
    ap.add_argument("--no-chat-warm", action="store_true", help="don't preload the on-device chat model")
    ap.add_argument("--gap-log", default="logs/unanswered.jsonl", help='searches with no reliable match; "" to disable')
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm-model", default=None, help="e.g. qwen3:1.7b (needs Ollama running locally)")
    ap.add_argument("--dense", default=None, help="tag of vectors built with build_dense.py, e.g. e5 or m2v")
    ap.add_argument("--dense-high", type=float, default=None, help="cosine threshold for a strong semantic match")
    a = ap.parse_args()
    dense = None
    if a.dense:
        from .dense import DenseIndex
        dense = DenseIndex.load(a.index, a.dense, a.dense_high)
    qvec = None
    if a.qvec_model and Path(a.qvec_model).exists():  # optional layer: core needs no numpy / model2vec
        try:
            from .qvec import QVecIndex
            qvec = QVecIndex.load(str(Path(a.index).parent), a.qvec_model, a.qvec_threshold)
        except ImportError:
            pass
    s = Searcher(a.index, a.glossary, dense=dense, abbreviations_path=a.abbrev, qvec=qvec,
                 profiles_path=a.profiles)
    print("question vectors:", f"on ({len(qvec.chunk_ids)} vectors)" if qvec else "off", flush=True)
    if not a.no_chat_warm:  # load the chat model in the background so the first answer is fast
        threading.Thread(target=lambda: print("chat model:", chat.warm(), flush=True), daemon=True).start()
    orig = [Path(f) for f in (a.originals, a.inbox) if f and Path(f).is_dir()]
    print(f"MSF library: {s.n_docs} documents. Open http://localhost:{a.port}")
    ThreadingHTTPServer((a.host, a.port), make_handler(s, orig, a.llm_model, a.gap_log or None)).serve_forever()


if __name__ == "__main__":
    main()
