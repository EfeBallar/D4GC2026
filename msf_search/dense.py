"""OPTIONAL dense (semantic, cross-lingual) layer. The core never needs it.

Two hardware tiers, same interface:
  * Model2Vec static embeddings (e.g. minishlab/potion-multilingual-128M):
      no PyTorch at query time, encoding = token lookup + average -> Pi-class hardware.
  * Transformer encoder (default intfloat/multilingual-e5-small, MIT licence):
      better quality, needs sentence-transformers (PyTorch) or an ONNX export.

Build at HQ:   python scripts/build_dense.py --model <name> --out-name <tag>
Use in field:  python -m msf_search.server --dense <tag>
Files written next to index.db:  <tag>.npy (float16), <tag>_ids.npy, <tag>.json
"""
import json
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Encoder:
    """Wraps either a Model2Vec StaticModel or a sentence-transformers model."""

    def __init__(self, name: str):
        self.name = name
        self.kind = "model2vec" if name.lower().startswith("minishlab/") or "potion" in name.lower() else "st"
        if self.kind == "model2vec":
            from model2vec import StaticModel  # pip install model2vec
            self.m = StaticModel.from_pretrained(name)
        else:
            from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
            self.m = SentenceTransformer(name, device="cpu")
        self.e5 = "e5" in name.lower()

    def _prefix(self, texts, kind):
        return [f"{kind}: {t}" for t in texts] if self.e5 else list(texts)

    def encode(self, texts, kind="passage", batch=64, progress=False) -> np.ndarray:
        texts = self._prefix(texts, kind)
        if self.kind == "model2vec":
            v = np.asarray(self.m.encode(texts), dtype=np.float32)
        else:
            v = np.asarray(self.m.encode(texts, batch_size=batch, show_progress_bar=progress), dtype=np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v


def build(index_path: str, model_name: str = DEFAULT_MODEL, out_name: str = "dense", batch: int = 64):
    con = sqlite3.connect(index_path)
    rows = con.execute("SELECT c.id, d.title, c.heading, c.text FROM chunks c JOIN docs d ON d.id=c.doc_id").fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int32)
    texts = [f"{r[1]} | {r[2]}\n" + zlib.decompress(r[3]).decode("utf-8")[:1500] for r in rows]
    enc = Encoder(model_name)
    t0 = time.time()
    vecs = enc.encode(texts, "passage", batch=batch, progress=True)
    base = Path(index_path).with_name(out_name)
    np.save(f"{base}.npy", vecs.astype(np.float16))
    np.save(f"{base}_ids.npy", ids)
    Path(f"{base}.json").write_text(json.dumps({"model": model_name, "n": len(ids), "dim": int(vecs.shape[1]),
                                                "build_seconds": round(time.time() - t0, 1)}))
    print(f"saved {len(ids)} x {vecs.shape[1]} vectors -> {base}.npy "
          f"({Path(f'{base}.npy').stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


class DenseIndex:
    def __init__(self, vecs, ids, model_name, high_threshold=None):
        self.vecs, self.ids, self.model_name = vecs, ids, model_name
        self.encoder = Encoder(model_name)
        # Absolute cosine above which a passage alone counts as a strong match.
        # Calibrate per model with scripts/compare_dense.py (it prints cosine stats).
        self.high_threshold = high_threshold if high_threshold is not None else (0.86 if self.encoder.e5 else 0.6)
        self.last_abs = {}

    @classmethod
    def load(cls, index_path: str, name: str = "dense", high_threshold=None):
        base = Path(index_path).with_name(name)
        meta = json.loads(Path(f"{base}.json").read_text())
        return cls(np.load(f"{base}.npy").astype(np.float32), np.load(f"{base}_ids.npy"), meta["model"], high_threshold)

    def search(self, query: str, top: int = 200, allowed=None) -> dict:
        q = self.encoder.encode([query], "query")[0]
        sims = self.vecs @ q
        top = min(top, len(sims))
        idx = np.argpartition(-sims, top - 1)[:top]
        top_s = sims[idx]
        lo, hi = float(top_s.min()), float(top_s.max())
        self.last_abs = {int(self.ids[i]): float(sims[i]) for i in idx}
        return {int(self.ids[i]): (float(sims[i]) - lo) / (hi - lo + 1e-9) for i in idx}
