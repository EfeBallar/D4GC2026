"""HyDE at HQ: embed the HQ-made questions (and summaries), so a field query phrased like a question
("kids not getting enough food") can match a document whose text never uses those words.

HQ:    python scripts/build_qvec.py      -> build/qvec.npy (int8) + build/qvec_ids.json
Field: the same small static model (Model2Vec, numpy only, no GPU) embeds the query, loaded from a
       local folder with HF_HUB_OFFLINE=1. Optional layer: without model2vec or the files, search
       simply runs without it.
Safety: a vector match can lift a document to "partial" at most, never override a query term that
appears nowhere in the library (unknown_w guard), and is labelled "matched via AI-generated question".
"""
import json
import os
import sqlite3
import zlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL_DIR = "models/potion-multilingual-128M"


def _load_model(model_dir: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never reach for the internet in the field
    from model2vec import StaticModel
    return StaticModel.from_pretrained(model_dir)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def build(index_path: str, model_dir: str = DEFAULT_MODEL_DIR, out_dir: str = "build") -> dict:
    con = sqlite3.connect(index_path)
    rows = con.execute("SELECT id, doc_id, kind, heading, text FROM chunks WHERE kind IN ('qa', 'summary')").fetchall()
    if not rows:
        return {"vectors": 0, "reason": "no HQ questions or summaries in the index"}
    texts = [h if kind == "qa" else zlib.decompress(t).decode("utf-8") for _, _, kind, h, t in rows]
    model = _load_model(model_dir)
    vecs = _unit(np.asarray(model.encode(texts), dtype=np.float32))
    q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)  # int8: 4x smaller, cosine barely changes
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "qvec.npy", q)
    (out / "qvec_ids.json").write_text(json.dumps({"model": Path(model_dir).name, "chunk_ids": [r[0] for r in rows],
                                                   "doc_ids": [r[1] for r in rows], "kinds": [r[2] for r in rows]}))
    return {"vectors": len(rows), "dim": int(q.shape[1]), "kb": round((out / "qvec.npy").stat().st_size / 1024, 1)}


class QVecIndex:
    def __init__(self, vecs: np.ndarray, meta: dict, model, threshold: float):
        self.vecs = vecs.astype(np.float32) / 127.0
        self.chunk_ids, self.doc_ids, self.kinds = meta["chunk_ids"], meta["doc_ids"], meta["kinds"]
        self.model, self.model_name, self.threshold = model, meta["model"], threshold

    @classmethod
    def load(cls, build_dir: str = "build", model_dir: str = DEFAULT_MODEL_DIR, threshold: float = 0.55):
        """None (layer off) when the vectors, the model folder or model2vec are missing."""
        b = Path(build_dir)
        if not (b / "qvec.npy").exists() or not Path(model_dir).exists():
            return None
        try:
            model = _load_model(model_dir)
        except ImportError:
            return None
        return cls(np.load(b / "qvec.npy"), json.loads((b / "qvec_ids.json").read_text()), model, threshold)

    def doc_scores(self, query: str) -> dict:
        """{doc_id: (max cosine, chunk_id of the best HQ question/summary)}"""
        qv = _unit(np.asarray(self.model.encode([query]), dtype=np.float32))[0]
        sims = self.vecs @ qv
        best = {}
        for i in np.argsort(-sims)[:200]:
            d = self.doc_ids[i]
            if d not in best:
                best[d] = (float(sims[i]), self.chunk_ids[i])
        return best
