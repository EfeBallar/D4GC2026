"""Score a hand-made eval set (data/eval_queries.csv: query,lang,type,relevant_ids). The questions are
written by people from the original documents, never by the hq_enrich prompt, so enrichment can't leak."""
import csv
import statistics


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "relevant": {x for x in r["relevant_ids"].split("|") if x}} for r in csv.DictReader(f)]


def score(searcher, rows: list[dict], k: int = 5) -> dict:
    hits, rr, fp, lat, n_ans, n_un = 0, 0.0, 0, [], 0, 0
    misses = []
    for r in rows:
        res = searcher.search(r["query"], k=10)
        lat.append(res["ms"])
        ids = [x["doc_id"] for x in res["results"]]
        if r["relevant"]:
            n_ans += 1
            rank = next((i + 1 for i, d in enumerate(ids) if d in r["relevant"]), None)
            hits += bool(rank and rank <= k)
            rr += 1 / rank if rank else 0.0
            if not rank or rank > k:
                misses.append(r["query"])
        else:
            n_un += 1
            fp += not res["abstained"]
    return {f"recall@{k}": round(hits / max(1, n_ans), 3), "mrr": round(rr / max(1, n_ans), 3),
            "false_positive_rate": round(fp / max(1, n_un), 3), "answerable": n_ans, "unanswerable": n_un,
            "latency_ms_median": round(statistics.median(lat), 1), "misses": misses}
