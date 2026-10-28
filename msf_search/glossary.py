"""Multilingual query expansion from a plain CSV glossary.

A query for "cholera" also searches "choléra", "cólera" and "الكوليرا", so an English
query can surface French, Spanish and Arabic documents without any ML model.
The glossary is a CSV that a medical coordinator can edit in a spreadsheet; no
retraining, no vendor.
"""
import csv
from pathlib import Path

from .textnorm import STOPWORDS, tokenize


class Glossary:
    def __init__(self, path: str | None = None):
        # normalised phrase (tuple of tokens) -> concept name
        self.phrase_to_concept: dict[tuple, str] = {}
        # concept -> list of normalised variant phrases
        self.variants: dict[str, list[tuple]] = {}
        if path and Path(path).exists():
            self.load(path)

    def load(self, path: str):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                concept = row["concept"].strip()
                vs = []
                for col, val in row.items():
                    if col == "concept" or not val:
                        continue
                    for v in val.split("|"):
                        toks = tuple(tokenize(v))
                        if toks and toks not in vs:
                            vs.append(toks)
                self.variants[concept] = vs
                for v in vs:
                    self.phrase_to_concept.setdefault(v, concept)
        self.max_len = max((len(p) for p in self.phrase_to_concept), default=1)

    def add_abbreviations(self, mapping: dict):
        """{abbr: [long forms]} from HQ enrichment (stored in index.db); same one-way rule as the CSV."""
        for abbr, longs in mapping.items():
            key = tuple(tokenize(abbr))
            if not key or len(abbr) < 2 or key[0] in STOPWORDS:  # "WHO" must not hijack "who"
                continue
            concept = self.phrase_to_concept.get(key)
            if concept is None:
                concept = abbr
                self.variants.setdefault(concept, [key])
                self.phrase_to_concept[key] = concept
            vs = self.variants[concept]
            vs.extend(v for v in (tuple(tokenize(l)) for l in longs) if v and v not in vs)
        self.max_len = max((len(p) for p in self.phrase_to_concept), default=1)

    def load_abbreviations(self, path: str):
        """Corpus-mined abbreviations (scripts/build_abbreviations.py). One-way on purpose: typing
        "SAM" also searches "severe acute malnutrition", but a long-form query keeps its glossary
        meaning. An abbreviation already in the glossary ("tb") gets the long forms added."""
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = tuple(tokenize(row["abbr"]))
                if not key:
                    continue
                concept = self.phrase_to_concept.get(key) or self.phrase_to_concept.get((key[0] + "s",))
                if concept is None:
                    concept = row["abbr"].strip()
                    self.variants.setdefault(concept, [key])
                vs = self.variants[concept]
                for v in row["expansions"].split("|"):
                    toks = tuple(tokenize(v))
                    if toks and toks not in vs:
                        vs.append(toks)
                for k in (key, (key[0] + "s",)) if len(key) == 1 else (key,):  # "idp" and "idps"
                    self.phrase_to_concept.setdefault(k, concept)
        self.max_len = max((len(p) for p in self.phrase_to_concept), default=1)

    def find(self, tokens: list[str]) -> list[tuple[int, int, str]]:
        """Greedy longest-match of glossary phrases in a token list -> (start, end, concept)."""
        hits, i = [], 0
        while i < len(tokens):
            for L in range(min(getattr(self, "max_len", 1), len(tokens) - i), 0, -1):
                c = self.phrase_to_concept.get(tuple(tokens[i:i + L]))
                if c:
                    hits.append((i, i + L, c))
                    i += L
                    break
            else:
                i += 1
        return hits
