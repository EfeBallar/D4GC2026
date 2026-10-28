"""Mine abbreviation definitions from the collection itself: "severe acute malnutrition (SAM)".

Preprocessing step (runs at HQ, no LLM, no external dictionary). Every expansion is one the
MSF documents define themselves, at least twice. The output, data/abbreviations.csv, is loaded
by the glossary so a query for "SAM" also searches "severe acute malnutrition". Coordinators
can edit or delete rows in a spreadsheet, just like glossary.csv.
"""
import re
from collections import Counter, defaultdict

from .textnorm import STOPWORDS, tokenize

# "(SAM)", "(MoH)", "(IDPs)": 2-10 chars, starts with a capital.
_SHORT_IN_PARENS = re.compile(r"\(([A-Z][A-Za-z0-9]{1,9})\)")
# "SAM (severe acute malnutrition)": the reverse order.
_LONG_IN_PARENS = re.compile(r"\b([A-Z][A-Za-z0-9]{1,9}) \(([^()\n]{4,100})\)")
_BOUNDARY = re.compile(r"[.;:!?()\[\]\n]")
_ROMAN = re.compile(r"^[IVXLC]+$")
_WORD = re.compile(r"\w+")
_CLEAN = re.compile(r"^[^\W\d_]+(?:[ \-–][^\W\d_]+)*$")  # letters, spaces, hyphens only


def _short_form(s: str) -> str | None:
    if s[-1] == "s" and s[:-1].isupper():  # plural: IDPs -> IDP
        s = s[:-1]
    if sum(ch.isupper() for ch in s) < 2 or _ROMAN.match(s) or s.isdigit():
        return None
    return s


def best_long_form(short: str, long: str) -> str | None:
    """Schwartz & Hearst (2003): shortest tail of `long` containing the short form's letters in
    order, with its first letter at the start of a word. "postpartum haemorrhage" -> PPH."""
    s, l = len(short) - 1, len(long) - 1
    while s >= 0:
        c = short[s].lower()
        if not c.isalnum():
            s -= 1
            continue
        while l >= 0 and (long[l].lower() != c or (s == 0 and l > 0 and long[l - 1].isalnum())):
            l -= 1
        if l < 0:
            return None
        l -= 1
        s -= 1
    return long[long.rfind(" ", 0, l + 1) + 1:]


def _plausible(short: str, long: str) -> bool:
    toks = tokenize(long)
    return (bool(toks) and toks[0] not in STOPWORDS and len(long) > len(short)
            and short.lower() not in toks and len(toks) <= len(short) + 3)


def mine_abbreviations(texts, min_count: int = 2, min_uses: int = 5, max_expansions: int = 4) -> list[dict]:
    pairs = Counter()                     # (short, normalised long) -> occurrences
    pair_docs = defaultdict(set)
    surface = defaultdict(Counter)        # normalised long -> how it was written
    words = Counter()                     # case-sensitive word counts, for the collision guard
    for doc_i, text in enumerate(texts):
        words.update(_WORD.findall(text))
        found = []
        for m in _SHORT_IN_PARENS.finditer(text):
            short = _short_form(m.group(1))
            if not short:
                continue
            before = text[max(0, m.start() - 250):m.start()]
            cut = [b.end() for b in _BOUNDARY.finditer(before)]
            window = before[cut[-1] if cut else 0:].split()
            window = window[-min(len(short) + 5, 2 * len(short)):]
            long = best_long_form(short, " ".join(window))
            if long:
                found.append((short, long))
        for m in _LONG_IN_PARENS.finditer(text):
            short, inside = _short_form(m.group(1)), m.group(2).strip()
            if short and best_long_form(short, inside) == inside:
                found.append((short, inside))
        for short, long in found:
            long = long.strip(" ,-–'\"‘’“”")
            if not _plausible(short, long):
                continue
            key = (short, " ".join(tokenize(long)))
            pairs[key] += 1
            pair_docs[key].add(doc_i)
            surface[key[1]][long.lower()] += 1

    by_short = defaultdict(list)
    for (short, norm), n in pairs.items():
        by_short[short].append((n, norm))

    rows = []
    for short, exps in by_short.items():
        uses = words[short] + words[short + "s"]
        # Collision guard: queries are lower-cased, so skip abbreviations used more often as an
        # ordinary word in the library ("WHO" vs "who", "CAR" vs French "car" = because).
        lower = short.lower()
        if lower in STOPWORDS or words[lower] > max(3, uses):
            continue
        exps = [(n, norm) for n, norm in exps if _CLEAN.match(surface[norm].most_common(1)[0][0])]
        # Defined at least twice; failing that, defined once but used often (P23 defines PPH once).
        strong = [(n, norm) for n, norm in exps if n >= min_count]
        exps = sorted(strong or (exps if uses >= min_uses else []), reverse=True)
        if not exps:
            continue
        top = exps[0][0]
        keep = [(n, norm) for n, norm in exps if n >= 0.25 * top][:max_expansions]
        rows.append({
            "abbr": short,
            "expansions": "|".join(surface[norm].most_common(1)[0][0] for _, norm in keep),
            "count": sum(n for n, _ in keep),
            "docs": len(set().union(*(pair_docs[(short, norm)] for _, norm in keep))),
        })
    rows.sort(key=lambda r: (-r["count"], r["abbr"]))
    return rows
