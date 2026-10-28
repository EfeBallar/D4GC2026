"""Text normalisation shared by indexing and querying.

Everything here is pure standard library so the field device needs nothing extra.
The same function is applied to documents at build time and to queries at search
time, which is what makes accent-insensitive, Arabic-prefix-insensitive matching work.
"""
import re
import unicodedata

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Arabic: diacritics (tashkeel), tatweel, letter variants, common clitic prefixes.
_AR_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def _norm_arabic_token(tok: str) -> str:
    tok = _AR_TASHKEEL.sub("", tok)
    tok = (tok.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
              .replace("ى", "ي").replace("ة", "ه"))
    for p in _AR_PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            tok = tok[len(p):]
            break
    else:
        # bare conjunction "و" + word
        if tok.startswith("و") and len(tok) >= 5:
            tok = tok[1:]
    return tok


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_token(tok: str) -> str:
    tok = tok.lower()
    if _ARABIC_CHAR.search(tok):
        return _norm_arabic_token(tok)
    return strip_accents(tok)


def tokenize(text: str) -> list[str]:
    """Lower-cased, accent-stripped, Arabic-normalised tokens."""
    return [normalize_token(t) for t in TOKEN_RE.findall(text)]


def normalize_for_index(text: str) -> str:
    return " ".join(tokenize(text))


STOPWORDS = set("""
a an the and or of to in on for with by at from as is are was were be been being this that these those it its
into than then there their them they we our you your he she his her not no do does did can could should would
may might will shall what which who whom whose when where why how about over under after before between during
also such any all some more most other only very per via vs if so up out each both
best find show me need want please information info document documents give tell i my
le la les un une des du de d l et ou en dans sur pour par avec au aux ce ces cet cette est sont qui que quoi
dont ou plus moins pas ne se sa son ses leur leurs nous vous il elle ils elles comme chez
comment quel quelle quels quelles combien pourquoi quand faut doit atteint atteinte atteints
el los las una unos unas del y o en por para con sin que es son su sus al lo como mas pero
cual cuales cuando donde porque debo hay
في من على الى إلى عن مع هذا هذه ذلك التي الذي ان أن او أو ما لا كما قد
""".split())
STOPWORDS = {normalize_token(w) for w in STOPWORDS}
