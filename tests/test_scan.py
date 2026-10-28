"""Label retrieval regressions. Run: python3 -m unittest discover -s tests -v"""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from msf_search.extras import handle_post
from msf_search.extras.scan import _reference_only, _read_label, _read_score, match_label, scan, terms_from
from msf_search.index import build_index
from msf_search.ingest import Doc
from msf_search.search import Searcher

ROOT = Path(__file__).resolve().parents[1]
LABEL = "ORS Oral Rehydration Salts IP Sodium Chloride Potassium Chloride Citrate Glucose Dextrose 20.5g"


class LabelSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        db = str(Path(cls.tmp.name) / "index.db")
        docs = [
            Doc("ors", title="Oral rehydration", genre="Clinical Guideline Chapter", language="en",
                text="Oral rehydration salts (ORS). This is a retrieval test source, not clinical guidance."),
            Doc("ingredients", title="Laboratory supplies", language="en", text="Citrate glucose dextrose."),
            Doc("dictionary", title="Oral rehydration salts", language="en",
                text="Abbreviations and acronyms\nORS\nOral rehydration salts\nIV\nIntravenous"),
            Doc("drug", title="Paracetamol", language="en", text="Paracetamol label retrieval test."),
        ]
        enrichment = {"dictionary": {"summary": "Oral rehydration salts ORS rehydration instructions generated at HQ",
                                      "model": "test", "status": "ok"}}
        with patch("msf_search.index.load_collection", return_value=docs), \
                patch("msf_search.index.load_enrichment", return_value=enrichment):
            build_index("", None, db, verbose=False)
        cls.searcher = Searcher(db, str(ROOT / "data/glossary.csv"),
                               abbreviations_path=str(ROOT / "data/abbreviations.csv"))

    @classmethod
    def tearDownClass(cls):
        cls.searcher.con.close()
        cls.tmp.cleanup()

    def test_ingredients_do_not_hide_recognised_ors(self):
        r = match_label(self.searcher, LABEL)
        self.assertEqual([t["term"] for t in r["terms"]], ["rehydration"])
        self.assertFalse(r["abstained"])
        self.assertEqual(r["results"][0]["doc_id"], "ors")
        self.assertNotIn("ingredients", [x["doc_id"] for x in r["results"]])

    def test_ocr_spellings_and_translations(self):
        for name in ["ORS", "O.R.S.", "O R S", "S.R.O.", "SRO", "Sels de réhydratation orale"]:
            with self.subTest(name=name):
                self.assertEqual(terms_from(self.searcher, name)[0]["term"], "rehydration")

    def test_does_not_guess_a_damaged_name(self):
        for text in ["0RS", "[?]", "XXZZQQ batch 1234", "Store below 25 C IP BP USP"]:
            with self.subTest(text=text):
                r = match_label(self.searcher, text)
                self.assertTrue(r["abstained"])
                self.assertEqual(r["results"], [])

    def test_word_fallback_keeps_drug_names(self):
        r = match_label(self.searcher, "PARACETAMOL 500mg batch 1234 store below 25 C")
        self.assertEqual(r["results"][0]["doc_id"], "drug")

    def test_generated_summary_is_replaced_by_original_evidence(self):
        rows = match_label(self.searcher, "ORS", k=10)["results"]
        ref = next(x for x in rows if x["doc_id"] == "dictionary")
        self.assertTrue(ref["reference_only"])
        self.assertFalse(ref["provenance"]["machine_generated"])
        self.assertEqual(ref["matched_via"], "text")
        self.assertNotIn("generated at HQ", ref["snippet"])

    def test_low_confidence_photo_never_searches(self):
        with patch("msf_search.extras.scan.read_label", return_value={
                "ok": True, "tier": "photo_only", "text": LABEL, "mean_conf": 30}):
            r = scan(self.searcher, b"fake photo")
        self.assertFalse(r["ok"])
        self.assertEqual(r["results"], [])

    def test_upload_and_corrected_text_have_same_matches(self):
        with patch("msf_search.extras.scan.read_label", return_value={
                "ok": True, "tier": "ok", "text": LABEL, "mean_conf": 90}):
            photo = scan(self.searcher, b"fake photo")
        body = json.dumps({"text": LABEL}).encode()
        code, payload, _ = handle_post("/api/scan/text", {"Content-Length": str(len(body))},
                                       io.BytesIO(body), self.searcher)
        typed = json.loads(payload)
        self.assertEqual(code, 200)
        self.assertEqual(typed["input_source"], "user")
        self.assertEqual(photo["query"], typed["query"])
        self.assertEqual(photo["results"], typed["results"])

    def test_invalid_manual_requests(self):
        for body in [b"[]", b"null", b"{}", b'{"text": 123}', b'{"text":""}', b"\xff"]:
            code, _, _ = handle_post("/api/scan/text", {"Content-Length": str(len(body))},
                                      io.BytesIO(body), self.searcher)
            self.assertEqual(code, 400)


class LabelOCRTests(unittest.TestCase):
    def test_retries_unclear_layout(self):
        tess = Mock()
        tess.transcribe.side_effect = [
            {"text": "[?]", "mean_conf": 20}, {"text": "ORS", "mean_conf": 90}]
        with patch("msf_search.extras.scan._engines", return_value=(tess, None, "")):
            r = _read_label([Path("unused.jpg")], "eng")
        self.assertEqual(r["text"], "ORS")
        self.assertEqual(r["tier"], "ok")
        self.assertEqual([call.kwargs["psm"] for call in tess.transcribe.call_args_list], [11, 6])

    def test_missing_engine_and_failed_retry(self):
        with patch("msf_search.extras.scan._engines", return_value=(None, None, "not installed")):
            self.assertFalse(_read_label([Path("unused.jpg")], "eng")["ok"])
        tess = Mock()
        tess.transcribe.side_effect = [{"text": "[?]", "mean_conf": 20}, TimeoutError()]
        with patch("msf_search.extras.scan._engines", return_value=(tess, None, "")):
            self.assertEqual(_read_label([Path("unused.jpg")], "eng")["tier"], "photo_only")

    def test_full_name_beats_high_confidence_logo(self):
        self.assertGreater(_read_score({"mean_conf": 85, "text": "PARACETAMOL 500mg tablets", "reliable_chars": 20}),
                           _read_score({"mean_conf": 99, "text": "RX", "reliable_chars": 2}))

    def test_rotations_checked_despite_confident_first_fragment(self):
        tess = Mock(langs="eng", missing=[])
        tess.transcribe.side_effect = [
            {"text": "RX", "mean_conf": 99, "reliable_chars": 2},
            {"text": "PARACETAMOL 500mg", "mean_conf": 92, "reliable_chars": 15}]
        with patch("msf_search.extras.scan._engines", return_value=(tess, None, "")):
            r = _read_label([Path("sideways.png"), Path("upright.png")], "eng")
        self.assertIn("PARACETAMOL", r["text"])

    def test_reference_lists_with_flattened_lines(self):
        self.assertTrue(_reference_only("", "| ACE | enzyme | | ACT | treatment | | ORS | salts | | IV | intravenous |"))
        self.assertTrue(_reference_only("", "On this page Appendix 1. Samples Appendix 2. Supplies Appendix 3. Rehydration"))
        self.assertFalse(_reference_only("Rehydration", "See Appendix 1 for further information."))


if __name__ == "__main__":
    unittest.main()
