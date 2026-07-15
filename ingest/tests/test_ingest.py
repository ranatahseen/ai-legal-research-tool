import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest.filters import is_pi_case, validate_extracted_fields
from ingest.loader import clean_text


# test for keyword filtering

class TestIsPiCase:
    # 1 STRONG keyword
    def test_strong_keyword_passes_alone(self):
        text = "This case involves a catastrophic impairment determination."
        assert is_pi_case(text) is True
    
    # 2 gneral keywords
    def test_two_general_keywords_pass(self):
        # Two GENERAL words meets the minimum threshold
        text = "The plaintiff suffered damages in a slip and fall incident."
        assert is_pi_case(text) is True

    # 1 GENERAL keyword
    def test_one_general_keyword_fails(self):
        # Only one GENERAL word — not enough
        text = "The plaintiff filed a breach of contract claim."
        assert is_pi_case(text) is False

    # no PI terms
    def test_non_pi_case_fails(self):
        text = "The respondent terminated the employee without cause."
        assert is_pi_case(text) is False

    # checking case sensitive 
    def test_case_insensitive(self):
        # Keywords should match regardless of case
        text = "CATASTROPHIC IMPAIRMENT was assessed by the designated centre."
        assert is_pi_case(text) is True


# tests for field validation

class TestValidateExtractedFields:

    def _base_record(self):
        return {
            "case_name": "Smith v. Jones",
            "plaintiff_won": True,
            "damages_awarded": 150000,
            "case_summary": "Plaintiff was injured in a slip and fall.",
        }

    def test_valid_record_passes(self):
        passes, missing_soft = validate_extracted_fields(self._base_record())
        assert passes is True

    def test_missing_hard_field_drops_case(self):
        record = self._base_record()
        del record["plaintiff_won"]
        passes, _ = validate_extracted_fields(record)
        assert passes is False

    def test_missing_soft_field_keeps_case(self):
        record = self._base_record()
        passes, missing_soft = validate_extracted_fields(record)
        assert passes is True
        assert "citation" in missing_soft

    def test_all_hard_fields_missing_drops_case(self):
        passes, _ = validate_extracted_fields({})
        assert passes is False


# cleaning text

class TestCleanText:

    def test_removes_null_bytes(self):
        assert "\x00" not in clean_text("hello\x00world")

    def test_collapses_whitespace(self):
        assert clean_text("too   many    spaces") == "too many spaces"

    def test_collapses_newlines(self):
        assert clean_text("line one\n\n\nline two") == "line one line two"

    def test_strips_edges(self):
        assert clean_text("  trimmed  ") == "trimmed"

    def test_empty_string(self):
        assert clean_text("") == ""