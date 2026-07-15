import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from embed.chunker import chunk_case
from embed.config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

# shared fixtures

SAMPLE_CASE = {
    "metadata": {
        "case_name":    "Smith v. TTC",
        "source_file":  "smith_v_ttc.txt",
        "court":        "Ontario Superior Court",
        "year":         2019,
        "plaintiff_won": True,
        "damages_awarded": 187000,
        "injury_type":  ["soft_tissue", "chronic_pain"],
        "defendant_type": ["municipality"],
    },
    "full_text": "A" * 10000,
}

SPARSE_CASE = {
    "metadata": {
        "case_name":   "Jones v. City",
        "source_file": "jones_v_city.txt",
    },
    "full_text": "",
}

NO_NAME_CASE = {
    "metadata": {
        "source_file": "unknown.txt",
    },
    "full_text": "Some text",
}


# metadata chunk tests

# every valid case produces exactly one metadata chunk
def test_metadata_chunk_always_produced():
    chunks = chunk_case(SAMPLE_CASE)
    metadata_chunks = [c for c in chunks if c.metadata["chunk_type"] == "metadata"]
    assert len(metadata_chunks) == 1


# metadata chunk text includes the case name
def test_metadata_chunk_contains_case_name():
    chunks = chunk_case(SAMPLE_CASE)
    meta = next(c for c in chunks if c.metadata["chunk_type"] == "metadata")
    assert "Smith v. TTC" in meta.text

# even a case with only case_name produces a metadata chunk
def test_metadata_chunk_produced_for_sparse_case():
    chunks = chunk_case(SPARSE_CASE)
    metadata_chunks = [c for c in chunks if c.metadata["chunk_type"] == "metadata"]
    assert len(metadata_chunks) == 1, "sparse case should still produce a metadata chunk"

# boolean flags reflect the injury_type and defendant_type lists
def test_boolean_flags_set_correctly():
    chunks = chunk_case(SAMPLE_CASE)
    meta = next(c for c in chunks if c.metadata["chunk_type"] == "metadata")
    assert meta.metadata["has_soft_tissue"]  is True
    assert meta.metadata["has_chronic_pain"] is True
    assert meta.metadata["has_municipality"] is True
    assert meta.metadata["has_fracture"]     is False

# a case with no case_name should produce zero chunks
def test_no_case_name_returns_empty():
    chunks = chunk_case(NO_NAME_CASE)
    assert chunks == [], "expected no chunks for a case with no case_name"


# text chunk tests

# a case with full_text produces at least one text chunk
def test_text_chunks_produced():
    chunks = chunk_case(SAMPLE_CASE)
    text_chunks = [c for c in chunks if c.metadata["chunk_type"] == "text"]
    assert len(text_chunks) > 0

# no text chunk exceeds CHUNK_SIZE_CHARS
def test_text_chunk_size():
    chunks = chunk_case(SAMPLE_CASE)
    for c in chunks:
        if c.metadata["chunk_type"] == "text":
            assert len(c.text) <= CHUNK_SIZE_CHARS, f"chunk {c.chunk_id} exceeds max size"

# 10,000 chars with size=3200 and overlap=400 (step=2800) should produce 4 chunks
def test_text_chunk_count():
    chunks = chunk_case(SAMPLE_CASE)
    text_chunks = [c for c in chunks if c.metadata["chunk_type"] == "text"]
    assert len(text_chunks) == 4, f"expected 4 text chunks, got {len(text_chunks)}"

# text chunk ids are sequentially numbered
def test_text_chunk_id_format():
    chunks = chunk_case(SAMPLE_CASE)
    text_chunks = [c for c in chunks if c.metadata["chunk_type"] == "text"]
    for i, c in enumerate(text_chunks):
        assert c.chunk_id == f"smith_v_ttc.txt::text::{i}"

# a case with no full_text produces zero text chunks
def test_no_text_chunks_when_full_text_empty():
    chunks = chunk_case(SPARSE_CASE)
    text_chunks = [c for c in chunks if c.metadata["chunk_type"] == "text"]
    assert len(text_chunks) == 0

# text chunks carry the same structured metadata as the metadata chunk
def test_text_chunks_carry_metadata_payload():
    chunks = chunk_case(SAMPLE_CASE)
    text_chunk = next(c for c in chunks if c.metadata["chunk_type"] == "text")
    assert text_chunk.metadata["case_name"]       == "Smith v. TTC"
    assert text_chunk.metadata["plaintiff_won"]   is True
    assert text_chunk.metadata["damages_awarded"] == 187000