# extractor.py

# Responsible for all LLM interaction during ingestion.

# Three public functions:
#   extract_case_metadata()   — main extraction pass (20+ fields)
#   extract_damages_fallback() — targeted second pass when damages are null
#   extract_query_facts()     — query-time structured fact extraction

# Private helpers handle HTTP, response parsing, and validation.
# Nothing here touches the filesystem or knows about Chroma.


import json
import re
from typing import Optional

import httpx

from .config import (
    DAMAGES_PLAINTIFF_LOST,
    DAMAGES_REVIEW_FLAG,
    DAMAGES_SPLIT_TRIAL,
    EXTRACTION_TIMEOUT,
    FALLBACK_TIMEOUT,
    MODEL_NAME,
    OLLAMA_URL,
)
from .filters import extract_damages_sections, extract_relevant_sections, validate_extracted_fields
from .prompts import case_extraction_prompt, damages_fallback_prompt, query_extraction_prompt


def _call_ollama(prompt: str, timeout: int) -> str:
    """
    Send a prompt to the local Ollama instance and return the raw response
    string.

    Raises httpx.HTTPError on network failure (caller handles it).
    Uses temperature=0 for deterministic, reproducible extraction.
    """
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    return response.json()["response"].strip()


def _parse_response(raw: str) -> dict:
    """
    Parse a raw LLM response string into a Python dict.
    """
    if "...done thinking." in raw:
        raw = raw.split("...done thinking.")[-1].strip()

    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    raw = raw.replace("```json", "").replace("```", "").strip()

    return json.loads(raw)



def extract_case_metadata(case_text: str, filepath: str) -> Optional[dict]:
    """
    Main extraction pass for a single PI judgment.

    Builds a combined context (general sections + damages sections), calls the
    LLM with the full extraction schema, validates the result, and runs a
    targeted damages fallback if damages_awarded comes back null.

    Returns a fully populated metadata dict, or None if extraction fails.

    Args:
        case_text: Full cleaned text of the judgment.
        filepath:  Source PDF path — stored in the record for resume support.
    """
    general_context = extract_relevant_sections(case_text)
    damages_context = extract_damages_sections(case_text)

    combined_context = (
        f"=== GENERAL CASE TEXT ===\n{general_context}\n\n"
        f"=== DAMAGES-FOCUSED EXCERPTS ===\n{damages_context}"
    )

    prompt = case_extraction_prompt(combined_context)

    try:
        raw = _call_ollama(prompt, timeout=EXTRACTION_TIMEOUT)
        data = _parse_response(raw)
    except Exception as e:
        print(f"  extraction failed: {e}")
        return None


    data["source_file"] = filepath

    passes, missing_soft = validate_extracted_fields(data)
    if not passes:
        return None

    if missing_soft:
        data["needs_review"] = True
        data["missing_fields"] = missing_soft
        print(f"  flagged — missing soft fields: {missing_soft}")

    if data.get("damages_awarded") is None:
        print("  damages null after main pass — running fallback...")
        fallback = extract_damages_fallback(case_text, data.get("case_name", filepath))

        if fallback:
            data["damages_awarded"] = fallback["damages_awarded"]
            data["damages_to_be_assessed"] = fallback.get("damages_to_be_assessed", False)
            data["damages_notes"] = fallback.get("damages_notes", "")
            # Lower confidence to reflect the two-pass uncertainty
            data["extraction_confidence"] = min(
                data.get("extraction_confidence") or 0.5,
                fallback.get("confidence", 0.5),
            )
            print(f"  fallback resolved damages: {data['damages_awarded']}")
        else:
            # Last resort: flag for manual review
            data["damages_awarded"] = (
                DAMAGES_PLAINTIFF_LOST if not data.get("plaintiff_won")
                else DAMAGES_REVIEW_FLAG
            )
            data["damages_notes"] = "Could not extract — flagged for manual review"
            print("  fallback failed — flagged for review")

    return data


def extract_damages_fallback(case_text: str, case_name: str) -> Optional[dict]:
    """
    Focused second-pass extraction for damages_awarded only.

    Called when the main pass returns null for damages_awarded. Pre-extracts
    all dollar amounts via regex and injects them as a cheat sheet so the
    model doesn't have to hunt through the text.

    Returns a dict with:
        damages_awarded (int): amount, 0 (lost), -1 (split trial), never null
        damages_to_be_assessed (bool)
        damages_notes (str)
        confidence (float)

    Returns None only if the HTTP call itself fails.

    Args:
        case_text:  Full cleaned text of the judgment.
        case_name:  Used only for logging.
    """
    dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', case_text)
    damages_section = extract_damages_sections(case_text)

    prompt = damages_fallback_prompt(dollar_amounts, damages_section)

    try:
        raw = _call_ollama(prompt, timeout=FALLBACK_TIMEOUT)
        data = _parse_response(raw)
    except Exception as e:
        print(f"  damages fallback failed for {case_name}: {e}")
        return None

    if "damages_awarded" not in data:
        print(f"  fallback missing damages_awarded for {case_name}")
        return None

    return data


def extract_query_facts(lawyer_query: str) -> Optional[dict]:
    """
    Query-time structured fact extraction.

    Converts a lawyer's plain-English case description into a structured dict
    that drives Chroma metadata pre-filtering (Layer 1 of v2 retrieval).

    Returns a dict with injury_type, defendant_type, location_type, boolean
    flags, and a query_summary — or None if the call fails.

    Args:
        lawyer_query: Raw plain-English input from the lawyer via Telegram.
    """
    prompt = query_extraction_prompt(lawyer_query)

    try:
        raw = _call_ollama(prompt, timeout=FALLBACK_TIMEOUT)
        return _parse_response(raw)
    except Exception as e:
        print(f"  query fact extraction failed: {e}")
        return None