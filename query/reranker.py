
# Layer 3 of the query pipeline.

# Takes the top VECTOR_SEARCH_K candidates from retriever.py and asks the
# local LLM to rank them by genuine legal relevance to the query.

# Why rerank after vector search?
#   Cosine similarity matches language. Legal relevance requires judgment.
#   Two cases can be linguistically similar but legally incomparable —
#   same words, different liability theory, different causation doctrine.
#   The reranker reads both the query and each candidate and makes a legal
#   assessment that cosine similarity cannot.

# Returns the top RERANK_TOP_K cases with a one-sentence relevance reason
# for each — these reasons are used in the memo to explain why each case
# was selected.


import json
import time
from typing import Optional

import httpx

from .config import LLM_RETRIES, LLM_TIMEOUT, MODEL_NAME, OLLAMA_URL, RERANK_TOP_K


# prompt

def _format_candidate(case: dict, index: int) -> str:
    """
    Format a single candidate case for the reranking prompt.
    Only includes fields the LLM needs to assess legal relevance —
    keeping the prompt focused reduces hallucination.
    """
    m = case.get("metadata", {})
    return f"""
[{index}] {m.get('case_name', 'Unknown')} ({m.get('year', '?')} {m.get('court', '?')})
  Injury: {m.get('injury_type', '')} | Severity: {m.get('injury_severity', '')}
  Defendant: {m.get('defendant_type', '')} | Location: {m.get('location_type', '')}
  Liability theory: {m.get('liability_theory', '')}
  Causation theory: {m.get('causation_theory', '')}
  Outcome: {'Plaintiff won' if m.get('plaintiff_won') else 'Plaintiff lost'} | Damages: ${m.get('damages_awarded', 0):,}
  Credibility issue: {m.get('credibility_issue', False)} | Pre-existing: {m.get('pre_existing_condition', False)}
  Surveillance: {m.get('surveillance_used', False)} | Treatment gap: {m.get('treatment_gap_present', False)}
  Deciding factor: {m.get('deciding_factor', '')}
  Weakest point for plaintiff: {m.get('weakest_point_for_plaintiff', '')}
""".strip()


def _build_prompt(lawyer_query: str, candidates: list[dict]) -> str:
    cases_block = "\n\n".join(
        _format_candidate(c, i + 1)
        for i, c in enumerate(candidates)
    )

    return f"""
You are a senior Ontario personal injury litigator reviewing case precedents.

A lawyer has submitted the following case for stress-testing:
"{lawyer_query}"

Below are {len(candidates)} candidate cases retrieved from a database of
Ontario PI decisions. Rank them by genuine legal relevance to the submitted
fact pattern.

Legal relevance means:
- Same injury mechanism (not just similar surface language)
- Same liability theory (occupiers liability vs negligence vs MVA tort)
- Comparable causation doctrine (thin skull, crumbling skull, but-for)
- Similar credibility and evidence profile
- Comparable damages profile (severity, heads of damage)
- Prefer ONSC over LAT for damages guidance

PENALISE cases that:
- Have a different defendant type than the query (e.g. do not select a
  transit/tour bus case when the query involves a private driver)
- Have a different injury mechanism (e.g. do not select a slip-and-fall
  case when the query is about an MVA)
- Are from WSIAT or tribunals (court = 'other' with tribunal-style names)
- Have injury severity dramatically different from the query

Do NOT rank by surface similarity. A case using the same words but a
different liability theory or defendant type is NOT relevant.

Candidate cases:
{cases_block}

Return ONLY valid JSON. No markdown. No explanation.

{{
  "ranked_cases": [
    {{
      "index": 1,
      "relevance_reason": "one sentence explaining why legally analogous"
    }}
  ]
}}

Return exactly {min(RERANK_TOP_K, len(candidates))} cases by their index number.
Most legally relevant first.
""".strip()


# llm call

def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model":   MODEL_NAME,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",
            "options": {"temperature": 0},
        },
        timeout=LLM_TIMEOUT,
    )
    body = response.json()
    if "response" not in body:
        raise ValueError(f"Ollama error: {body.get('error', body)}")
    return body["response"].strip()


def _parse(raw: str) -> dict:
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if "...done thinking." in raw:
        raw = raw.split("...done thinking.")[-1].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def rerank_candidates(
    lawyer_query: str,
    candidates:   list[dict],
) -> list[dict]:
    """
    Rerank candidate cases by legal relevance using the local LLM.

    Each returned case gets a 'relevance_reason' field added to its metadata
    explaining why it was selected. This reason is used in the memo.

    Falls back gracefully:
      - If reranking fails, returns the top RERANK_TOP_K candidates in their
        original vector search order (still better than nothing).
      - If fewer than RERANK_TOP_K candidates exist, returns all of them.

    Args:
        lawyer_query: Original plain-English query.
        candidates:   List of case dicts from retriever.py.

    Returns:
        List of up to RERANK_TOP_K cases, most relevant first, each with
        a 'relevance_reason' in their metadata.
    """
    if not candidates:
        return []

    if len(candidates) <= RERANK_TOP_K:
        # Not enough candidates to rerank — return as-is
        return candidates

    prompt = _build_prompt(lawyer_query, candidates)

    for attempt in range(LLM_RETRIES):
        try:
            raw    = _call_ollama(prompt)
            data   = _parse(raw)
            ranked = data.get("ranked_cases", [])

            if not ranked:
                raise ValueError("reranker returned empty ranked_cases")

            # Reconstruct ordered list from the indices the LLM returned
            reranked = []
            for item in ranked[:RERANK_TOP_K]:
                idx = item.get("index")
                if idx is None or not (1 <= idx <= len(candidates)):
                    # FIX: log invalid indices instead of silently dropping them.
                    # This makes it visible when the reranker hallucinates an
                    # out-of-range index, which helps debug prompt issues.
                    print(f"  reranker returned invalid index {idx} (valid range: 1–{len(candidates)}) — skipped")
                    continue
                case = dict(candidates[idx - 1])  # copy to avoid mutation
                case["metadata"] = dict(case["metadata"])
                case["metadata"]["relevance_reason"] = item.get(
                    "relevance_reason", ""
                )
                reranked.append(case)

            if reranked:
                print(f"  reranked {len(reranked)} cases")
                return reranked

            raise ValueError("no valid indices in reranker response")

        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  reranking attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  reranking failed — using vector search order")
                return candidates[:RERANK_TOP_K]