
# Layer 0 of the query pipeline.

import json
import time
from typing import Optional

import httpx

from .config import LLM_RETRIES, LLM_TIMEOUT, MODEL_NAME, OLLAMA_URL


# prompt

def _build_prompt(lawyer_query: str) -> str:
    return f"""
You are a legal intake specialist for an Ontario personal injury research tool.

Extract structured facts from the lawyer's case description below.
These facts will filter a database of Ontario PI decisions to find the most
legally comparable cases.

CRITICAL — DETERMINE PERSPECTIVE FIRST:
- "my client was injured / slipped / was hit / suffered"  → perspective: "plaintiff"
- "my client ran a red light / is being sued / hit someone / the other party is claiming"  → perspective: "defense"
- If unclear → perspective: "plaintiff"

Return ONLY valid JSON. No markdown. No explanation. No backticks.

{{
  "perspective": "plaintiff",
  "injury_type":     ["slip_and_fall", "fracture"],
  "injury_severity": "moderate",
  "defendant_type":  ["retailer"],
  "location_type":   ["retail"],
  "liability_theory": "occupiers_liability",
  "plaintiff_age_group": "elderly",
  "plaintiff_employed_at_injury": false,
  "credibility_issue":               false,
  "pre_existing_condition":          false,
  "treatment_gap_present":           false,
  "future_income_loss_claimed":      false,
  "contributory_negligence_found":   true,
  "threshold_motion_brought":        false,
  "municipal_liability_case":        false,
  "causation_issue_likely":          false,
  "query_summary": "Elderly retired nurse fractured hip in slip and fall outside retail store in winter conditions; contributory negligence likely raised."
}}

ALLOWED VALUES:
  perspective:     plaintiff | defense
  injury_type:     slip_and_fall | chronic_pain | soft_tissue | mTBI |
                   fracture | orthopedic | psychological | spinal_cord |
                   burns | amputation | wrongful_death | other
                   NOTE: Do NOT use MVA as an injury_type — it is a mechanism,
                   not an injury. Use soft_tissue, mTBI, fracture etc. instead.
                   MVA is already captured by liability_theory: mva_tort.
  injury_severity: minor | moderate | serious | catastrophic | unknown
  defendant_type:  driver | municipality | retailer | property_owner |
                   insurer | employer | contractor | school_board | other
  location_type:   road | sidewalk | retail | parking_lot | workplace |
                   residential | pedestrian_ramp | stairwell | elevator |
                   public_transit | other
  liability_theory: occupiers_liability | negligence | mva_tort |
                    vicarious_liability | products_liability |
                    medical_malpractice | other

RULES:
- injury_type, defendant_type, location_type are arrays — include all that apply
- liability_theory is a single string — pick the most likely one
- For boolean fields use your best inference. Use false if no signal.
- query_summary: one factual sentence, no legal conclusions.

Lawyer's case description:
{lawyer_query}
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


def extract_query_facts(lawyer_query: str) -> Optional[dict]:
    """
    Convert a lawyer's plain-English case description into a structured fact
    dict for Chroma metadata filtering.

    Retries LLM_RETRIES times on failure before returning None.

    Args:
        lawyer_query: Raw input from the lawyer via Telegram or CLI.

    Returns:
        Structured fact dict, or None if extraction fails after all retries.
    """
    prompt = _build_prompt(lawyer_query)

    for attempt in range(LLM_RETRIES):
        try:
            raw  = _call_ollama(prompt)
            data = _parse(raw)
            return data
        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  query extraction attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  query extraction failed after {LLM_RETRIES} attempts: {e}")
                return None