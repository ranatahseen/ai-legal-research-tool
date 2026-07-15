
# Follow-up helpers for the query pipeline.

# Public functions:
#   merge_whatif_into_query()       — LLM merges a change into a description,
#                                     removing contradictions
#   compute_diff()                  — stat delta between two pipeline runs
#   format_diff()                   — human-readable diff summary
#   answer_clarification()          — answers a question from the existing memo
#   generate_adverse_stress_test()  — defense's strongest case against the file
#   generate_final_description()    — merges selected follow-ups at export time
#                                     (kept for compatibility; export flow now

# Removed from previous version:
#   classify_followup()      — replaced by explicit prefix routing in __main__
#   build_modified_query()   — replaced by merge_whatif_into_query() everywhere

import time
from typing import Optional

import httpx

from .config import LLM_RETRIES, LLM_TIMEOUT, MODEL_NAME, OLLAMA_URL



def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model":   MODEL_NAME,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.2},
        },
        timeout=LLM_TIMEOUT,
    )
    body = response.json()
    if "response" not in body:
        raise ValueError(f"Ollama error: {body.get('error', body)}")
    return body["response"].strip()


def interpret_whatif(original_query: str, whatif: str) -> Optional[str]:
    """
    Convert a vague what-if into an explicit, unambiguous fact statement
    that can be cleanly applied to the original case description.

    This is the first step in the what-if flow. The interpreter reads the
    original case and the what-if together and produces a precise statement
    of what changes — removing ambiguity before the merge runs.

    Examples:
      "she went through 3 months of physiotherapy"
      → "She had no gap in physiotherapy treatment. She attended
         physiotherapy consistently throughout her recovery."

      "she has a social media post showing her hiking"
      → "There is a social media post showing the plaintiff hiking,
         which the defense could use as a credibility attack."

    Args:
        original_query: Current canonical case description.
        whatif:         Raw what-if text from the lawyer.

    Returns:
        Explicit fact statement string, or None if the call fails.
    """
    prompt = f"""You are a legal assistant helping clarify a hypothetical change to a case.

ORIGINAL CASE:
{original_query}

THE LAWYER ASKS: "what if {whatif}"

Your job is to interpret exactly what factual change the lawyer is proposing,
given the context of the original case.

Rules:
- Read the original case to understand what is currently true
- Determine what the what-if is changing relative to the original
- State the change as a precise, explicit fact (not a question)
- If the original mentions a problem/gap/risk and the what-if removes it,
  explicitly state that the problem/gap/risk does NOT exist
- Be specific — do not be vague or hedge

Examples:
- Original has "3-month gap in physiotherapy" + what-if "she went through physiotherapy"
  → "The plaintiff had no gap in physiotherapy. She attended physiotherapy
     consistently for the full duration of her recovery without interruption."

- Original has "defense IME says fully recovered" + what-if "there is no defense IME"
  → "There is no defense IME. The defense has not obtained an independent
     medical examination."

- Original has no social media mention + what-if "she has hiking photos on social media"
  → "The plaintiff has social media posts showing her hiking, which post-date
     the accident and could be used by the defense as a credibility attack."

Return ONLY the explicit fact statement. No preamble, no explanation.

Explicit fact statement:""".strip()

    for attempt in range(LLM_RETRIES):
        try:
            return _call_ollama(prompt)
        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  interpret attempt {attempt + 1} failed ({e}) — retrying...")
                import time
                time.sleep(1)
            else:
                print(f"  interpret failed: {e}")
                return None



def merge_whatif_into_query(original_query: str, change: str) -> Optional[str]:
    """
    Use the LLM to merge a change (new fact or what-if) into the original
    case description, producing a single coherent description with no
    contradictions.

    Used for both 'fact' additions (canonical update) and 'whatif' runs
    (temporary hypothetical only).

    Rules applied by the LLM:
      - Keep all original facts unless the change directly contradicts one
      - Where a contradiction exists, use the new version, remove the old
      - Do not add legal conclusions or analysis
      - Match the plain-English style of the original

    Args:
        original_query: Current canonical case description.
        change:         New fact or what-if modification from the lawyer.

    Returns:
        Rewritten case description, or None if the LLM call fails.
    """
    # Split original into sentences so we can show the LLM exactly what
    # it is working with, sentence by sentence. This prevents it from
    # "forgetting" sentences that contradict the change.
    import re as _re
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', original_query.strip()) if s.strip()]
    numbered_sentences = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))

    prompt = f"""You are a legal editor. Your job is to rewrite a case description by applying exactly one change.

ORIGINAL CASE DESCRIPTION (each sentence numbered):
{numbered_sentences}

CHANGE TO APPLY:
"{change}"

STEP 1 — IDENTIFY CONTRADICTIONS:
Read every numbered sentence above. Mark any sentence that directly contradicts the change.
A contradiction means: the sentence asserts the OPPOSITE of what the change asserts.
Example: change says "she attended physiotherapy" → any sentence mentioning a gap in physiotherapy is a contradiction.
Example: change says "no defense IME" → any sentence mentioning a defense IME is a contradiction.

STEP 2 — REWRITE:
- DELETE every sentence you marked as a contradiction. Do not soften or qualify it — delete it entirely.
- ADD the change as a new sentence in a natural position.
- KEEP all other sentences exactly as written.

OUTPUT RULES:
- Return ONLY the rewritten case description as flowing prose.
- Do NOT include sentence numbers.
- Do NOT include any explanation, preamble, or commentary.
- Do NOT add any new facts beyond what the change states.
- Do NOT add legal conclusions or analysis.

Rewritten case description:""".strip()

    for attempt in range(LLM_RETRIES):
        try:
            return _call_ollama(prompt)
        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  merge attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  merge failed: {e}")
                return None



def compute_diff(baseline_stats: dict, scenario_stats: dict) -> dict:
    """
    Compute the difference between two sets of aggregated stats.

    Called after a what-if re-run to show what changed relative to the
    current canonical baseline.

    Args:
        baseline_stats: Stats dict from the canonical case pipeline run.
        scenario_stats: Stats dict from the hypothetical pipeline run.

    Returns:
        Diff dict with before/after/delta for key metrics.
    """
    def _delta(a, b):
        if a is not None and b is not None:
            return round(b - a, 1)
        return None

    b_win = baseline_stats.get("win_rate")
    s_win = scenario_stats.get("win_rate")

    b_dmg = baseline_stats.get("damages", {}).get("median")
    s_dmg = scenario_stats.get("damages", {}).get("median")

    b_cn  = baseline_stats.get("cn_rate")
    s_cn  = scenario_stats.get("cn_rate")

    b_tot = baseline_stats.get("total_cases")
    s_tot = scenario_stats.get("total_cases")

    return {
        "win_rate_before":       b_win,
        "win_rate_after":        s_win,
        "win_rate_delta":        _delta(b_win, s_win),

        "median_damages_before": b_dmg,
        "median_damages_after":  s_dmg,
        "damages_delta":         _delta(b_dmg, s_dmg) if b_dmg and s_dmg else None,

        "cn_rate_before":        round(b_cn * 100, 1) if b_cn is not None else None,
        "cn_rate_after":         round(s_cn * 100, 1) if s_cn is not None else None,

        "cases_before":          b_tot,
        "cases_after":           s_tot,
    }


def format_diff(diff: dict, modification: str) -> str:
    """
    Format the stat diff as a readable summary shown before the
    hypothetical memo.

    FIX: win_rate is stored as a decimal (e.g. 0.733). Multiply by 100
    before display so it shows as 73.3% not 0.733%.
    """
    _SUBDIV = "─" * 58

    lines = [
        f"\n{_SUBDIV}",
        f"  WHAT CHANGED: {modification}",
        _SUBDIV,
    ]

    # Win rate — stored as decimal, display as percentage
    b_win = diff.get("win_rate_before")
    a_win = diff.get("win_rate_after")
    if b_win is not None and a_win is not None:
        b_pct   = round(b_win * 100, 1)
        a_pct   = round(a_win * 100, 1)
        delta   = round(a_pct - b_pct, 1)
        arrow   = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(
            f"  Win rate:      {b_pct}% → {a_pct}%  {arrow} {delta:+.1f}%"
        )

    # Median damages
    b_dmg = diff.get("median_damages_before")
    a_dmg = diff.get("median_damages_after")
    if b_dmg and a_dmg:
        delta = diff.get("damages_delta") or 0
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(
            f"  Median award:  ${b_dmg:,} → ${a_dmg:,}  {arrow} ${delta:+,.0f}"
        )

    # CN rate (already multiplied by 100 in compute_diff)
    b_cn = diff.get("cn_rate_before")
    a_cn = diff.get("cn_rate_after")
    if b_cn is not None and a_cn is not None:
        lines.append(f"  CN rate:       {b_cn}% → {a_cn}%")

    # Sample size
    b_tot = diff.get("cases_before")
    a_tot = diff.get("cases_after")
    if b_tot and a_tot:
        lines.append(f"  Comparable cases: {b_tot} → {a_tot}")

    lines.append(_SUBDIV)
    return "\n".join(lines)


def answer_clarification(
    question:       str,
    canonical_query: str,
    memo:           str,
) -> Optional[str]:
    """
    Answer a question using the existing memo as context.
    Does NOT re-run the pipeline. One LLM call.

    Args:
        question:        The lawyer's question.
        canonical_query: Current canonical case description.
        memo:            Full memo already generated for this case.

    Returns:
        Answer string, or None if the call fails.
    """
    prompt = f"""You are a senior Ontario personal injury litigator.

A lawyer submitted this case:
{canonical_query}

You produced this research memo:
{memo}

The lawyer now asks: {question}

Answer directly and specifically based on the memo above.
- Reference specific cases and statistics from the memo where relevant
- Be concise — this is a follow-up, not a new memo
- If the question cannot be answered from the memo, say so plainly
  and suggest what additional research would help
- Use precise legal language
""".strip()

    for attempt in range(LLM_RETRIES):
        try:
            return _call_ollama(prompt)
        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  clarification attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  clarification failed: {e}")
                return None



def generate_adverse_stress_test(
    canonical_query: str,
    memo:            str,
) -> Optional[str]:
    """
    Generate an adverse stress test — the defense's strongest case.

    Always runs against the canonical case description (original +
    accumulated facts). What-ifs do not affect this.

    Args:
        canonical_query: Current canonical case description.
        memo:            Full memo already generated for the canonical case.

    Returns:
        Adverse stress test string, or None if the call fails.
    """
    prompt = f"""You are the most aggressive defense counsel in Ontario.

A plaintiff's lawyer has submitted this case:
{canonical_query}

Their research memo contains this analysis:
{memo}

Your job is to destroy this case. Argue the defense's strongest possible
version of events. Be ruthless, specific, and legally precise.

Find every weakness. Exploit every gap. Use the precedents in the memo
against the plaintiff where possible.

Write the adverse stress test using EXACTLY this structure:

ADVERSE STRESS TEST
===================

OPENING ASSESSMENT
One sentence: is this case strong, moderate, or weak for the plaintiff,
and what is the single biggest vulnerability.

DEFENSE ARGUMENTS (ranked by impact)

For each argument:
- State the argument precisely as defense counsel would frame it
- Name the specific evidence or facts that support it
- Cite a case from the memo where this argument succeeded if possible
- State the likely impact: FATAL | HIGH | MEDIUM | LOW

THRESHOLD MOTION ANALYSIS
Will defense bring a threshold motion? What are the odds it succeeds?
What specific facts make the threshold argument strong or weak here?

CREDIBILITY ATTACK PLAN
How will defense attack the plaintiff's credibility specifically?
What inconsistencies, gaps, or facts will they exploit?

IME STRATEGY
How will defense use their IME? What questions will they ask their expert?
What will they argue the IME proves?

PROBABILITY OF $0 OUTCOME
State a percentage and explain the specific combination of factors
that would lead to a complete defense win.

WHAT PLAINTIFF COUNSEL MUST DO
Three specific actions the plaintiff's lawyer must take NOW to prevent
the defense from succeeding with these arguments. Be direct.

Be a ruthless defense counsel. Do not soften the analysis.
The plaintiff's lawyer needs to see the worst case to prepare for it.
""".strip()

    for attempt in range(LLM_RETRIES):
        try:
            return _call_ollama(prompt)
        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  stress test attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  stress test failed: {e}")
                return None


def generate_final_description(
    original_query:     str,
    selected_followups: list[str],
) -> Optional[str]:
    """
    Merge a list of selected follow-ups into the original query.

    Kept for backwards compatibility. In the new flow, the canonical_query
    is updated incrementally on each 'fact' addition, so this function is
    only needed if the caller wants to reconstruct a description from a
    log of changes rather than using the live canonical query.

    Args:
        original_query:     The original plain-English case description.
        selected_followups: Follow-up texts to incorporate.

    Returns:
        Rewritten case description, or None if the call fails.
    """
    if not selected_followups:
        return original_query

    followups_text = "\n".join(f"- {f}" for f in selected_followups)

    prompt = f"""A lawyer submitted this original case description:

{original_query}

They want to incorporate the following changes and additional facts:
{followups_text}

Rewrite the case description to incorporate ALL of these changes.
Rules:
- The original facts are the base — keep everything unless a change
  contradicts it
- Where a change contradicts an original fact, use the new version and
  remove the old one
- Do not add new facts, legal conclusions, or analysis
- Write in the same plain-English style as the original
- Return only the rewritten case description, nothing else

Rewritten case description:""".strip()

    for attempt in range(LLM_RETRIES):
        try:
            return _call_ollama(prompt)
        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  final description attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  final description generation failed: {e}")
                return None