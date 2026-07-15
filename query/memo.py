
import json
import time
from typing import Optional

import httpx

from .config import LLM_RETRIES, LLM_TIMEOUT, MODEL_NAME, OLLAMA_URL


def _truncate_case_name(name: str, max_len: int = 60) -> str:
    """
    Truncate a case name to a readable length.
    For multi-plaintiff cases like "WHITNEY HORNICK, PAMELA HORNICK..."
    extract the first plaintiff and first defendant only.
    """
    if len(name) <= max_len:
        return name
    # Try to extract first plaintiff v. first defendant
    if " v. " in name or " v " in name:
        sep = " v. " if " v. " in name else " v "
        parts = name.split(sep, 1)
        plaintiff = parts[0].split(",")[0].strip()
        defendant = parts[1].split(",")[0].strip() if len(parts) > 1 else ""
        short = f"{plaintiff}{sep}{defendant}"
        return short if len(short) <= max_len else short[:max_len - 3] + "..."
    return name[:max_len - 3] + "..."


def _format_key_cases(cases: list[dict]) -> str:
    """Format the top cases for inclusion in the memo prompt."""
    lines = []
    for i, case in enumerate(cases, 1):
        m = case.get("metadata", {})
        dmg = m.get("damages_awarded", 0)
        dmg_str = f"${dmg:,}" if isinstance(dmg, int) and dmg > 0 else (
            "split trial" if m.get("damages_to_be_assessed") else "$0"
        )
        citation = m.get('citation', '')
        citation_str = f" [{citation}]" if citation else ""
        case_name = _truncate_case_name(m.get('case_name', 'Unknown'))
        lines.append(
            f"[{i}] {case_name}{citation_str} "
            f"({m.get('year', '?')} {m.get('court', '?')}) — "
            f"{'Won' if m.get('plaintiff_won') else 'Lost'} | {dmg_str}\n"
            f"    Deciding factor: {m.get('deciding_factor', 'N/A')}\n"
            f"    Weakness: {m.get('weakest_point_for_plaintiff', 'N/A')}\n"
            f"    Why relevant: {m.get('relevance_reason', 'N/A')}"
        )
    return "\n\n".join(lines)


def build_memo_prompt(
    lawyer_query: str,
    facts:        dict,
    cases:        list[dict],
    stats:        dict,
) -> str:
    """
    Build the LLM prompt that writes the research memo.

    Args:
        lawyer_query: Original plain-English query from the lawyer.
        facts:        Structured facts extracted by extractor.py.
        cases:        Reranked cases from reranker.py.
        stats:        Aggregated statistics from aggregator.py.
    """
    cases_text   = _format_key_cases(cases)
    damages      = stats.get("damages", {})
    low_sample   = stats.get("low_sample", False)
    sample_note  = (
        f"\nNOTE: Only {stats.get('total_cases')} comparable cases found. "
        "Win rate should be treated as directional, not statistical."
        if low_sample else ""
    )
    excluded_limitations = stats.get("excluded_limitations", 0)
    limitations_note = (
        f"\nNOTE: {excluded_limitations} additional case(s) excluded from win rate "
        "— dismissed on limitations grounds (missed 2-year filing deadline). "
        "These are procedural dismissals, not merits losses."
        if excluded_limitations else ""
    )

    credibility_rate_str = stats.get("credibility_rate", "0%")
    credibility_note = (
        " (near-universal in this case type — treat as baseline, not a specific risk signal)"
        if stats.get("credibility_rate_is_noise")
        else ""
    )

    damages_count = damages.get('count', 0)
    damages_sample_note = " low sample" if damages_count < 5 else ""
    damages_summary = (
        f"Median award: ${damages.get('median', 0):,} | "
        f"Range: ${damages.get('min', 0):,} – ${damages.get('max', 0):,} | "
        f"Based on {damages_count} awards{damages_sample_note}"
        if damages else "Insufficient damages data in comparable cases."
    )

    return f"""
You are a senior Ontario personal injury litigator writing a case stress-test
memo for a colleague. Be direct, specific, and honest. Do not soften bad news.
Use precise legal language. Reference the specific cases provided.
Always include full CanLII citations in the format: Case Name [Year Court Docket].
The CanLII citation is provided for each case — use it exactly as given.
Do not abbreviate or omit the docket number.


CRITICAL FORMATTING RULE:
Every statistic you state must include its sample size in parentheses.
Examples:
  "Win rate: 67% (8 comparable cases)"
  "Median award: $185,000 (5 damages awards)"
  "Credibility contested in 75% of cases (6 of 8 cases)"
Never state a percentage or average without the sample size.
If the sample is below 5, add: "(low sample — treat as directional only)"

CASE SUBMITTED FOR STRESS-TESTING:
{lawyer_query}

EXTRACTED FACT PATTERN:
{json.dumps(facts, indent=2)}

OUTCOME STATISTICS FROM {stats.get('total_cases')} COMPARABLE ONTARIO CASES:
- Win rate on liability: {stats.get('win_rate_pct')} ({stats.get('wins')} outright wins + {stats.get('split_trials')} split trials = {stats.get('liability_wins')} liability wins, {stats.get('losses')} losses) — based on {stats.get('total_cases')} cases{sample_note}{limitations_note}
- Contributory negligence found: {stats.get('cn_rate_pct')} of cases (avg reduction: {stats.get('avg_cn_percentage', 'N/A')}%)
- {damages_summary}
- Credibility contested: {credibility_rate_str} of cases{credibility_note}
- Pre-existing condition raised: {stats.get('pre_existing_rate')} of cases
- Surveillance used: {stats.get('surveillance_rate')} of cases
- Treatment gap raised: {stats.get('treatment_gap_rate')} of cases
- IME conducted: {stats.get('ime_rate')} of cases
- Threshold motion brought: {stats.get('threshold_rate')} of cases | succeeded (plaintiff failed): {stats.get('threshold_success_rate', 'N/A')} of motions brought ({stats.get('threshold_motions_with_outcome', 0)} motions)
- Adverse credibility finding by judge: {stats.get('adverse_credibility_rate', 'N/A')} of cases

MOST RELEVANT PRECEDENTS:
{cases_text}

Write a structured research memo with EXACTLY these seven sections.
Use the section headers exactly as written.

## 1. WIN RATE ASSESSMENT
Start with this exact sentence, filled in:
"Win rate: [X]% ([outright wins] outright wins + [split trials] split trials = [liability wins] liability wins, [losses] losses) from [total] comparable Ontario cases."
Then explain what drives wins and losses. Cite specific deciding factors from the cases above.
If the win rate is below 50%, say so plainly.

## 2. ARGUMENTS IN YOUR FAVOUR
List every argument that supports the plaintiff's case, in order of strength.
For each argument, cite a specific case where that argument succeeded.
Include: legal duty arguments, evidence of systemic failure, case law on
the standard of care, and any facts from the submitted description that
strengthen the claim. This section is as important as the weaknesses.

## 3. DAMAGES BENCHMARK
State the median, range, and what drives higher or lower awards.
Break down which heads of damage (non-pecuniary, future care, income loss)
are typically awarded in cases like this. Flag if damages data is limited.

## 4. OPPOSING COUNSEL PLAYBOOK
List every argument the defense will make, in order of likely impact.
For each argument, cite a specific case where that argument succeeded or failed.
Be exhaustive — the lawyer needs to prepare for all of it.

## 5. CASE WEAKNESSES
Identify the specific vulnerabilities in THIS case based on the submitted
facts. Not generic weaknesses — specific ones given what was described.
Rank them by how likely each is to affect the outcome.
This is the most important section. Be direct.

## 6. KEY PRECEDENTS
List the {len(cases)} most relevant cases with:
- Full citation in format: Case Name [Year Court Docket] if available
- Outcome and damages awarded
- Why it is legally analogous to this case
- What the lawyer can use from it or what it warns against

## 7. SETTLEMENT GUIDANCE
Based on the win rate, damages range, and case weaknesses identified above,
give a specific settlement range recommendation. Explain the reasoning.
Flag any factors that suggest settling early vs. proceeding to trial.

Write the memo now. Be a trusted colleague, not a cautious AI.
""".strip()


def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model":   MODEL_NAME,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.2},  # slight creativity for prose
        },
        timeout=LLM_TIMEOUT,
    )
    body = response.json()
    if "response" not in body:
        raise ValueError(f"Ollama error: {body.get('error', body)}")
    return body["response"].strip()



def format_memo(
    lawyer_query: str,
    facts:        dict,
    cases:        list[dict],
    stats:        dict,
) -> Optional[str]:
    """
    Generate the research memo by calling the local LLM.

    Args:
        lawyer_query: Original plain-English query.
        facts:        Structured facts from extractor.py.
        cases:        Reranked cases from reranker.py.
        stats:        Aggregated stats from aggregator.py.

    Returns:
        The formatted memo as a string, or None if generation fails.
    """
    if not cases:
        return (
            "**No comparable Ontario PI cases found for this fact pattern.**\n\n"
            "This may indicate a rare injury type or unusual liability theory. "
            "Consider broadening the case description or consulting primary sources directly."
        )

    prompt = build_memo_prompt(lawyer_query, facts, cases, stats)

    for attempt in range(LLM_RETRIES):
        try:
            memo = _call_ollama(prompt)

            # Sanity check — verify the first AND last sections are present.
            # Checking only the first two sections (previous behaviour) would
            # let a memo truncated at section 4 or 5 pass silently.
            # With LLM_TIMEOUT = 180 truncation is rare, but this catches it.
            if "WIN RATE ASSESSMENT" not in memo:
                raise ValueError("memo missing section 1 (WIN RATE ASSESSMENT)")
            if "SETTLEMENT GUIDANCE" not in memo:
                raise ValueError("memo missing section 7 (SETTLEMENT GUIDANCE) — likely truncated")

            return memo

        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  memo generation attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  memo generation failed: {e}")
                return None