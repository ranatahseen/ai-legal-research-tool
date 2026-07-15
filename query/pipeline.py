# flow:
#   layer 0 — extract structured facts from plain English query
#   layer 1 — metadata pre-filter in Chroma
#   layer 2 — vector similarity search within filtered subset
#   layer 3 — LLM reranks candidates by legal relevance
#   layer 4 — aggregate stats from reranked cases
#   layer 5 — LLM writes the research memo


import time
from typing import Optional

from .config import MEMO_TOP_K
from .extractor  import extract_query_facts
from .retriever  import retrieve_candidates
from .reranker   import rerank_candidates
from .aggregator import compute_stats
from .memo       import format_memo


def run_query(
    lawyer_query: str,
    return_stats: bool = False,
) -> Optional[str | tuple]:
    """
    Run the full query pipeline for a single lawyer input.

    Args:
        lawyer_query:  Plain-English case description from the lawyer.
        return_stats:  If True, returns (memo, stats, reranked_cases) tuple
                       instead of just the memo string. Used by __main__.py
                       to capture cases for chat mode and baseline stats for
                       what-if diffing.

    Returns:
        memo string, or (memo, stats, reranked_cases) tuple if return_stats=True,
        or None if the pipeline fails at any critical step.
    """
    start = time.time()
    print(f"\n{'='*56}")
    print(f"  QUERY PIPELINE")
    print(f"{'='*56}")
    print(f"  Input: {lawyer_query[:100]}{'...' if len(lawyer_query) > 100 else ''}\n")

    # layer 0: extract structured facts
    print("[ Layer 0 ] Extracting structured facts...")
    facts = extract_query_facts(lawyer_query)

    if facts is None:
        print("  FAILED — could not extract facts from query")
        return None

    print(f"  perspective:      {facts.get('perspective', 'plaintiff')}")
    print(f"  injury_type:      {facts.get('injury_type')}")
    print(f"  liability_theory: {facts.get('liability_theory')}")
    print(f"  defendant_type:   {facts.get('defendant_type')}")
    print(f"  query_summary:    {facts.get('query_summary')}\n")

    if facts.get("perspective") == "defense":
        print(
            "  ⚠  Defense-side query detected. This tool is optimised for\n"
            "     plaintiff-side analysis. Win rates and arguments reflect\n"
            "     the plaintiff's perspective in comparable cases.\n"
            "     Use results to understand what the plaintiff will argue.\n"
        )

    # layers 1 + 2: retrieve candidates
    print("[ Layers 1+2 ] Metadata filter + vector search...")
    candidates = retrieve_candidates(lawyer_query, facts)

    if not candidates:
        print("  FAILED — no candidates retrieved")
        return None

    print(f"  {len(candidates)} candidates retrieved\n")

    # layer 3: rerank
    print("[ Layer 3 ] Reranking by legal relevance...")
    reranked = rerank_candidates(lawyer_query, candidates)

    if not reranked:
        print("  FAILED — reranking returned no results")
        return None

    print(f"  {len(reranked)} cases selected for memo\n")

    # layer 4: aggregate stats
    print("[ Layer 4 ] Aggregating outcome statistics...")
    stats = compute_stats(candidates)

    print(f"  win rate:  {stats.get('win_rate_pct')}  ({stats.get('total_cases')} cases)")
    if stats.get("damages"):
        print(f"  median $:  ${stats['damages'].get('median', 0):,}")
    print()

    # layer 5: generate memo
    memo_cases = reranked[:MEMO_TOP_K]
    print("[ Layer 5 ] Generating research memo...")
    memo = format_memo(lawyer_query, facts, memo_cases, stats)
    elapsed = time.time() - start

    if memo:
        print(f"\n  Memo generated in {elapsed:.1f}s")
    else:
        print(f"\n  Memo generation failed after {elapsed:.1f}s")

    if return_stats:
        return (memo, stats, reranked) if memo else None
    return memo


def run_what_if_with_diff(
    original_query: str,
    modification:   str,
    baseline_stats: dict,
    merged_query:   str | None = None,
) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    Re-run the full pipeline with a modified fact pattern and return
    a stat diff, the new memo, and the new stats.

    Args:
        original_query: The original plain-English case description.
        modification:   The what-if change from the lawyer.
        baseline_stats: Stats dict from the original run for diffing.
        merged_query:   LLM-merged coherent query (preferred). If None,
                        falls back to appending modification to original.

    Returns:
        (diff_summary, new_memo, scenario_stats) — any may be None on failure.
    """
    from .followup import compute_diff, format_diff

    modified_query = merged_query or f"{original_query}\n\nMODIFIED FACT FOR THIS ANALYSIS: {modification}"

    print(f"\n  Modification: {modification}")
    if merged_query:
        print(f"  Merged query:\n{merged_query}\n")
    else:
        print()

    facts = extract_query_facts(modified_query)
    if not facts:
        return None, None, None

    candidates = retrieve_candidates(modified_query, facts)
    if not candidates:
        return None, None, None

    reranked = rerank_candidates(modified_query, candidates)
    if not reranked:
        return None, None, None

    scenario_stats = compute_stats(candidates)
    diff           = compute_diff(baseline_stats, scenario_stats)
    diff_summary   = format_diff(diff, modification)

    memo = format_memo(modified_query, facts, reranked[:MEMO_TOP_K], scenario_stats)

    return diff_summary, memo, scenario_stats