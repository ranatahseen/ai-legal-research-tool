from collections import Counter
from .config import WIN_RATE_MIN_SAMPLE


def compute_stats(cases: list[dict]) -> dict:
    """
    Compute outcome statistics from a list of reranked cases.

    Args:
        cases: List of case dicts from reranker.py. Each has a 'metadata' key
               with all stored Chroma fields.

    Returns:
        A stats dict consumed by memo.py to write the research memo.
    """
    if not cases:
        return {}

    metadata = [c["metadata"] for c in cases]
    total    = len(metadata)


    excluded_limitations = sum(
        1 for m in metadata
        if m.get("dismissed_on_limitations") is True
    )
    valid = [
        m for m in metadata
        if not m.get("dismissed_on_limitations")
    ]
    valid_total = len(valid)

    wins         = sum(
        1 for m in valid
        if m.get("plaintiff_won") is True
        and not m.get("damages_to_be_assessed")
    )
    split_trials  = sum(1 for m in valid if m.get("damages_to_be_assessed") is True)
    losses        = sum(
        1 for m in valid
        if m.get("plaintiff_won") is False
        and not m.get("damages_to_be_assessed")
    )

    liability_wins = wins + split_trials
    win_rate       = liability_wins / valid_total if valid_total else 0
    low_sample     = valid_total < WIN_RATE_MIN_SAMPLE


    awards = [
        m["damages_awarded"]
        for m in valid
        if isinstance(m.get("damages_awarded"), int)
        and m["damages_awarded"] > 0
    ]

    damages_stats = {}
    if awards:
        awards_sorted          = sorted(awards)
        damages_stats["min"]   = awards_sorted[0]
        damages_stats["max"]   = awards_sorted[-1]
        damages_stats["mean"]  = int(sum(awards) / len(awards))
        mid                    = len(awards_sorted) // 2
        damages_stats["median"] = (
            awards_sorted[mid]
            if len(awards_sorted) % 2 != 0
            else (awards_sorted[mid - 1] + awards_sorted[mid]) // 2
        )
        damages_stats["count"] = len(awards)


    cn_cases = [
        m for m in valid
        if m.get("contributory_negligence_found") is True
    ]
    cn_rate = len(cn_cases) / valid_total if valid_total else 0

    cn_percentages = [
        m["contributory_negligence_percentage"]
        for m in cn_cases
        if isinstance(m.get("contributory_negligence_percentage"), int)
    ]
    avg_cn_percentage = (
        int(sum(cn_percentages) / len(cn_percentages))
        if cn_percentages else None
    )

    deciding_factors = [
        m["deciding_factor"]
        for m in valid
        if m.get("deciding_factor")
    ]

    defense_arguments = [
        m["primary_defense_argument"]
        for m in valid
        if m.get("primary_defense_argument")
    ]

    weaknesses = [
        m["weakest_point_for_plaintiff"]
        for m in valid
        if m.get("weakest_point_for_plaintiff")
    ]

    credibility_rate = sum(
        1 for m in valid if m.get("credibility_issue") is True
    ) / valid_total if valid_total else 0

    pre_existing_rate = sum(
        1 for m in valid if m.get("pre_existing_condition") is True
    ) / valid_total if valid_total else 0

    surveillance_rate = sum(
        1 for m in valid if m.get("surveillance_used") is True
    ) / valid_total if valid_total else 0

    treatment_gap_rate = sum(
        1 for m in valid if m.get("treatment_gap_present") is True
    ) / valid_total if valid_total else 0

    ime_rate = sum(
        1 for m in valid if m.get("ime_used") is True
    ) / valid_total if valid_total else 0

    threshold_rate = sum(
        1 for m in valid if m.get("threshold_motion_brought") is True
    ) / valid_total if valid_total else 0

    causation_theories = Counter(
        m["causation_theory"]
        for m in valid
        if m.get("causation_theory")
    )

    return {
        "total_cases":           valid_total,
        "total_cases_raw":       total,
        "excluded_limitations":  excluded_limitations,
        "wins":                  wins,
        "liability_wins":        liability_wins,
        "losses":                losses,
        "split_trials":          split_trials,
        "win_rate":              round(win_rate, 3),
        "win_rate_pct":          f"{win_rate * 100:.1f}%",
        "low_sample":            low_sample,

        "damages":          damages_stats,

        "cn_rate":          round(cn_rate, 3),
        "cn_rate_pct":      f"{cn_rate * 100:.1f}%",
        "avg_cn_percentage": avg_cn_percentage,

        "deciding_factors":   deciding_factors,
        "defense_arguments":  defense_arguments,
        "weaknesses":         weaknesses,
        "causation_theories": dict(causation_theories),

        "credibility_rate":              f"{credibility_rate * 100:.0f}%",
        "credibility_rate_is_noise":     credibility_rate > 0.85,
        "adverse_credibility_rate":      f"{sum(1 for m in valid if m.get('credibility_finding_adverse') is True) / valid_total * 100:.0f}%" if valid_total else "0%",
        "threshold_success_rate":        f"{sum(1 for m in valid if m.get('threshold_motion_outcome') == 'plaintiff_failed') / max(sum(1 for m in valid if m.get('threshold_motion_brought')), 1) * 100:.0f}%",
        "threshold_motions_with_outcome":sum(1 for m in valid if m.get('threshold_motion_brought')),
        "pre_existing_rate":  f"{pre_existing_rate * 100:.0f}%",
        "surveillance_rate":  f"{surveillance_rate * 100:.0f}%",
        "treatment_gap_rate": f"{treatment_gap_rate * 100:.0f}%",
        "ime_rate":           f"{ime_rate * 100:.0f}%",
        "threshold_rate":     f"{threshold_rate * 100:.0f}%",
    }