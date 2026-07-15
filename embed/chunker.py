# Splits a case record into chunks ready for embedding.

# Two chunk types per case:
#   1. Metadata chunk  — structured summary of extracted fields
#   2. Text chunks     — overlapping windows of the full case text


from dataclasses import dataclass, field
from typing import Any

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


@dataclass
class Chunk:
    """A single embeddable unit of content from a case."""
    text:     str
    chunk_id: str
    metadata: dict[str, Any]


def _make_metadata_payload(m: dict) -> dict:
    """
    Build a flat Chroma-compatible metadata dict.

    All values must be str, int, float, or bool — no lists, no nulls.
    Multi-value fields stored as both display strings AND boolean flags.
    """
    def _str(val) -> str:
        if val is None: return ""
        if isinstance(val, list): return ", ".join(str(v) for v in val)
        return str(val)

    def _int(val) -> int:
        return val if isinstance(val, int) else 0

    def _bool(val) -> bool:
        return bool(val) if val is not None else False

    def _has(field_val, value: str) -> bool:
        """Check if value is in a list or comma-separated string field."""
        if not field_val:
            return False
        if isinstance(field_val, list):
            return value in field_val
        if isinstance(field_val, str):
            return value in [v.strip() for v in field_val.split(",")]
        return False

    injury_types    = m.get("injury_type", []) or []
    defendant_types = m.get("defendant_type", []) or []
    location_types  = m.get("location_type", []) or []

    return {
        # identity
        "case_name":     _str(m.get("case_name")),
        "citation":      _str(m.get("citation")),
        "court":         _str(m.get("court")),
        "year":          _int(m.get("year")),
        "decision_type": _str(m.get("decision_type")),
        "source_file":   _str(m.get("source_file")),

        # outcome
        "plaintiff_won":          _bool(m.get("plaintiff_won")),
        "damages_awarded":        _int(m.get("damages_awarded") or 0),
        "damages_to_be_assessed": _bool(m.get("damages_to_be_assessed")),

        # injury profile
        "injury_type":     _str(injury_types),
        "injury_severity": _str(m.get("injury_severity")),
        "location_type":   _str(location_types),
        "defendant_type":  _str(defendant_types),
        "liability_theory": _str(m.get("liability_theory")),

        # injury type boolean flags
        "has_slip_and_fall":  _has(injury_types, "slip_and_fall"),
        "has_fracture":       _has(injury_types, "fracture"),
        "has_soft_tissue":    _has(injury_types, "soft_tissue"),
        "has_mTBI":           _has(injury_types, "mTBI"),
        "has_chronic_pain":   _has(injury_types, "chronic_pain"),
        "has_psychological":  _has(injury_types, "psychological"),
        "has_orthopedic":     _has(injury_types, "orthopedic"),
        "has_spinal_cord":    _has(injury_types, "spinal_cord"),
        "has_amputation":     _has(injury_types, "amputation"),
        "has_wrongful_death": _has(injury_types, "wrongful_death"),

        # defendant type boolean flags
        "has_driver":         _has(defendant_types, "driver"),
        "has_municipality":   _has(defendant_types, "municipality"),
        "has_retailer":       _has(defendant_types, "retailer"),
        "has_property_owner": _has(defendant_types, "property_owner"),
        "has_employer":       _has(defendant_types, "employer"),

        # plaintiff profile
        "plaintiff_age_group":          _str(m.get("plaintiff_age_group")),
        "plaintiff_employed_at_injury":  _bool(m.get("plaintiff_employed_at_injury")),

        # liability
        "causation_theory":               _str(m.get("causation_theory")),
        "contributory_negligence_found":  _bool(m.get("contributory_negligence_found")),
        "municipal_liability_case":       _bool(m.get("municipal_liability_case")),
        "threshold_motion_brought":       _bool(m.get("threshold_motion_brought")),
        "threshold_motion_outcome":       _str(m.get("threshold_motion_outcome")),

        # credibility and evidence
        "credibility_issue":               _bool(m.get("credibility_issue")),
        "credibility_finding_adverse":     _bool(m.get("credibility_finding_adverse")),
        "dismissed_on_limitations":        _bool(m.get("dismissed_on_limitations")),
        "pre_existing_condition":          _bool(m.get("pre_existing_condition")),
        "surveillance_used":               _bool(m.get("surveillance_used")),
        "treatment_gap_present":           _bool(m.get("treatment_gap_present")),
        "inconsistent_statements_present": _bool(m.get("inconsistent_statements_present")),
        "ime_used":                        _bool(m.get("ime_used")),

        # income loss
        "future_income_loss_claimed": _bool(m.get("future_income_loss_claimed")),

        # extraction quality
        "extraction_confidence": float(m.get("extraction_confidence") or 0.0),
        "needs_review":          _bool(m.get("needs_review")),
    }


def _build_metadata_text(m: dict) -> str:
    """
    Build a structured summary from extracted fields for the metadata chunk.
    """
    injury_list = m.get("injury_type") or []
    if isinstance(injury_list, str):
        injury_list = [v.strip() for v in injury_list.split(",")]

    location_list = m.get("location_type") or []
    if isinstance(location_list, str):
        location_list = [v.strip() for v in location_list.split(",")]

    defendant_list = m.get("defendant_type") or []
    if isinstance(defendant_list, str):
        defendant_list = [v.strip() for v in defendant_list.split(",")]

    dmg = m.get("damages_awarded")
    if isinstance(dmg, int) and dmg > 0:
        damages_str = f"${dmg:,}"
    elif m.get("damages_to_be_assessed"):
        damages_str = "split trial"
    else:
        damages_str = "$0"

    lines = [
        f"Case: {m.get('case_name', 'Unknown')}",
        f"Court: {m.get('court', 'Unknown')}  Year: {m.get('year', 'Unknown')}",
        f"Outcome: {'Plaintiff won' if m.get('plaintiff_won') else 'Plaintiff lost'}",
        f"Damages: {damages_str}",
    ]

    # Add optional fields only when populated
    if injury_list:
        lines.append(f"Injury type: {', '.join(injury_list)}")
    if m.get("injury_severity"):
        lines.append(f"Injury severity: {m['injury_severity']}")
    if location_list:
        lines.append(f"Location: {', '.join(location_list)}")
    if defendant_list:
        lines.append(f"Defendant: {', '.join(defendant_list)}")
    if m.get("liability_theory"):
        lines.append(f"Liability theory: {m['liability_theory']}")
    if m.get("causation_theory"):
        lines.append(f"Causation theory: {m['causation_theory']}")
    if m.get("deciding_factor"):
        lines.append(f"Deciding factor: {m['deciding_factor']}")
    if m.get("primary_defense_argument"):
        lines.append(f"Primary defense: {m['primary_defense_argument']}")
    if m.get("weakest_point_for_plaintiff"):
        lines.append(f"Weakest point: {m['weakest_point_for_plaintiff']}")
    if m.get("credibility_finding_adverse"):
        lines.append("Adverse credibility finding: yes")
    elif m.get("credibility_issue"):
        lines.append("Credibility issue: yes")
    if m.get("dismissed_on_limitations"):
        lines.append("Dismissed on limitations: yes")
    if m.get("threshold_motion_outcome"):
        lines.append(f"Threshold motion outcome: {m['threshold_motion_outcome']}")
    if m.get("pre_existing_condition"):
        lines.append("Pre-existing condition: yes")
    if m.get("surveillance_used"):
        lines.append("Surveillance used: yes")
    if m.get("treatment_gap_present"):
        lines.append("Treatment gap: yes")
    if m.get("ime_used"):
        lines.append("IME used: yes")
    if m.get("future_income_loss_claimed"):
        lines.append("Future income loss claimed: yes")
    if m.get("case_summary"):
        lines.append(f"Summary: {m['case_summary']}")

    return "\n".join(lines)


def chunk_case(case: dict) -> list[Chunk]:
    """
    Split a single case record into Chunk objects ready for embedding.

    Produces:
      - One metadata chunk (always — minimum is case name + outcome)
      - N text chunks (overlapping windows of full_text)
    """
    m           = case.get("metadata", {})
    full_text   = case.get("full_text", "")
    source_file = m.get("source_file", "unknown")

    if not m.get("case_name"):
        return []

    payload = _make_metadata_payload(m)
    chunks: list[Chunk] = []

    # metadata chunk
    metadata_text = _build_metadata_text(m)
    chunks.append(Chunk(
        text=metadata_text,
        chunk_id=f"{source_file}::metadata::0",
        metadata={**payload, "chunk_type": "metadata"},
    ))

    # text chunks
    if full_text:
        start = 0
        idx   = 0
        while start < len(full_text):
            end   = min(start + CHUNK_SIZE_CHARS, len(full_text))
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(Chunk(
                    text=chunk,
                    chunk_id=f"{source_file}::text::{idx}",
                    metadata={**payload, "chunk_type": "text"},
                ))
            start += CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
            idx   += 1

    return chunks