# prompts.py


def case_extraction_prompt(combined_context: str) -> str:
    """
    Main extraction prompt. Extracts all v2 schema fields from a single
    Ontario PI judgment. Called once per case during ingestion.

    Args:
        combined_context: General case excerpts + damages-focused excerpts,
                          pre-assembled by extractor.py.
    """
    return f"""
You are extracting structured litigation intelligence from Ontario personal
injury decisions for a legal research database used by PI lawyers.

Return ONLY VALID JSON. No markdown. No explanation. No backticks.
Every field must be present. Use null for genuinely unknown values.

══════════════════════════════════════════════
BLOCK 1 — CASE IDENTITY
══════════════════════════════════════════════

"case_name"           Full style of cause as it appears in the header.
"citation"            CanLII neutral citation e.g. "2022 ONSC 1234". null if absent.
"court"               One of: ONSC | ONCA | LAT | FSCO | other
"year"                Integer. Decision year only.
"judge_name"          Full name and title e.g. "Justice Smith". null if not found.
"trial_duration_days" Integer number of trial days. null if not stated.
"decision_type"       One of: full_trial | summary_judgment | costs_only |
                      quantum_only | appeal

══════════════════════════════════════════════
BLOCK 2 — OUTCOME
══════════════════════════════════════════════

"plaintiff_won"            true | false. Did plaintiff succeed on liability?
"damages_awarded"          Integer. Total gross damages before contributory
                           negligence reduction. See DAMAGES RULES below.
"damages_to_be_assessed"   true if liability found but quantum sent to separate
                           hearing.
"non_pecuniary_damages"    Integer or null. General / pain and suffering.
"special_damages"          Integer or null. Out-of-pocket expenses.
"future_care_costs"        Integer or null. Future cost of care award.
"future_income_loss_awarded" Integer or null. Actual income loss awarded.
"fla_damages"              Integer or null. Family Law Act claims.
"aggravated_punitive_damages" Integer or null. 0 if not awarded.
"housekeeping_damages"     Integer or null.
"gross_up_amount"          Integer or null. Tax gross-up on future awards.
"prejudgment_interest"     Integer or null. Do NOT include in damages_awarded.
"appeal_outcome"           One of: affirmed | reversed | varied |
                           not_appealed | null
"dismissed_on_limitations" true | false.
                           true ONLY if the claim was dismissed because
                           the plaintiff missed the 2-year limitation
                           period under the Limitations Act, 2002.
                           Look for: "statute-barred", "limitation period
                           has expired", "discoverability", "s.4 of the
                           Limitations Act", "two-year limitation period".
                           false if the case was decided on the merits,
                           even if limitations was argued and rejected.
                           false if you are uncertain.

══════════════════════════════════════════════
BLOCK 3 — INJURY AND PLAINTIFF PROFILE
══════════════════════════════════════════════

"injury_type"     Array. One or more of:
                  slip_and_fall | chronic_pain | soft_tissue | mTBI |
                  fracture | orthopedic | psychological | spinal_cord |
                  burns | amputation | wrongful_death | other

                  CRITICAL RULES FOR injury_type:
                  - MVA is NOT an injury type — it is a mechanism of injury.
                    Use the actual injuries instead: soft_tissue, mTBI,
                    fracture, chronic_pain, orthopedic, psychological, etc.
                  - "other" is a LAST RESORT only. Use it when the injury
                    genuinely does not fit any listed category.
                    Do NOT use "other" for:
                      whiplash            → soft_tissue
                      neck/back pain      → soft_tissue or chronic_pain
                      concussion          → mTBI
                      torn ligament / ACL → orthopedic
                      PTSD/anxiety/depression → psychological
                      broken bone         → fracture
                  - Always include chronic_pain if the judgment describes
                    ongoing pain lasting more than 3 months.
                  - Include ALL applicable types — a rear-end victim with
                    neck pain, headaches, and depression should have:
                    ["soft_tissue", "chronic_pain", "mTBI", "psychological"]

"injury_severity" One of: minor | moderate | serious | catastrophic | fatal | unknown
                  Note: "catastrophic" has a specific legal definition under
                  Ontario SABS. "minor" aligns with the Minor Injury Guideline.

"location_type"   Array. One or more of:
                  road | sidewalk | retail | parking_lot | workplace |
                  residential | pedestrian_ramp | stairwell | elevator |
                  public_transit | other

"defendant_type"  Array. One or more of:
                  driver | municipality | retailer | property_owner |
                  insurer | employer | contractor | school_board | other

"plaintiff_age_group"        One of: minor | young | middle | elderly | unknown
                             minor = under 18.
"plaintiff_occupation"       General category e.g. "tradesperson", "professional",
                             "self-employed", "retired". null if not stated.
"plaintiff_employed_at_injury" true | false | null

══════════════════════════════════════════════
BLOCK 4 — LIABILITY
══════════════════════════════════════════════

"contributory_negligence_found"      true | false
"contributory_negligence_percentage" Integer or null. Plaintiff's share of fault.
"liability_theory"      One of: occupiers_liability | negligence | mva_tort |
                        vicarious_liability | products_liability |
                        medical_malpractice | other
"standard_of_care_disputed" true | false. Was the standard itself contested?
"causation_issue_present"   true | false
"causation_theory"     One of: but_for | material_contribution |
                       crumbling_skull | thin_skull | null
"municipal_liability_case"  true | false
"notice_of_claim_issue"     true | false. Was the 10-day Municipal Act notice raised?

══════════════════════════════════════════════
BLOCK 5 — CREDIBILITY AND EVIDENCE
══════════════════════════════════════════════

"credibility_issue"    true | false. Was plaintiff credibility seriously contested?
"credibility_outcome"  One of: accepted | partially_accepted | rejected | null
"credibility_finding_adverse" true | false.
                       true ONLY if the judge made an explicit adverse
                       credibility finding against the plaintiff — i.e. the
                       judge stated the plaintiff was not believable, was
                       exaggerating, or found their evidence unreliable.
                       false if credibility was merely raised or contested
                       without a specific finding against the plaintiff.
                       false if the plaintiff's credibility was accepted.
                       This is a rare finding — most cases should be false.
"pre_existing_condition"        true | false
"pre_existing_condition_impact" One of: none | minor_reduction |
                                significant_reduction | claim_defeated | null
"surveillance_used"    true | false
"surveillance_impact"  One of: none | credibility_damaged | claim_defeated | null
"treatment_gap_present" true | false
"treatment_gap_impact"  One of: none | damages_reduced | claim_defeated | null
"inconsistent_statements_present" true | false. Statements to insurer, discovery,
                                  or social media inconsistent with claimed injuries.
"social_media_evidence_used" true | false

══════════════════════════════════════════════
BLOCK 6 — EXPERT EVIDENCE
══════════════════════════════════════════════

"expert_evidence_decisive"   true | false
"plaintiff_expert_accepted"  true | false | null (null if no plaintiff expert called)
"defense_expert_accepted"    true | false | null
"ime_used"     true | false. Was an Independent Medical Examination conducted?
"ime_outcome"  One of: accepted | rejected | partially_accepted | null
"expert_disciplines" Array of strings or null.
                     e.g. ["orthopedic surgery", "physiatry", "neuropsychology"]

══════════════════════════════════════════════
BLOCK 7 — INCOME LOSS
══════════════════════════════════════════════

"future_income_loss_claimed" true | false. Was it pleaded?
"income_loss_theory" One of: lost_years | loss_of_competitive_advantage |
                     earnings_approach | null

══════════════════════════════════════════════
BLOCK 8 — DEFENSE STRATEGY
══════════════════════════════════════════════

"primary_defense_argument"    One concise sentence.
"secondary_defense_arguments" Array of concise sentences or null.
"threshold_motion_brought"    true | false. s.267.5 Insurance Act motion.
"threshold_motion_outcome"    One of: plaintiff_cleared | plaintiff_failed | null
                              plaintiff_cleared = plaintiff met the threshold
                              plaintiff_failed  = threshold motion succeeded,
                                                  non-pecuniary claim dismissed
                              null if no threshold motion was brought.
"offer_to_settle_present"     true | false. Rule 49 offer.
"costs_awarded_to"            One of: plaintiff | defendant | no_order | null

══════════════════════════════════════════════
BLOCK 9 — LEGAL REASONING
══════════════════════════════════════════════

"deciding_factor"           One concise sentence. The single factor that most
                            determined the outcome.
"weakest_point_for_plaintiff" One concise sentence. The biggest vulnerability
                              in the plaintiff's case as identified by the court.
                              null if plaintiff won decisively.
"case_summary"              5-10 sentences covering: facts, liability theory,
                            key evidence, court's reasoning, and outcome.
"key_cases_cited"           Array of case names cited in the judgment or null.
                            e.g. ["Andrews v. Grand & Toy", "Athey v. Leonati"]
"legislation_applied"       Array of statutes applied or null.
                            e.g. ["Occupiers' Liability Act", "Insurance Act s.267.5"]
"extraction_confidence"     Float 0.0-1.0. Your confidence in this extraction.

══════════════════════════════════════════════
DAMAGES RULES (read carefully)
══════════════════════════════════════════════

damages_awarded — search exhaustively through the DAMAGES-FOCUSED EXCERPTS:
  1. First look for a single total/judgment line:
     "I award $X", "judgment for the plaintiff in the amount of $X",
     "total damages of $X", "damages are assessed at $X"
  2. If no total exists, SUM all heads:
     general + special + future care + future income loss + FLA +
     housekeeping + gross-up.
     Do NOT include prejudgment interest or costs.
  3. If contributory negligence applies, use the PRE-REDUCTION gross amount.
  4. Plaintiff lost → 0
  5. Split trial (quantum deferred) → damages_to_be_assessed: true,
     damages_awarded: -1
  6. NEVER return null for damages_awarded.
     A rough estimate with lower extraction_confidence beats null.

CASE TEXT:
{combined_context}
""".strip()


def damages_fallback_prompt(dollar_amounts: list[str], damages_section: str) -> str:
    """
    Focused second-pass prompt called only when the main extraction returns
    null for damages_awarded.

    Args:
        dollar_amounts:  List of "$X,XXX" strings found by regex. Capped at
                         80 items by the caller.
        damages_section: Output of extract_damages_sections().
    """
    amounts_display = dollar_amounts[:80]

    return f"""
You are a legal data extraction specialist. Your ONLY job is to find the
total damages awarded to the plaintiff in this Ontario court case.

SEARCH STRATEGY:
1. Look for a total/final award line:
   "I award $X", "judgment for the plaintiff in the amount of $X",
   "total damages of $X", "damages are assessed at $X"
2. If no total, ADD UP: general damages + special damages + future care +
   future income loss + FLA claims + housekeeping.
   Do NOT include prejudgment interest or costs.
3. If contributory negligence applies, use the PRE-REDUCTION gross amount.
4. Plaintiff lost → return 0
5. Split trial / separate assessment → return -1

All dollar amounts found in the document:
{amounts_display}

Relevant excerpts:
{damages_section}

Return ONLY valid JSON, no markdown, no explanation:
{{
  "damages_awarded": 150000,
  "damages_to_be_assessed": false,
  "damages_notes": "one sentence explaining what you found or why uncertain",
  "confidence": 0.85
}}

- damages_awarded must be an integer (dollars only, no cents)
- damages_awarded = -1 means split trial
- damages_awarded = 0 means plaintiff lost
- NEVER return null
""".strip()


def query_extraction_prompt(lawyer_query: str) -> str:
    """
    Query-time prompt. Extracts structured facts from a lawyer's plain-English
    case description to drive Chroma metadata pre-filtering (Layer 1).

    Fields must match the metadata fields stored in Chroma during ingestion.

    Args:
        lawyer_query: Raw plain-English input from the lawyer via Telegram.
    """
    return f"""
You are a legal intake specialist for an Ontario personal injury research tool.

Extract structured facts from the lawyer's case description.
These facts will filter a database of 5,000+ Ontario PI decisions to find
the most legally comparable cases.

Return ONLY valid JSON. No markdown. No explanation. No backticks.

{{
  "injury_type":     ["soft_tissue", "mTBI"],
  "injury_severity": "minor | moderate | serious | catastrophic | unknown",
  "defendant_type":  ["driver"],
  "location_type":   ["road"],
  "liability_theory": "occupiers_liability | negligence | mva_tort | other | null",
  "plaintiff_age_group": "minor | young | middle | elderly | unknown",
  "plaintiff_employed_at_injury": true,
  "credibility_issue":              false,
  "pre_existing_condition":         false,
  "treatment_gap_present":          false,
  "future_income_loss_claimed":     true,
  "contributory_negligence_found":  false,
  "threshold_motion_brought":       false,
  "municipal_liability_case":       false,
  "causation_issue_present":        false,
  "query_summary": "one sentence describing the core legal issue"
}}

ALLOWED VALUES:
  injury_type:    slip_and_fall | chronic_pain | soft_tissue | mTBI |
                  fracture | orthopedic | psychological | spinal_cord |
                  burns | amputation | wrongful_death | other
  defendant_type: driver | municipality | retailer | property_owner |
                  insurer | employer | contractor | school_board | other
  location_type:  road | sidewalk | retail | parking_lot | workplace |
                  residential | pedestrian_ramp | stairwell | elevator |
                  public_transit | other

RULES:
- injury_type, defendant_type, location_type are arrays — include all that apply.
- For boolean fields, infer from context. Use false if there is no signal.
- query_summary: one factual sentence, no legal conclusions.

Lawyer's case description:
{lawyer_query}
""".strip()


def reranking_prompt(lawyer_query: str, cases_block: str) -> str:
    """
    Layer 3 reranking prompt. Called after vector search returns the top 15
    candidates. Asks Claude to rank by genuine legal relevance, not just
    semantic similarity.

    Args:
        lawyer_query:  Original plain-English case description.
        cases_block:   Pre-formatted string of the 15 candidate cases.
                       Each case should include: case_name, year, court,
                       injury_type, injury_severity, liability_theory,
                       deciding_factor, weakest_point_for_plaintiff,
                       plaintiff_won, damages_awarded.
    """
    return f"""
You are a senior Ontario personal injury litigator reviewing case precedents.

A lawyer has submitted the following case for stress-testing:
{lawyer_query}

Below are 15 candidate cases retrieved from a database of Ontario PI decisions.
Rank them by genuine legal relevance to the submitted fact pattern.

Legal relevance means:
- Same injury mechanism (not just similar surface language)
- Same liability theory (occupier's liability vs. negligence vs. MVA tort)
- Comparable causation issues (thin skull, crumbling skull, but-for)
- Similar credibility and evidence profile
- Comparable damages profile (severity, heads of damage)
- Same court level where possible (prefer ONSC over LAT for damages guidance)

Do NOT rank by surface similarity. A case using the same words but a
different liability theory is NOT relevant.

Candidate cases:
{cases_block}

Return ONLY valid JSON. No markdown. No explanation.

{{
  "ranked_cases": [
    {{
      "case_name": "Smith v. Jones, 2021 ONSC 1234",
      "rank": 1,
      "relevance_reason": "one sentence explaining why legally analogous"
    }}
  ]
}}

Return exactly 15 cases. Most relevant first.
""".strip()