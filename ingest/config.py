

# llm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:31b-cloud"

# Seconds to wait for the main extraction call (long — full case text)
EXTRACTION_TIMEOUT = 600

# Seconds to wait for the focused damages fallback call (shorter prompt)
FALLBACK_TIMEOUT = 300

# paths

INPUT_FOLDER = "cases"       
OUTPUT_FILE = "pi_cases.json"  

# dataset filters

MIN_CASE_YEAR = 1990

# context window

# Maximum characters fed to the LLM in the general extraction prompt.
# Gemma 4 31B has a 128k context, but keeping this tight reduces hallucination
# and speeds up inference.
MAX_CONTEXT_CHARS = 16_000

# Characters taken from damages-focused excerpts (separate from the above)
MAX_DAMAGES_CHARS = 12_000

# Minimum character length for a PDF to be considered a real judgment
MIN_CASE_LENGTH = 1_500

# pi relevance filter
# this checks if the pdf is relevant

# two tier keyword system:

#   STRONG: PI exclusive terms. A single match passes the document automatically.

#   GENERAL: Common legal terms. Requires PI_KEYWORD_MIN_MATCHES hits to pass.


# To audit what is being dropped: set PI_LOG_SKIPPED = True. Every skipped
# file will be logged with its keyword score so you can spot gaps.

PI_LOG_SKIPPED = True
PI_KEYWORD_MIN_MATCHES = 2

PI_KEYWORDS_STRONG = [
    # These terms are PI-exclusive — one match is enough to pass
    "catastrophic impairment",
    "mild traumatic brain injury",
    "occupier's liability",
    "occupiers' liability",
    "statutory accident benefits",
    "loss of earning capacity",
    "loss of competitive advantage",
    "chronic pain syndrome",
    "soft tissue injury",
    "functional capacity evaluation",
    "designated assessment centre",
    "minor injury guideline",
]

PI_KEYWORDS_GENERAL = [
    # Common legal terms — need PI_KEYWORD_MIN_MATCHES hits to pass
    "slip and fall",
    "personal injury",
    "motor vehicle accident",
    "rear-end collision",
    "negligence",
    "soft tissue",
    "chronic pain",
    "concussion",
    "fracture",
    "future income loss",
    "plaintiff",
    "tort",
    "whiplash",
    "damages",
    "occupier liability",
]

# Dataset validation
# this does a check to see if the fields in the JSON are filled correctly

# two tier field requirement system:

#   HARD: case is dropped if any of these are missing. Without them the record is analytically useless (can't compute win rate, can't summarise the case).

#   SOFT: case is kept but flagged with needs_review=True if any are missing. These are genuinely absent in some real Ontario decisions

REQUIRED_FIELDS_HARD = [
    "case_name",        # can't identify the record
    "plaintiff_won",    # core label — useless without it
    "damages_awarded",  # core outcome — useless without it
    "case_summary",     # minimum viable content for the memo
]

REQUIRED_FIELDS_SOFT = [
    # Keep but flag needs_review=True if any are missing
    # Identity
    "citation",
    "court",
    "year",
    "decision_type",
    # Injury profile
    "injury_type",
    "injury_severity",
    "liability_theory",
    # Legal reasoning
    "deciding_factor",
    "primary_defense_argument",
    "weakest_point_for_plaintiff",
    # Causation
    "causation_theory",
    "dismissed_on_limitations",
]

DAMAGES_SPLIT_TRIAL = -1   # liability found, but quantum sent to separate hearing
DAMAGES_REVIEW_FLAG = -2   # extraction failed; needs manual review
DAMAGES_PLAINTIFF_LOST = 0  # plaintiff did not succeed