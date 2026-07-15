# filters.py
# Two responsibilities:
#   1. Deciding whether a document is a PI case worth processing
#   2. Extracting the most legally relevant sections before LLM ingestion


import httpx

from .config import (
    MAX_CONTEXT_CHARS,
    MAX_DAMAGES_CHARS,
    PI_KEYWORD_MIN_MATCHES,
    PI_KEYWORDS_GENERAL,
    PI_KEYWORDS_STRONG,
    PI_LOG_SKIPPED,
    REQUIRED_FIELDS_HARD,
    REQUIRED_FIELDS_SOFT,
    OLLAMA_URL, 
    MODEL_NAME, 
    FALLBACK_TIMEOUT
)
 
# section keyword
# used to locate the legally meaningful parts of a judgment.
 
_GENERAL_SECTION_KEYWORDS = [
    "overview", "facts", "analysis", "damages", "liability",
    "causation", "credibility", "conclusion", "i find",
    "the plaintiff", "the defendant", "judgment", "decision",
]
 
_DAMAGES_SECTION_KEYWORDS = [
    "i award", "i assess", "i find damages", "total damages",
    "judgment for", "damages of", "entitled to recover",
    "general damages", "special damages", "future care",
    "future income", "non-pecuniary", "pecuniary",
    "pain and suffering", "loss of income", "gross up",
    "prejudgment interest", "cost of care", "housekeeping",
    "fla", "family law act", "aggravated", "punitive",
]
 
 
# pi relevance filter
 
def _score_document(lower_text: str) -> tuple[int, int]:
    """
    Count how many STRONG and GENERAL keywords appear in the document.
 
    Returns:
        strong_count:  Number of PI_KEYWORDS_STRONG matched.
        general_count: Number of PI_KEYWORDS_GENERAL matched.
    """
    strong_count  = sum(1 for kw in PI_KEYWORDS_STRONG  if kw.lower() in lower_text)
    general_count = sum(1 for kw in PI_KEYWORDS_GENERAL if kw.lower() in lower_text)
    return strong_count, general_count
 
 
def is_pi_case(text: str, filepath: str = "") -> bool:
    """
    Return True if the document is likely a PI case worth processing.
 
    Two-tier logic:
      - A single STRONG keyword match passes automatically.
        These terms (e.g. "catastrophic impairment") don't appear outside PI.
      - Otherwise, requires PI_KEYWORD_MIN_MATCHES GENERAL keyword hits.
        This filters out contract/employment disputes that share surface
        language ("plaintiff", "damages", "negligence") with PI cases.
 
    If PI_LOG_SKIPPED is True, every rejected document is logged with its
    scores so you can audit what the filter is dropping.
 
    Args:
        text:     Full cleaned text of the document.
        filepath: Used only for logging — pass the PDF path for useful output.
    """
    lower = text.lower()
    strong_count, general_count = _score_document(lower)
 
    if strong_count > 0:
        return True
 
    if general_count >= PI_KEYWORD_MIN_MATCHES:
        return True
 
    if PI_LOG_SKIPPED:
        print(
            f"  skipped  {filepath or '(unknown)'}  "
            f"(strong={strong_count}, general={general_count}/{PI_KEYWORD_MIN_MATCHES})"
        )
 
    return False


def classify_pi_case(text: str, filepath: str = "") -> bool:
    """
    LLM binary classifier — second gate after keyword pre-screening.

    Sends the first 2,000 characters to the local Ollama instance and asks
    a single yes/no question. Much more accurate than keyword matching for
    edge cases (limitations motions in PI cases, employment disputes that
    mention injury, etc.).

    Only called when the keyword filter passes — never on zero-keyword docs.
    Returns True on any failure (fail open) so an Ollama outage doesn't
    silently drop valid cases.

    Args:
        text:     Full cleaned text of the document.
        filepath: Used only for logging.
    """
    prompt = f"""Is this an Ontario personal injury civil lawsuit or related motion?
Answer only YES or NO. No explanation. No punctuation.

{text[:2_000]}"""

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=FALLBACK_TIMEOUT,
        )
        answer = response.json()["response"].strip().upper()
        is_pi = answer.startswith("YES")

        if not is_pi:
            print(f"  classifier rejected  {filepath or '(unknown)'}")

        return is_pi

    except Exception as e:
        print(f"  classifier failed ({e}) — defaulting to pass")
        return True   # fail open — don't silently drop cases on Ollama error
 
 
# field validation
 
def validate_extracted_fields(data: dict) -> tuple[bool, list[str]]:
    """
    Validate an extracted case record against the two-tier field requirements.
 
    Hard fields: case is unusable without them — return False to drop it.
    Soft fields: case is kept but flagged with needs_review=True.
 
    Returns:
        passes (bool):        False if any HARD field is missing.
        missing_soft (list):  Names of any missing SOFT fields (may be empty).
 
    The caller is responsible for setting needs_review on the record and
    deciding whether to log the soft misses.
    """
    missing_hard = [f for f in REQUIRED_FIELDS_HARD if data.get(f) is None]
    missing_soft = [f for f in REQUIRED_FIELDS_SOFT if data.get(f) is None]
 
    if missing_hard:
        print(f"  dropped — missing hard fields: {missing_hard}")
        return False, missing_soft
 
    return True, missing_soft
 
 
# ── Context extraction ────────────────────────────────────────────────────────
 
def extract_relevant_sections(text: str) -> str:
    """
    Extract the most legally meaningful portions of a judgment for the main
    extraction prompt.
 
    Strategy: locate each section keyword, grab a window of text around it,
    deduplicate overlapping windows, cap at MAX_CONTEXT_CHARS.
 
    Falls back to the first and last 6,000 chars if no keywords are found
    (unusual but possible in older formatted judgments).
    """
    lower = text.lower()
    chunks: list[str] = []
 
    for kw in _GENERAL_SECTION_KEYWORDS:
        idx = lower.find(kw)
        if idx == -1:
            continue
        start = max(0, idx - 1_500)
        end = min(len(text), idx + 3_500)
        chunk = text[start:end]
        if chunk not in chunks:
            chunks.append(chunk)
 
    if not chunks:
        chunks = [text[:6_000], text[-6_000:]]
 
    return "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]
 
 
def extract_damages_sections(text: str) -> str:
    """
    Extract the portions of a judgment most likely to contain the damages
    award, for use in both the main prompt and the fallback pass.
 
    Three-pronged strategy:
      1. Always include the last 4,000 chars — awards almost always appear
         at the end of a judgment.
      2. Include any sentence containing a dollar sign.
      3. Include windows around damages-specific keywords.
 
    Capped at MAX_DAMAGES_CHARS (separate, smaller budget than the general
    context, since this section is injected alongside it in the main prompt).
    """
    lower = text.lower()
    chunks: list[str] = []
 
    # 1. Tail of document
    chunks.append(text[-4_000:])
 
    # 2. Dollar-sign sentences
    for sentence in text.split(". "):
        if "$" in sentence and len(sentence) > 20:
            chunks.append(sentence)
 
    # 3. Keyword windows
    for kw in _DAMAGES_SECTION_KEYWORDS:
        idx = lower.find(kw)
        if idx == -1:
            continue
        start = max(0, idx - 200)
        end = min(len(text), idx + 800)
        chunk = text[start:end]
        if chunk not in chunks:
            chunks.append(chunk)
 
    return "\n\n".join(chunks)[:MAX_DAMAGES_CHARS]