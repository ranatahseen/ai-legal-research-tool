import time
from typing import Optional

import httpx

from .config import CHAT_HISTORY_TURNS, LLM_RETRIES, LLM_TIMEOUT, MODEL_NAME, OLLAMA_URL


def _format_cases_for_context(cases: list[dict]) -> str:
    """Format the reranked cases as a compact reference block."""
    lines = []
    for i, case in enumerate(cases, 1):
        m = case.get("metadata", {})
        dmg = m.get("damages_awarded", 0)
        dmg_str = (
            f"${dmg:,}" if isinstance(dmg, int) and dmg > 0
            else ("split trial" if m.get("damages_to_be_assessed") else "$0")
        )
        citation = m.get("citation", "")
        citation_str = f" [{citation}]" if citation else ""

        lines.append(
            f"[{i}] {m.get('case_name', 'Unknown')}{citation_str} "
            f"({m.get('year', '?')} {m.get('court', '?')})\n"
            f"    Outcome: {'Plaintiff won' if m.get('plaintiff_won') else 'Plaintiff lost'} | "
            f"Damages: {dmg_str}\n"
            f"    Injury: {m.get('injury_type', 'N/A')} | "
            f"Severity: {m.get('injury_severity', 'N/A')}\n"
            f"    Liability: {m.get('liability_theory', 'N/A')} | "
            f"Causation: {m.get('causation_theory', 'N/A')}\n"
            f"    Deciding factor: {m.get('deciding_factor', 'N/A')}\n"
            f"    Weakest point: {m.get('weakest_point_for_plaintiff', 'N/A')}\n"
            f"    Defense argument: {m.get('primary_defense_argument', 'N/A')}\n"
            f"    Pre-existing: {m.get('pre_existing_condition', False)} | "
            f"Credibility issue: {m.get('credibility_issue', False)} | "
            f"Treatment gap: {m.get('treatment_gap_present', False)}\n"
            f"    Summary: {m.get('case_summary', 'N/A')}"
        )
    return "\n\n".join(lines)


def build_chat_context(canonical_query: str, cases: list[dict]) -> str:
    """
    Build the fixed system context for a chat session.

    Called once per session (CLI) or once per request (web, where the
    client caches the context string to avoid rebuilding it each turn).

    Args:
        canonical_query: The lawyer's canonical case description.
        cases:           Reranked cases from the query pipeline.

    Returns:
        System context string injected at the start of every prompt.
    """
    cases_block = _format_cases_for_context(cases)
    return f"""You are a senior Ontario personal injury litigator acting as a research assistant.

A lawyer is working on the following case:
{canonical_query}

The following {len(cases)} Ontario PI decisions are the most legally comparable cases
retrieved for this fact pattern. Answer ALL questions using ONLY these cases as your
source of precedent. Do not reference or invent cases outside this list.

COMPARABLE CASES:
{cases_block}

RULES:
- Answer conversationally but with legal precision
- Always cite the specific case(s) from the list above that support your answer
- If the answer cannot be found in these cases, say so plainly — do not speculate
- Keep answers focused and direct — this is a research conversation, not a memo
- Use full citations in the format: Case Name [Year Court Docket] when referencing cases
- If asked about a topic not covered by these cases, say: "The comparable cases
  retrieved for this fact pattern don't directly address that. You may want to
  run a new query focused on [topic]."
"""


def _build_prompt(
    system_context: str,
    history:        list[dict],
    question:       str,
) -> str:
    """Assemble the full prompt for a single chat turn."""
    history_block = ""
    if history:
        turns = []
        for turn in history:
            turns.append(f"Lawyer: {turn['question']}")
            turns.append(f"Assistant: {turn['answer']}")
        history_block = "\nCONVERSATION SO FAR:\n" + "\n\n".join(turns) + "\n"

    return f"""{system_context}{history_block}
Lawyer's question: {question}

Answer:""".strip()


def _call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama and return the response string, or None on failure."""
    for attempt in range(LLM_RETRIES):
        try:
            response = httpx.post(
                OLLAMA_URL,
                json={
                    "model":   MODEL_NAME,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.3},
                },
                timeout=LLM_TIMEOUT,
            )
            body = response.json()
            if "response" not in body:
                raise ValueError(f"Ollama error: {body.get('error', body)}")
            return body["response"].strip()

        except Exception as e:
            if attempt < LLM_RETRIES - 1:
                print(f"  chat attempt {attempt + 1} failed ({e}) — retrying...")
                time.sleep(1)
            else:
                print(f"  chat failed: {e}")
                return None


def get_chat_response(
    canonical_query: str,
    reranked_cases:  list[dict],
    history:         list[dict],
    question:        str,
) -> Optional[str]:
    """
    Single stateless chat turn for the web API.

    The caller (server.py) owns history and cases — nothing is stored here.
    History is trimmed to CHAT_HISTORY_TURNS before building the prompt.

    Args:
        canonical_query: The lawyer's canonical case description.
        reranked_cases:  Cases from the query pipeline (stored client-side).
        history:         Conversation history [{question, answer}].
                         Trimmed to CHAT_HISTORY_TURNS internally.
        question:        The lawyer's current question.

    Returns:
        Answer string, or None if the LLM call fails.
    """
    system_context  = build_chat_context(canonical_query, reranked_cases)
    trimmed_history = history[-CHAT_HISTORY_TURNS:]
    prompt          = _build_prompt(system_context, trimmed_history, question)
    return _call_ollama(prompt)


def run_chat_session(
    canonical_query: str,
    reranked_cases:  list[dict],
) -> None:
    """
    Interactive CLI chat session.

    History is a local list — cleared automatically when the function returns.
    Type 'exit', 'quit', or 'back' to return to the main loop.

    Args:
        canonical_query: The lawyer's canonical case description.
        reranked_cases:  Cases returned by reranker.py for this case.
    """
    if not reranked_cases:
        print("\n  No cases loaded — run a case query first before entering chat.\n")
        return

    _DIVIDER = "=" * 58
    _SUBDIV  = "─" * 58

    print(f"\n{_DIVIDER}")
    print("  CASE CHAT")
    print(f"  {len(reranked_cases)} comparable cases loaded as context")
    print(_DIVIDER)
    print("  Ask anything about your case or the comparable precedents.")
    print("  Type 'back' to return to the main menu.")
    print(f"{_SUBDIV}\n")

    system_context = build_chat_context(canonical_query, reranked_cases)
    history: list[dict] = []

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "back", "done", "return"):
            print(f"\n{_SUBDIV}")
            print("  Returning to main menu. Chat history cleared.")
            print(f"{_SUBDIV}\n")
            break

        trimmed_history = history[-CHAT_HISTORY_TURNS:]
        prompt  = _build_prompt(system_context, trimmed_history, user_input)
        answer  = _call_ollama(prompt)

        if answer:
            print(f"\n  Assistant: {answer}\n")
            history.append({"question": user_input, "answer": answer})
        else:
            print("\n  Failed to get a response. Please try again.\n")