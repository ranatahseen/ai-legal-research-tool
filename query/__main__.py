# query/__main__.py
# CLI entry point for the query pipeline.

# Two modes:
#   python -m query "..."   — single query, prints memo, exits
#   python -m query         — interactive conversational mode

# Interactive mode commands (explicit prefix routing — no intent guessing):

#   <case description>        Initial case description → full memo
#   fact <text>               Add a fact to the canonical case, re-runs memo
#   whatif <text>             Hypothetical analysis — does NOT modify canonical case
#   worst case                Adverse stress test against canonical case
#   chat                      Case-aware chat using the retrieved cases as context
#   export                    Export PDF of canonical case + memo
#   new                       Clear state, start a new case
#   help                      Print command reference
#   quit                      Exit


import sys

from .pipeline  import run_query, run_what_if_with_diff
from .followup  import (
    answer_clarification,
    generate_adverse_stress_test,
    interpret_whatif,
    merge_whatif_into_query,
)
from .exporter  import export_pdf
from .chat      import run_chat_session


_DIVIDER     = "=" * 58
_SUBDIV      = "─" * 58

def _print_banner() -> None:
    print(f"\n{_DIVIDER}")
    print("  PI CASE STRESS-TESTER")
    print("  Ontario Personal Injury Research")
    print(_DIVIDER)
    print("  Describe your case to begin.")
    print(f"{_DIVIDER}\n")


def _print_help() -> None:
    print(f"\n{_SUBDIV}")
    print("  COMMANDS")
    print(_SUBDIV)
    print("  fact <text>     Add a fact to the case — updates analysis")
    print("  whatif <text>   Hypothetical — doesn't change your case")
    print("  worst case      Adverse stress test")
    print("  chat            Ask questions about your case and precedents")
    print("  export          Save PDF report")
    print("  new             Start a new case")
    print("  help            Show this reference")
    print("  quit            Exit")
    print(_SUBDIV + "\n")


def _print_memo(memo: str, label: str = "RESEARCH MEMO") -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {label}")
    print(f"{_DIVIDER}\n")
    print(memo)
    print(f"\n{_DIVIDER}")
    print("  fact · whatif · worst case · chat · export · help")
    print(f"{_DIVIDER}\n")


def _print_unknown_command() -> None:
    print(
        "\n  Unknown command. Use:\n"
        "    fact <text>   — add a fact to your case\n"
        "    whatif <text> — run a hypothetical\n"
        "    worst case    — adverse stress test\n"
        "    chat          — ask questions about your case\n"
        "    export        — save PDF\n"
        "    help          — full command list\n"
    )


def _run_single(lawyer_query: str) -> None:
    """Run a single query and print the memo. No conversation loop."""
    memo = run_query(lawyer_query)
    if memo:
        print(f"\n{_DIVIDER}")
        print("  RESEARCH MEMO")
        print(f"{_DIVIDER}\n")
        print(memo)
    else:
        print("\nFailed to generate memo.")
        sys.exit(1)


def _handle_export(
    canonical_query: str,
    current_memo:    str,
    current_stats:   dict,
    current_stress:  str | None,
) -> None:
    """
    Export the canonical case description + memo to PDF.
    Shows the description and asks for confirmation before writing.
    """
    print(f"\n{_SUBDIV}")
    print("  CASE DESCRIPTION FOR PDF COVER")
    print(_SUBDIV)
    print(f"\n{canonical_query}\n")
    print(_SUBDIV)

    try:
        happy = input("\n  Export this description? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        happy = "y"

    if happy != "y":
        print("\n  Enter the exact description you want on the PDF cover:")
        try:
            override = input("  > ").strip()
            if override:
                canonical_query = override
        except (EOFError, KeyboardInterrupt):
            pass

    print("\n  Generating PDF...\n")
    try:
        path = export_pdf(
            canonical_query,
            current_memo,
            current_stats,
            stress_test=current_stress,
        )
        print(f"\n  Report saved: {path}\n")
    except Exception as e:
        print(f"\n  Failed to generate PDF: {e}\n")


def _run_interactive() -> None:
    """
    Conversational loop.

    State:
        canonical_query  — original description + all 'fact' additions.
                           This is what export and worst case always use.
        current_memo     — memo for the current canonical query.
        current_stats    — stats dict for the current canonical query.
        current_stress   — most recent adverse stress test output (if any).
        current_cases    — reranked cases from the last pipeline run.
                           Passed to chat mode as context — no new retrieval.
        baseline_stats   — stats used as the diff baseline for what-ifs.
                           Updated when a new fact is added (not on what-ifs).
    """
    _print_banner()

    canonical_query: str | None  = None
    current_memo:    str | None  = None
    current_stats:   dict | None = None
    current_stress:  str | None  = None
    current_cases:   list        = []   # reranked cases for chat context
    baseline_stats:  dict | None = None
    state = "no_case"

    while True:

        prompt_text = (
            "Describe your case: "
            if state == "no_case"
            else "  > "
        )
        try:
            user_input = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        lower = user_input.lower().strip()

        if lower == "quit":
            print("\nExiting.")
            break

        if lower == "help":
            _print_help()
            continue

        if lower == "new":
            canonical_query = None
            current_memo    = None
            current_stats   = None
            current_stress  = None
            current_cases   = []
            baseline_stats  = None
            state           = "no_case"
            print(f"\n{_SUBDIV}")
            print("  Starting new case.")
            print(f"{_SUBDIV}\n")
            continue

        if lower == "chat":
            if not current_cases:
                print("\n  No cases loaded yet. Describe a case first.\n")
                continue
            # Pass the canonical query and reranked cases — no new retrieval.
            # History is managed inside run_chat_session and cleared on exit.
            run_chat_session(canonical_query, current_cases)
            continue


        if lower in ("export", "export report", "save", "save report", "pdf"):
            if not (current_memo and current_stats and canonical_query):
                print("\n  No memo to export yet. Describe a case first.\n")
                continue
            _handle_export(canonical_query, current_memo, current_stats, current_stress)
            continue


        if lower in (
            "worst case", "worst case scenario",
            "stress test", "adverse", "devil's advocate",
            "how will i lose", "how do i lose",
            "defense view", "defence view",
        ):
            if not (current_memo and canonical_query):
                print("\n  No memo yet. Describe a case first.\n")
                continue
            print("\n  Running adverse stress test against your case...\n")
            stress = generate_adverse_stress_test(canonical_query, current_memo)
            if stress:
                current_stress = stress
                print(f"\n{_DIVIDER}")
                print("  ADVERSE STRESS TEST")
                print("  Defense's strongest case against you")
                print(f"{_DIVIDER}\n")
                print(stress)
                print(f"\n{_DIVIDER}\n")
            else:
                print("\n  Failed to generate stress test. Please try again.\n")
            continue


        if state == "no_case":
            print("\n  Searching Ontario PI decisions...\n")
            result = run_query(user_input, return_stats=True)
            if result:
                memo, stats, cases = result
                canonical_query  = user_input
                current_memo     = memo
                current_stats    = stats
                current_cases    = cases   # store for chat
                baseline_stats   = stats
                state            = "has_memo"
                _print_memo(memo)
            else:
                print("\n  Failed to generate memo. Please try again.\n")
            continue

        if lower.startswith("fact "):
            new_fact = user_input[5:].strip()
            if not new_fact:
                print("\n  Usage: fact <description of new fact>\n")
                continue

            print("\n  Merging fact into case description...\n")
            merged = merge_whatif_into_query(canonical_query, new_fact)

            if merged:
                canonical_query = merged
            else:
                canonical_query = f"{canonical_query}\nAdditional fact: {new_fact}"
                print("  (Merge failed — fact appended directly)\n")

            print("  Updating analysis...\n")
            result = run_query(canonical_query, return_stats=True)

            if result:
                memo, stats, cases = result
                current_memo   = memo
                current_stats  = stats
                current_cases  = cases
                baseline_stats = stats
                current_stress = None
                _print_memo(memo, label="UPDATED MEMO")
            else:
                print("\n  Failed to update memo. Please try again.\n")
            continue

        if lower.startswith("whatif "):
            modification = user_input[7:].strip()
            if not modification:
                print("\n  Usage: whatif <hypothetical change>\n")
                continue

            print("\n  Interpreting hypothetical...\n")
            interpreted = interpret_whatif(canonical_query, modification)

            if interpreted:
                print(f"  Understood as:\n  {interpreted}\n")
                try:
                    confirm = input("  Is this correct? (y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "y"
                if confirm != "y":
                    print("\n  Try rephrasing your what-if more specifically.\n")
                    continue
            else:
                print("  (Interpretation failed — proceeding with raw input)\n")
                interpreted = modification

            print("\n  Building hypothetical case description...\n")
            merged_query = merge_whatif_into_query(canonical_query, interpreted)

            if merged_query:
                print(f"  Hypothetical description:\n{merged_query}\n")
            else:
                print("  (Merge failed — running with appended modification)\n")

            print("  Running hypothetical analysis...\n")
            diff_summary, memo, new_stats = run_what_if_with_diff(
                original_query  = canonical_query,
                modification    = modification,
                baseline_stats  = baseline_stats or current_stats,
                merged_query    = merged_query,
            )

            if diff_summary:
                print(diff_summary)

            if memo:
                _print_memo(memo, label="HYPOTHETICAL MEMO  (your case is unchanged)")
            else:
                print("\n  Failed to generate hypothetical memo. Please try again.\n")
            continue

        _print_unknown_command()


def main() -> None:
    if len(sys.argv) > 1:
        _run_single(" ".join(sys.argv[1:]))
    else:
        _run_interactive()


if __name__ == "__main__":
    main()