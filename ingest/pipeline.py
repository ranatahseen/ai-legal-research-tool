# pipeline.py

# Orchestrates the full ingestion pipeline.

# Flow:
#   1 load existing dataset (resume support)
#   2 load PDFs from INPUT_FOLDER
#   3 for each new PDF: filter → extract → validate → save
#   4 print stats on completion

# This file contains no LLM logic, no file parsing, and no prompt strings.
# It only coordinates the modules that do those things.

import time

from .config import DAMAGES_REVIEW_FLAG, DAMAGES_SPLIT_TRIAL, INPUT_FOLDER, MIN_CASE_YEAR
from .dataset import load_dataset, print_stats, save_dataset
from .extractor import extract_case_metadata
from .filters import is_pi_case, classify_pi_case
from .loader import load_local_cases


def _format_damages(result: dict) -> str:
    """
    Format the damages_awarded field for a one-line progress log.

    Handles all sentinel values so the main loop stays readable.
    """
    awarded = result.get("damages_awarded")
    if result.get("damages_to_be_assessed"):
        return "split trial"
    if awarded == DAMAGES_REVIEW_FLAG:
        return "REVIEW"
    if isinstance(awarded, int) and awarded >= 0:
        return f"${awarded:,}"
    return str(awarded)


def run_pipeline(
    input_folder: str = INPUT_FOLDER,
    output_file: str | None = None,
) -> None:
    """
    Run the full ingestion pipeline.

    Resume-safe: already-processed files (identified by source_file path) are
    skipped. The dataset is saved after every successfully processed case, so
    a crash mid-run loses at most one case.

    Args:
        input_folder: Folder containing raw PDF judgments.
        output_file:  Path to the output JSON. Defaults to OUTPUT_FILE from config. Overridable for testing.
    """

    from .config import OUTPUT_FILE
    json_file = output_file or OUTPUT_FILE

    # resume support
    dataset = load_dataset(json_file)
    processed_files = {case["metadata"]["source_file"] for case in dataset}

    if dataset:
        print(f"Resuming — {len(dataset)} cases already processed\n")

    # load pdfs
    raw_cases = load_local_cases(input_folder)
    total = len(raw_cases)

    if not raw_cases:
        print("No cases to process.")
        return

    print(f"Starting extraction ({total} PDFs, {len(processed_files)} already done)...\n")

    # process each case
    for i, raw in enumerate(raw_cases, start=1):
        label = f"[{i}/{total}]"

        if raw.filepath in processed_files:
            print(f"{label} skip  {raw.filepath}")
            continue

        print(f"\n{label} processing  {raw.filepath}")

        # Stage 1 fast keyword pre-screen
        if not is_pi_case(raw.text, raw.filepath):
            print("  not PI-related (keyword filter) — skipped")
            continue

        # Stage 2 LLM classifier confirms before full extraction
        if not classify_pi_case(raw.text, raw.filepath):
            print("  not PI-related — skipped")
            continue

        start = time.time()
        result = extract_case_metadata(case_text=raw.text, filepath=raw.filepath)
        elapsed = time.time() - start

        if result is None:
            print("  extraction returned no result — skipped")
            continue

        case_year = result.get("year")
        if case_year and case_year < MIN_CASE_YEAR:
            print(f"  skipped — pre-{MIN_CASE_YEAR} decision ({case_year})")
            continue

        dataset.append({"metadata": result, "full_text": raw.text})
        save_dataset(dataset, json_file)

        print(
            f"  done  {result['case_name']}  |  "
            f"won={result['plaintiff_won']}  |  "
            f"damages={_format_damages(result)}  |  "
            f"confidence={result.get('extraction_confidence')}  |  "
            f"{elapsed:.0f}s"
        )

    # summary
    print(f"\nDone — {len(dataset)} cases saved to {json_file}")
    print_stats(json_file)