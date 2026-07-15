# dataset.py
# Responsible for all reads and writes to pi_cases.json.

# Four public functions:
#   load_dataset()        — load existing JSON (or return empty list)
#   save_dataset()        — write dataset to disk (called after every case)
#   patch_null_damages()  — fix null damages_awarded in an existing dataset
#   print_stats()         — print a dataset health report


import json
import os

from .config import DAMAGES_PLAINTIFF_LOST, DAMAGES_REVIEW_FLAG, OUTPUT_FILE
from .extractor import extract_damages_fallback


# load and save

def load_dataset(json_file: str = OUTPUT_FILE) -> list[dict]:
    """
    Load the existing dataset from disk.

    Returns an empty list (not an error) if the file doesn't exist yet —
    this is the normal state on a fresh run.
    """
    if not os.path.exists(json_file):
        return []

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset: list[dict], json_file: str = OUTPUT_FILE) -> None:
    """
    Write the full dataset to disk.

    Called after every successfully processed case (not at the end of the
    run) so a crash at case 347 doesn't lose the previous 346.
    """
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


# patch

def patch_null_damages(json_file: str = OUTPUT_FILE) -> None:
    """
    Re-run damages extraction on every case where damages_awarded is null.

    Safe to run multiple times — only touches records that are still null.
    Saves after each resolved case so a mid-patch crash doesn't lose work.

    Typical use: run this once on an existing dataset before scaling up
    ingestion, to clean up records where the main pass failed.
    """
    dataset = load_dataset(json_file)

    if not dataset:
        print(f"No dataset found at {json_file}")
        return

    null_indices = [
        i for i, case in enumerate(dataset)
        if case["metadata"].get("damages_awarded") is None
    ]

    print(f"\nPATCH MODE: {len(null_indices)} cases with null damages_awarded\n")

    if not null_indices:
        print("Nothing to patch.")
        return

    patched = 0

    for i in null_indices:
        case = dataset[i]
        name = case["metadata"].get("case_name", f"case_{i}")
        text = case.get("full_text", "")

        print(f"  [{i}] {name}")

        if not text:
            print("    no full_text stored — cannot re-extract")
            continue

        fallback = extract_damages_fallback(text, name)

        if fallback:
            dataset[i]["metadata"]["damages_awarded"] = fallback["damages_awarded"]
            dataset[i]["metadata"]["damages_to_be_assessed"] = fallback.get("damages_to_be_assessed", False)
            dataset[i]["metadata"]["damages_notes"] = fallback.get("damages_notes", "")
            save_dataset(dataset, json_file)
            patched += 1
            confidence = fallback.get("confidence", "?")
            print(f"    resolved: ${fallback['damages_awarded']:,}  (confidence: {confidence})")
        else:
            plaintiff_won = case["metadata"].get("plaintiff_won", False)
            dataset[i]["metadata"]["damages_awarded"] = (
                DAMAGES_PLAINTIFF_LOST if not plaintiff_won else DAMAGES_REVIEW_FLAG
            )
            dataset[i]["metadata"]["damages_notes"] = "Flagged for manual review"
            save_dataset(dataset, json_file)
            print("    fallback failed — flagged for manual review")

    print(f"\nPatch complete: {patched} of {len(null_indices)} resolved\n")


# stats

def print_stats(json_file: str = OUTPUT_FILE) -> None:
    """
    Print a dataset health report to stdout.

    Run this after ingestion to verify data quality before using the dataset
    for retrieval. Key signals:
      - Null damages > 0  → run patch_null_damages()
      - Flagged > 5%      → extraction prompt may need tuning
      - Avg confidence < 0.75 → consider spot-checking more cases manually
    """
    dataset = load_dataset(json_file)

    if not dataset:
        print(f"No dataset found at {json_file}")
        return

    total = len(dataset)
    metadata = [c["metadata"] for c in dataset]

    won          = sum(1 for m in metadata if m.get("plaintiff_won") is True)
    lost         = sum(1 for m in metadata if m.get("plaintiff_won") is False)
    null_damages = sum(1 for m in metadata if m.get("damages_awarded") is None)
    split_trial  = sum(1 for m in metadata if m.get("damages_to_be_assessed") is True)
    flagged      = sum(1 for m in metadata if m.get("damages_awarded") == DAMAGES_REVIEW_FLAG)
    avg_conf     = (
        sum(m.get("extraction_confidence") or 0 for m in metadata) / total
        if total else 0
    )

    separator = "=" * 52

    print(f"\n{separator}")
    print(f"  DATASET STATS: {json_file}")
    print(separator)
    print(f"  Total cases            {total}")
    print(f"  Plaintiff won          {won}  ({won / total * 100:.1f}%)" if total else "")
    print(f"  Plaintiff lost         {lost}")
    print(f"  Null damages           {null_damages}  {'← run patch_null_damages()' if null_damages else ''}")
    print(f"  Split trial (no $)     {split_trial}")
    print(f"  Flagged for review     {flagged}")
    print(f"  Avg confidence         {avg_conf:.2f}")
    print(f"{separator}\n")