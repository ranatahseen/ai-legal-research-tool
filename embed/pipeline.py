# Flow:
#   1. Load pi_cases.json
#   2. Connect to Chroma, get already-embedded source files (resume support)
#   3. For each new case: chunk → embed → upsert into Chroma
#   4. Print summary


import json

from .config import PI_CASES_FILE
from .chunker import chunk_case
from .embedder import embed_texts
from .store import get_collection, get_embedded_ids, upsert_chunks


def run_pipeline(cases_file: str = PI_CASES_FILE) -> None:
    """
    Embed all cases from pi_cases.json into Chroma.

    Resume-safe: cases whose source_file is already in Chroma are skipped.
    Safe to re-run after a crash or after adding new cases to the dataset.

    Args:
        cases_file: Path to pi_cases.json. Defaults to PI_CASES_FILE in config.
    """
    # load dataset
    print(f"Loading {cases_file}...")
    with open(cases_file, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  {len(dataset)} cases loaded\n")

    # connect to chroma
    collection     = get_collection()
    embedded_files = get_embedded_ids(collection)

    if embedded_files:
        print(f"Resuming — {len(embedded_files)} cases already in Chroma\n")

    # process each case
    total        = len(dataset)
    embedded     = 0
    skipped      = 0
    failed       = 0

    for i, case in enumerate(dataset, start=1):
        source_file = case.get("metadata", {}).get("source_file", f"case_{i}")
        label       = f"[{i}/{total}]"

        if source_file in embedded_files:
            print(f"{label} skip  {source_file}")
            skipped += 1
            continue

        print(f"{label} embedding  {source_file}")

        # chunk
        chunks = chunk_case(case)
        if not chunks:
            print("  no chunks produced — skipped")
            failed += 1
            continue

        # embed
        texts      = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        if embeddings is None:
            print("  embedding failed — skipped")
            failed += 1
            continue

        if len(embeddings) != len(chunks):
            print(f"  embedding count mismatch ({len(embeddings)} vs {len(chunks)}) — skipped")
            failed += 1
            continue

        # upsert into chroma
        upsert_chunks(collection, chunks, embeddings)
        embedded += 1

        case_name = case.get("metadata", {}).get("case_name", source_file)
        print(f"  done  {case_name}  ({len(chunks)} chunks)")

    # prints
    print(f"\n{'='*52}")
    print(f"  EMBEDDING COMPLETE")
    print(f"  Total cases:    {total}")
    print(f"  Newly embedded: {embedded}")
    print(f"  Skipped:        {skipped}  (already in Chroma)")
    print(f"  Failed:         {failed}")
    print(f"  Chroma path:    {collection._client._settings.persist_directory if hasattr(collection._client, '_settings') else 'see config'}")
    print(f"{'='*52}\n")