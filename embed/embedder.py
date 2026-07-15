import time
from typing import Optional

import voyageai

from .config import (
    VOYAGE_API_KEY,
    VOYAGE_BATCH_DELAY,
    VOYAGE_MAX_BATCH_SIZE,
    VOYAGE_MODEL,
)


_client = voyageai.Client(api_key=VOYAGE_API_KEY)


def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Embed a list of texts using the Voyage AI API.

    Automatically batches into groups of VOYAGE_MAX_BATCH_SIZE with a short
    delay between batches to stay within rate limits.

    Args:
        texts: List of strings to embed. Can be any length — batching is
               handled internally.

    Returns:
        List of embedding vectors in the same order as the input texts,
        or None if the API call fails entirely.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + VOYAGE_MAX_BATCH_SIZE - 1) // VOYAGE_MAX_BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * VOYAGE_MAX_BATCH_SIZE
        end   = min(start + VOYAGE_MAX_BATCH_SIZE, len(texts))
        batch = texts[start:end]

        try:
            result = _client.embed(
                texts=batch,
                model=VOYAGE_MODEL,
                input_type="document",   # "document" for storage, "query" for search
            )
            all_embeddings.extend(result.embeddings)
            print(f"  embedded batch {batch_idx + 1}/{total_batches} ({len(batch)} texts)")

        except Exception as e:
            print(f"  embedding failed on batch {batch_idx + 1}: {e}")
            return None

        # Avoid hammering the API on large datasets
        if batch_idx < total_batches - 1:
            time.sleep(VOYAGE_BATCH_DELAY)

    return all_embeddings


def embed_query(query: str) -> Optional[list[float]]:
    """
    Embed a single query string for retrieval.

    Uses input_type="query" which tells Voyage AI to optimise the embedding
    for search (asymmetric retrieval) rather than storage. This is important
    for voyage-law-2 — query and document embeddings are in the same space
    but the model applies different weighting for each input type.

    Args:
        query: The lawyer's plain-English case description, or the structured
               summary produced by query/extractor.py.

    Returns:
        A single embedding vector, or None on failure.
    """
    try:
        result = _client.embed(
            texts=[query],
            model=VOYAGE_MODEL,
            input_type="query",
        )
        return result.embeddings[0]

    except Exception as e:
        print(f"  query embedding failed: {e}")
        return None