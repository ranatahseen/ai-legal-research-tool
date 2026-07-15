# Two public functions for the embedding pipeline:
#   get_collection()         — returns the Chroma collection (creates if new)
#   upsert_chunks()          — adds or updates chunks in the collection

# Two public functions for the query pipeline:
#   get_embedded_ids()       — returns set of source_file values already in Chroma
#   query_collection()       — metadata filter + vector search

# Nothing here calls Voyage AI or knows about case JSON structure.

import chromadb
from chromadb.config import Settings

from .config import CHROMA_COLLECTION, CHROMA_PATH, VOYAGE_EMBEDDING_DIM
from .chunker import Chunk

# client

_client: chromadb.Client | None = None

def _get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """
    Return the PI cases Chroma collection, creating it if it doesn't exist.

    Uses cosine similarity — appropriate for normalised Voyage AI embeddings.
    """
    client = _get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# write to chroma

def upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """
    Add or update chunks in the Chroma collection.

    Uses upsert (not add) so re-running the pipeline on an existing collection
    updates records rather than duplicating them.

    Args:
        collection:  The Chroma collection returned by get_collection().
        chunks:      List of Chunk objects from chunker.py.
        embeddings:  Parallel list of embedding vectors from embedder.py.
                     Must be the same length as chunks.
    """
    if not chunks:
        return

    collection.upsert(
        ids        = [c.chunk_id  for c in chunks],
        documents  = [c.text      for c in chunks],
        embeddings = embeddings,
        metadatas  = [c.metadata  for c in chunks],
    )


# read from chroma

def get_embedded_ids(collection: chromadb.Collection) -> set[str]:
    """
    Return the set of source_file values already embedded in Chroma.

    Used by the embedding pipeline for resume support — cases whose
    source_file is already in Chroma are skipped on re-run.

    Note: we track by source_file (not chunk_id) because a single case
    produces many chunks. Filtering to chunk_type="metadata" means we
    fetch exactly one record per case instead of all 10-15 chunks —
    this keeps the resume check fast as the dataset grows.
    """
    result = collection.get(
        where={"chunk_type": {"$eq": "metadata"}},
        include=["metadatas"],
    )
    return {
        m["source_file"]
        for m in result["metadatas"]
        if m.get("source_file")
    }


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    where: dict | None = None,
    n_results: int = 15,
) -> list[dict]:
    """
    Run a metadata-filtered vector similarity search.

    Layer 1 (metadata filter) and Layer 2 (vector search) happen in a single
    Chroma call — Chroma applies the where filter first, then ranks by cosine
    similarity within the filtered subset.

    Args:
        collection:      The Chroma collection.
        query_embedding: Embedding of the lawyer's query (from embedder.py).
        where:           Chroma metadata filter dict. None = no filter.
                         Use boolean flag fields — $contains does NOT do
                         substring matching on strings in Chroma 1.5.9.
                         Example: {"has_soft_tissue": {"$eq": True}}
        n_results:       Number of results to return (default 15 for reranking).

    Returns:
        List of result dicts, each containing:
            chunk_id, text, metadata, distance
        Sorted by similarity (most similar first).
    """
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        print(f"  Chroma query failed: {e}")
        return []

    # Unpack Chroma's nested response format into a flat list of dicts
    output = []
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        output.append({
            "chunk_id": chunk_id,
            "text":     doc,
            "metadata": meta,
            "distance": dist,
        })

    return output