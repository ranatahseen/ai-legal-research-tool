# layer 1 — metadata pre-filter
#   Uses the structured facts from extractor.py to build a Chroma `where`
#   clause. This shrinks the search space from all chunks to only the legally
#   comparable subset before any vector math runs.

# Layer 2 — Vector similarity search
#   Runs cosine similarity only within the Layer 1 filtered subset.
#   Returns the top VECTOR_SEARCH_K chunks for reranking.


from typing import Optional

import chromadb
import voyageai
from chromadb.config import Settings

from .config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    VECTOR_SEARCH_K,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)

_chroma_client: chromadb.Client | None = None


def _get_client() -> chromadb.Client:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    return _get_client().get_collection(name=CHROMA_COLLECTION)


_voyage = voyageai.Client(api_key=VOYAGE_API_KEY)


# layer 1 build metadata filter

def _build_where_clause(facts: dict) -> Optional[dict]:
    """
    Build a Chroma metadata filter using boolean flags.

    Uses has_* boolean flags instead of $contains on strings.
    Chroma 1.5.9 $contains is broken for substring matching — boolean flags
    stored during embedding are the only reliable filter strategy.

    Two-tier logic:
      Tier 1 ($and): injury_type flags AND liability_theory when both present
      Tier 2 ($or):  fallback when only one signal available
    """
    injury_types  = [t for t in facts.get("injury_type", []) if t and t != "other"]
    liability     = facts.get("liability_theory")
    if liability in ("other", "null", "", None):
        liability = None

    injury_condition = None
    valid_injuries = [t for t in injury_types if t in {
        "slip_and_fall", "fracture", "soft_tissue", "mTBI", "chronic_pain",
        "psychological", "orthopedic", "spinal_cord", "amputation", "wrongful_death"
    }]
    if len(valid_injuries) == 1:
        injury_condition = {f"has_{valid_injuries[0]}": {"$eq": True}}
    elif len(valid_injuries) > 1:
        injury_condition = {"$or": [
            {f"has_{t}": {"$eq": True}} for t in valid_injuries
        ]}

    liability_condition = None
    if liability:
        liability_condition = {"liability_theory": {"$eq": liability}}

    defendant_types = [d for d in facts.get("defendant_type", []) if d and d != "other"]
    defendant_condition = None
    valid_defendants = [d for d in defendant_types if d in {
        "driver", "municipality", "retailer", "property_owner", "employer"
    }]
    if len(valid_defendants) == 1:
        defendant_condition = {f"has_{valid_defendants[0]}": {"$eq": True}}
    elif len(valid_defendants) > 1:
        defendant_condition = {"$or": [
            {f"has_{d}": {"$eq": True}} for d in valid_defendants
        ]}


    municipal_condition = None
    if facts.get("municipal_liability_case") is True:
        municipal_condition = {"municipal_liability_case": {"$eq": True}}

    if injury_condition and liability_condition:
        and_conditions = [injury_condition, liability_condition]
        if municipal_condition:
            and_conditions.append(municipal_condition)
        return {"$and": and_conditions}

    or_conditions = [c for c in [
        injury_condition,
        liability_condition,
        defendant_condition,
        municipal_condition,
    ] if c is not None]

    if not or_conditions:
        return None
    if len(or_conditions) == 1:
        return or_conditions[0]
    return {"$or": or_conditions}


def _embed_query(query_text: str) -> Optional[list[float]]:
    """
    Embed the query using Voyage AI in query mode (asymmetric retrieval).
    """
    try:
        result = _voyage.embed(
            texts=[query_text],
            model=VOYAGE_MODEL,
            input_type="query",
        )
        return result.embeddings[0]
    except Exception as e:
        print(f"  query embedding failed: {e}")
        return None


def _deduplicate_by_case(results: list[dict]) -> list[dict]:
    """
    Chroma returns chunks, but we want cases.

    Multiple chunks from the same case may appear in the top K results.
    Keep only the highest-scoring chunk per case (identified by source_file),
    then return in score order.

    This ensures the reranker sees K distinct cases, not K chunks from
    potentially fewer cases.
    """
    seen:   dict[str, dict] = {}  # source_file → best chunk

    for result in results:
        source = result["metadata"].get("source_file", "")
        if source not in seen:
            seen[source] = result
        # Chroma returns results sorted by distance (lower = more similar).
        # First occurrence is already the best for this case.

    return list(seen.values())


def retrieve_candidates(
    lawyer_query: str,
    facts: dict,
    n_results: int = VECTOR_SEARCH_K,
) -> list[dict]:
    """
    Run Layer 1 + Layer 2 retrieval and return candidate cases.

    Layer 1: build metadata filter from structured facts
    Layer 2: embed query and run cosine similarity within filtered subset

    Returns a list of unique cases (deduplicated by source_file), each with:
        chunk_id, text, metadata (all stored fields), distance

    Falls back to unfiltered vector search if:
      - Layer 1 produces no meaningful filter
      - Filtered search returns fewer than MIN_CASES_FOR_MEMO results

    Args:
        lawyer_query: Original plain-English query (used for embedding).
        facts:        Structured facts from extractor.py (used for filtering).
        n_results:    Number of candidates to return before reranking.
    """
    from .config import MIN_CASES_FOR_MEMO

    query_embedding = _embed_query(lawyer_query)
    if query_embedding is None:
        return []

    try:
        collection = _get_collection()
    except Exception as e:
        print(f"  Chroma connection failed: {e}")
        return []
    
    where = _build_where_clause(facts)

    def _search(where_clause):
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results":        n_results,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where_clause:
            kwargs["where"] = where_clause
        results = collection.query(**kwargs)
        return [
            {
                "chunk_id": cid,
                "text":     doc,
                "metadata": meta,
                "distance": dist,
            }
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    if where:
        print(f"  Layer 1 filter: {where}")
        try:
            results = _search(where)
            cases   = _deduplicate_by_case(results)
            print(f"  Layer 2: {len(cases)} unique cases after filtered search")

            if len(cases) >= MIN_CASES_FOR_MEMO:
                return cases

            print(f"  filtered search returned only {len(cases)} cases — falling back to unfiltered")
        except Exception as e:
            print(f"  filtered search failed ({e}) — falling back to unfiltered")

    print("  running unfiltered vector search")
    try:
        results = _search(None)
        cases   = _deduplicate_by_case(results)
        print(f"  Layer 2: {len(cases)} unique cases (unfiltered)")
        return cases
    except Exception as e:
        print(f"  unfiltered search failed: {e}")
        return []