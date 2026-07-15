# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import pytest
# from embed.embedder import embed_texts, embed_query
# from embed.config import VOYAGE_EMBEDDING_DIM


# # embed_texts returns exactly as many vectors as texts passed in
# def test_embed_texts_returns_one_vector_per_input():
#     texts = [
#         "Plaintiff slipped on ice outside a grocery store.",
#         "Defendant failed to maintain safe premises.",
#         "Soft tissue injury to the lower back.",
#     ]
#     embeddings = embed_texts(texts)
#     assert embeddings is not None, "embed_texts returned None — check api key"
#     assert len(embeddings) == len(texts)

# # every vector returned by embed_texts has VOYAGE_EMBEDDING_DIM dimensions
# def test_embed_texts_correct_dimension():
#     texts = ["Occupier's liability in Ontario slip and fall."]
#     embeddings = embed_texts(texts)
#     assert embeddings is not None
#     assert len(embeddings[0]) == VOYAGE_EMBEDDING_DIM

# # embed_texts returns an empty list when given no texts
# def test_embed_texts_empty_input():
#     embeddings = embed_texts([])
#     assert embeddings == []

# # embed_query returns exactly one vector of the correct dimension
# def test_embed_query_returns_single_vector():
#     vector = embed_query("Slip and fall on municipal sidewalk, soft tissue injury.")
#     assert vector is not None, "embed_query returned None — check api key"
#     assert isinstance(vector, list)
#     assert len(vector) == VOYAGE_EMBEDDING_DIM