import os
from dotenv import load_dotenv

load_dotenv()

# voyage ai

VOYAGE_API_KEY   = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL     = "voyage-4-large"

# Voyage AI hard limits for voyage-4-large
VOYAGE_MAX_BATCH_TOKENS  = 120_000   # max tokens per batch request
VOYAGE_MAX_BATCH_SIZE    = 128       # max texts per batch request
VOYAGE_EMBEDDING_DIM     = 2048      # output dimension for voyage-4-large

# Seconds to wait between Voyage API batch calls to avoid rate limiting
VOYAGE_BATCH_DELAY = 22

# chroma

CHROMA_PATH       = "chroma_db"
CHROMA_COLLECTION = "pi_cases"

# chunking

CHUNK_SIZE_CHARS    = 3_200   # ~800 tokens at ~4 chars/token
CHUNK_OVERLAP_CHARS = 400     # ~100 tokens of overlap between chunks

# source data

PI_CASES_FILE = "pi_cases.json"