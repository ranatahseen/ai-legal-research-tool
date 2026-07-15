import os
from dotenv import load_dotenv

load_dotenv()


OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
LLM_TIMEOUT  = 180
LLM_RETRIES  = 2

#Retrieval

# number of candidates returned by vector search
# set higher than RERANK_TOP_K to give deduplication room to work.
VECTOR_SEARCH_K = 30

# number of cases the reranker selects for memo generation
RERANK_TOP_K = 15
MEMO_TOP_K = 8

# minimum number of retrieved cases required to generate a memo.
MIN_CASES_FOR_MEMO = 3

# chroma

CHROMA_PATH       = "chroma_db"
CHROMA_COLLECTION = "pi_cases"

# voyage AI

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_MODEL   = "voyage-4-large"

# stats

# minimum cases in a filtered subset to report a win rate with confidence.
# below this threshold the memo flags the win rate as low-sample.
WIN_RATE_MIN_SAMPLE = 5

# chat

# number of conversation turns kept in chat history per session.
CHAT_HISTORY_TURNS = 6