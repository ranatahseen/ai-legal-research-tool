# PI Case Stress-Tester

An AI-powered legal research pipeline that ingests real Ontario personal injury (PI) case law, embeds it into a searchable vector database, and lets a lawyer describe a new case in plain English to get back a research memo: comparable precedents, an aggregated win rate, a damages range, and an AI-generated legal argument plus interactive follow-ups like "what if" scenario testing and an adverse stress test of the case's weak points.

This started as a personal project to explore how retrieval-augmented generation (RAG) holds up on a domain that punishes sloppy matching, case law where two documents can share almost all the same words and still be legally irrelevant to each other.

---

## What it does

A lawyer types something like:

> *"Plaintiff slipped on ice in a grocery store parking lot, fractured wrist, store admits no salting log exists."*

The tool then:

1. Extracts structured facts from that sentence (injury type, liability theory, defendant type).
2. Filters and searches a database of ~hundreds of real Ontario PI decisions for legally comparable cases.
3. Uses an LLM to **rerank** those candidates by actual legal relevance (not just word similarity).
4. Aggregates outcome statistics, win rate, median/range of damages awarded, across the comparable set.
5. Writes a research memo citing the most relevant precedents and explaining why each one applies.

From there, in the interactive UI or CLI, the lawyer can:

- Add a new fact to the case and get the memo automatically refreshed.
- Run a **"what if"** scenario (e.g. "what if there was a pre-existing back injury?") without altering the original case.
- Run a **"worst case" stress test**, where the tool argues the opposing side's strongest position.
- Chat with the tool using the retrieved cases as grounded context.
- Export the whole analysis as a formatted PDF memo.

## Why this project is interesting

- **Five-layer retrieval pipeline** (see [Architecture](#architecture) below) metadata pre-filtering, vector search, and LLM reranking are separate, inspectable stages rather than one black-box call.
- **Resume-safe batch pipelines** both the PDF ingestion and embedding stages can be killed mid-run and restarted without reprocessing or re-embedding work already done.
- **Two-tier data quality system** a keyword pre-filter plus an LLM classifier decide whether a scraped judgment is actually a PI case before spending a full extraction call on it, and a hard/soft required-field system flags incomplete extractions instead of silently corrupting the dataset.
- **Runs entirely on local/self-hosted inference** for the reasoning steps (via [Ollama](https://ollama.com)), with a managed embeddings API only for the vector search step.

## Architecture

The project is split into three independent pipelines plus a web layer:

```
 ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
 │  ingest/   │ --> │  embed/    │ --> │  query/    │ --> │ server.py  │
 │  PDFs →    │     │  chunks →  │     │  facts →   │     │ FastAPI +  │
 │  pi_cases  │     │  vectors → │     │  memo /    │     │ ui.html    │
 │  .json     │     │  Chroma    │     │  chat/...  │     │            │
 └────────────┘     └────────────┘     └────────────┘     └────────────┘
```

**1. `ingest/`** — turns raw case PDFs into structured JSON.
- `loader.py` pulls text out of each PDF (via PyMuPDF).
- `filters.py` runs a two-tier keyword screen, then an LLM classifier, to decide if a document is actually a relevant PI judgment.
- `extractor.py` prompts the LLM (`prompts.py`) to pull structured fields out of the judgment: case name, outcome, damages awarded, injury type, liability theory, deciding factors, etc.
- `dataset.py` validates each record against required fields (hard-required vs. soft/flag-for-review) and writes/resumes `pi_cases.json`.
- Entry point: `python -m ingest`

**2. `embed/`** — turns the structured case dataset into a searchable vector index.
- `chunker.py` splits each case's full text into overlapping chunks sized for retrieval.
- `embedder.py` calls the [Voyage AI](https://www.voyageai.com/) embeddings API (`voyage-4-large`) in batches.
- `store.py` upserts chunks + embeddings + metadata (including boolean `has_*` flags used for fast filtering later) into a persistent [ChromaDB](https://www.trychroma.com/) collection.
- Entry point: `python -m embed`

**3. `query/`** — the live research pipeline, run per lawyer query:
- **Layer 0** (`extractor.py`) — extract structured facts from the plain-English query.
- **Layers 1–2** (`retriever.py`) — build a metadata filter from those facts, then run vector similarity search inside that filtered subset (falling back to unfiltered search if the filter is too narrow).
- **Layer 3** (`reranker.py`) — an LLM reranks the vector-search candidates by genuine legal relevance (same liability theory, same causation doctrine, comparable injury severity) rather than surface wording.
- **Layer 4** (`aggregator.py`) — computes win rate and damages statistics across the retrieved set.
- **Layer 5** (`memo.py`) — an LLM writes the final research memo citing the top cases.
- `followup.py` handles what-if merging, stress tests, and clarifying questions. `chat.py` handles case-aware Q&A. `exporter.py` renders the PDF (via ReportLab).
- Entry points: `python -m query` (CLI) or via the web server below.

**4. `server.py` + `ui.html`** — a FastAPI backend that wraps the query pipeline behind a REST API, and a single-page browser UI for running queries, chatting, stress-testing, and exporting PDFs interactively.

## Tech stack

| Purpose | Tool |
|---|---|
| LLM inference (extraction, reranking, memo writing, chat) | [Ollama](https://ollama.com) (local/self-hosted) |
| Text embeddings | [Voyage AI](https://www.voyageai.com/) (`voyage-4-large`) |
| Vector database | [ChromaDB](https://www.trychroma.com/) (persistent, local) |
| PDF text extraction | PyMuPDF (`fitz`) |
| PDF report generation | ReportLab |
| Web backend | FastAPI + Uvicorn |
| HTTP client | httpx |
| Frontend | Single-file HTML/JS/CSS (`ui.html`) — no build step |

## Getting started

### Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/download)** installed and running locally, with a model available. The pipeline defaults to `gemma4:31b-cloud`, configurable via the `OLLAMA_MODEL` environment variable if you'd rather point it at a model you have pulled locally:
   ```bash
   ollama pull <your-model-name>
   ```
3. **A [Voyage AI](https://www.voyageai.com/) API key** (used for embeddings) — free tier is available.

### Installation

```bash
git clone https://github.com/ranatahseen/ai-legal-research-tool.git
cd ai-legal-research-tool

python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your Voyage API key:

```
VOYAGE_API_KEY=your-key-here
```

### 1. Build the case dataset

The `cases/` folder contains the raw PI judgment PDFs. Run the ingestion pipeline to extract structured data from them into `pi_cases.json` (this step calls the LLM once per candidate case, so it can take a while for a large folder — it's resume-safe if interrupted):

```bash
python -m ingest
```

Useful variants:
```bash
python -m ingest stats    # print a health report on pi_cases.json
python -m ingest patch    # attempt to fix cases with missing damages values
```

### 2. Embed the dataset into the vector store

```bash
python -m embed
```

This chunks every case in `pi_cases.json`, embeds each chunk via Voyage AI, and upserts everything into a local Chroma collection (`chroma_db/`). Also resume-safe.

### 3. Run a query

**Option A — interactive CLI:**
```bash
python -m query
```
Then describe a case in plain English. Once you have a memo, you can use:
- `fact <text>` — add a fact to the case and refresh the memo
- `whatif <text>` — explore a hypothetical without changing the case
- `worst case` — generate an adverse stress test
- `chat` — ask follow-up questions grounded in the retrieved cases
- `export` — save the memo as a PDF
- `new` / `help` / `quit`

**Option B — one-shot CLI:**
```bash
python -m query "Plaintiff slipped on ice in a grocery store parking lot..."
```

**Option C — web UI (recommended for demoing):**
```bash
python server.py
```
Then open **http://localhost:8000** in a browser.

## Project structure

```
.
├── cases/            # Raw PI judgment PDFs (source data)
├── ingest/           # PDF → structured JSON pipeline
├── embed/            # JSON → chunked, embedded Chroma vector store
├── query/            # Live research pipeline (retrieval, reranking, memo, chat, export)
├── pi_cases.json      # Extracted structured case dataset (output of ingest/)
├── server.py         # FastAPI backend wrapping the query pipeline
├── ui.html            # Browser UI served by server.py
└── requirements.txt
```

## Notes & limitations

- The reranker and memo-writer are only as good as the local model backing Ollama — results will vary noticeably depending on which model you point it at.
- This tool is built and tuned for **plaintiff-side** analysis of Ontario PI cases specifically; it will flag when a query looks like a defense-side question, but results skew toward plaintiff framing regardless.
- The case dataset is a snapshot of publicly available Ontario decisions (CanLII and court sources) — it is not exhaustive and is not updated automatically.
