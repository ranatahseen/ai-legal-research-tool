# server.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI backend for the PI Case Stress-Tester browser UI.
#
# Wraps the existing query pipeline — no logic lives here.
# All pipeline logic stays in query/pipeline.py, query/chat.py, etc.
#
# Endpoints:
#   POST /api/query          — run full pipeline, return memo + stats + cases
#   POST /api/chat           — single chat turn, return assistant response
#   POST /api/stress-test    — run adverse stress test
#   POST /api/whatif         — run what-if with diff
#   POST /api/export         — generate PDF, return file download
#   GET  /api/status         — health check (Ollama + Chroma)
#   GET  /                   — serve the UI
#
# Run with:
#   pip install fastapi uvicorn
#   python server.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Add project root to path so query/ is importable ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from query.pipeline  import run_query, run_what_if_with_diff
from query.followup  import generate_adverse_stress_test, merge_whatif_into_query, interpret_whatif
from query.chat      import build_chat_context, get_chat_response
from query.exporter  import export_pdf
from query.config    import OLLAMA_URL, MODEL_NAME, CHROMA_PATH, CHROMA_COLLECTION


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="PI Case Stress-Tester", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI at /
UI_FILE = Path(__file__).parent / "ui.html"


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    lawyer_query: str

class ChatRequest(BaseModel):
    canonical_query: str
    reranked_cases:  list[dict]
    history:         list[dict]   # [{question: str, answer: str}]
    question:        str

class StressTestRequest(BaseModel):
    canonical_query: str
    memo:            str

class WhatIfRequest(BaseModel):
    canonical_query: str
    modification:    str
    baseline_stats:  dict
    merged_query:    Optional[str] = None

class MergeFactRequest(BaseModel):
    canonical_query: str
    new_fact:        str

class ExportRequest(BaseModel):
    canonical_query: str
    memo:            str
    stats:           dict
    stress_test:     Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the browser UI."""
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="ui.html not found")
    return UI_FILE.read_text(encoding="utf-8")


@app.get("/api/status")
async def status():
    """
    Health check. Returns connection status for Ollama and Chroma.
    Called by the UI on load to show the status badges in the top bar.
    """
    ollama_ok = False
    chroma_ok = False
    case_count = 0

    # Check Ollama
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    # Check Chroma
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(CHROMA_COLLECTION)
        # Count unique cases via metadata chunks
        result = col.get(
            where={"chunk_type": {"$eq": "metadata"}},
            include=["metadatas"],
        )
        case_count = len(result["metadatas"])
        chroma_ok = True
    except Exception:
        pass

    return {
        "ollama": ollama_ok,
        "chroma": chroma_ok,
        "case_count": case_count,
        "model": MODEL_NAME,
    }


@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    """
    Run the full query pipeline.

    Returns memo text, aggregated stats, and the reranked cases.
    The reranked cases are stored client-side and sent back with each
    chat request — no server-side session state needed.
    """
    if not req.lawyer_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = run_query(req.lawyer_query, return_stats=True)

    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline failed — check Ollama and Chroma")

    memo, stats, reranked = result

    if memo is None:
        raise HTTPException(status_code=500, detail="Memo generation failed")

    return {
        "memo":    memo,
        "stats":   stats,
        "cases":   reranked,
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Single chat turn. Stateless — full history and cases sent by client.

    History is trimmed to CHAT_HISTORY_TURNS inside get_chat_response().
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not req.reranked_cases:
        raise HTTPException(status_code=400, detail="No cases loaded — run a query first")

    answer = get_chat_response(
        canonical_query = req.canonical_query,
        reranked_cases  = req.reranked_cases,
        history         = req.history,
        question        = req.question,
    )

    if answer is None:
        raise HTTPException(status_code=500, detail="Chat response failed")

    return {"answer": answer}


@app.post("/api/stress-test")
async def stress_test_endpoint(req: StressTestRequest):
    """Run the adverse stress test against the canonical case."""
    if not req.canonical_query.strip() or not req.memo.strip():
        raise HTTPException(status_code=400, detail="canonical_query and memo are required")

    result = generate_adverse_stress_test(req.canonical_query, req.memo)

    if result is None:
        raise HTTPException(status_code=500, detail="Stress test generation failed")

    return {"stress_test": result}


@app.post("/api/whatif")
async def whatif_endpoint(req: WhatIfRequest):
    """
    Run a what-if analysis with diff.

    Step 1: interpret the modification (optional — client may skip)
    Step 2: merge into canonical query
    Step 3: run full pipeline
    Step 4: return diff summary + new memo + new stats
    """
    if not req.modification.strip():
        raise HTTPException(status_code=400, detail="modification cannot be empty")

    # Merge the modification into the canonical query if not pre-merged
    merged = req.merged_query
    if not merged:
        merged = merge_whatif_into_query(req.canonical_query, req.modification)

    diff_summary, memo, new_stats = run_what_if_with_diff(
        original_query  = req.canonical_query,
        modification    = req.modification,
        baseline_stats  = req.baseline_stats,
        merged_query    = merged,
    )

    if memo is None:
        raise HTTPException(status_code=500, detail="What-if pipeline failed")

    return {
        "diff_summary": diff_summary,
        "memo":         memo,
        "stats":        new_stats,
        "merged_query": merged,
    }


@app.post("/api/merge-fact")
async def merge_fact_endpoint(req: MergeFactRequest):
    """
    Merge a new fact into the canonical query using the LLM.
    Falls back to plain append if the merge call fails.
    """
    merged = merge_whatif_into_query(req.canonical_query, req.new_fact)
    if not merged:
        merged = req.canonical_query + "\nAdditional fact: " + req.new_fact
    return {"merged_query": merged}


@app.post("/api/interpret-whatif")
async def interpret_whatif_endpoint(req: QueryRequest):
    """
    Interpret a what-if statement before merging.
    Returns the interpreted explicit fact statement for user confirmation.
    """
    result = interpret_whatif("", req.lawyer_query)
    return {"interpreted": result}


@app.post("/api/export")
async def export_endpoint(req: ExportRequest):
    """
    Generate a PDF report and return it as a file download.
    """
    os.makedirs("exports", exist_ok=True)
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"exports/pi_memo_{timestamp}.pdf"

    try:
        path = export_pdf(
            lawyer_query  = req.canonical_query,
            memo          = req.memo,
            stats         = req.stats,
            stress_test   = req.stress_test,
            output_path   = output_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return FileResponse(
        path        = path,
        filename    = f"pi_memo_{timestamp}.pdf",
        media_type  = "application/pdf",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  PI CASE STRESS-TESTER — Web Server")
    print("=" * 56)
    print(f"  UI:    http://localhost:8000")
    print(f"  API:   http://localhost:8000/api/status")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 56 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",   # suppress request noise — errors still show
    )