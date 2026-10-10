"""
main.py — The FastAPI app. This is the thin layer that exposes our RAG logic
as HTTP endpoints the React frontend calls.

ENDPOINTS:
  POST /upload   — accepts a .txt or .pdf, chunks + embeds it, stores it
  POST /chat     — runs the full orchestration pipeline for a question
  GET  /health   — simple liveness check (useful once this is deployed)
"""

import os
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

from vector_store import store
from orchestrator import answer_question

load_dotenv()  # reads backend/.env into environment variables

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

app = FastAPI(title="RAG Document Chat API")

# CORS: allows the frontend (running on a different port during dev, e.g.
# localhost:5173) to call this backend (e.g. localhost:8000). Browsers block
# cross-origin requests by default; this explicitly allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for a portfolio demo; in production you'd list exact domains
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


def _extract_text(filename: str, raw_bytes: bytes) -> str:
    """Pulls plain text out of a .txt or .pdf upload."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw_bytes.decode("utf-8", errors="ignore")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_stored": len(store.chunks)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    raw_bytes = await file.read()
    text = _extract_text(file.filename, raw_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in this file.")

    try:
        chunk_count = store.add_document(text, doc_name=file.filename)
    except RuntimeError as exc:
        # Raised by VectorStore when VOYAGE_API_KEY isn't set — surface it clearly
        # instead of a generic 500, since this is a common first-run setup mistake.
        raise HTTPException(status_code=500, detail=str(exc))

    return {"filename": file.filename, "chunks_created": chunk_count}


@app.post("/chat")
def chat(request: ChatRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(request.question, api_key=ANTHROPIC_API_KEY)
    return result


@app.delete("/clear")
def clear_documents():
    """Resets the in-memory store — handy while testing, since restarting the
    server also clears it (nothing is persisted to disk)."""
    store.clear()
    return {"status": "cleared"}
