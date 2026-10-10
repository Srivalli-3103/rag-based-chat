# Marginnote — Chat With Your Document

A RAG (Retrieval-Augmented Generation) app: upload a document, ask questions,
get answers grounded in that document with citations back to the exact
excerpts used.

Built as a learning + portfolio project to go from "I studied RAG in a
certification course" to "I built RAG and can explain every design decision
in it" — for a Frontend Developer transitioning toward Frontend+AI roles.

## Concept → code map

| Concept | File | What to look at |
|---|---|---|
| Chunking | `backend/vector_store.py` | `chunk_text()` — word-based, 800-word chunks, 150-word overlap |
| Embeddings | `backend/vector_store.py` | `VectorStore.embed_text()` — Voyage AI `voyage-3.5` |
| Semantic search / RAG retrieval | `backend/vector_store.py` | `VectorStore.search()` — cosine similarity over all stored chunks |
| Tool integration | `backend/tools.py` | `CALCULATE_TOOL` schema + `calculate()` — AST-based, not `eval()` |
| Orchestration | `backend/orchestrator.py` | `answer_question()` — retrieve → reason → tool loop → structure |
| Structured outputs | `backend/orchestrator.py` | `SYSTEM_PROMPT` + `_parse_structured_response()` |
| Frontend: loading/error states | `frontend/src/App.jsx` | `isThinking`, `isUploading`, `uploadError` state |
| Frontend: rendering structured AI output | `frontend/src/App.jsx` | `MessageBubble` — turns `sources[]` into citation footnotes |
| Frontend: upload UX | `frontend/src/App.jsx` | `handleFile`, drag/drop handlers |

## Why these design choices (the questions an interviewer will ask)

**Why 800-word chunks with 150-word overlap?**
Small chunks lose surrounding context; very large chunks dilute the
embedding's meaning (it becomes an average of too many ideas). Overlap
prevents an idea from being orphaned if it falls right on a chunk boundary.

**Why Voyage AI for embeddings instead of writing our own?**
Embedding quality is a solved, specialized problem — nobody trains their own
in production. `voyage-3.5` is tuned specifically for retrieval, and using
`input_type="document"` vs `"query"` lets the model encode the two differently
since they're phrased differently.

**Why cosine similarity?**
It measures the *angle* between two vectors — i.e. do they point in a similar
semantic direction — independent of vector length. That's what "similar
meaning" means in embedding space.

**Why an in-memory store instead of a real vector database?**
For a single-session demo, numpy in memory is instant and has zero
infrastructure cost. A production app with millions of chunks across many
users would need a real vector DB (Pinecone, Weaviate, pgvector) for
persistence and fast approximate search at scale — that's a scaling decision,
not a correctness one.

**Why AST parsing instead of `eval()` for the calculator tool?**
The tool's input comes from the model, which is ultimately reacting to
untrusted document text (a prompt-injection vector). `eval()` would let
injected text execute arbitrary Python. Parsing into an AST and only
evaluating whitelisted arithmetic node types makes that attack impossible.

**Why force structured JSON output instead of free text?**
The frontend needs to reliably render `sources` as citation chips and
`confidence` as a score — that's only possible with a predictable shape, not
by parsing prose.

**Why orchestration as explicit steps instead of one big prompt?**
If an answer is wrong, you can tell whether retrieval found the wrong chunks
(bad context) or the model reasoned poorly over the right chunks (bad
reasoning). A single monolithic call gives you no way to isolate the failure.

## Architecture

```
Browser (React + Vite)
      │ fetch()
      ▼
FastAPI backend (one process)
      ├── vector_store.py  (in-memory, numpy)
      ├── Voyage AI API    (embeddings)
      └── Anthropic API    (Claude — reasoning, tool calls, structured output)
```

Monorepo, monolithic backend — deliberately simple. Micro-frontends and
microservices are enterprise-scale patterns this project doesn't need.

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your real API keys
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                     # runs on http://localhost:5173
```

Open `http://localhost:5173`, upload a `.txt` or `.pdf`, and start asking
questions.

### Required API keys (both free-tier available)
- `ANTHROPIC_API_KEY` — console.anthropic.com
- `VOYAGE_API_KEY` — voyageai.com

## API reference

```bash
# Health check
curl http://localhost:8000/health

# Upload a document
curl -F "file=@yourdoc.pdf" http://localhost:8000/upload

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund window?"}'
```

## What this project is, honestly

This is a learning + portfolio project, not a production system. The backend
(chunking, embeddings, orchestration) was built to deeply understand RAG
concepts end-to-end — in a real Frontend+AI role, that layer is typically
owned by backend/AI engineers and exposed to the frontend as an API. The
frontend (`App.jsx`) is the part that reflects actual day-to-day
Frontend+AI work: consuming an AI-backed API, handling its unpredictability
gracefully, and rendering structured AI output as real UI.
