"""
orchestrator.py — The "orchestration" concept: the sequence of steps that turns
a user's question into a grounded, structured answer.

THE SEQUENCE:
  1. RETRIEVE  — search the vector store for the most relevant document chunks.
  2. REASON    — send those chunks + the question to Claude, with a tool available.
  3. TOOL LOOP — if Claude asks to call `calculate`, run it and hand the result back.
                 This can repeat (Claude might ask for more than one calculation).
  4. STRUCTURE — Claude is instructed to answer in a strict JSON shape so the
                 frontend can render it reliably instead of parsing free text.

WHY NOT JUST ONE BIG PROMPT WITH EVERYTHING?
  Each step has a distinct, checkable job. Separating them means: if an answer
  is wrong, you can tell whether retrieval found the wrong chunks (bad context)
  or Claude reasoned poorly over the right chunks (bad reasoning) — a single
  monolithic call gives you no way to isolate which part failed.
"""

import json
from anthropic import Anthropic

from vector_store import store
from tools import CALCULATE_TOOL, run_tool

MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 3  # safety cap so a confused model can't loop forever

SYSTEM_PROMPT = """You are a document Q&A assistant. You answer questions using \
ONLY the document excerpts provided in the user's message — never your own \
outside knowledge. If the excerpts don't contain the answer, say so clearly.

Use the `calculate` tool for any arithmetic instead of computing it yourself.

When you give your final answer, respond with ONLY a JSON object in this \
exact shape, and nothing else — no markdown fences, no preamble:
{
  "answer": "<your answer, grounded in the excerpts>",
  "sources": [<integers: which excerpt numbers you actually used, e.g. [1,3]>],
  "confidence": <float 0.0-1.0: how directly the excerpts support this answer>
}"""


def _build_context_block(chunks) -> str:
    """Numbers each retrieved chunk so Claude can cite them as [1], [2], etc.,
    and so we can map those numbers back to real sources for the frontend."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Excerpt {i} — from {chunk.doc_name}]\n{chunk.text}")
    return "\n\n".join(parts)


def _parse_structured_response(text: str) -> dict:
    """Claude is instructed to return raw JSON, but models sometimes wrap it
    in ```json fences anyway — strip those defensively before parsing."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: never let a parse failure crash the request — degrade gracefully.
        return {"answer": text, "sources": [], "confidence": 0.0}


def answer_question(question: str, api_key: str) -> dict:
    """Runs the full retrieve -> reason -> tool-loop -> structure pipeline.
    Returns a dict ready to send to the frontend as JSON."""
    client = Anthropic(api_key=api_key)

    # --- Step 1: RETRIEVE ---
    retrieved_chunks = store.search(question, top_k=4)
    if not retrieved_chunks:
        return {
            "answer": "No document has been uploaded yet, so I have nothing to search.",
            "sources": [],
            "confidence": 0.0,
            "retrieved_chunks": [],
        }

    context_block = _build_context_block(retrieved_chunks)
    user_message = f"{context_block}\n\nQuestion: {question}"

    messages = [{"role": "user", "content": user_message}]

    # --- Step 2 & 3: REASON, with a TOOL LOOP ---
    final_text = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[CALCULATE_TOOL],
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Claude wants to call a tool. Append its request to the conversation,
            # run every tool call it asked for, and append the results.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue  # loop back: send the tool result, let Claude continue reasoning

        # Not a tool call -> this is the final answer.
        final_text = "".join(block.text for block in response.content if block.type == "text")
        break

    if final_text is None:
        final_text = '{"answer": "I was unable to complete this within the allowed steps.", "sources": [], "confidence": 0.0}'

    # --- Step 4: STRUCTURE ---
    parsed = _parse_structured_response(final_text)

    # Map Claude's cited excerpt numbers back to real chunk data for the frontend.
    cited_sources = []
    for n in parsed.get("sources", []):
        if isinstance(n, int) and 1 <= n <= len(retrieved_chunks):
            chunk = retrieved_chunks[n - 1]
            cited_sources.append({
                "doc_name": chunk.doc_name,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            })

    return {
        "answer": parsed.get("answer", ""),
        "sources": cited_sources,
        "confidence": parsed.get("confidence", 0.0),
        "retrieved_chunks": [
            {"doc_name": c.doc_name, "chunk_index": c.chunk_index, "text": c.text[:200] + "..."}
            for c in retrieved_chunks
        ],
    }
