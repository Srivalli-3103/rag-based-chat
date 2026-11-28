import json
from openai import OpenAI

from vector_store import store
from tools import CALCULATE_TOOL, run_tool

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-20b" 
MAX_TOOL_ROUNDS = 3 

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
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Excerpt {i} — from {chunk.doc_name}]\n{chunk.text}")
    return "\n\n".join(parts)


def _parse_structured_response(text: str) -> dict:
    cleaned = (text or "").strip()
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
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    final_text = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
          model=MODEL,
          messages=messages,
          tools=[CALCULATE_TOOL],
        )
        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "tool_calls" and message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                tool_args = json.loads(tc.function.arguments)
                result = run_tool(tc.function.name, tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue  

        final_text = message.content
        break

    if final_text is None:
        final_text = '{"answer": "I was unable to complete this within the allowed steps.", "sources": [], "confidence": 0.0}'

    parsed = _parse_structured_response(final_text)

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
