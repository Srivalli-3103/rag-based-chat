"""
vector_store.py — The RAG core: chunking, embedding, and similarity search.

WHY THIS FILE EXISTS:
An LLM can't "read" a 50-page document in one go effectively, and you don't want
to send the whole document on every question (slow, expensive, and the model
may ignore relevant parts buried in a huge prompt).

Instead, RAG (Retrieval-Augmented Generation) works like this:
  1. Break the document into small chunks ahead of time.
  2. Convert each chunk into a vector (a list of numbers) that captures its MEANING.
  3. When a question comes in, convert the question into a vector too.
  4. Find the chunks whose vectors are most similar to the question's vector.
  5. Only send THOSE chunks to the LLM, not the whole document.

This file implements steps 1, 2, and 4.
"""

import os
import re
from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# STEP 1: CHUNKING
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One piece of a document, plus metadata about where it came from."""
    id: str                # e.g. "doc1_chunk_3" — unique identifier
    text: str               # the actual chunk text
    doc_name: str           # which document this came from (for citations)
    chunk_index: int        # position in the document (0, 1, 2, ...)
    embedding: list = field(default=None, repr=False)  # filled in later


def chunk_text(text: str, doc_name: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """
    Splits text into overlapping word-based chunks.

    WHY WORD-BASED (not character-based)?
    Word boundaries keep chunks readable and avoid cutting words in half.

    WHY CHUNK_SIZE = 800 WORDS?
    A tradeoff: too small (e.g. 100 words) and chunks lose context — a sentence
    about "the policy" might not say WHICH policy. Too large (e.g. 3000 words)
    and you dilute relevance — the embedding represents an average of many
    ideas, making it harder to match a specific question precisely.
    800 words (~1000-1200 tokens) is a common middle ground for Q&A-style RAG.

    WHY OVERLAP = 150 WORDS?
    Without overlap, a sentence split across two chunks loses context in both
    halves. Overlap means each chunk shares some text with its neighbor, so
    an idea near a chunk boundary is still captured whole in at least one chunk.
    """
    words = re.findall(r"\S+", text)  # split on whitespace, keep words intact
    if not words:
        return []

    chunks = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(Chunk(
            id=f"{doc_name}_chunk_{index}",
            text=chunk_text_str,
            doc_name=doc_name,
            chunk_index=index,
        ))

        index += 1
        start += step
        if end == len(words):
            break  # reached the end, stop

    return chunks


# ---------------------------------------------------------------------------
# STEP 2 & 4: EMBEDDING + SIMILARITY SEARCH
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Holds embedded chunks in memory and answers "which chunks are most
    relevant to this question?" using cosine similarity.

    WHY VOYAGE AI FOR EMBEDDINGS (not building embeddings ourselves)?
    Embedding models are trained on massive text corpora to learn what makes
    two pieces of text "similar in meaning." This is a solved problem with
    specialized models (Voyage, OpenAI, Cohere) — nobody builds this from
    scratch in production. Voyage's voyage-3.5 model is tuned for retrieval
    tasks specifically (as opposed to e.g. classification).

    WHY IN-MEMORY (not a vector database like Pinecone)?
    For a single-document, single-session demo, a real vector DB is
    overkill — it adds infrastructure (hosting, indexing, network calls)
    for a problem a numpy array solves in milliseconds. In production,
    with millions of chunks across many users, you'd switch to a real
    vector DB for persistence and fast approximate search at scale.
    """

    def __init__(self):
        api_key = os.environ.get("VOYAGE_API_KEY")
        self.client = None
        if api_key:
            import voyageai  # imported lazily so the module loads (and tests run) without the package
            self.client = voyageai.Client(api_key=api_key)
        self.chunks: list[Chunk] = []

    def embed_text(self, texts: list[str], input_type: str) -> list[list[float]]:
        """
        input_type is either "document" (for chunks being stored) or
        "query" (for the user's question). Voyage's model treats these
        differently internally — documents and queries aren't phrased the
        same way, so encoding them with the right mode improves match quality.
        """
        if not self.client:
            raise RuntimeError("VOYAGE_API_KEY not set — cannot create embeddings.")
        result = self.client.embed(texts, model="voyage-3.5", input_type=input_type)
        return result.embeddings

    def add_document(self, text: str, doc_name: str):
        """Chunks a document, embeds every chunk, and stores them."""
        new_chunks = chunk_text(text, doc_name)
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        embeddings = self.embed_text(texts, input_type="document")

        for chunk, emb in zip(new_chunks, embeddings):
            chunk.embedding = emb
            self.chunks.append(chunk)

        return len(new_chunks)

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """
        Returns the top_k chunks most relevant to the query.

        COSINE SIMILARITY, explained simply:
        Each embedding is a vector (direction in high-dimensional space).
        Cosine similarity measures the ANGLE between two vectors, not their
        length — so it asks "do these two pieces of text point in a similar
        semantic direction?" rather than "are they the same length/magnitude?"
        Score ranges from -1 (opposite meaning) to 1 (identical meaning).
        """
        if not self.chunks:
            return []

        query_embedding = self.embed_text([query], input_type="query")[0]

        doc_matrix = np.array([c.embedding for c in self.chunks])
        query_vec = np.array(query_embedding)

        # cosine similarity = dot product / (magnitude_a * magnitude_b)
        dot_products = doc_matrix @ query_vec
        norms = np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(query_vec)
        similarities = dot_products / norms

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def clear(self):
        self.chunks = []


# A single shared instance used across the app (simple approach for a demo —
# in production you'd scope this per-user or per-session instead of global).
store = VectorStore()
