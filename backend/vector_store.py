import os
import re
from dataclasses import dataclass, field
import numpy as np

# STEP 1: CHUNKING

@dataclass
class Chunk:
    """One piece of a document, plus metadata about where it came from."""
    id: str                
    text: str               
    doc_name: str           # which document this came from 
    chunk_index: int        # position in the document (0, 1, 2, ...)
    embedding: list = field(default=None, repr=False)  # filled in later


def chunk_text(text: str, doc_name: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    words = re.findall(r"\S+", text)  
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
            break  

    return chunks



#  EMBEDDING + SIMILARITY SEARCH


class VectorStore:
    def __init__(self):
        api_key = os.environ.get("VOYAGE_API_KEY")
        self.client = None
        if api_key:
            import voyageai  
            self.client = voyageai.Client(api_key=api_key)
        self.chunks: list[Chunk] = []

    def embed_text(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not self.client:
            raise RuntimeError("VOYAGE_API_KEY not set — cannot create embeddings.")
        result = self.client.embed(texts, model="voyage-3.5", input_type=input_type)
        return result.embeddings

    def add_document(self, text: str, doc_name: str):
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
        if not self.chunks:
            return []

        query_embedding = self.embed_text([query], input_type="query")[0]

        doc_matrix = np.array([c.embedding for c in self.chunks])
        query_vec = np.array(query_embedding)
        dot_products = doc_matrix @ query_vec
        norms = np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(query_vec)
        similarities = dot_products / norms

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    def clear(self):
        self.chunks = []

store = VectorStore()
