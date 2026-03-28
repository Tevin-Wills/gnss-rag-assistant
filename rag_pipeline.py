"""
GNSS RAG Pipeline
Core functions: retrieve, prompt construction, and LLM generation.
Uses Groq (free tier) with OpenAI-compatible API.
"""

import os
import time

# Disable tqdm progress bars — prevents OSError on Windows when
# sys.stderr.flush() is called inside Streamlit's thread context.
os.environ["TQDM_DISABLE"] = "1"

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_PATH, EMBEDDING_MODEL,
    TOP_K, LLM_MODEL, MAX_GENERATION_TOKENS,
    GROQ_API_KEY, GROQ_BASE_URL, RAG_SYSTEM_PROMPT, PLAIN_LLM_SYSTEM_PROMPT,
    CHUNKING_STRATEGY, get_collection_name,
)

# Module-level caches (loaded once, reused)
_embedding_model = None
_chroma_client = None
_llm_client = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _chroma_client


def _get_collection():
    client = _get_chroma_client()
    return client.get_collection(get_collection_name())


def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _llm_client


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query.

    Returns a list of dicts with keys:
        text, doc_name, page_start, page_end, score
    """
    model = _get_embedding_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = 1.0 - distance  # cosine distance -> similarity

        meta = results["metadatas"][0][i]
        chunks.append({
            "text": results["documents"][0][i],
            "doc_name": meta["doc_name"],
            "page_start": meta["page_start"],
            "page_end": meta["page_end"],
            "score": round(score, 4),
        })

    return chunks


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build the LLM messages with retrieved context."""
    # Format retrieved sources
    source_sections = []
    for i, chunk in enumerate(chunks, 1):
        pages = (f"p. {chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
                 else f"pp. {chunk['page_start']}-{chunk['page_end']}")
        source_sections.append(
            f"[Source {i}: {chunk['doc_name']}, {pages}]\n{chunk['text']}"
        )

    context_block = "\n\n".join(source_sections)

    user_message = f"""## Retrieved Source Documents

{context_block}

## Question
{query}"""

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate(query: str, chunks: list[dict]) -> str:
    """Generate a RAG-grounded answer using Groq LLM."""
    client = _get_llm_client()
    messages = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_GENERATION_TOKENS,
        messages=messages,
    )

    return response.choices[0].message.content


def generate_without_rag(query: str) -> str:
    """Generate an answer WITHOUT retrieved context (for comparison)."""
    client = _get_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_GENERATION_TOKENS,
        messages=[
            {"role": "system", "content": PLAIN_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message.content


def query_pipeline(query: str, top_k: int = TOP_K) -> dict:
    """Full RAG pipeline: retrieve -> generate, with timing.

    Returns:
        dict with keys: answer, sources, retrieval_time_ms, generation_time_ms, total_time_ms
    """
    total_start = time.time()

    # Retrieve
    retrieval_start = time.time()
    sources = retrieve(query, top_k=top_k)
    retrieval_time = (time.time() - retrieval_start) * 1000

    # Generate
    generation_start = time.time()
    answer = generate(query, sources)
    generation_time = (time.time() - generation_start) * 1000

    total_time = (time.time() - total_start) * 1000

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_time_ms": round(retrieval_time),
        "generation_time_ms": round(generation_time),
        "total_time_ms": round(total_time),
    }


def get_db_stats() -> dict:
    """Get database statistics for the UI sidebar."""
    collection = _get_collection()
    count = collection.count()

    # Count documents by source
    all_meta = collection.get(include=["metadatas"])
    doc_chunk_counts = {}
    for m in all_meta["metadatas"]:
        name = m["doc_name"]
        doc_chunk_counts[name] = doc_chunk_counts.get(name, 0) + 1

    return {
        "total_chunks": count,
        "total_documents": len(doc_chunk_counts),
        "document_names": sorted(doc_chunk_counts.keys()),
        "doc_chunk_counts": doc_chunk_counts,
        "strategy": CHUNKING_STRATEGY,
    }
