"""
GNSS RAG Ingestion Pipeline
Run once: extracts text from PDFs, chunks it, embeds it, and stores in ChromaDB.

Usage:
    python ingest.py                           # uses default strategy from config
    python ingest.py --strategy fixed          # fixed-size token windows (512 tokens)
    python ingest.py --strategy sentence       # sentence-based grouping
    python ingest.py --strategy semantic       # semantic similarity breakpoints
    python ingest.py --chunk-size 200          # fixed strategy at 200 tokens
    python ingest.py --chunk-size 1000         # fixed strategy at 1000 tokens
"""

import os
import re
import time
import argparse
import numpy as np
import pdfplumber
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    PDF_FOLDER, CHROMA_DB_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    TIKTOKEN_ENCODING, DOC_SHORT_NAMES, CHUNKING_STRATEGY,
    SENTENCE_CHUNK_TARGET, SEMANTIC_SIMILARITY_THRESHOLD,
    get_collection_name,
)


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF.
    Returns a list of {page: int, text: str} dicts.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                # Light cleaning
                text = " ".join(text.split())  # collapse whitespace
                pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages: list[dict], doc_name: str, doc_file: str,
                chunk_size: int, chunk_overlap: int, encoding) -> list[dict]:
    """Chunk extracted pages into fixed-size token windows with overlap.
    Tracks page boundaries in metadata.
    """
    # Build a list of (token_id, page_number) pairs
    token_page_pairs = []
    for page_info in pages:
        tokens = encoding.encode(page_info["text"])
        for tok in tokens:
            token_page_pairs.append((tok, page_info["page"]))

    # Slide window
    stride = chunk_size - chunk_overlap
    chunks = []
    idx = 0
    chunk_index = 0

    while idx < len(token_page_pairs):
        window = token_page_pairs[idx: idx + chunk_size]
        if not window:
            break

        token_ids = [pair[0] for pair in window]
        page_numbers = [pair[1] for pair in window]
        text = encoding.decode(token_ids)

        page_start = min(page_numbers)
        page_end = max(page_numbers)

        chunks.append({
            "text": text,
            "doc_name": doc_name,
            "doc_file": doc_file,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
        })

        chunk_index += 1
        idx += stride

        # Stop if remaining tokens are too small to be useful
        if idx < len(token_page_pairs) and len(token_page_pairs) - idx < chunk_overlap:
            break

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on period/question/exclamation followed by space and uppercase,
    # or followed by end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages_sentence(pages: list[dict], doc_name: str, doc_file: str,
                         target_sentences: int, encoding) -> list[dict]:
    """Chunk by grouping sentences together.
    Each chunk contains ~target_sentences sentences, with page tracking.
    """
    # Build list of (sentence_text, page_number)
    sentence_page_pairs = []
    for page_info in pages:
        sents = _split_sentences(page_info["text"])
        for s in sents:
            sentence_page_pairs.append((s, page_info["page"]))

    chunks = []
    chunk_index = 0
    i = 0
    while i < len(sentence_page_pairs):
        group = sentence_page_pairs[i:i + target_sentences]
        text = " ".join(s for s, _ in group)
        pages_in_chunk = [p for _, p in group]

        chunks.append({
            "text": text,
            "doc_name": doc_name,
            "doc_file": doc_file,
            "page_start": min(pages_in_chunk),
            "page_end": max(pages_in_chunk),
            "chunk_index": chunk_index,
        })
        chunk_index += 1
        i += target_sentences

    return chunks


def chunk_pages_semantic(pages: list[dict], doc_name: str, doc_file: str,
                         threshold: float, encoding, embed_model) -> list[dict]:
    """Chunk by detecting semantic topic shifts between sentences.
    Breaks when cosine similarity between consecutive sentence embeddings
    drops below the threshold.
    """
    # Build sentence list with page info
    sentence_page_pairs = []
    for page_info in pages:
        sents = _split_sentences(page_info["text"])
        for s in sents:
            sentence_page_pairs.append((s, page_info["page"]))

    if not sentence_page_pairs:
        return []

    # Embed all sentences
    sentence_texts = [s for s, _ in sentence_page_pairs]
    embeddings = embed_model.encode(sentence_texts, batch_size=64)

    # Compute cosine similarities between consecutive sentences
    chunks = []
    chunk_index = 0
    current_group = [0]  # indices into sentence_page_pairs

    for i in range(1, len(sentence_page_pairs)):
        # Cosine similarity between consecutive embeddings
        sim = float(np.dot(embeddings[i - 1], embeddings[i]) /
                    (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-10))

        if sim < threshold and len(current_group) >= 2:
            # Topic shift detected -- flush current group as a chunk
            text = " ".join(sentence_texts[j] for j in current_group)
            pages_in_chunk = [sentence_page_pairs[j][1] for j in current_group]
            chunks.append({
                "text": text,
                "doc_name": doc_name,
                "doc_file": doc_file,
                "page_start": min(pages_in_chunk),
                "page_end": max(pages_in_chunk),
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            current_group = [i]
        else:
            current_group.append(i)

    # Flush remaining sentences
    if current_group:
        text = " ".join(sentence_texts[j] for j in current_group)
        pages_in_chunk = [sentence_page_pairs[j][1] for j in current_group]
        chunks.append({
            "text": text,
            "doc_name": doc_name,
            "doc_file": doc_file,
            "page_start": min(pages_in_chunk),
            "page_end": max(pages_in_chunk),
            "chunk_index": chunk_index,
        })

    return chunks


def run_ingestion(strategy: str = None, chunk_size_override: int = None):
    """Main ingestion pipeline.

    Args:
        strategy: Chunking strategy override. If None, uses config default.
        chunk_size_override: Override chunk size for fixed strategy (for
            Session 8 slide 27-28 comparison across 200/500/1000 tokens).
    """
    strategy = strategy or CHUNKING_STRATEGY
    cs = chunk_size_override or CHUNK_SIZE
    collection_name = get_collection_name(strategy, chunk_size=chunk_size_override)

    print("=" * 60)
    size_info = f", chunk_size: {cs}" if strategy == "fixed" else ""
    print(f"GNSS RAG Ingestion Pipeline  [strategy: {strategy}{size_info}]")
    print("=" * 60)

    # Initialize tokenizer
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    # Load embedding model early (needed for semantic chunking)
    print(f"\n[Step 1/4] Loading embedding model: {EMBEDDING_MODEL}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded ({embed_model.get_sentence_embedding_dimension()} dimensions)")

    # --- Step 2: Extract text from PDFs ---
    print("\n[Step 2/4] Extracting text from PDFs...")
    pdf_files = sorted([
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ])
    print(f"  Found {len(pdf_files)} PDF files\n")

    all_chunks = []
    for doc_idx, filename in enumerate(pdf_files):
        filepath = os.path.join(PDF_FOLDER, filename)
        doc_name = DOC_SHORT_NAMES.get(filename, filename)

        pages = extract_text_from_pdf(filepath)
        total_tokens = sum(len(encoding.encode(p["text"])) for p in pages)

        print(f"  [{doc_idx + 1}/{len(pdf_files)}] {doc_name}")
        print(f"    Pages: {len(pages)}, Tokens: {total_tokens:,}")

        # --- Step 3: Chunk using selected strategy ---
        if strategy == "fixed":
            overlap = max(1, cs // 8)  # ~12.5% overlap, scales with chunk size
            chunks = chunk_pages(
                pages, doc_name, filename,
                cs, overlap, encoding
            )
        elif strategy == "sentence":
            chunks = chunk_pages_sentence(
                pages, doc_name, filename,
                SENTENCE_CHUNK_TARGET, encoding
            )
        elif strategy == "semantic":
            chunks = chunk_pages_semantic(
                pages, doc_name, filename,
                SEMANTIC_SIMILARITY_THRESHOLD, encoding, embed_model
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(f"    Chunks: {len(chunks)}")
        all_chunks.extend(chunks)

    print(f"\n  Total chunks across all documents: {len(all_chunks)}")

    # --- Step 4: Embed and store ---
    print(f"\n[Step 3/4] Embedding {len(all_chunks)} chunks...")
    start_time = time.time()

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True)
    embed_time = time.time() - start_time
    print(f"  Embedding completed in {embed_time:.1f}s")

    print(f"\n[Step 4/4] Storing in ChromaDB collection '{collection_name}'...")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Delete existing collection if it exists (fresh ingest)
    try:
        client.delete_collection(collection_name)
        print(f"  Cleared existing collection '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []
    embedding_list = []

    for i, chunk in enumerate(all_chunks):
        ids.append(f"chunk_{i:04d}")
        documents.append(chunk["text"])
        metadatas.append({
            "doc_name": chunk["doc_name"],
            "doc_file": chunk["doc_file"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_index": chunk["chunk_index"],
            "strategy": strategy,
        })
        embedding_list.append(embeddings[i].tolist())

    # ChromaDB has a batch limit; add in batches of 500
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embedding_list[start:end],
        )

    total_time = time.time() - start_time
    print(f"  Stored {collection.count()} chunks in ChromaDB at: {CHROMA_DB_PATH}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Strategy:   {strategy}")
    print(f"  Collection: {collection_name}")
    print(f"  Documents:  {len(pdf_files)}")
    print(f"  Chunks:     {collection.count()}")
    print(f"  Embed dims: {embed_model.get_sentence_embedding_dimension()}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  DB path:    {CHROMA_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS RAG Ingestion Pipeline")
    parser.add_argument(
        "--strategy", choices=["fixed", "sentence", "semantic"],
        default=None,
        help="Chunking strategy (default: uses config.py CHUNKING_STRATEGY)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Override chunk size for fixed strategy (e.g., 200, 512, 1000)"
    )
    args = parser.parse_args()
    if args.chunk_size and args.strategy and args.strategy != "fixed":
        parser.error("--chunk-size only applies to the 'fixed' strategy")
    run_ingestion(strategy=args.strategy, chunk_size_override=args.chunk_size)
