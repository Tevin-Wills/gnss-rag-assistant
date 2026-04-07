# GNSS Technical Knowledge Assistant — RAG Demo

**Assignment 3** for the Artificial Intelligence and Advanced Large Models course (BUAA, Spring 2026).

A Retrieval-Augmented Generation (RAG) prototype that answers GNSS engineering questions grounded in 5 technical documents. Built from scratch (no LangChain) to demonstrate understanding of each RAG pipeline component.

---

## Architecture

```
┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│  5 GNSS  │───▶│  Chunking │───▶│ Embeddings │───▶│ ChromaDB │
│   PDFs   │    │ (3 modes) │    │ MiniLM-L6  │    │Vector DB │
└──────────┘    └───────────┘    └────────────┘    └────┬─────┘
                                                        │
┌──────────┐    ┌───────────┐    ┌────────────┐         │
│ Grounded │◀───│  Llama 3.3│◀───│  Top-K     │◀────────┘
│  Answer  │    │  70B (Groq│    │  Retrieval │
│ + cites  │    │  free)    │    │  (cosine)  │
└──────────┘    └───────────┘    └────────────┘
```

---

## Project Structure

```
ASSIGNMENT 3 (TAKE2)/
├── app.py               # Streamlit UI (3 tabs: Q&A, Evaluation, Strategy Comparison)
├── config.py            # Central configuration (models, chunking, prompts)
├── ingest.py            # PDF → Chunks → Embeddings → ChromaDB (supports --chunk-size)
├── rag_pipeline.py      # Retrieve → Prompt → Generate (with granular timing)
├── generate_report.js   # Node.js report generator (Assignment 3 AI Process Log)
├── requirements.txt     # Python dependencies
├── .env                 # GROQ_API_KEY (not committed)
├── GNSS PDF/            # 5 source documents (~598 pages total)
├── chroma_db/           # Persistent vector database (6 collections)
└── README.md            # This file
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your-key-here
```

Get a free API key from [Groq Console](https://console.groq.com).

### 3. Run ingestion

```bash
# Default: fixed-size token windows
python ingest.py

# Or choose a strategy:
python ingest.py --strategy fixed      # 512-token windows, 64-token overlap
python ingest.py --strategy sentence   # 5-sentence groups
python ingest.py --strategy semantic   # topic-shift breakpoints via embedding similarity

# Chunk-size variants for Session 8 slides 27-28 comparison:
python ingest.py --chunk-size 200      # 200-token windows (1,728 chunks)
python ingest.py --chunk-size 1000     # 1,000-token windows (346 chunks)
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Usage

### Ask a Question tab
- Select one of 6 pre-validated scenario-based questions or type a custom query
- View the RAG-grounded answer with source citations
- Inspect retrieved chunks with colour-coded relevance scores
- Granular timing breakdown: embedding, retrieval, generation, and total (Session 8, slide 22)
- Low-confidence warning when top relevance score < 50% (Session 8, slide 24)
- Toggle "Compare: RAG vs. Plain LLM" to see hallucination reduction

### Evaluation Dashboard tab
- Click "Run Evaluation" to test all 6 questions automatically
- View scorecard table with retrieval/generation times, top relevance scores, and citation detection
- IR metrics: Hit Rate @k and Mean Reciprocal Rank (MRR) per Session 8, slide 26
- Interactive Plotly charts: latency donut chart, document retrieval frequency, colour-coded relevance distribution

### Strategy Comparison tab
- Compare retrieval across 3 chunking strategies (fixed, sentence, semantic) — retrieval only, no LLM tokens burned
- Grouped Plotly bar charts showing top relevance score per question per strategy
- Aggregate metrics table with Hit Rate @k, MRR, and average search latency
- Chunk-size comparison (200, 512, 1000 tokens) with hit rate bar charts — Session 8, slides 27-28 deliverable

---

## Course Concept Mapping

| Session | Concept | Where Demonstrated |
|---------|---------|-------------------|
| Session 6 — Transformers | Embeddings / vector representations | Chunks encoded to 384-dim vectors via sentence-transformers |
| Session 6 — Transformers | Hallucination | Side-by-side RAG vs. plain LLM comparison shows grounding effect |
| Session 7 — Alignment | System prompt safety | RAG prompt refuses to fabricate; states what is missing |
| Session 7 — Alignment | Grounding & citation | Every claim must cite [Source: Doc, pp. X-Y] |
| Session 8 — RAG | Full RAG architecture | PDF → Chunk → Embed → Store → Retrieve → Generate pipeline |
| Session 8 — RAG | Chunking strategies | 3 strategies: fixed-size, sentence-based, semantic breakpoints |
| Session 8 — RAG | Vector database & retrieval | ChromaDB with cosine similarity, Top-K retrieval |
| Session 8 — RAG | Chunk-size comparison | 200/512/1000 token variants with hit rate bar charts (slides 27-28) |
| Session 8 — RAG | IR Metrics | Hit Rate @k and MRR evaluation (slide 26) |
| Session 8 — RAG | Latency breakdown | Donut chart: embedding vs retrieval vs generation (slide 22) |
| Session 8 — RAG | Evaluation | Built-in dashboard running all 6 questions with scorecard |

---

## Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | No LangChain (from scratch) | Educational clarity; shows understanding of each component |
| Embedding | all-MiniLM-L6-v2 (local, free) | 384 dims, fast on CPU, good quality for demo |
| Vector DB | ChromaDB (persistent) | Simple Python API, built-in cosine search, metadata filtering |
| LLM | Llama 3.3 70B via Groq (free) | High quality, OpenAI-compatible API, zero cost |
| Chunking | 3 strategies + 3 chunk sizes | Demonstrates trade-offs (Session 8, slides 7-8, 27-28) |
| Visualisation | Plotly (dark-themed) | Interactive charts with hover tooltips, colour-coded bars |
| PDF extraction | pdfplumber | Reliable text extraction, handles complex layouts |
| Tokenizer | tiktoken (cl100k_base) | Accurate token counting for chunk boundaries |
| Frontend | Streamlit | Rapid prototyping, built-in widgets for data apps |

---

## Sample Questions

1. **Urban NLOS/Multipath** — Signal bands and 3D mapping-aided algorithms for dense urban environments
2. **Spoofing Defense** — Security features and authentication for timing receivers
3. **Maritime Reliability** — Navigation message degradation and operational modes for autonomous vessels
4. **Static Noise Reduction** — Receiver platform models and position publication settings for geological monitoring
5. **GPS Denial Resilience** — Backup PNT technologies and network resilience strategies
6. **RTK Base Station Setup** — Survey-in mode for establishing base station coordinates

---

## Source Documents

| # | Document | Pages | Focus |
|---|----------|-------|-------|
| 1 | GPS SPS Performance Standard (5th ed., 2020) | 196 | Signal specs, accuracy, integrity |
| 2 | 3D Mapping-Aided GNSS (Groves, Inside GNSS) | 7 | Urban canyon solutions |
| 3 | 2021 Federal Radionavigation Plan | 242 | US PNT policy, backup systems |
| 4 | ZED-F9P Integration Manual (u-blox) | 129 | Receiver configuration, RTK |
| 5 | Urban Positioning 3DMA GNSS | 23 | Shadow matching, 3D mapping |

---

## Limitations

- **Groq free tier rate limits** — Evaluation dashboard adds 3-second delays between queries
- **Single embedding model** — No comparison of different embedding models
- **No reranking** — Retrieved chunks ranked by cosine similarity only, no cross-encoder reranking
- **Fixed Top-K** — Retrieval always returns K=8 chunks regardless of query complexity
- **PDF extraction quality** — Tables and figures may not extract cleanly from all documents
