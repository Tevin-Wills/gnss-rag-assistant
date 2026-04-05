"""
Configuration for the GNSS RAG Technical Knowledge Assistant.
Central place for all tunable parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent
PDF_FOLDER = BASE_DIR / "GNSS PDF"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# --- ChromaDB ---
COLLECTION_NAME = "gnss_documents"  # base name; strategy suffix added at runtime

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dims, ~80MB, local/free

# --- Chunking Parameters ---
CHUNKING_STRATEGY = "fixed"  # "fixed", "sentence", or "semantic"
CHUNK_SIZE = 512       # tokens per chunk (fixed strategy)
CHUNK_OVERLAP = 64     # token overlap between consecutive chunks (fixed strategy)
TIKTOKEN_ENCODING = "cl100k_base"  # tokenizer for counting
SENTENCE_CHUNK_TARGET = 5  # sentences per chunk (sentence strategy)
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # cosine similarity breakpoint (semantic strategy)


# Chunk-size variants for Session 8 slide 27-28 comparison
CHUNK_SIZE_VARIANTS = [200, 512, 1000]


def get_collection_name(strategy: str = None, chunk_size: int = None) -> str:
    """Return the ChromaDB collection name for a given chunking strategy.

    For fixed strategy with non-default chunk sizes (200, 1000), appends
    the size to create separate collections for the slide 27-28 comparison.
    """
    s = strategy or CHUNKING_STRATEGY
    if s == "fixed":
        if chunk_size is not None and chunk_size != CHUNK_SIZE:
            return f"{COLLECTION_NAME}_fixed_{chunk_size}"
        return COLLECTION_NAME  # backward compatible with existing DB
    return f"{COLLECTION_NAME}_{s}"

# --- Retrieval ---
TOP_K = 8  # number of chunks to retrieve per query

# --- LLM (Groq - free tier, OpenAI-compatible API) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_GENERATION_TOKENS = 1500

# --- Document Short Names (for display) ---
DOC_SHORT_NAMES = {
    "1.GPS Standard Positioning Service Performance Standard (5th ed., 2020).pdf":
        "GPS SPS Performance Standard",
    "2.3D Mapping\u2013Aided GNSS (Paul D. Groves, Inside GNSS).pdf":
        "3D Mapping-Aided GNSS (Groves)",
    "3.2021_Federal_Rdionavigation_Plan.pdf":
        "Federal Radionavigation Plan 2021",
    "4.ZED-F9P_IntegrationManual_UBX-18010802.pdf":
        "ZED-F9P Integration Manual",
    "5.Urban positioning 3D mapping-aided GNSS.pdf":
        "Urban Positioning 3DMA GNSS",
}

# --- Sample Questions ---
SAMPLE_QUESTIONS = [
    "We are deploying delivery robots in a dense city center with tall glass buildings, "
    "and our standard GNSS is experiencing severe positioning errors due to signal reflections. "
    "Based on the sources, what specific signal bands or 3D mapping-aided algorithms should we "
    "implement to mitigate these non-line-of-sight (NLOS) and multipath errors?",

    "Our stationary power grid timing receivers are at risk of intentional signal spoofing. "
    "What specific security features, authentication protocols, or configuration settings can "
    "we enable on our receiver to detect and mitigate these threats?",

    "We are deploying a fleet of autonomous maritime surface vessels that will operate for weeks "
    "without communication to ground control. What are the primary reliability concerns regarding "
    "the degradation of the broadcast navigation message, and what operational mode defines this state?",

    "We are installing fixed GNSS monitors to measure slight geological shifts, but the position "
    "output exhibits constant low-speed wander due to environmental noise. What specific receiver "
    "platform models or position publication settings should we configure to reduce this noise?",

    "Our national transportation infrastructure is heavily dependent on GPS, making it vulnerable "
    "to intentional jamming and spoofing attacks. What are the recommended backup PNT technologies "
    "or network resilience strategies to ensure continuous operations during a prolonged GPS denial event?",

    "We need to set up a temporary RTK base station in the field for a drone survey, but we do not "
    "know our exact absolute coordinates in advance. What receiver configuration mode should we "
    "initialize to establish our position before broadcasting RTCM corrections to the rover?",
]

# --- System Prompts ---
RAG_SYSTEM_PROMPT = """You are a GNSS technical knowledge assistant. Answer questions using ONLY \
the provided source documents. Follow these rules strictly:

1. Base your answer exclusively on the retrieved document excerpts below.
2. Cite your sources using [Source: Document Name, pp. X-Y] after each claim.
3. If the retrieved excerpts do not contain enough information to fully answer \
the question, explicitly state what is missing and do not fabricate details.
4. Use precise technical language appropriate for GNSS engineering.
5. Structure your answer with clear sections if the question has multiple parts.
6. Be thorough but concise — prioritize actionable engineering information."""

PLAIN_LLM_SYSTEM_PROMPT = """You are a GNSS technical assistant. Answer the following question \
based on your training knowledge. Be honest about uncertainty. If you are not sure \
about specific details, say so rather than guessing."""
