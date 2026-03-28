"""
GNSS Technical Knowledge Assistant
Streamlit RAG Demo — Retrieval-Augmented Generation Pipeline

Usage:
    streamlit run app.py
"""

import base64
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import (
    SAMPLE_QUESTIONS, EMBEDDING_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, GROQ_API_KEY,
    CHUNKING_STRATEGY, SENTENCE_CHUNK_TARGET,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from rag_pipeline import query_pipeline, generate_without_rag, get_db_stats

# ── Paths ──
BASE_DIR = Path(__file__).parent

# ── Logo loader ──
def _load_logo_b64(filename: str) -> str:
    path = BASE_DIR / filename
    if not path.exists():
        return ""
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


LOGO_BUAA = _load_logo_b64("university logo.png")
LOGO_RCSSTEAP = _load_logo_b64("RCSSTEAP.jpg")

# ── Page Config ──
st.set_page_config(
    page_title="GNSS RAG Assistant",
    page_icon="🛰️",
    layout="wide",
)




# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Navy + Cyan theme (matching Assignment 2)
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global dark theme ── */
    .stApp {
        background-color: #0D1B2A;
    }

    /* ── Animations ── */
    @keyframes breathe {
        0%, 100% { opacity: 0.75; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.03); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 4px rgba(0,188,212,0.15); }
        50% { box-shadow: 0 0 14px rgba(0,188,212,0.45); }
    }
    @keyframes textShimmer {
        0%, 100% { opacity: 0.85; text-shadow: 0 0 0 transparent; }
        50% { opacity: 1; text-shadow: 0 0 8px rgba(0,188,212,0.3); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Landing page ── */
    .landing-container {
        background: linear-gradient(180deg, #0D1B2A 0%, #142A3E 50%, #0D1B2A 100%);
        border: 1px solid #1E3A5F;
        border-radius: 16px;
        padding: 35px 40px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .landing-logos {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .landing-logos img {
        height: 72px;
        transition: transform 0.3s ease;
    }
    .landing-logos img:hover { transform: scale(1.08); }
    .landing-uni {
        font-size: 14px;
        color: #E0E7EE;
        margin: 0;
    }
    .landing-sub {
        font-size: 13px;
        color: #A0AEBB;
        margin: 2px 0;
    }
    .landing-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00BCD4, transparent);
        margin: 18px auto;
        max-width: 400px;
    }
    .landing-course {
        font-size: 13px;
        color: #A0AEBB;
        margin: 8px 0 4px 0;
    }
    .landing-title {
        font-size: 24px;
        font-weight: 700;
        color: #E0E7EE;
        margin: 8px 0 4px 0;
    }
    .landing-subtitle {
        font-size: 16px;
        font-weight: 600;
        color: #00BCD4;
        margin: 4px 0 8px 0;
    }
    .landing-group {
        font-size: 13px;
        color: #A0AEBB;
        margin: 4px 0 12px 0;
    }

    /* Member table */
    table.member-table {
        margin: 10px auto;
        border-collapse: collapse;
        max-width: 520px;
    }
    table.member-table th {
        background: #0A2E50;
        color: #00BCD4;
        padding: 8px 14px;
        border: 1px solid #1E3A5F;
        font-size: 13px;
    }
    table.member-table td {
        background: #142A3E;
        color: #E0E7EE;
        padding: 8px 14px;
        border: 1px solid #1E3A5F;
        font-size: 13px;
        text-align: center;
    }
    table.member-table tr:hover td {
        background: #1A3A5A;
    }

    /* About card */
    .brief-card {
        background: linear-gradient(135deg, #0A2E50, #142A3E);
        border: 1px solid #00BCD4;
        border-radius: 10px;
        padding: 18px 24px;
        max-width: 750px;
        margin: 16px auto 12px auto;
        text-align: left;
        transition: all 0.35s ease;
    }
    .brief-card:hover {
        box-shadow: 0 6px 20px rgba(0,188,212,0.2);
        transform: translateY(-2px);
    }
    .brief-card h4 {
        color: #00BCD4;
        margin: 0 0 8px 0;
        font-size: 15px;
    }
    .brief-card p {
        color: #A0AEBB;
        font-size: 13px;
        line-height: 1.6;
        margin: 0;
    }

    /* Guide text */
    .guide-text {
        text-align: center;
        font-size: 14px;
        font-weight: 600;
        color: #00BCD4;
        background: linear-gradient(135deg, #0A2E50, #142A3E);
        border: 1px solid #00BCD4;
        border-radius: 10px;
        padding: 14px 24px;
        max-width: 750px;
        margin: 0 auto 1rem auto;
        animation: glowPulse 3s infinite;
    }
    .guide-text .hl { color: #FFB74D; }

    /* ── Card system (4 variants) ── */
    .info-card {
        background: linear-gradient(135deg, #142A3E, #1A354C);
        border-left: 4px solid #00BCD4;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.35s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,188,212,0.25);
    }
    .success-card {
        background: linear-gradient(135deg, #0A2A15, #0A3E20);
        border-left: 4px solid #66BB6A;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.35s ease;
    }
    .success-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102,187,106,0.25);
    }
    .warn-card {
        background: linear-gradient(135deg, #2A1A0A, #3E250A);
        border-left: 4px solid #FFB74D;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.35s ease;
    }
    .warn-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255,183,77,0.25);
    }
    .card-title {
        color: #E0E7EE;
        font-size: 15px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .card-text {
        color: #A0AEBB;
        font-size: 13px;
        line-height: 1.5;
        margin: 0;
    }

    /* ── Pipeline diagram (full-width, dark) ── */
    .pipeline-wrapper {
        background: linear-gradient(135deg, #0A2E50 0%, #142A3E 100%);
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 1.5rem 1rem 1.2rem 1rem;
        margin-bottom: 1rem;
    }
    .pipeline-title {
        text-align: center;
        font-size: 0.85rem;
        font-weight: 700;
        color: #00BCD4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1.2rem;
        animation: textShimmer 3s infinite;
    }
    .pipeline-flow {
        display: flex;
        align-items: stretch;
        justify-content: center;
        gap: 0;
        flex-wrap: nowrap;
    }
    .pipeline-stage {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        max-width: 140px;
        position: relative;
    }
    .pipeline-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
        border: 2px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .pipeline-stage:hover .pipeline-icon {
        transform: scale(1.12);
        box-shadow: 0 4px 16px rgba(0,188,212,0.35);
    }
    .pipeline-icon.ingest { background: #0A2E50; border-color: #29B6F6; }
    .pipeline-icon.process { background: #2A1A0A; border-color: #FFB74D; }
    .pipeline-icon.store { background: #0A2A15; border-color: #66BB6A; }
    .pipeline-icon.retrieve { background: #1A0A2A; border-color: #CE93D8; }
    .pipeline-icon.generate { background: #2A0A1A; border-color: #EF5350; }
    .pipeline-icon.output { background: #0A2A15; border-color: #00BCD4; }
    .pipeline-stage-name {
        font-size: 0.8rem;
        font-weight: 700;
        color: #E0E7EE;
        margin-bottom: 2px;
        text-align: center;
    }
    .pipeline-stage-detail {
        font-size: 0.65rem;
        color: #6B7B8D;
        text-align: center;
        line-height: 1.3;
    }
    .pipeline-connector {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 2px;
        margin-top: -12px;
    }
    .pipeline-connector svg.flow-arrow {
        width: 48px;
        height: 24px;
    }
    .pipeline-phase-label {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px dashed #1E3A5F;
    }
    .phase-tag {
        font-size: 0.65rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 10px;
        letter-spacing: 0.5px;
    }
    .phase-tag.offline { background: #0A2E50; color: #29B6F6; border: 1px solid #29B6F6; }
    .phase-tag.online { background: #0A2A15; color: #66BB6A; border: 1px solid #66BB6A; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #101E30 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #E0E7EE !important;
    }
    section[data-testid="stSidebar"] img {
        display: block;
        margin: 0 auto 8px auto;
        animation: breathe 3.5s ease-in-out infinite;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E3A5F !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        background: #142A3E !important;
        color: #E0E7EE !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stTabs"] button:hover {
        background: #1A3A5A !important;
        border-color: #00BCD4 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,188,212,0.25);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        background: #0A2E50 !important;
        border-bottom: 2px solid #00BCD4 !important;
        color: #00BCD4 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #142A3E;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(20,42,62,0.8);
        border-color: #00BCD4;
        box-shadow: 0 4px 12px rgba(0,188,212,0.15);
    }
    [data-testid="stMetric"] label {
        color: #A0AEBB !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #E0E7EE !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #142A3E;
        border: 1px solid #1E3A5F !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: #00BCD4 !important;
        box-shadow: 0 4px 15px rgba(0,188,212,0.15);
    }
    [data-testid="stExpander"] summary span {
        color: #E0E7EE !important;
    }

    /* ── Text elements ── */
    .stMarkdown, .stMarkdown p, .stText {
        color: #E0E7EE !important;
    }
    h1, h2, h3, .stSubheader {
        color: #E0E7EE !important;
    }

    /* ── Selectbox / Text inputs ── */
    [data-testid="stSelectbox"] label,
    [data-testid="stTextArea"] label {
        color: #A0AEBB !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E3A5F;
        border-radius: 8px;
    }

    /* ── Chunk bar (sidebar) ── */
    .chunk-bar-container { margin: 2px 0; }
    .chunk-bar-label { font-size: 0.7rem; color: #A0AEBB; margin-bottom: 1px; }
    .chunk-bar {
        height: 14px;
        border-radius: 3px;
        background: linear-gradient(90deg, #00BCD4, #29B6F6);
    }
    .chunk-bar-count { font-size: 0.65rem; color: #6B7B8D; }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        color: #6B7B8D;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #1E3A5F;
        margin-top: 2rem;
    }

    /* ── Relevance bar (dark) ── */
    .relevance-track {
        background: #1E3A5F;
        border-radius: 4px;
        height: 8px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Check prerequisites ──
if not GROQ_API_KEY:
    st.error(
        "**GROQ_API_KEY not found.** "
        "Create a `.env` file in the project directory with:\n\n"
        "```\nGROQ_API_KEY=gsk_your-key-here\n```"
    )
    st.stop()


# ── Cache heavy resources ──
@st.cache_data(show_spinner=False)
def load_db_stats():
    return get_db_stats()


# ── Load stats ──
try:
    db_stats = load_db_stats()
except Exception as e:
    st.error(
        f"**Vector database not found.** Run the ingestion script first:\n\n"
        f"```\npython ingest.py\n```\n\nError: {e}"
    )
    st.stop()


# ── Helpers ──
def _relevance_color(score: float) -> str:
    if score >= 0.8:
        return "#66BB6A"
    elif score >= 0.5:
        return "#FFB74D"
    return "#EF5350"


def _strategy_description() -> str:
    strategy = db_stats.get("strategy", CHUNKING_STRATEGY)
    if strategy == "fixed":
        return f"Fixed-size token windows ({CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap)"
    elif strategy == "sentence":
        return f"Sentence-based grouping ({SENTENCE_CHUNK_TARGET} sentences/chunk)"
    elif strategy == "semantic":
        return f"Semantic similarity breakpoints (threshold: {SEMANTIC_SIMILARITY_THRESHOLD})"
    return strategy


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Animated GNSS constellation scene
    _sidebar_icon = """
    <html><head><style>
    *{margin:0;padding:0;box-sizing:border-box;}
    body{background:transparent;display:flex;flex-direction:column;align-items:center;
        justify-content:center;padding:6px 0 2px 0;}
    .scene{position:relative;width:240px;height:180px;overflow:hidden;}

    /* Starfield */
    .star{position:absolute;width:2px;height:2px;background:#E0E7EE;border-radius:50%;
        animation:twinkle 3s ease-in-out infinite;}
    .star:nth-child(1){top:12%;left:8%;animation-delay:0s;}
    .star:nth-child(2){top:6%;left:28%;animation-delay:0.7s;width:1.5px;height:1.5px;}
    .star:nth-child(3){top:18%;left:52%;animation-delay:1.4s;}
    .star:nth-child(4){top:8%;left:72%;animation-delay:0.3s;width:1.5px;height:1.5px;}
    .star:nth-child(5){top:22%;left:90%;animation-delay:2.1s;}
    .star:nth-child(6){top:30%;left:15%;animation-delay:1.8s;width:1px;height:1px;}
    .star:nth-child(7){top:5%;left:44%;animation-delay:0.5s;width:1px;height:1px;}
    .star:nth-child(8){top:15%;left:85%;animation-delay:1.1s;}
    .star:nth-child(9){top:28%;left:62%;animation-delay:2.5s;width:1px;height:1px;}
    .star:nth-child(10){top:10%;left:95%;animation-delay:0.9s;width:1.5px;height:1.5px;}
    @keyframes twinkle{0%,100%{opacity:0.3;}50%{opacity:1;}}

    /* Earth arc at bottom */
    .earth{position:absolute;bottom:-90px;left:50%;transform:translateX(-50%);
        width:260px;height:130px;border-radius:50%;
        background:linear-gradient(180deg,#0A3D2A 0%,#0A2E50 40%,#142A3E 100%);
        border-top:2px solid rgba(0,188,212,0.3);
        box-shadow:0 -8px 30px rgba(0,188,212,0.1);}
    /* Atmosphere glow */
    .earth::before{content:'';position:absolute;top:-6px;left:10%;right:10%;height:12px;
        background:linear-gradient(90deg,transparent,rgba(0,188,212,0.15),rgba(41,182,246,0.1),transparent);
        border-radius:50%;filter:blur(4px);}

    /* Ground station on earth */
    .ground-station{position:absolute;bottom:32px;left:50%;transform:translateX(-50%);z-index:5;}
    .gs-dish{width:16px;height:10px;border:2px solid #FFB74D;border-bottom:none;
        border-radius:50% 50% 0 0;margin:0 auto;}
    .gs-pole{width:2px;height:8px;background:#FFB74D;margin:0 auto;}
    .gs-base{width:12px;height:3px;background:#FFB74D;border-radius:1px;margin:0 auto;}
    /* Ground station signal */
    .gs-signal{position:absolute;top:-8px;left:50%;transform:translateX(-50%);
        width:6px;height:6px;border:1.5px solid #FFB74D;border-radius:50%;
        opacity:0;animation:gsBeam 2.5s ease-out infinite;}
    .gs-signal:nth-child(2){width:14px;height:14px;top:-12px;animation-delay:0.5s;}
    .gs-signal:nth-child(3){width:22px;height:22px;top:-16px;animation-delay:1s;}
    @keyframes gsBeam{0%{opacity:0.8;transform:translateX(-50%) scale(0.5);}
        100%{opacity:0;transform:translateX(-50%) scale(1.3);}}

    /* Satellite orbits (elliptical paths) */
    .orbit-path{position:absolute;border:1px dashed rgba(0,188,212,0.12);border-radius:50%;
        top:50%;left:50%;transform:translate(-50%,-50%);}
    .orbit-path.o1{width:140px;height:80px;transform:translate(-50%,-50%) rotate(-15deg);}
    .orbit-path.o2{width:190px;height:100px;transform:translate(-50%,-50%) rotate(10deg);}
    .orbit-path.o3{width:110px;height:65px;transform:translate(-50%,-50%) rotate(-35deg);}

    /* Satellites */
    .sat{position:absolute;z-index:3;}
    .sat-body{width:14px;height:14px;display:flex;align-items:center;justify-content:center;
        font-size:11px;filter:drop-shadow(0 0 4px rgba(0,188,212,0.6));}

    /* Satellite 1 — main orbit */
    .sat.s1{top:50%;left:50%;animation:orbit1 8s linear infinite;}
    @keyframes orbit1{
        0%{transform:translate(-70px,-40px);}
        25%{transform:translate(0px,-48px);}
        50%{transform:translate(70px,-40px);}
        75%{transform:translate(0px,0px);}
        100%{transform:translate(-70px,-40px);}
    }
    /* Satellite 2 — wider orbit, opposite direction */
    .sat.s2{top:50%;left:50%;animation:orbit2 11s linear infinite;}
    @keyframes orbit2{
        0%{transform:translate(90px,-30px);}
        25%{transform:translate(0px,-55px);}
        50%{transform:translate(-90px,-30px);}
        75%{transform:translate(0px,10px);}
        100%{transform:translate(90px,-30px);}
    }
    /* Satellite 3 — tighter orbit */
    .sat.s3{top:50%;left:50%;animation:orbit3 6.5s linear infinite;}
    @keyframes orbit3{
        0%{transform:translate(-50px,-20px);}
        25%{transform:translate(10px,-38px);}
        50%{transform:translate(55px,-25px);}
        75%{transform:translate(5px,5px);}
        100%{transform:translate(-50px,-20px);}
    }

    /* Signal beams from satellites to ground */
    .signal-beam{position:absolute;bottom:38px;left:50%;width:1px;
        transform:translateX(-50%);z-index:2;opacity:0;}
    .signal-beam.b1{height:60px;background:linear-gradient(180deg,rgba(0,188,212,0.5),transparent);
        animation:beam1 8s linear infinite;transform-origin:bottom center;}
    .signal-beam.b2{height:70px;background:linear-gradient(180deg,rgba(41,182,246,0.4),transparent);
        animation:beam2 11s linear infinite;}
    .signal-beam.b3{height:50px;background:linear-gradient(180deg,rgba(206,147,216,0.4),transparent);
        animation:beam3 6.5s linear infinite;}
    @keyframes beam1{0%,20%{opacity:0;}25%,45%{opacity:0.6;}50%,100%{opacity:0;}}
    @keyframes beam2{0%,40%{opacity:0;}45%,65%{opacity:0.5;}70%,100%{opacity:0;}}
    @keyframes beam3{0%,60%{opacity:0;}65%,85%{opacity:0.5;}90%,100%{opacity:0;}}

    /* Center radar pulse (from earth) */
    .radar{position:absolute;bottom:36px;left:50%;transform:translateX(-50%);z-index:1;}
    .radar-ring{position:absolute;bottom:0;left:50%;transform:translateX(-50%);
        border:1.5px solid rgba(0,188,212,0.3);border-radius:50%;
        opacity:0;animation:radarPulse 4s ease-out infinite;}
    .radar-ring:nth-child(1){width:30px;height:30px;animation-delay:0s;}
    .radar-ring:nth-child(2){width:55px;height:55px;animation-delay:1s;}
    .radar-ring:nth-child(3){width:80px;height:80px;animation-delay:2s;}
    @keyframes radarPulse{0%{opacity:0.6;transform:translateX(-50%) scale(0.4);}
        100%{opacity:0;transform:translateX(-50%) scale(1.2);}}

    /* Data stream particles */
    .particle{position:absolute;width:3px;height:3px;border-radius:50%;opacity:0;}
    .particle.p1{background:#00BCD4;left:35%;animation:dataFlow 3s ease-in infinite;}
    .particle.p2{background:#29B6F6;left:55%;animation:dataFlow 3s ease-in 1s infinite;}
    .particle.p3{background:#66BB6A;left:65%;animation:dataFlow 3s ease-in 2s infinite;}
    .particle.p4{background:#CE93D8;left:42%;animation:dataFlow 3s ease-in 0.5s infinite;}
    @keyframes dataFlow{
        0%{top:20%;opacity:0;}
        20%{opacity:0.8;}
        80%{opacity:0.6;}
        100%{top:72%;opacity:0;}
    }

    /* Title styling */
    .icon-title{color:#00BCD4;font-family:Calibri,"Source Sans Pro",sans-serif;
        font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
        text-align:center;margin-top:4px;
        text-shadow:0 0 8px rgba(0,188,212,0.3);}
    .icon-sub{color:#6B7B8D;font-family:Calibri,"Source Sans Pro",sans-serif;
        font-size:9px;letter-spacing:1px;text-transform:uppercase;text-align:center;margin-top:2px;}
    </style></head><body>
    <div class="scene">
        <!-- Stars -->
        <div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div><div class="star"></div><div class="star"></div>
        <div class="star"></div>

        <!-- Orbit paths -->
        <div class="orbit-path o1"></div>
        <div class="orbit-path o2"></div>
        <div class="orbit-path o3"></div>

        <!-- Data particles -->
        <div class="particle p1"></div>
        <div class="particle p2"></div>
        <div class="particle p3"></div>
        <div class="particle p4"></div>

        <!-- Signal beams -->
        <div class="signal-beam b1"></div>
        <div class="signal-beam b2"></div>
        <div class="signal-beam b3"></div>

        <!-- Satellites -->
        <div class="sat s1"><div class="sat-body">🛰️</div></div>
        <div class="sat s2"><div class="sat-body">🛰️</div></div>
        <div class="sat s3"><div class="sat-body">📡</div></div>

        <!-- Radar pulses from ground -->
        <div class="radar">
            <div class="radar-ring"></div>
            <div class="radar-ring"></div>
            <div class="radar-ring"></div>
        </div>

        <!-- Earth surface -->
        <div class="earth"></div>

        <!-- Ground station -->
        <div class="ground-station">
            <div class="gs-signal"></div>
            <div class="gs-signal"></div>
            <div class="gs-signal"></div>
            <div class="gs-dish"></div>
            <div class="gs-pole"></div>
            <div class="gs-base"></div>
        </div>
    </div>
    <div class="icon-title">GNSS RAG Assistant</div>
    <div class="icon-sub">Satellite Navigation Intelligence</div>
    </body></html>
    """
    components.html(_sidebar_icon, height=220)
    st.divider()

    st.markdown("### Pipeline Configuration")

    # Knowledge base stats
    st.markdown("**Knowledge Base**")
    col1, col2 = st.columns(2)
    col1.metric("Documents", db_stats["total_documents"])
    col2.metric("Chunks", db_stats["total_chunks"])

    # Chunk distribution
    doc_counts = db_stats.get("doc_chunk_counts", {})
    if doc_counts:
        max_count = max(doc_counts.values())
        st.markdown("**Chunks per Document:**")
        for doc_name in sorted(doc_counts.keys()):
            count = doc_counts[doc_name]
            pct = count / max_count * 100
            short_name = doc_name[:25] + "..." if len(doc_name) > 25 else doc_name
            st.markdown(f"""
            <div class="chunk-bar-container">
                <div class="chunk-bar-label">{short_name}</div>
                <div class="chunk-bar" style="width: {pct}%;"></div>
                <div class="chunk-bar-count">{count} chunks</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Indexed Documents"):
        for doc in db_stats["document_names"]:
            st.markdown(f"- {doc}")

    st.divider()

    # Chunking strategy
    st.markdown("**Chunking Strategy**")
    st.markdown(f"Active: {_strategy_description()}")
    st.markdown("Tokenizer: cl100k_base")

    st.divider()

    # Models
    st.markdown("**Models**")
    st.markdown(f"Embedding: `{EMBEDDING_MODEL}` (384d, local)")
    st.markdown(f"LLM: `{LLM_MODEL}` (Groq)")
    st.markdown(f"Retrieval: Top-{TOP_K} cosine")

    st.divider()

    compare_mode = st.toggle(
        "Compare: RAG vs. Plain LLM",
        value=False,
        help="Show a side-by-side answer from the LLM without retrieval.",
    )

    st.divider()
    st.caption(
        "Assignment 3 — AI & Large Models  \n"
        "Sessions 6-8: Transformers, Alignment & RAG"
    )


# ══════════════════════════════════════════════════════════════════════
# LANDING PAGE (rendered via components.html to handle large base64)
# ══════════════════════════════════════════════════════════════════════
_landing_html = f"""
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: transparent;
        font-family: Calibri, "Source Sans Pro", sans-serif;
        color: #E0E7EE;
    }}
    .landing-container {{
        background: linear-gradient(180deg, #0D1B2A 0%, #142A3E 50%, #0D1B2A 100%);
        border: 1px solid #1E3A5F;
        border-radius: 16px;
        padding: 35px 40px;
        text-align: center;
    }}
    .landing-logos {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }}
    .landing-logos img {{
        height: 72px;
        transition: transform 0.3s ease;
    }}
    .landing-logos img:hover {{ transform: scale(1.08); }}
    .landing-uni {{ font-size: 14px; color: #E0E7EE; margin: 0; }}
    .landing-sub {{ font-size: 13px; color: #A0AEBB; margin: 2px 0; }}
    .landing-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, #00BCD4, transparent);
        margin: 18px auto;
        max-width: 400px;
    }}
    .landing-course {{ font-size: 13px; color: #A0AEBB; margin: 8px 0 4px 0; }}
    .landing-course strong {{ color: #E0E7EE; }}
    .landing-title {{ font-size: 24px; font-weight: 700; color: #E0E7EE; margin: 8px 0 4px 0; }}
    .landing-subtitle {{ font-size: 16px; font-weight: 600; color: #00BCD4; margin: 4px 0 8px 0; }}
    .landing-group {{ font-size: 13px; color: #A0AEBB; margin: 4px 0 12px 0; }}

    table.member-table {{
        margin: 10px auto;
        border-collapse: collapse;
        max-width: 520px;
    }}
    table.member-table th {{
        background: #0A2E50;
        color: #00BCD4;
        padding: 8px 14px;
        border: 1px solid #1E3A5F;
        font-size: 13px;
    }}
    table.member-table td {{
        background: #142A3E;
        color: #E0E7EE;
        padding: 8px 14px;
        border: 1px solid #1E3A5F;
        font-size: 13px;
        text-align: center;
        transition: background 0.3s ease;
    }}
    table.member-table tr:hover td {{ background: #1A3A5A; }}

    .brief-card {{
        background: linear-gradient(135deg, #0A2E50, #142A3E);
        border: 1px solid #00BCD4;
        border-radius: 10px;
        padding: 18px 24px;
        max-width: 750px;
        margin: 20px auto 16px auto;
        text-align: left;
        transition: all 0.35s ease;
    }}
    .brief-card:hover {{
        box-shadow: 0 6px 20px rgba(0,188,212,0.2);
        transform: translateY(-2px);
    }}
    .brief-card h4 {{ color: #00BCD4; margin: 0 0 8px 0; font-size: 15px; }}
    .brief-card p {{ color: #A0AEBB; font-size: 13px; line-height: 1.6; margin: 0; }}

    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 4px rgba(0,188,212,0.15); }}
        50% {{ box-shadow: 0 0 14px rgba(0,188,212,0.45); }}
    }}
    .guide-text {{
        text-align: center;
        font-size: 14px;
        font-weight: 600;
        color: #00BCD4;
        background: linear-gradient(135deg, #0A2E50, #142A3E);
        border: 1px solid #00BCD4;
        border-radius: 10px;
        padding: 14px 24px;
        max-width: 750px;
        margin: 0 auto;
        animation: glowPulse 3s infinite;
    }}
    .guide-text .hl {{ color: #FFB74D; }}
</style>
</head>
<body>
<div class="landing-container">
    <div class="landing-logos">
        <img src="{LOGO_BUAA}" alt="Beihang University">
        <img src="{LOGO_RCSSTEAP}" alt="RCSSTEAP" style="border-radius:6px;">
    </div>
    <p class="landing-uni">Beihang University (BUAA)</p>
    <p class="landing-sub">Regional Centre for Space Science and Technology Education<br>
    in Asia and the Pacific (China) &mdash; RCSSTEAP</p>

    <div class="landing-divider"></div>

    <p class="landing-course">Course: <strong>Artificial Intelligence and Advanced Large Models</strong>
    &nbsp;|&nbsp; Spring 2025</p>
    <p class="landing-title">Assignment 3 &mdash; The RAG Concept Demo</p>
    <p class="landing-subtitle">GNSS Technical Knowledge Assistant using Retrieval-Augmented Generation</p>
    <p class="landing-group">Group 14</p>

    <table class="member-table">
        <tr><th>Name</th><th>Admission Number</th></tr>
        <tr><td>Granny Tlou Molokomme</td><td>LS2525256</td></tr>
        <tr><td>Letsoalo Maile</td><td>LS2525231</td></tr>
        <tr><td>Lemalasia Tevin Muchera</td><td>LS2525229</td></tr>
    </table>

    <div class="brief-card" style="text-align:center;">
        <h4>About This Dashboard</h4>
        <p>This interactive demo implements a complete
        <strong style="color:#00BCD4;">Retrieval-Augmented Generation (RAG)</strong>
        pipeline for GNSS technical knowledge. It ingests 5 engineering documents (~598 pages),
        chunks them using configurable strategies (fixed-size, sentence-based, or semantic),
        embeds them into a vector database, and retrieves relevant context to ground LLM answers
        with precise source citations &mdash; eliminating hallucination through
        document-grounded generation.</p>
    </div>
</div>

<p class="guide-text">
    Ask a <span class="hl">GNSS question</span> in the Q&amp;A tab to see retrieval-augmented answers with source citations,
    or run the <span class="hl">Evaluation Dashboard</span> to benchmark pipeline performance across all test scenarios.
</p>
</body>
</html>
"""
components.html(_landing_html, height=780, scrolling=False)


# ══════════════════════════════════════════════════════════════════════
# PIPELINE ARCHITECTURE DIAGRAM
# ══════════════════════════════════════════════════════════════════════
arrow_svg = """<svg viewBox="0 0 50 24" fill="none" class="flow-arrow">
<defs>
  <linearGradient id="ag" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#0A2E50"/>
    <stop offset="40%" stop-color="#00BCD4"/>
    <stop offset="100%" stop-color="#00E5FF"/>
  </linearGradient>
  <filter id="glow1"><feGaussianBlur stdDeviation="2.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="glow2"><feGaussianBlur stdDeviation="3.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<!-- Broad outer glow line -->
<path d="M2 12h36" stroke="#00BCD4" stroke-width="6" stroke-linecap="round" opacity="0.15" filter="url(#glow2)">
  <animate attributeName="opacity" values="0.08;0.25;0.08" dur="2s" repeatCount="indefinite"/>
</path>
<!-- Main flowing line -->
<path d="M2 12h36" stroke="url(#ag)" stroke-width="3" stroke-linecap="round" filter="url(#glow1)">
  <animate attributeName="stroke-dasharray" values="0,50;50,0" dur="1.8s" repeatCount="indefinite"/>
</path>
<!-- Arrowhead outer glow -->
<path d="M34 4l12 8-12 8" stroke="#00BCD4" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.2" filter="url(#glow2)">
  <animate attributeName="opacity" values="0.1;0.35;0.1" dur="2s" repeatCount="indefinite"/>
</path>
<!-- Arrowhead -->
<path d="M34 4l12 8-12 8" stroke="#00E5FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow1)">
  <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>
</path>
<!-- Traveling dot -->
<circle r="2.5" fill="#00E5FF" filter="url(#glow1)">
  <animateMotion dur="1.8s" repeatCount="indefinite" path="M2,12 L38,12"/>
  <animate attributeName="opacity" values="0;1;1;0" dur="1.8s" repeatCount="indefinite"/>
</circle>
</svg>"""

with st.expander("RAG Pipeline Architecture", expanded=True):
    st.markdown(f"""
    <div class="pipeline-wrapper">
        <div class="pipeline-title">Retrieval-Augmented Generation Pipeline</div>
        <div class="pipeline-flow">
            <div class="pipeline-stage">
                <div class="pipeline-icon ingest">📄</div>
                <div class="pipeline-stage-name">PDF Documents</div>
                <div class="pipeline-stage-detail">5 GNSS sources<br>598 pages</div>
            </div>
            <div class="pipeline-connector">{arrow_svg}</div>
            <div class="pipeline-stage">
                <div class="pipeline-icon process">✂️</div>
                <div class="pipeline-stage-name">Chunking</div>
                <div class="pipeline-stage-detail">{_strategy_description()}</div>
            </div>
            <div class="pipeline-connector">{arrow_svg}</div>
            <div class="pipeline-stage">
                <div class="pipeline-icon process">🔢</div>
                <div class="pipeline-stage-name">Embeddings</div>
                <div class="pipeline-stage-detail">all-MiniLM-L6-v2<br>384 dimensions</div>
            </div>
            <div class="pipeline-connector">{arrow_svg}</div>
            <div class="pipeline-stage">
                <div class="pipeline-icon store">🗄️</div>
                <div class="pipeline-stage-name">Vector Store</div>
                <div class="pipeline-stage-detail">ChromaDB<br>{db_stats['total_chunks']} chunks</div>
            </div>
            <div class="pipeline-connector">{arrow_svg}</div>
            <div class="pipeline-stage">
                <div class="pipeline-icon retrieve">🔍</div>
                <div class="pipeline-stage-name">Retrieval</div>
                <div class="pipeline-stage-detail">Top-{TOP_K} cosine<br>similarity search</div>
            </div>
            <div class="pipeline-connector">{arrow_svg}</div>
            <div class="pipeline-stage">
                <div class="pipeline-icon generate">🤖</div>
                <div class="pipeline-stage-name">LLM Generation</div>
                <div class="pipeline-stage-detail">Llama 3.3 70B<br>via Groq API</div>
            </div>
            <div class="pipeline-connector">{arrow_svg}</div>
            <div class="pipeline-stage">
                <div class="pipeline-icon output">✅</div>
                <div class="pipeline-stage-name">Grounded Answer</div>
                <div class="pipeline-stage-detail">With source<br>citations</div>
            </div>
        </div>
        <div class="pipeline-phase-label">
            <span class="phase-tag offline">OFFLINE &mdash; Ingestion &amp; Indexing</span>
            <span class="phase-tag online">ONLINE &mdash; Query &amp; Generation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab_qa, tab_eval = st.tabs(["📡 Ask a Question", "📊 Evaluation Dashboard"])


# ──────────────────────────────────────────────────────────────────────
# TAB 1: Ask a Question
# ──────────────────────────────────────────────────────────────────────
with tab_qa:
    question_options = ["Select a sample question..."] + [
        f"Q{i+1}: {q[:80]}..." for i, q in enumerate(SAMPLE_QUESTIONS)
    ] + ["Custom question"]

    selected = st.selectbox("Choose a question:", question_options)

    if selected == "Custom question":
        user_query = st.text_area(
            "Enter your GNSS question:",
            height=100,
            placeholder="e.g., How does shadow matching improve urban GNSS positioning?",
        )
    elif selected.startswith("Q"):
        idx = int(selected[1]) - 1
        user_query = SAMPLE_QUESTIONS[idx]
        st.markdown(f"""
        <div class="info-card">
            <p class="card-title">Selected Question</p>
            <p class="card-text">{user_query}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        user_query = ""

    ask_button = st.button("Ask", type="primary", disabled=not user_query)

    if ask_button and user_query:
        with st.spinner("Retrieving relevant documents and generating answer..."):
            try:
                result = query_pipeline(user_query)
            except Exception as e:
                st.error(f"Error during RAG pipeline: {e}")
                st.stop()

        # Timing metrics
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Retrieval Time", f"{result['retrieval_time_ms']} ms")
        col_t2.metric("Generation Time", f"{result['generation_time_ms']} ms")
        col_t3.metric("Total Time", f"{result['total_time_ms']} ms")

        # RAG answer
        st.markdown("""
        <div class="success-card">
            <p class="card-title">RAG-Grounded Answer</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result["answer"])

        # Retrieved sources
        st.markdown(f"""
        <div class="info-card">
            <p class="card-title">Retrieved Sources ({len(result['sources'])} chunks)</p>
        </div>
        """, unsafe_allow_html=True)

        for i, source in enumerate(result["sources"], 1):
            pages = (f"p. {source['page_start']}" if source["page_start"] == source["page_end"]
                     else f"pp. {source['page_start']}-{source['page_end']}")
            score_pct = f"{source['score'] * 100:.1f}%"
            color = _relevance_color(source["score"])

            with st.expander(
                f"[{score_pct}] {source['doc_name']} — {pages}",
                expanded=(i <= 3),
            ):
                bar_width = max(source["score"] * 100, 5)
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-weight:600;color:#A0AEBB;">Relevance:</span>
                    <span style="color:{color};font-weight:700;">{score_pct}</span>
                    <div class="relevance-track">
                        <div style="background:{color};width:{bar_width}%;height:8px;border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Document:** {source['doc_name']}")
                st.markdown(f"**Pages:** {pages}")
                st.text(source["text"][:1500])

        # Comparison mode
        if compare_mode:
            st.markdown("""
            <div class="warn-card">
                <p class="card-title">Plain LLM Answer (No Retrieval)</p>
                <p class="card-text">This answer uses only the model's training knowledge &mdash;
                no documents were retrieved. Compare with the RAG answer above
                to see how retrieval reduces hallucination and adds citations.</p>
            </div>
            """, unsafe_allow_html=True)
            with st.spinner("Generating plain LLM answer for comparison..."):
                try:
                    plain_answer = generate_without_rag(user_query)
                    st.markdown(plain_answer)
                except Exception as e:
                    st.error(f"Error generating comparison: {e}")


# ──────────────────────────────────────────────────────────────────────
# TAB 2: Evaluation Dashboard
# ──────────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("""
    <div class="info-card">
        <p class="card-title">RAG Pipeline Evaluation</p>
        <p class="card-text">Run all 6 sample questions through the RAG pipeline and see a performance
        scorecard. This demonstrates systematic evaluation of retrieval quality and generation grounding.</p>
    </div>
    """, unsafe_allow_html=True)

    run_eval = st.button("Run Evaluation", type="primary", key="eval_btn")

    if run_eval:
        progress_bar = st.progress(0, text="Starting evaluation...")
        results = []

        for i, question in enumerate(SAMPLE_QUESTIONS):
            progress_bar.progress(
                (i) / len(SAMPLE_QUESTIONS),
                text=f"Running Q{i+1}/{len(SAMPLE_QUESTIONS)}..."
            )

            try:
                result = query_pipeline(question)
                top_source = result["sources"][0] if result["sources"] else None
                has_citations = bool(re.search(r'\[Source:', result["answer"]))

                results.append({
                    "question_num": i + 1,
                    "question": question[:80] + "...",
                    "top_source_doc": top_source["doc_name"] if top_source else "N/A",
                    "top_score": top_source["score"] if top_source else 0,
                    "retrieval_ms": result["retrieval_time_ms"],
                    "generation_ms": result["generation_time_ms"],
                    "total_ms": result["total_time_ms"],
                    "sources_found": len(result["sources"]),
                    "has_citations": has_citations,
                    "all_sources": result["sources"],
                })
            except Exception as e:
                results.append({
                    "question_num": i + 1,
                    "question": question[:80] + "...",
                    "top_source_doc": "ERROR",
                    "top_score": 0,
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "total_ms": 0,
                    "sources_found": 0,
                    "has_citations": False,
                    "all_sources": [],
                })
                st.warning(f"Q{i+1} failed: {e}")

            if i < len(SAMPLE_QUESTIONS) - 1:
                time.sleep(3)

        progress_bar.progress(1.0, text="Evaluation complete!")

        # Scorecard
        st.markdown("""
        <div class="success-card">
            <p class="card-title">Scorecard</p>
        </div>
        """, unsafe_allow_html=True)

        df = pd.DataFrame([{
            "Q#": r["question_num"],
            "Question": r["question"],
            "Top Source": r["top_source_doc"],
            "Top Score": f"{r['top_score']*100:.1f}%",
            "Retrieval (ms)": r["retrieval_ms"],
            "Generation (ms)": r["generation_ms"],
            "Sources": r["sources_found"],
            "Citations": "Yes" if r["has_citations"] else "No",
        } for r in results])

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Aggregate metrics
        valid = [r for r in results if r["top_source_doc"] != "ERROR"]
        if valid:
            st.markdown("""
            <div class="info-card">
                <p class="card-title">Aggregate Metrics</p>
            </div>
            """, unsafe_allow_html=True)

            avg_retrieval = sum(r["retrieval_ms"] for r in valid) / len(valid)
            avg_generation = sum(r["generation_ms"] for r in valid) / len(valid)
            avg_top_score = sum(r["top_score"] for r in valid) / len(valid)
            citation_rate = sum(1 for r in valid if r["has_citations"]) / len(valid)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Avg Retrieval", f"{avg_retrieval:.0f} ms")
            mc2.metric("Avg Generation", f"{avg_generation:.0f} ms")
            mc3.metric("Avg Top Relevance", f"{avg_top_score*100:.1f}%")
            mc4.metric("Citation Rate", f"{citation_rate*100:.0f}%")

        # Document retrieval frequency
        st.markdown("""
        <div class="info-card">
            <p class="card-title">Document Retrieval Frequency</p>
            <p class="card-text">How often each document appears in the top-K results across all questions.</p>
        </div>
        """, unsafe_allow_html=True)

        doc_freq = {}
        for r in valid:
            for src in r["all_sources"]:
                name = src["doc_name"]
                doc_freq[name] = doc_freq.get(name, 0) + 1

        if doc_freq:
            freq_df = pd.DataFrame(
                sorted(doc_freq.items(), key=lambda x: -x[1]),
                columns=["Document", "Times Retrieved"]
            )
            st.bar_chart(freq_df.set_index("Document"))

        # Relevance score distribution
        st.markdown("""
        <div class="info-card">
            <p class="card-title">Relevance Score Distribution</p>
            <p class="card-text">Top relevance score per question.</p>
        </div>
        """, unsafe_allow_html=True)

        score_df = pd.DataFrame({
            "Question": [f"Q{r['question_num']}" for r in valid],
            "Top Relevance Score": [r["top_score"] for r in valid],
        })
        st.bar_chart(score_df.set_index("Question"))


# ── Footer ──
st.markdown("""
<div class="app-footer">
    Assignment 3 &mdash; AI &amp; Large Models | GNSS Technical Knowledge Assistant (RAG) |
    Group 14 &mdash; BUAA 2026
</div>
""", unsafe_allow_html=True)
