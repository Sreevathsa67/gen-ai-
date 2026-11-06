import os
 
import pandas as pd
import streamlit as st
 
from agent import answer_followup, reset_client, run_agent
from rag import MedicalRetriever
from tools import SAMPLE_REPORT, extract_text_from_pdf
 
st.set_page_config(page_title="Medical Report Explainer", page_icon="🩺", layout="wide")
 
STATUS_LABEL = {"LOW": "🔵 Low", "HIGH": "🔴 High", "NORMAL": "🟢 Normal", "NO_RANGE": "⚪ No range"}
 
 
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def load_retriever():
    return MedicalRetriever()


with st.sidebar:
    st.header("⚙️ Settings")
    key = st.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    if key and key != os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = key
        reset_client()
    st.caption("Without a key the agent still runs, using rule-based fallbacks.")
    st.divider()
    st.warning("**Educational tool only.** It does not diagnose or replace a doctor. "
               "Always discuss your results with a qualified medical professional.")
    st.divider()
    st.markdown("**Pipeline**  \nPDF → extract → range-check tool → agent plans retrieval → "
                "FAISS RAG → grade / re-retrieve → Gemini explanation → doctor questions")