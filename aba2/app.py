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
