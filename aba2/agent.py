import json
import os
import re
from typing import Dict, List, TypedDict
 
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
 
from rag import format_context
from tools import check_ranges, make_result, parse_report, results_to_text, summary_text
 
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 2
MIN_SCORE = 0.35

class ReportState(TypedDict, total=False):
    report_text: str
    results: List[dict]
    flagged: List[dict]
    summary: str
    queries: List[dict]               # [{"test": name, "query": text}]
    retrieved: Dict[str, List[dict]]  # test name -> docs
    weak: List[str]
    attempts: int
    explanation: str
    questions: List[str]
    trace: List[str]
 
 
def _log(state, msg):
    return state.get("trace", []) + [msg]
 
 
def _word(status):
    return {"LOW": "below", "HIGH": "above"}.get(status, "within")
 
 
def _rule_query(r):
    base = (r.get("canonical") or r["name"]).replace("_", " ")
    if r["status"] in ("LOW", "HIGH"):
        return f"{r['status'].lower()} {base} meaning causes"
    return f"{base} blood test what it measures"
