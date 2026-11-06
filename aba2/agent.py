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
