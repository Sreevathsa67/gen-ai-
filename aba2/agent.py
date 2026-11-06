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


def build_agent(retriever):
 
    def extract_node(state: ReportState):
        text = state.get("report_text", "")
        results, method = parse_report(text), "rule-based parser"
        if len(results) < 2:
            try:
                results, method = llm_extract(text), "Gemini extraction (parser fallback)"
            except Exception as e:  # noqa: BLE001
                method = f"rule-based parser (Gemini fallback failed: {e})"
        return {"results": results, "trace": _log(state, f"Extracted {len(results)} tests using {method}")}
 
    def check_node(state: ReportState):
        results = check_ranges(state.get("results", []))
        flagged = [r for r in results if r["status"] in ("LOW", "HIGH")]
        return {"results": results, "flagged": flagged, "summary": summary_text(results),
                "trace": _log(state, f"Range-check tool: {len(flagged)} of {len(results)} outside reference range")}
 
    def plan_node(state: ReportState):
        results, flagged = state.get("results", []), state.get("flagged", [])
        if not results:
            return {"queries": [], "trace": _log(state, "Nothing to retrieve")}
        names = {r["name"] for r in results}
        plan, how = [], "Gemini planner"
        try:
            prompt = (
                "You are the retrieval planner of a lab-report agent. Decide which tests need background knowledge "
                "from the medical knowledge base. Always include every test that is LOW or HIGH; add at most 2 "
                "normal tests only if they help explain the abnormal ones (e.g. MCV with hemoglobin).\n"
                'Return a JSON list of {"test": <exact test name from the list>, "query": <short search query>}.\n\n'
                f"Tests:\n{results_to_text(results)}"
            )
            plan = [p for p in parse_json(call_llm(prompt, json_mode=True, temperature=0))
                    if isinstance(p, dict) and p.get("test") in names and p.get("query")]
        except Exception:  # noqa: BLE001
            plan, how = [], "rule-based planner (Gemini unavailable)"
        planned = {p["test"] for p in plan}
        targets = flagged or [r for r in results if r["canonical"]][:3]
        plan += [{"test": r["name"], "query": _rule_query(r)} for r in targets if r["name"] not in planned]
        plan = plan[:12]  # every flagged test is kept (planner prompt asks for <= 8)
        return {"queries": plan, "attempts": 0,
                "trace": _log(state, f"{how} chose {len(plan)} tests to look up: "
                                     + ", ".join(p["test"] for p in plan))}
 
    def retrieve_node(state: ReportState):
        retrieved = dict(state.get("retrieved", {}))
        for q in state.get("queries", []):
            retrieved[q["test"]] = retriever.retrieve(q["query"], k=3)
        if state.get("flagged") and "_general" not in retrieved:
            retrieved["_general"] = retriever.retrieve(
                "what does a value outside the reference range mean, is it a diagnosis", k=2)
        return {"retrieved": retrieved,
                "trace": _log(state, f"FAISS retrieval for {len(state.get('queries', []))} queries")}
 
    def grade_node(state: ReportState):
        by_name = {r["name"]: r for r in state.get("results", [])}
        weak = []
        for q in state.get("queries", []):
            docs = state["retrieved"].get(q["test"], [])
            canon = by_name.get(q["test"], {}).get("canonical")
            best = max((d["score"] for d in docs), default=0.0)
            if (canon and canon not in {d["id"] for d in docs}) or best < MIN_SCORE:
                weak.append(q["test"])
        msg = f"Grader: weak retrieval for {', '.join(weak)}" if weak else "Grader: all retrievals relevant"
        return {"weak": weak, "trace": _log(state, msg)}
 
    def route_after_grade(state: ReportState):
        if state.get("weak") and state.get("attempts", 0) < MAX_RETRIES:
            return "rewrite"
        return "explain"
 
    def rewrite_node(state: ReportState):
        by_name = {r["name"]: r for r in state.get("results", [])}
        new_queries = []
        for test in state.get("weak", []):
            r = by_name.get(test, {"name": test, "canonical": None, "status": "UNKNOWN"})
            query = ""
            try:
                query = call_llm(
                    f"A knowledge-base search for the lab test '{test}' returned irrelevant documents. Write ONE "
                    "better search query (max 10 words) using the full standard English name of the test and common "
                    "synonyms. Reply with the query only.", temperature=0
                ).strip().strip('"')
            except Exception:  # noqa: BLE001
                pass
            if not query:
                canon = (r.get("canonical") or "").replace("_", " ")
                query = f"{canon} {test} blood test what it measures".strip()
            new_queries.append({"test": test, "query": query})
        attempts = state.get("attempts", 0) + 1
        return {"queries": new_queries, "attempts": attempts,
                "trace": _log(state, f"Rewrote {len(new_queries)} queries, retrieving again (attempt {attempts})")}
 
    def explain_node(state: ReportState):
        results = state.get("results", [])
        if not results:
            return {"explanation": "No test values could be read from this report. If it is a scanned image, "
                                   "try a text-based PDF.", "trace": _log(state, "No results to explain")}
        all_docs = [d for docs in state.get("retrieved", {}).values() for d in docs]
        prompt = (
            f"Lab results (status computed against the report's own reference ranges):\n{results_to_text(results)}\n\n"
            f"Knowledge base context:\n{format_context(all_docs)}\n\n"
            "Write a patient-friendly explanation in Markdown with these sections:\n"
            "### Overview\n2-3 sentences.\n"
            "### Values outside the reference range\nFor each LOW/HIGH test: **Test - value unit (range)**, then what "
            "the test measures, what being low/high can generally be linked to (several possibilities, from the "
            "context), and why it is worth discussing.\n"
            "### Values within range\nOne short bullet list.\n"
            "### Keep in mind\nOne result is not a diagnosis; ranges vary between labs; talk to your doctor."
        )
        try:
            text, how = call_llm(prompt), "Gemini wrote the explanation"
        except Exception as e:  # noqa: BLE001
            text, how = _fallback_explanation(state, retriever), f"Template explanation (Gemini unavailable: {e})"
        return {"explanation": text, "trace": _log(state, how)}
 
    def questions_node(state: ReportState):
        flagged = state.get("flagged", [])
        try:
            qs = parse_json(call_llm(
                "Based on these lab results, write 5 short, specific questions the patient could ask their doctor. "
                f"Focus on the out-of-range values. Return a JSON list of strings.\n\n"
                f"{results_to_text(state.get('results', []))}", json_mode=True))
            qs, how = [str(q) for q in qs][:6], "Gemini suggested doctor questions"
        except Exception:  # noqa: BLE001
            qs = [f"My {r['name']} is {_word(r['status'])} the range. What could be causing this in my case?"
                  for r in flagged[:3]]
            qs += ["Do any of these tests need to be repeated, and when?",
                   "Could my medicines, diet or a recent illness have affected these results?",
                   "Is there anything in my lifestyle I should change based on these results?"]
            how = "Rule-based doctor questions (Gemini unavailable)"
        return {"questions": qs, "trace": _log(state, how)}
 
    g = StateGraph(ReportState)
    g.add_node("extract_tests", extract_node)
    g.add_node("check_ranges", check_node)
    g.add_node("plan_retrieval", plan_node)
    g.add_node("retrieve_docs", retrieve_node)
    g.add_node("grade_docs", grade_node)
    g.add_node("rewrite_queries", rewrite_node)
    g.add_node("write_explanation", explain_node)
    g.add_node("suggest_questions", questions_node)
 
    g.set_entry_point("extract_tests")
    g.add_edge("extract_tests", "check_ranges")
    g.add_edge("check_ranges", "plan_retrieval")
    g.add_edge("plan_retrieval", "retrieve_docs")
    g.add_edge("retrieve_docs", "grade_docs")
    g.add_conditional_edges("grade_docs", route_after_grade,
                            {"rewrite": "rewrite_queries", "explain": "write_explanation"})
    g.add_edge("rewrite_queries", "retrieve_docs")
    g.add_edge("write_explanation", "suggest_questions")
    g.add_edge("suggest_questions", END)
    return g.compile()
 
 
def _fallback_explanation(state, retriever):
    lines = ["### Overview", state.get("summary", ""), "", "### Values outside the reference range"]
    flagged = state.get("flagged", [])
    for r in flagged:
        doc = retriever.get(r["canonical"]) if r.get("canonical") else None
        if doc is None:
            docs = state.get("retrieved", {}).get(r["name"], [])
            doc = docs[0] if docs else None
        lines.append(f"**{r['name']} - {r['value']:g} {r['unit']} (range {r['range_text']})** is "
                     f"{_word(r['status'])} the reference range.")
        lines.append((doc["text"] if doc else "Your doctor can explain what this test measures.") + "\n")
    if not flagged:
        lines.append("None - all values with a printed range are within it.")
    lines += ["", "### Keep in mind",
              "One result outside the range is not a diagnosis. Please discuss these results with your doctor.",
              "", "_Gemini was unavailable, so this shows knowledge-base text directly._"]
    return "\n".join(lines)
 
 
def run_agent(report_text: str, retriever, on_step=None):
    """Run the graph; on_step(node_name, message) is called after every node (for live UI updates)."""
    graph = build_agent(retriever)
    final: dict = {"report_text": report_text, "attempts": 0, "trace": []}
    for chunk in graph.stream(final, stream_mode="updates"):
        for node, update in chunk.items():
            final.update(update or {})
            if on_step and update and update.get("trace"):
                on_step(node, update["trace"][-1])
    return final
 
 
def answer_followup(question: str, state: dict, retriever, history=None):
    """Answer a follow-up question using the report + freshly retrieved knowledge. Returns (answer, docs)."""
    docs = retriever.retrieve(question, k=3)
    q_lower = question.lower()
    for r in state.get("results", []):
        canon = (r.get("canonical") or "").replace("_", " ")
        if r["name"].lower() in q_lower or (canon and canon in q_lower):
            docs += retriever.retrieve(_rule_query(r), k=1)
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-6:])
    prompt = (
        f"The patient's lab results:\n{results_to_text(state.get('results', []))}\n\n"
        f"Knowledge base context:\n{format_context(docs)}\n\n"
        f"Conversation so far:\n{history_text or '(none)'}\n\n"
        f"Patient's question: {question}\n\n"
        "Answer in under 150 words, simply and kindly, using only the context and their results. "
        "If the question is outside the context or asks for a diagnosis or medicine, say so and suggest asking "
        "their doctor."
    )
    try:
        return call_llm(prompt), docs
    except Exception as e:  # noqa: BLE001
        return f"Gemini is unavailable right now ({e}). Here is what the knowledge base says:\n\n" \
               + format_context(docs[:2]), docs


     
