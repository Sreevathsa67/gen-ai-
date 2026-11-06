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

st.title("🩺 Medical Report Explainer")
st.caption("AI agent (LangGraph) · RAG (FAISS + Sentence Transformers) · Generative AI (Gemini)")
 
retriever = load_retriever()
 
uploaded = st.file_uploader("Upload your lab report (text-based PDF)", type=["pdf"])
col1, col2 = st.columns(2)
analyze = col1.button("Analyze report", type="primary", disabled=uploaded is None)
use_sample = col2.button("Try a sample blood report")
 
report_text = None
if analyze and uploaded is not None:
    report_text = extract_text_from_pdf(uploaded.getvalue())
    if not report_text.strip():
        st.error("No text found in this PDF. It may be a scanned image - please upload a text-based PDF.")
        st.stop()
elif use_sample:
    report_text = SAMPLE_REPORT
 
if report_text:
    with st.status("Agent is analysing your report...", expanded=True) as status:
        try:
            result = run_agent(report_text, retriever,
                               on_step=lambda node, msg: st.write(f"✅ **{node}** - {msg}"))
        except Exception as e:  # noqa: BLE001
            status.update(label="Analysis failed", state="error")
            st.error(f"Something went wrong: {e}")
            st.stop()
        status.update(label="Analysis complete", state="complete", expanded=False)
    st.session_state.result = result
    st.session_state.chat = []
 
# ---------------- Results ----------------
result = st.session_state.get("result")
if result:
    results = result.get("results", [])
    if not results:
        st.warning("No test values could be read from this report.")
    else:
        out = sum(r["status"] in ("LOW", "HIGH") for r in results)
        normal = sum(r["status"] == "NORMAL" for r in results)
        m1, m2, m3 = st.columns(3)
        m1.metric("Tests found", len(results))
        m2.metric("Within range", normal)
        m3.metric("Outside range", out)
        st.info(result.get("summary", ""))
 
        df = pd.DataFrame([{
            "Status": STATUS_LABEL.get(r["status"], r["status"]),
            "Test": r["name"],
            "Value": f"{r['value']:g}",
            "Unit": r["unit"],
            "Reference range (from report)": r["range_text"],
            "Difference": f"{r['deviation_pct']:+.0f}%" if r["deviation_pct"] is not None else "",
        } for r in results])
        st.dataframe(df, hide_index=True)
 
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Explanation", "❓ Questions for your doctor",
                                      "📚 Retrieved knowledge", "🧠 Agent trace"])
    with tab1:
        st.markdown(result.get("explanation", ""))
    with tab2:
        for i, q in enumerate(result.get("questions", []), 1):
            st.markdown(f"{i}. {q}")
    with tab3:
        for test, docs in result.get("retrieved", {}).items():
            label = "General guidance" if test == "_general" else test
            with st.expander(f"{label} ({len(docs)} documents)"):
                for d in docs:
                    st.markdown(f"**{d['title']}** · similarity {d['score']:.2f}  \n{d['text']}")
                    st.caption(d["source"])
    with tab4:
        for i, step in enumerate(result.get("trace", []), 1):
            st.markdown(f"{i}. {step}")
            
    st.divider()
    st.subheader("💬 Ask a follow-up question")
    chat = st.session_state.setdefault("chat", [])
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
 
    if question := st.chat_input("e.g. What does a low MCV mean?"):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and thinking..."):
                answer, docs = answer_followup(question, result, retriever, history=chat)
            st.markdown(answer)
            if docs:
                st.caption("Sources: " + ", ".join(dict.fromkeys(d["title"] for d in docs)))
        chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
 
st.caption("⚠️ Educational explanations only - not a medical diagnosis.")