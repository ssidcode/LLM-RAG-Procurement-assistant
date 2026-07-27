"""
app.py
Streamlit UI for the GeM Procurement RAG Assistant.

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

from rag import answer_question, PERSIST_DIR

st.set_page_config(
    page_title="GeM Procurement Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 GeM Procurement Assistant")
st.caption(
    "Ask questions about buyer/seller/service provider policies, SOPs, "
    "and guidelines. Answers are generated only from your ingested GeM documents."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    category = st.selectbox(
        "Search scope",
        options=["all", "buyer", "seller", "service_provider"],
        index=0,
        help="Restrict retrieval to documents from one category.",
    )
    top_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=10, value=5)

    st.divider()
    db_exists = Path(PERSIST_DIR).exists() and any(Path(PERSIST_DIR).iterdir())
    if db_exists:
        st.success("Vector store found ✅")
    else:
        st.warning("No vector store found.\nRun `python ingest.py` first.")

    st.divider()
    st.markdown(
        "**Setup checklist**\n"
        "1. `ollama serve` running\n"
        "2. `ollama pull nomic-embed-text`\n"
        "3. `ollama pull llama3.1`\n"
        "4. `python ingest.py`\n"
    )

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source_file']}** — page {s['page']} ({s['category']})")

# --- Chat input ---
question = st.chat_input("Ask about GeM procurement policy, SOPs, approval workflows...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            if not db_exists:
                answer = "No vector store found. Please run `python ingest.py` first."
                sources = []
            else:
                answer, docs = answer_question(question, category=category, k=top_k)
                sources = [
                    {
                        "source_file": d.metadata.get("source_file", "unknown"),
                        "page": d.metadata.get("page", "?"),
                        "category": d.metadata.get("category", "unknown"),
                    }
                    for d in docs
                ]
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- **{s['source_file']}** — page {s['page']} ({s['category']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
