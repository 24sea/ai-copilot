"""Streamlit frontend for Enterprise AI Copilot."""

from __future__ import annotations

import streamlit as st

from src.logger import get_logger
from src.rag_pipeline import answer_question, ingest_documents


logger = get_logger(__name__)


st.set_page_config(
    page_title="Enterprise AI Copilot",
    layout="wide",
)

st.title("Enterprise AI Copilot")
st.caption("Upload PDFs, ask questions, and get grounded answers with citations.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    process_clicked = st.button(
        "Process Documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = 0

if process_clicked:
    try:
        with st.spinner("Processing documents..."):
            inserted_chunks = ingest_documents(list(uploaded_files))
            st.session_state.processed_chunks += inserted_chunks
        if inserted_chunks:
            st.success(f"Processed {inserted_chunks} document chunks.")
        else:
            st.warning("No extractable text was found in the uploaded PDFs.")
    except Exception as exc:
        logger.exception("Document processing failed")
        st.error(f"Document processing failed: {exc}")

st.subheader("Ask a question")
question = st.text_input(
    "Question",
    placeholder="Example: What are the key obligations described in the document?",
    label_visibility="collapsed",
)

ask_clicked = st.button(
    "Ask Copilot",
    disabled=not question.strip(),
)

if ask_clicked:
    try:
        with st.spinner("Searching documents and generating answer..."):
            result = answer_question(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Source citations")
        if result["sources"]:
            for index, source in enumerate(result["sources"], start=1):
                metadata = source["metadata"]
                title = (
                    f"{index}. {metadata.get('source', 'Unknown source')} "
                    f"- page {metadata.get('page', 'unknown')} "
                    f"- chunk {metadata.get('chunk', 'unknown')}"
                )
                with st.expander(title):
                    st.caption(f"Similarity score: {source['score']:.4f}")
                    st.write(source["text"])
        else:
            st.info("No source chunks were retrieved.")
    except Exception as exc:
        logger.exception("Question answering failed")
        st.error(f"Question answering failed: {exc}")

with st.sidebar:
    st.divider()
    st.metric("Chunks indexed this session", st.session_state.processed_chunks)
    st.caption("Vector data persists locally in the configured ChromaDB folder.")
