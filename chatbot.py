import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import fitz
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Industrial Chatbot", page_icon="🏭")
st.title("Industrial AI Chatbot")
st.caption("Ask questions from your industrial documents")

# ── Load everything once ──────────────────────────────────
@st.cache_resource
def load_pipeline(pdf_path):
    # Load + chunk
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Embed + index
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed_model.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    # LLM
    llm = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        do_sample=False
    )
    return chunks, index, embed_model, llm

# ── File uploader ─────────────────────────────────────────
uploaded = st.file_uploader("Upload an industrial PDF", type="pdf")

if uploaded:
    pdf_path = f"/tmp/{uploaded.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Loading document and models... (first time takes 3-4 min)"):
        chunks, index, embed_model, llm = load_pipeline(pdf_path)

    st.success(f"Ready! Loaded {len(chunks)} chunks from {uploaded.name}")

    # ── Chat history ──────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    # ── Display history ───────────────────────────────────
    for q, a, sources in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
        with st.expander("Source chunks"):
            for s in sources:
                st.write(s)
                st.divider()

    # ── Input ─────────────────────────────────────────────
    question = st.chat_input("Ask a question about the document...")

    if question:
        # Retrieve
        q_vec = embed_model.encode([question])
        _, indices = index.search(np.array(q_vec), 3)
        retrieved = [chunks[i] for i in indices[0]]

        # Answer
        context = "\n".join(retrieved)
        prompt = f"""<|system|>
You are an industrial robotics assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."</s>
<|user|>
Context:
{context}

Question: {question}</s>
<|assistant|>"""
        result = llm(prompt)
        ans = result[0]["generated_text"].split("<|assistant|>")[-1].strip()

        st.session_state.history.append((question, ans, retrieved))
        st.rerun()