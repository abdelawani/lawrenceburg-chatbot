# app.py

import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ── 1) Load & cache the embedding model ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── 2) Load & cache the RAG LLM pipeline (Flan‑T5‑Base) ────────────────
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    model_name = "google/flan-t5-base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=768,
        do_sample=True,
        temperature=0.7,
    )

# ── 3) Load pre‑computed FAISS index + metadata ────────────────────────
@st.cache_resource(show_spinner=False)
def build_vector_index():
    index = faiss.read_index("index.faiss")
    with open("texts.json", "r") as f:
        texts = json.load(f)
    with open("metas.json", "r") as f:
        metas = json.load(f)
    return index, texts, metas

# ── 4) Retrieve top‑k relevant chunks ──────────────────────────────────
def retrieve(query, index, texts, metas, k=5):
    embed_model = load_embedding_model()
    q_emb       = embed_model.encode([query])
    _, I        = index.search(np.array(q_emb, dtype="float32"), k)
    return [(texts[i], metas[i]) for i in I[0]]

# ── 5) Generate an answer grounded in context ──────────────────────────
def generate_answer(query, contexts):
    qa   = load_qa_pipeline()
    ctxs = "\n\n".join(f"[{m['source']}] {txt}" for txt, m in contexts)
    prompt = f"""
You are the Lawrenceburg County Extension virtual agent.
Use ONLY the context below to answer the question in numbered steps.
Cite the source in brackets after each step.

CONTEXT:
{ctxs}

QUESTION:
{query}

ANSWER:
"""
    out = qa(prompt)[0]["generated_text"]
    return out.strip()

# ── 6) Streamlit user interface ────────────────────────────────────────
def main():
    st.set_page_config(page_title="Lawrenceburg Extension Chatbot")
    st.title("🌾 Lawrenceburg County Extension Chatbot")
    st.write(
        "Ask questions about Extension services, resources, and programs "
        "in Lawrence County, TN."
    )

    index, texts, metas = build_vector_index()

    query = st.text_input("Your question here…")
    if st.button("Ask") and query:
        with st.spinner("Retrieving relevant info…"):
            contexts = retrieve(query, index, texts, metas, k=5)

        st.markdown("**🔍 Retrieved contexts:**")
        for i, (txt, m) in enumerate(contexts, start=1):
            snippet = txt[:200].replace("\n", " ") + "…"
            st.markdown(f"{i}. [{m['source']}]({m['url']}) — “{snippet}”")

        with st.spinner("Generating answer…"):
            answer = generate_answer(query, contexts)

        st.markdown("**💡 Answer:**")
        st.write(answer)

if __name__ == "__main__":
    main()
