# app.py

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import trafilatura
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ── 1) Your source URLs ─────────────────────────────────────────────────
SOURCES = {
    "Lawrence County Extension":
        "https://lawrencecountytn.gov/government/departments/agricultural-extension/",
    "Lawrence County Homepage":
        "https://lawrencecountytn.gov/",
    "UTIA":
        "https://utia.tennessee.edu/",
    "TN State Univ.":
        "https://www.tnstate.edu/",
    "TN Ag. Dept.":
        "https://www.tn.gov/agriculture.html",
}

# ── 2) Session with retry & browser UA ──────────────────────────────────
def make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
    })
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# ── 3) Chunker (≈250 words, 50‑word overlap) ────────────────────────────
def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ── 4) Load & cache the faster MiniLM embedding model ──────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── 5) Load & cache RAG LLM (Flan‑T5‑Base) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=768,
        do_sample=True,
        temperature=0.7,
    )

# ── 6) Build FAISS index with chunk caps ───────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vector_index():
    sess           = make_session()
    embed_model    = load_embedding_model()
    texts, metas   = [], []
    max_per_site   = 20
    total_limit    = 150

    for name, url in SOURCES.items():
        try:
            raw = trafilatura.fetch_url(url)
            if not raw:
                st.warning(f"No content fetched from {url}")
                continue

            cleaned = trafilatura.extract(
                raw,
                include_comments=False,
                include_tables=False
            )
            if not cleaned:
                st.warning(f"Couldn’t extract text from {url}")
                continue

            chunks = chunk_text(cleaned)
            for chunk in chunks[:max_per_site]:
                texts.append(chunk)
                metas.append({"source": name, "url": url})
                if len(texts) >= total_limit:
                    break

            if len(texts) >= total_limit:
                break

        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")
            continue

    if not texts:
        st.error("No texts to index. Check sources or scraping.")
        st.stop()

    embs       = embed_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embs, dtype="float32")
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, texts, metas

# ── 7) Retrieve top‑k chunks ────────────────────────────────────────────
def retrieve(query, index, texts, metas, k=5):
    embed_model = load_embedding_model()
    q_emb       = embed_model.encode([query])
    _, I        = index.search(np.array(q_emb, dtype="float32"), k)
    return [(texts[i], metas[i]) for i in I[0]]

# ── 8) Generate answer with expanded prompt ────────────────────────────
def generate_answer(query, contexts):
    qa    = load_qa_pipeline()
    ctxs  = "\n\n".join(f"[{m['source']}] {txt}" for txt, m in contexts)
    prompt = f"""
You are the Lawrenceburg County Extension virtual agent.
Use ONLY the context below to answer in numbered steps.
Cite the source in brackets after each step.

CONTEXT:
{ctxs}

QUESTION:
{query}

ANSWER:
"""
    out = qa(prompt)[0]["generated_text"]
    return out.strip()

# ── 9) Streamlit UI ────────────────────────────────────────────────────
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
        with st.spinner("Retrieving…"):
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
