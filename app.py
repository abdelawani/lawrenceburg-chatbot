# app.py

import os
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import trafilatura
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ── 1) Source URLs ───────────────────────────────────────────────────────
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

# ── 2) HTTP session with retries & browser UA ────────────────────────────
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

# ── 3) Simple sliding‑window chunker (≈250 words, 50‑word overlap) ──────
def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ── 4) Fast embedding model (MiniLM) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── 5) Load & cache quantized Llama 2‑7B‑Chat pipeline ─────────────────
@st.cache_resource(show_spinner=False)
def load_chat_model():
    # **You must host your GGUF** on HF Hub under your repo (or adjust repo_id)
    repo_id   = "your-hf-username/llama2-7b-chat-gguf"
    filename  = "llama2-7b-chat-q4_0.gguf"
    models_dir = "models"
    local_path = os.path.join(models_dir, filename)

    # 1) download if not already present
    if not os.path.exists(local_path):
        os.makedirs(models_dir, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            repo_type="model"
        )

    # 2) load quantized Llama 2 on CPU
    llm = Llama(
        model_path=local_path,
        n_ctx=2048,
        n_threads=4,
        temperature=0.7,
    )

    # 3) wrap to unify interface
    def chat_fn(prompt: str) -> str:
        resp = llm(prompt, max_tokens=256)
        return resp["choices"][0]["text"].strip()

    return chat_fn

# ── 6) Build FAISS index with chunk caps ───────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vector_index():
    sess         = make_session()
    embed_model  = load_embedding_model()
    texts, metas = [], []
    per_site_cap = 20
    total_cap    = 150

    for name, url in SOURCES.items():
        try:
            raw = trafilatura.fetch_url(url)
            if not raw:
                st.warning(f"No content from {url}")
                continue

            cleaned = trafilatura.extract(raw,
                                          include_comments=False,
                                          include_tables=False)
            if not cleaned:
                st.warning(f"Couldn’t extract text from {url}")
                continue

            chunks = chunk_text(cleaned)
            for chunk in chunks[:per_site_cap]:
                texts.append(chunk)
                metas.append({"source": name, "url": url})
                if len(texts) >= total_cap:
                    break

            if len(texts) >= total_cap:
                break

        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")
            continue

    if not texts:
        st.error("No texts to index. Check your SOURCES or scraping.")
        st.stop()

    embeddings = embed_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
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

# ── 8) Generate answer via quantized chat model ───────────────────────
def generate_answer(query, contexts):
    chat = load_chat_model()
    ctxs = "\n\n".join(f"[{m['source']}] {txt}" for txt, m in contexts)
    prompt = f"""
You are Sam, a friendly UT Extension agronomist based in Lawrence County, TN.
Use ONLY the context below to give a step-by-step answer. Cite the source
in brackets after each step.

CONTEXT:
{ctxs}

QUESTION:
{query}

ANSWER:
"""
    return chat(prompt)

# ── 9) Streamlit UI ─────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Lawrenceburg Extension Chatbot")
    st.title("🌾 Lawrenceburg County Extension Chatbot")
    st.write(
        "Ask questions about Extension services, resources, and programs "
        "in Lawrence County, TN."
    )

    # build (or load) index once
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
