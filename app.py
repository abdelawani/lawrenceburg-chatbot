import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import trafilatura
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1) Your six source URLs
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

# 2) HTTP session w/ retry & browser UA
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
    })
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# 3) Simple chunker (~250 words, 50-word overlap)
def chunk_text(text, chunk_size=250, overlap=50):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# 4) Fast embedder
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# 5) Chat‑optimized Llama2‑7B‑Chat on GPU via HF pipeline
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 6) Build FAISS index (cap 20 chunks/site, 150 total)
@st.cache_resource(show_spinner=False)
def build_vector_index():
    sess       = make_session()
    embedder   = load_embedder()
    texts, metas = [], []
    per_site, total = 20, 150

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
                st.warning(f"Extract failed for {url}")
                continue

            for chunk in chunk_text(cleaned)[:per_site]:
                texts.append(chunk)
                metas.append({"source": name, "url": url})
                if len(texts) >= total:
                    break
            if len(texts) >= total:
                break

        except Exception as e:
            st.warning(f"{name} → {e}")

    if not texts:
        st.error("No text to index. Check SOURCES or scraper.")
        st.stop()

    embs = embedder.encode(texts, show_progress_bar=False)
    idx  = faiss.IndexFlatL2(len(embs[0]))
    idx.add(np.array(embs, dtype="float32"))
    return idx, texts, metas

# 7) Retrieval
def retrieve(q, idx, texts, metas, k=5):
    q_emb  = load_embedder().encode([q])
    _, I   = idx.search(np.array(q_emb, dtype="float32"), k)
    return [(texts[i], metas[i]) for i in I[0]]

# 8) Answer generation
def generate_answer(q, contexts):
    qa  = load_qa_pipeline()
    ctx = "\n\n".join(f"[{m['source']}] {txt}" for txt, m in contexts)
    prompt = f"""
You are Sam, a friendly UT Extension agronomist in Lawrence County, TN.
Use ONLY the context below to answer in clear numbered steps.
Cite each source in brackets after the step.

CONTEXT:
{ctx}

QUESTION:
{q}

ANSWER:
"""
    return qa(prompt)[0]["generated_text"].strip()

# 9) Streamlit UI
def main():
    st.set_page_config(page_title="Lawrenceburg Extension Chatbot")
    st.title("🌾 Lawrenceburg County Extension Chatbot")
    st.write("Ask questions about Extension services in Lawrence County, TN.")

    idx, texts, metas = build_vector_index()

    q = st.text_input("Your question here…")
    if st.button("Ask") and q:
        with st.spinner("Retrieving…"):
            ctxs = retrieve(q, idx, texts, metas)
        st.markdown("**🔍 Retrieved contexts:**")
        for i,(txt,m) in enumerate(ctxs,1):
            snippet = txt[:200].replace("\n"," ") + "…"
            st.markdown(f"{i}. [{m['source']}]({m['url']}) — “{snippet}”")
        with st.spinner("Generating…"):
            ans = generate_answer(q, ctxs)
        st.markdown("**💡 Answer:**")
        st.write(ans)

if __name__=="__main__":
    main()
