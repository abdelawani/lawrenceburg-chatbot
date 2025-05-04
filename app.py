import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1) Configure your six source URLs
SOURCES = {
    "Lawrence County Extension": "https://lawrencecountytn.gov/government/departments/agricultural-extension/",
    "Lawrence County Homepage":   "https://lawrencecountytn.gov/",
    "UTIA":                       "https://utia.tennessee.edu/",
    "TN State Univ.":             "https://www.tnstate.edu/",
    "TN Government":              "https://www.tn.gov/",
    "TN Ag. Dept.":               "https://www.tn.gov/agriculture.html",
}

# 2) Load & cache embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-mpnet-base-v2")

# 3) Load & cache RAG LLM (Flan-T5-Small for CPU friendliness)
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
    )

# 4) Scrape each site, chunk text, embed, and build FAISS index
@st.cache_resource(show_spinner=False)
def build_vector_index():
    embed_model = load_embedding_model()
    texts, metas = [], []
    for name, url in SOURCES.items():
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.find_all(["h1","h2","h3","p","li"]):
                txt = tag.get_text(strip=True)
                if len(txt) >= 50:
                    texts.append(txt)
                    metas.append({"source": name, "url": url})
        except Exception as e:
            st.warning(f"Couldn’t fetch {url}: {e}")
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    return index, texts, metas

# 5) Retrieve top‑k similar chunks
def retrieve(query, index, texts, metas, k=5):
    embed_model = load_embedding_model()
    q_emb       = embed_model.encode([query])
    D, I        = index.search(np.array(q_emb, dtype="float32"), k)
    return [(texts[i], metas[i]) for i in I[0]]

# 6) Generate answer with contexts
def generate_answer(query, contexts):
    qa = load_qa_pipeline()
    ctx_str = " ".join([c[0] for c in contexts])
    prompt  = (
        "Answer the question using the context below.\n\n"
        f"Context: {ctx_str}\n\n"
        f"Question: {query}\nAnswer:"
    )
    out = qa(prompt)[0]["generated_text"]
    return out.strip()

# 7) Streamlit UI
def main():
    st.set_page_config(page_title="Lawrenceburg Extension Chatbot")
    st.title("Lawrenceburg County Extension Chatbot")
    st.write("Ask questions about Extension services, resources, and programs in Lawrence County, TN.")

    # Build (or load) our index once per session
    index, texts, metas = build_vector_index()

    query = st.text_input("Your question:", "")
    if st.button("Ask") and query:
        with st.spinner("Retrieving…"):
            contexts = retrieve(query, index, texts, metas, k=5)

        st.markdown("**🔍 Sources Retrieved:**")
        for i, (txt, m) in enumerate(contexts, start=1):
            st.markdown(f"{i}. [{m['source']}]({m['url']}) — “{txt[:200]}…”")

        with st.spinner("Generating answer…"):
            answer = generate_answer(query, contexts)

        st.markdown("**💡 Answer:**")
        st.write(answer)

if __name__ == "__main__":
    main()
