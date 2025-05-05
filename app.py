import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import trafilatura
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr

# 1) Source URLs
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

# 2) HTTP session w/ retry & UA
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

# 3) Simple chunker (≈250 words, 50 overlap)
def chunk_text(text, chunk_size=250, overlap=50):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# 4) Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 5) Chat‑optimized Falcon 7B‑Instruct
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model     = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    device_map="auto",
    load_in_8bit=True
)
chat_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 6) Build FAISS index (cache in memory)
_index, _texts, _metas = None, None, None
def build_index():
    global _index, _texts, _metas
    if _index is not None:
        return _index, _texts, _metas

    sess = make_session()
    texts, metas = [], []
    per_site, total = 20, 150

    for name, url in SOURCES.items():
        try:
            raw = trafilatura.fetch_url(url)
            cleaned = trafilatura.extract(raw,
                                          include_comments=False,
                                          include_tables=False)
            for chunk in chunk_text(cleaned)[:per_site]:
                texts.append(chunk)
                metas.append({"source": name, "url": url})
                if len(texts) >= total:
                    break
            if len(texts) >= total:
                break
        except:
            continue

    embs = embedder.encode(texts, show_progress_bar=False)
    idx  = faiss.IndexFlatL2(len(embs[0]))
    idx.add(np.array(embs, dtype="float32"))
    _index, _texts, _metas = idx, texts, metas
    return idx, texts, metas

# 7) Retrieve
def retrieve(query, idx, texts, metas, k=5):
    q_emb = embedder.encode([query])
    _, I  = idx.search(np.array(q_emb, dtype="float32"), k)
    return [(texts[i], metas[i]) for i in I[0]]

# 8) Generate answer
def generate_answer(query):
    idx, texts, metas = build_index()
    contexts = retrieve(query, idx, texts, metas)
    ctx = "\n\n".join(f"[{m['source']}] {txt}" for txt, m in contexts)
    prompt = f"""
You are Sam, a friendly UT Extension agronomist in Lawrence County, TN.
Use ONLY the context below to answer in clear numbered steps.
Cite each source in brackets after the step.

CONTEXT:
{ctx}

QUESTION:
{query}

ANSWER:
"""
    return chat_pipe(prompt)[0]["generated_text"].strip()

# 9) Gradio interface
iface = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question…"),
    outputs="text",
    title="🌾 Lawrenceburg County Extension Chatbot",
    description="Ask about Extension services, resources, and programs in Lawrence County, TN.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(share=True)
