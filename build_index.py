#!/usr/bin/env python3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import trafilatura
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ── 1) Your source URLs ────────────────────────────────────────────────
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
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=[429,500,502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# ── 3) Simple sliding‑window chunker ───────────────────────────────────
def chunk_text(text, chunk_size=250, overlap=50):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def main():
    sess = make_session()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts, metas = [], []
    max_per_site, total_limit = 20, 150

    for name, url in SOURCES.items():
        try:
            raw = trafilatura.fetch_url(url)
            if not raw:
                print(f"[WARN] no content from {url}")
                continue

            cleaned = trafilatura.extract(raw,
                                          include_comments=False,
                                          include_tables=False)
            if not cleaned:
                print(f"[WARN] couldn’t extract {url}")
                continue

            for chunk in chunk_text(cleaned)[:max_per_site]:
                texts.append(chunk)
                metas.append({"source": name, "url": url})
                if len(texts) >= total_limit:
                    break
            if len(texts) >= total_limit:
                break

        except Exception as e:
            print(f"[ERROR] fetching {url}: {e}")

    if not texts:
        raise RuntimeError("No texts collected for indexing!")

    # embed & index
    embs = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embs, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # write outputs
    faiss.write_index(index, "index.faiss")
    with open("texts.json", "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    with open("metas.json", "w") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"✅ Built index with {len(texts)} chunks")

if __name__ == "__main__":
    main()
