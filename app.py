"""
PDF Reader RAG  (100% free stack)
- PDF parsing : pypdf
- Embeddings  : fastembed (BAAI/bge-small-en-v1.5, runs locally on CPU)
- Vector search: NumPy cosine similarity (no paid vector DB)
- LLM         : Google Gemini (AI Studio free key) OR Groq free tier
- Frontend    : Streamlit
Accuracy: per-answer faithfulness + retrieval scores, and a one-click
automatic evaluation (Hit@k and MRR) on the uploaded PDF.
"""
import json
import os
import random
import re

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="PDF Reader RAG", page_icon="📄", layout="wide")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedder():
    return TextEmbedding(EMBED_MODEL)


def embed(texts):
    vecs = np.array(list(load_embedder().embed(texts)), dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # normalised


def read_pdf(file):
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, re.sub(r"\s+", " ", text)))
    return pages


def chunk_pages(pages, size=800, overlap=150):
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if len(piece) > 50:
                chunks.append({"page": page_no, "text": piece})
            start += size - overlap
    return chunks


def retrieve(query, chunks, index, k):
    q = embed([query])[0]
    scores = index @ q
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def get_client(provider, api_key):
    if provider.startswith("Gemini"):
        return {"provider": "gemini", "obj": genai.Client(api_key=api_key)}
    return {"provider": "groq", "obj": Groq(api_key=api_key)}


def llm(client, model, system, user, temperature=0.0):
    if client["provider"] == "gemini":
        r = client["obj"].models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature),
        )
        return (r.text or "").strip()
    r = client["obj"].chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()


def answer_question(client, model, question, chunks, hits):
    context = "\n\n".join(
        f"[Page {chunks[i]['page']}] {chunks[i]['text']}" for i, _ in hits)
    system = ("You answer questions using ONLY the provided context from a PDF. "
              "If the answer is not in the context, say: "
              "'I could not find this in the document.' "
              "Mention page numbers like (p. 3) when you use them.")
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return llm(client, model, system, user), context


def judge_faithfulness(client, model, question, answer, context):
    """LLM-as-judge: is every claim in the answer supported by the context?"""
    system = ("You are a strict fact-checker. Reply ONLY with JSON: "
              '{"score": <0-100>, "reason": "<one sentence>"}. '
              "100 = every claim is supported by the context, "
              "0 = answer is unsupported or contradicts the context.")
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    raw = llm(client, model, system, user)
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        return float(data["score"]), data.get("reason", "")
    except Exception:
        return None, raw


def make_test_question(client, model, chunk_text):
    system = ("Write ONE specific question that can be answered using only the "
              "passage. Do not mention 'the passage'. Output only the question.")
    return llm(client, model, system, chunk_text, temperature=0.3)


# ---------------------------------------------------------------- sidebar
st.title("📄 PDF Reader RAG")
st.caption("Ask questions about a single PDF — free stack: pypdf + fastembed + Gemini/Groq + Streamlit")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider (both free)",
                            ["Gemini (Google AI Studio)", "Groq"])
    secret_name = "GEMINI_API_KEY" if provider.startswith("Gemini") else "GROQ_API_KEY"
    default_model = ("gemini-2.5-flash" if provider.startswith("Gemini")
                     else "llama-3.1-8b-instant")
    api_key = None
    try:
        api_key = st.secrets[secret_name]
    except Exception:
        api_key = os.getenv(secret_name)
    if not api_key:
        api_key = st.text_input(f"{provider} API key (free)", type="password")
    model = st.text_input("Model name", value=default_model, key=f"model_{provider}",
                          help="If the model is retired, pick a current free one from the provider's docs")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, 4)
    chunk_size = st.slider("Chunk size (characters)", 400, 1500, 800, 100)
    st.markdown("---")
    uploaded = st.file_uploader("Upload ONE PDF", type="pdf")

# ---------------------------------------------------------------- indexing
if uploaded is None:
    st.info("👈 Upload a PDF in the sidebar to begin.")
    st.stop()

doc_key = f"{uploaded.name}-{uploaded.size}-{chunk_size}"
if st.session_state.get("doc_key") != doc_key:
    with st.spinner("Reading PDF and building index..."):
        pages = read_pdf(uploaded)
        if not pages:
            st.error("No extractable text found (scanned PDF?). Try a text-based PDF.")
            st.stop()
        chunks = chunk_pages(pages, size=chunk_size)
        index = embed([c["text"] for c in chunks])
    st.session_state.update(doc_key=doc_key, chunks=chunks, index=index, history=[])

chunks, index = st.session_state["chunks"], st.session_state["index"]
st.success(f"Indexed **{uploaded.name}**: {len(chunks)} chunks.")

if not api_key:
    st.warning(f"Add your free {provider} API key in the sidebar (or Streamlit secrets as {secret_name}).")
    st.stop()
client = get_client(provider, api_key)

tab_ask, tab_eval = st.tabs(["💬 Ask the PDF", "📊 Evaluate accuracy"])

# ---------------------------------------------------------------- Ask tab
with tab_ask:
    question = st.chat_input("Ask something about your PDF...")
    for item in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            st.write(item["a"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Retrieval relevance", f"{item['rel']:.0f}%")
            c2.metric("Faithfulness (LLM judge)",
                      "n/a" if item["faith"] is None else f"{item['faith']:.0f}%")
            c3.metric("Overall confidence", f"{item['overall']:.0f}%")
            if item["reason"]:
                st.caption(f"Judge: {item['reason']}")
            with st.expander("Sources used"):
                for i, s in item["hits"]:
                    st.markdown(f"**Page {chunks[i]['page']}** — similarity {s:.2f}")
                    st.write(chunks[i]["text"])

    if question:
        with st.spinner("Thinking..."):
            hits = retrieve(question, chunks, index, top_k)
            answer, context = answer_question(client, model, question, chunks, hits)
            faith, reason = judge_faithfulness(client, model, question, answer, context)
            # cosine similarity of best chunk -> rough 0-100 relevance
            rel = max(0.0, min(1.0, hits[0][1])) * 100
            overall = rel * 0.4 + faith * 0.6 if faith is not None else rel
        st.session_state["history"].append(
            dict(q=question, a=answer, hits=hits, rel=rel,
                 faith=faith, reason=reason, overall=overall))
        st.rerun()

# ---------------------------------------------------------------- Eval tab
with tab_eval:
    st.markdown(
        "Automatically tests the RAG on **your PDF**: random chunks are turned into "
        "questions by the LLM, then we check whether retrieval brings back the "
        "original chunk.\n\n"
        "- **Hit@k** = % of questions where the source chunk is in the top-k results\n"
        "- **MRR** = average of 1/rank of the source chunk (1.0 is perfect)\n"
        "- **Faithfulness** = % of generated answers supported by retrieved context")
    n_q = st.slider("Number of test questions", 3, 15, 8)
    if st.button("Run evaluation"):
        random.seed(42)
        sample = random.sample(range(len(chunks)), min(n_q, len(chunks)))
        rows, hits_ct, rr, faiths = [], 0, [], []
        bar = st.progress(0.0)
        for n, ci in enumerate(sample, start=1):
            q = make_test_question(client, model, chunks[ci]["text"])
            res = retrieve(q, chunks, index, top_k)
            ids = [i for i, _ in res]
            hit = ci in ids
            rank = ids.index(ci) + 1 if hit else None
            hits_ct += hit
            rr.append(1 / rank if rank else 0)
            ans, ctx = answer_question(client, model, q, chunks, res)
            f, _ = judge_faithfulness(client, model, q, ans, ctx)
            if f is not None:
                faiths.append(f)
            rows.append({"Question": q, "Source page": chunks[ci]["page"],
                         "Retrieved?": "✅" if hit else "❌",
                         "Rank": rank, "Faithfulness %": f})
            bar.progress(n / len(sample))
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Hit@{top_k}", f"{100 * hits_ct / len(sample):.0f}%")
        m2.metric("MRR", f"{np.mean(rr):.2f}")
        m3.metric("Avg faithfulness",
                  f"{np.mean(faiths):.0f}%" if faiths else "n/a")
        st.dataframe(rows, use_container_width=True)
