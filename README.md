# 📄 PDF Reader RAG (100% free)

A Retrieval-Augmented Generation app: upload **one PDF**, ask questions, get answers
with page citations and **accuracy scores**.

| Part | Tool (all free) |
|---|---|
| PDF parsing | `pypdf` |
| Embeddings | `fastembed` – BAAI/bge-small-en-v1.5 (runs locally) |
| Vector search | NumPy cosine similarity |
| LLM | Groq free tier (Llama 3.1) |
| Frontend / hosting | Streamlit + Streamlit Community Cloud |
| Code hosting | GitHub |

## How it works
1. PDF text is extracted page by page.
2. Text is split into overlapping chunks (default 800 characters, 150 overlap).
3. Each chunk is converted to an embedding vector.
4. Your question is embedded; the top-k most similar chunks are retrieved.
5. The LLM answers **only** from those chunks and cites pages.

## How accuracy is measured
There is no single "accuracy" number for RAG, so the app reports three things:

1. **Retrieval relevance** – cosine similarity of the best chunk (0–100%).
2. **Faithfulness** – a second LLM call checks if every claim in the answer is supported by the retrieved text (0–100%).
3. **Overall confidence** – 40% retrieval + 60% faithfulness.

The **Evaluate accuracy** tab runs an automatic test on your PDF:
the LLM writes questions from random chunks and the app checks whether
retrieval finds the original chunk (**Hit@k**, **MRR**) and whether answers are faithful.

> Note: the LLM judge is an estimate, not a human-verified ground truth. Spot-check a few answers against the PDF yourself.

---

## PART A – Get a free Groq API key
1. Go to https://console.groq.com and sign up (free, no card).
2. Open **API Keys → Create API Key**. Copy it and keep it private.

## PART B – Put the code on GitHub
1. Sign in at https://github.com (create a free account if needed).
2. Click **+ → New repository**. Name it `pdf-rag`. Choose **Public**. Click **Create repository**.
3. Click **uploading an existing file**.
4. Drag in these 3 files: `app.py`, `requirements.txt`, `README.md`. (`requirements.txt` is mandatory – Streamlit uses it to install libraries.)
5. Click **Commit changes**.

⚠️ Never upload your API key to GitHub.

## PART C – Deploy the frontend on Streamlit Community Cloud
1. Go to https://share.streamlit.io and click **Continue with GitHub**. Authorize it.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `your-username/pdf-rag`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "paste_your_key_here"
   ```
5. Click **Deploy**. The first build takes about 3–5 minutes.
6. You get a public link like `https://your-app.streamlit.app` – share it.

## PART D – Use it
1. Upload one text-based PDF in the sidebar.
2. Ask questions in the **Ask the PDF** tab; check scores under each answer.
3. Open **Evaluate accuracy** and click **Run evaluation** for Hit@k, MRR and faithfulness.

## Run locally (optional)
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key"      # Windows: set GROQ_API_KEY=your_key
streamlit run app.py
```

## Troubleshooting
- **"No extractable text"** – the PDF is scanned images. Use a text-based PDF.
- **Model not found / decommissioned** – change the model name in the sidebar (see https://console.groq.com/docs/models).
- **Rate limit error** – free tier has limits; wait a minute or reduce test questions.
- **App was changed on GitHub but not updated** – Streamlit redeploys automatically; or use **Reboot app**.
- **Low Hit@k** – try a larger top-k or a different chunk size in the sidebar.

## Ideas to improve (still free)
Add a re-ranker, hybrid keyword + vector search, or conversation memory.
