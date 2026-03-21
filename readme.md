# 🧠 DocuMind

> Ask questions about your documents. Get precise answers with citations. Runs 100% locally — no API keys, no internet required.

---

## Overview

DocuMind is a local RAG (Retrieval-Augmented Generation) system built with Python. Upload any PDF, TXT or Markdown file and ask questions about it in plain English. Every answer includes an exact citation showing which page and section it came from.

---

## Features

- 📄 **Multi-format support** — PDF, TXT and Markdown files
- 🔍 **Semantic search** — finds meaning, not just keywords
- 🎯 **Two-stage retrieval** — vector search + cross-encoder reranking
- 📌 **Exact citations** — every answer cites its source page
- 💬 **Chat history** — conversations saved and restored automatically
- 🔒 **100% local** — Mistral 7B runs on your machine via Ollama

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Mistral 7B via Ollama |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Evaluation

Tested on 10 ground truth Q&A pairs across two chunk sizes:

| Metric | Chunk 512 | Chunk 256 |
|---|---|---|
| Retrieval precision | 90% | **100%** |
| Answer quality | 87% | **93%** |

---

## Setup

**Requirements:** Python 3.11 and [Ollama](https://ollama.ai) installed.

```bash
# Clone and enter the project
git clone https://github.com/Jackson008-OP/DocuMind.git
cd DocuMind

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows


# Install dependencies
pip install -r requirements.txt

# Pull Mistral model
ollama pull mistral

# Launch
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## How It Works

```
Document → Load → Clean → Chunk → Embed → ChromaDB
Query    → Embed → Search → Rerank → Prompt → Mistral → Answer + Citation
```

---

## Usage

1. Click **📎** to upload a document
2. Wait a few seconds for indexing
3. Ask anything about your document
4. Get a cited, grounded answer
