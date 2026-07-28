# GeM Procurement Assistant (RAG + Ollama + ChromaDB + Streamlit)

A retrieval-augmented generation assistant over Government e-Marketplace (GeM)
policy documents — buyer guidelines, seller registration/validation policies,
service provider SOPs, GST/TDS notifications, integrity pact guidelines, etc.

Fully local and free: **Ollama** for both embeddings and the chat model,
**ChromaDB** as the vector store, **Streamlit** for the UI.

## Architecture

```
PDFs (data/buyer, data/seller, data/service_provider)
        │
        ▼
  ingest.py  ── PyPDFLoader → chunk (RecursiveCharacterTextSplitter)
        │                 → embed (Ollama: nomic-embed-text)
        ▼
   Chroma vector store (chroma_db/, persisted locally)
        │
        ▼
     rag.py  ── retrieve top-k chunks (optionally filtered by category)
        │      → prompt Ollama chat model (llama3.1) with context
        ▼
    app.py (Streamlit) ── chat UI, shows answer + cited source documents/pages
```

## 1. Install Ollama

Download from https://ollama.com and install it, then pull the two models
this project uses:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.1           # chat/generation (or use a smaller model, e.g. mistral, phi3)
```

Make sure the Ollama server is running (it usually starts automatically,
or run `ollama serve`).

## 2. Set up the Python environment

```bash
cd gem-rag-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your documents

Copy your GeM PDFs into the matching folders (already created for you):

```
data/
  buyer/
    buyer-rating-framework-document.pdf
    GeM_GST_TDS_Notification.pdf
    ...
  seller/
    final-user-manual-05-09-2024-2_1726048300.pdf
    ...
  service_provider/
    integrity-pact-guidelines_1741768865.pdf
    ...
```

The folder name each PDF sits in becomes its `category` metadata, which
powers the "search scope" filter in the app.

## 4. Ingest the documents

```bash
python ingest.py
```

This loads every PDF, splits it into ~1000-character chunks (150-char
overlap), embeds each chunk, and persists them into `chroma_db/`. Re-run
this any time you add or update PDFs.

## 5. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).
Ask things like:

- "What is the GST/TDS process for sellers on GeM?"
- "What are the pre-requisites for seller registration?"
- "Summarize the integrity pact guidelines for service providers."
- "What does the technical evaluator do in the buyer workflow?"

Each answer includes an expandable **Sources** panel citing the exact PDF
and page number the model used.

