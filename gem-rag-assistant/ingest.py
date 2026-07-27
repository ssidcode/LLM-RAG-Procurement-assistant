"""
ingest.py
Walks the data/ folder (buyer/, seller/, service_provider/ subfolders),
loads each PDF, splits it into chunks, embeds the chunks with an Ollama
embedding model, and persists everything into a local Chroma vector store.

Run this once whenever you add/update PDFs:
    python ingest.py
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "gem_procurement_docs"

# Model used purely for embeddings (small + fast). Pull it first:
#   ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents():
    """Load every PDF under data/<category>/ and tag it with metadata."""
    all_docs = []
    categories = [d for d in DATA_DIR.iterdir() if d.is_dir()]

    if not categories:
        print(f"No category folders found in {DATA_DIR}. "
              f"Expected e.g. data/buyer, data/seller, data/service_provider.")
        return all_docs

    for category_dir in categories:
        category = category_dir.name
        pdf_files = list(category_dir.glob("*.pdf"))
        print(f"[{category}] found {len(pdf_files)} PDF(s)")

        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
            except Exception as e:
                print(f"  ! Failed to load {pdf_path.name}: {e}")
                continue

            for page in pages:
                # Attach useful metadata for filtering + citations later
                page.metadata["category"] = category
                page.metadata["source_file"] = pdf_path.name
                # PyPDFLoader already sets metadata["page"] (0-indexed)
            all_docs.extend(pages)
            print(f"  - {pdf_path.name}: {len(pages)} page(s)")

    return all_docs

def clean_text(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    print(f"\nSplit {len(documents)} pages into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


BATCH_SIZE = 20

def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"embedded {i + len(batch)}/{len(chunks)}")
    return vectorstore


def main():
    print(f"Loading documents from {DATA_DIR} ...\n")
    documents = load_documents()

    if not documents:
        print("\nNo documents were loaded. Put your GeM PDFs into "
              "data/buyer, data/seller, and data/service_provider, then re-run.")
        return

    chunks = chunk_documents(documents)
    build_vectorstore(chunks)
    print("\nIngestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
