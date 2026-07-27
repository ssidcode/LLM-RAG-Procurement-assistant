"""
rag.py
Core retrieval-augmented generation logic: given a question (and optional
category filter), retrieve relevant chunks from Chroma and ask an Ollama
chat model to answer using only that context, citing sources.
"""
from __future__ import annotations
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "gem_procurement_docs"

EMBED_MODEL = "nomic-embed-text"
# Chat/generation model. Pull it first:
#   ollama pull llama3.1
LLM_MODEL = "llama3.1"

TOP_K = 5

SYSTEM_PROMPT = """You are a procurement assistant for the Government e-Marketplace (GeM).
Answer the user's question using ONLY the context below, which comes from official
GeM policy documents, SOPs, and user manuals.

Rules:
- If the context does not contain the answer, say you don't have enough information
  in the provided documents rather than guessing.
- Be precise and cite the source document for each claim, e.g. (Source: file.pdf, page 3).
- Keep answers focused and well-organized. Use bullet points for multi-part answers.

Context:
{context}
"""


def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def format_context(docs):
    """Turn retrieved chunks into a single context string with source tags."""
    blocks = []
    for d in docs:
        source = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        category = d.metadata.get("category", "unknown")
        blocks.append(
            f"[Source: {source}, page {page}, category: {category}]\n{d.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, category: str | None = None, k: int = TOP_K):
    """
    Retrieve relevant chunks and generate an answer.
    Returns (answer_text, list_of_source_docs).
    """
    vectorstore = get_vectorstore()

    search_kwargs = {"k": k}
    if category and category != "all":
        search_kwargs["filter"] = {"category": category}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)

    if not docs:
        return (
            "I couldn't find any relevant information in the ingested documents "
            "for this question. Try rephrasing, or check that ingest.py has been run.",
            [],
        )

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content, docs


if __name__ == "__main__":
    # Quick CLI test
    q = input("Ask a procurement question: ")
    ans, sources = answer_question(q)
    print("\n--- ANSWER ---\n")
    print(ans)
    print("\n--- SOURCES ---")
    for d in sources:
        print(f"- {d.metadata.get('source_file')} (page {d.metadata.get('page')}, "
              f"{d.metadata.get('category')})")
