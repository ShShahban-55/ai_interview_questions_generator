"""Build the Chroma vector store (the RAG knowledge base) from knowledge/knowledge.json.

Usage:
    python build_vectorstore.py
"""
import json
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

KNOWLEDGE = "knowledge/knowledge.json"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    items = json.loads(Path(KNOWLEDGE).read_text(encoding="utf-8"))
    docs = [
        Document(page_content=i["text"], metadata={"topic": i["topic"], "level": i["level"]})
        for i in items
    ]

    # Start clean so re-running does not duplicate documents.
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    print(f"Indexed {len(docs)} documents into '{PERSIST_DIR}/'")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    for query in ["Django ORM optimization", "system design for a senior candidate"]:
        print(f"\nQuery: {query}")
        for d in retriever.invoke(query):
            print(f"  [{d.metadata['topic']} | {d.metadata['level']}] {d.page_content[:90]}...")


if __name__ == "__main__":
    main()
