"""Chunk and embed docs pages + GitHub issues into Chroma.

Each chunk carries metadata (source_type, source_id, extra) so answers can be
cited back to either a doc page (e.g. "docs: tutorial/dependencies") or a
GitHub issue (e.g. "issue #4213").
"""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = Path("data/raw_docs")
ISSUES_DIR = Path("data/raw_issues")
DB_DIR = "chroma_db"
COLLECTION_NAME = "devdocs"
CHUNK_SIZE = 500  # characters; override per-experiment during week 2 comparison
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def load_docs() -> list[dict]:
    items = []
    for f in sorted(DOCS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        items.append({
            "source_type": "docs",
            "source_id": data["slug"],
            "text": data["text"],
            "url": f"https://fastapi.tiangolo.com/{data['slug']}/",
        })
    return items


def load_issues() -> list[dict]:
    items = []
    for f in sorted(ISSUES_DIR.glob("issue_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        text = f"Title: {data['title']}\n\n{data['body']}"
        if data.get("comments"):
            text += "\n\nDiscussion:\n" + "\n---\n".join(data["comments"][:3])
        items.append({
            "source_type": "issue",
            "source_id": str(data["number"]),
            "text": text,
            "url": data["url"],
        })
    return items


def main() -> None:
    docs = load_docs()
    issues = load_issues()
    print(f"Loaded {len(docs)} doc pages, {len(issues)} issues")
    if not docs and not issues:
        raise SystemExit(
            "No source data found. Run fetch_docs.py and fetch_issues.py first."
        )

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)

    ids, chunk_docs, metas = [], [], []
    for item in docs + issues:
        chunks = chunk_text(item["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{item['source_type']}_{item['source_id']}_{i}")
            chunk_docs.append(chunk)
            metas.append({
                "source_type": item["source_type"],
                "source_id": item["source_id"],
                "chunk_id": i,
                "url": item["url"],
            })

    for i in range(0, len(ids), 500):
        collection.add(
            ids=ids[i:i + 500],
            documents=chunk_docs[i:i + 500],
            metadatas=metas[i:i + 500],
        )
    print(f"Ingested {len(ids)} chunks from {len(docs)} docs + {len(issues)} issues into {DB_DIR}/")


if __name__ == "__main__":
    main()
