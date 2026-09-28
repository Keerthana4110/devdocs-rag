"""Ask a question against the ingested docs + issues and get a cited answer.

Uses OpenRouter (https://openrouter.ai) as the LLM provider, calling Claude
through OpenRouter's OpenAI-compatible chat completions endpoint.

Usage: python src/ask.py "Why do I get a 422 error on a POST endpoint?"
"""
import os
import sys

import chromadb
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "devdocs"
TOP_K = 5

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# OpenRouter's model slug for Claude. Check https://openrouter.ai/models for
# the current slug if this one is renamed/deprecated.
MODEL = os.environ.get("MODEL", "anthropic/claude-3.5-sonnet")

SYSTEM_PROMPT = """You answer questions about FastAPI using ONLY the provided
context passages, which come from either official documentation or closed
GitHub issues. Every claim must be backed by a passage. Cite each claim using
the tag shown with its passage, e.g. [docs: tutorial/dependencies] or
[issue #4213]. If a doc page and an issue disagree (e.g. a bug fixed in a
newer version), point that out explicitly rather than picking one silently.
If the context does not contain the answer, say so plainly instead of
guessing."""


def retrieve(query: str, k: int = TOP_K):
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    results = collection.query(query_texts=[query], n_results=k)
    passages = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        passages.append({"text": doc, "meta": meta})
    return passages


def tag_for(meta: dict) -> str:
    if meta["source_type"] == "docs":
        return f"[docs: {meta['source_id']}]"
    return f"[issue #{meta['source_id']}]"


def build_context(passages: list[dict]) -> str:
    blocks = []
    for p in passages:
        tag = tag_for(p["meta"])
        blocks.append(f"{tag}\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def call_llm(context: str, query: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in .env")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {query}"},
            ],
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def ask(query: str) -> str:
    passages = retrieve(query)
    if not passages:
        return "No documents ingested yet. Run src/ingest.py first."
    context = build_context(passages)

    answer = call_llm(context, query)
    sources = ", ".join(tag_for(p["meta"]) for p in passages)
    return f"{answer}\n\n[Retrieved from: {sources}]"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/ask.py "your question here"')
    question = " ".join(sys.argv[1:])
    print(ask(question))
