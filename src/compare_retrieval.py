"""Compare vector search vs. keyword (BM25) search on the same query."""
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

DB_DIR = "chroma_db"
COLLECTION_NAME = "devdocs"


def load_all_chunks():
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    data = collection.get()  # pulls everything back out
    return data["documents"], data["metadatas"]


def build_bm25_index(docs):
    tokenized = [doc.lower().split() for doc in docs]
    return BM25Okapi(tokenized)


def bm25_search(bm25, docs, metas, query, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], metas[i], scores[i]) for i in top_indices]

def vector_search(docs, metas, query, k=3):
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    results = collection.query(query_texts=[query], n_results=k)
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))



if __name__ == "__main__":
    docs, metas = load_all_chunks()
    print(f"Loaded {len(docs)} chunks\n")

    bm25 = build_bm25_index(docs)

    query = "Why do I get a 422 error on a POST endpoint?"
    print(f"Query: {query}\n")

    print("=== BM25 (keyword) top 3 ===")
    for doc, meta, score in bm25_search(bm25, docs, metas, query, k=3):
        print(f"[score {score:.2f}] {meta['source_type']} {meta['source_id']} chunk {meta['chunk_id']}")
        print(doc[:150])
        print()

    print("=== Vector search top 3 ===")
    for doc, meta, distance in vector_search(docs, metas, query, k=3):
        print(f"[distance {distance:.3f}] {meta['source_type']} {meta['source_id']} chunk {meta['chunk_id']}")
        print(doc[:150])
        print()