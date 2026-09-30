"""Compare vector search vs. keyword (BM25) search on the same query."""
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "get", "gets", "getting", "got",
    "i", "you", "he", "she", "it", "we", "they", "my", "your",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "but",
    "this", "that", "these", "those", "why", "how", "what", "when",
}


def tokenize(text: str) -> list[str]:
    words = text.lower().split()
    return [w for w in words if w not in STOPWORDS]

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
    tokenized = [tokenize(doc) for doc in docs]
    return BM25Okapi(tokenized)


def bm25_search(bm25, docs, metas, query, k=5):
    tokenized_query = tokenize(query)
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

def hybrid_search(docs, metas, bm25, query, k=3, rrf_k=60):
    # Get a longer ranked list from each method (more than we need,
    # so fusion has enough overlap to work with)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    vector_results = collection.query(query_texts=[query], n_results=20)
    vector_ids = vector_results["ids"][0]  # chunk IDs, ranked best-first

    # We need all chunk IDs (from load_all_chunks doesn't give IDs, so pull them too)
    all_data = collection.get()
    id_to_index = {cid: i for i, cid in enumerate(all_data["ids"])}

    # RRF scoring: rank position (0-indexed) -> fusion score
    fusion_scores = {}
    for rank, idx in enumerate(bm25_ranked):
        cid = all_data["ids"][idx]
        fusion_scores[cid] = fusion_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
    for rank, cid in enumerate(vector_ids):
        fusion_scores[cid] = fusion_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)

    top_ids = sorted(fusion_scores, key=lambda c: fusion_scores[c], reverse=True)[:k]
    return [(docs[id_to_index[cid]], metas[id_to_index[cid]], fusion_scores[cid]) for cid in top_ids]

def rerank(query, candidates, top_k=3):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc] for doc, meta, score in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc, meta, rerank_score) for (doc, meta, old_score), rerank_score in scored[:top_k]]

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
    print("=== Hybrid (RRF) top 3 ===")
    for doc, meta, score in hybrid_search(docs, metas, bm25, query, k=3):
        print(f"[fusion score {score:.4f}] {meta['source_type']} {meta['source_id']} chunk {meta['chunk_id']}")
        print(doc[:150])
        print()
    print("=== Hybrid + Reranked top 3 ===")
    hybrid_candidates = hybrid_search(docs, metas, bm25, query, k=10)  # widen to 10 candidates first
    for doc, meta, score in rerank(query, hybrid_candidates, top_k=3):
        print(f"[rerank score {score:.4f}] {meta['source_type']} {meta['source_id']} chunk {meta['chunk_id']}")
        print(doc[:150])
        print()