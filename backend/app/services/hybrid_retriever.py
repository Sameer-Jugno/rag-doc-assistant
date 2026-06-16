import os
import pickle
from embeddings import get_embeddings
from vector_db import query_vector_db
from sparse_retriever import query_bm25

BM25_STORAGE_PATH = "../data/bm25_index.pkl"

def computing_rrf(query: str, top_k: int = 3):
    query_embeddings = get_embeddings(query)
    
    vector_results = query_vector_db(query_vector=query_embeddings, top_k=top_k)
    bm25_results = query_bm25(query, storage_path=BM25_STORAGE_PATH, top_k=top_k)
    
    rrf_scoreboard = {}
    
    for rank, point in enumerate(vector_results):
        content = point.payload["page_content"]
        metadata = point.payload["metadata"]
        score = 1.0 / (60.0 + rank)
        rrf_scoreboard[content] = {
            "score": score,
            "metadata": metadata
        }
        
    for rank, doc in enumerate(bm25_results):
        content = doc.page_content
        metadata = doc.metadata
        score = 1.0 / (60.0 + rank)
        if content in rrf_scoreboard:
            rrf_scoreboard[content]["score"] += score
        else:
            rrf_scoreboard[content] = {
                "score": score,
                "metadata": metadata
            }
            
    sorted_items = sorted(
        rrf_scoreboard.items(),
        key=lambda item: item[1]["score"],
        reverse=True
    )[:top_k]
    
    return sorted_items

if __name__ == "__main__":
    query_text = "What is the net income or cloud revenue growth?"
    final_results = computing_rrf(query_text, top_k=3)
    
    print(f"\nTOP HYBRID MATCHES FOR: '{query_text}'")
    for i, (content, data) in enumerate(final_results):
        print(f"\n[RANK {i+1}] Fused Score: {data['score']:.4f}")
        print(f"Source: {data['metadata']['source']} | Page: {data['metadata']['page']}")
        print(f"Content: {content[:140]}...")
