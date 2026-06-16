import os
import pickle
from rank_bm25 import BM25Okapi
from ingestion import extract_text_from_pdfs, chunk_documents

def build_bm25_index(path: str, storage_path: str):
    raw_content = extract_text_from_pdfs(path)
    chunks = chunk_documents(raw_content)
    
    tokenized_corpus = [chunk.page_content.lower().split(" ") for chunk in chunks]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    index_data = {
        "bm25_engine": bm25,
        "raw_chunks": chunks
    }
    
    print(f"Saving compiled keyword index binaries to: '{storage_path}'...")

    with open(storage_path, "wb") as file:
        pickle.dump(index_data, file)
    
    print("BM25 Sparse Retrieval Index built and saved successfully!")

def query_bm25(query_text: str, storage_path: str, top_k: int = 5):
    if not os.path.exists(storage_path):
        raise FileNotFoundError(f"The {storage_path} not exists.")
    
    with open(storage_path, "rb") as file:
        index_data = pickle.load(file)
    
    tokenize_query = query_text.lower().split(" ")
    top_chunks = index_data["bm25_engine"].get_top_n(tokenize_query, index_data["raw_chunks"], n=top_k)
    
    return top_chunks

if __name__ == "__main__":
    path = "../data/raw"
    storage_path = "../data/bm25_index.pkl"
    
    # build_bm25_index(path, storage_path)
    
    query_text = str(input("Ask Query : "))
    top_k = int(input("Enter top_k parameter : "))
    results = query_bm25(query_text, storage_path, top_k)
    
    for i, chunk in enumerate(results):
        print(f"\n[MATCH {i+1}] Source: {chunk.metadata['source']} | Page: {chunk.metadata['page']}")
        print(f"Content snippet: {chunk.page_content[:150]}...")
