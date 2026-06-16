import os
import pickle
from rank_bm25 import BM25Okapi
from ingestion import extract_text_from_pdfs, chunk_documents



def build_bm25_index(path : list, storage_path : list) : 
    pass 
    raw_content = extract_text_from_pdfs(path) 

    chunks = chunk_documents(raw_content) 

    tokenized_corpus = [chunk.page_content.lower().split(" ") for chunk in chunks] 

    bm25 = BM25Okapi(tokenized_corpus) 

    index_data = {
        "bm25_engine": bm25,
        "raw_chunks": chunks
    }

    # 5. Write the index data packet permanently to your laptop hard drive
    print(f"Saving compiled keyword index binaries to: '{storage_path}'...")
    with open(storage_path, "wb") as file:
        pickle.dump(index_data, file)
        
    print("✅ BM25 Sparse Retrieval Index built and saved successfully!")

    
if __name__ == "__main__" : 
    path="../data/raw"
    storage_path = "../data/bm25_index.pkl"
    build_bm25_index(path, storage_path)
