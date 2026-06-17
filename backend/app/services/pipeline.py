import time
import uuid
from qdrant_client.models import PointStruct
from app.services.ingestion import extract_text_from_pdfs, chunk_documents
from app.services.embeddings import get_embeddings
from app.services.vector_db import init_vector_db, client

def run_ingestion_pipeline(data_dir: str):
    print("\n" + "="*60)
    print("🚀 STARTING FINANCIAL DOCUMENT INGESTION PIPELINE")
    print("="*60)
    
    # --- Step 1: Database Initialization ---
    print("\n[STEP 1/5] 🛠️ Initializing Vector Database...")
    collection_name = "financial_reports"
    init_vector_db(collection_name)
    
    # --- Step 2: PDF Text Extraction ---
    print(f"\n[STEP 2/5] 📄 Extracting text from raw PDFs in: '{data_dir}'...")
    start_time = time.time()
    raw_content = extract_text_from_pdfs(data_dir)
    print(f"✅ Text extraction complete. Found {len(raw_content)} total pages.")
    
    # --- Step 3: Document Chunking ---
    print("\n[STEP 3/5] ✂️ Splitting documents into structural chunks...")
    chunks = chunk_documents(raw_content)
    total_chunks = len(chunks)
    print(f"✅ Chunking complete. Generated {total_chunks} small text segments.")
    
    # --- Step 4: Batch Processing & Vector Generation ---
    print(f"\n[STEP 4/5] 🧠 Generating BAAI/bge-m3 embeddings and streaming to Qdrant...")
    batch_size = 30
    print(f"-> Processing configurations: Batch Size = {batch_size}")
    
    for i in range(0, total_chunks, batch_size):
        batched_chunks = chunks[i : i + batch_size]
        text_chunks = [chunk.page_content for chunk in batched_chunks]
        
        print(f"   ⚡ Processing mini-batch {i // batch_size + 1}: Chunks {i} to {min(i + batch_size, total_chunks)}...")
        batched_embeddings = get_embeddings(text_chunks)
        
        points = []
        for chunk, vector in zip(batched_chunks, batched_embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # --- Step 5: Database Upsert ---
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    # --- Pipeline Wrap-up ---
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"🎉 SUCCESS: Ingestion pipeline executed completely!")
    print(f"📊 Total Chunks Indexed: {total_chunks}")
    print(f"⏱️ Total Time Elapsed:  {elapsed_time:.2f} seconds")
    print("="*60 + "\n")

if __name__ == "__main__":
    path = "../data/raw"
    run_ingestion_pipeline(path)
