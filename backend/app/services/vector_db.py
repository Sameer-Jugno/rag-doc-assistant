from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams
from embeddings import get_embeddings

client = QdrantClient(url="http://localhost:6333")

def init_vector_db(collection_name):
    existing_collections = [collection.name for collection in client.get_collections().collections]

    if collection_name not in existing_collections: 
        print(f"creating a collection of name : {collection_name}...")
        client.create_collection(
            collection_name=collection_name, 
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"creation of collection with name : {collection_name} is successfull...")
    else: 
        print(f"collection of name : {collection_name} already exist...")
        print(f"returning collection of name : {collection_name}...")
        collection_response = client.get_collections()
        print("Database Response:", collection_response)
        return collection_response

    # This calls the running Docker database API to fetch existing vector tables
    # collections_response = client.get_collections()
    
def query_vector_db(query_vector: list, collection_name: str = "financial_reports", top_k: int = 5) -> list:
    results = client.query_points(
        collection_name=collection_name, 
        limit=top_k, 
        query=query_vector
    )
    return results.points 

if __name__ == "__main__":
    collection_name = "financial_reports" 
    init_vector_db(collection_name) 

    query = str(input("Ask query : ")) 
    top_k = int(input("Enter top_k parameter : ")) 
    
    query_vector = get_embeddings([query])[0] 
    result = query_vector_db(query_vector, collection_name, top_k)   

    print("\n" + "="*60)
    print(f"SEMANTIC SEARCH RESULTS FOR: '{query}'")
    print("="*60)
    
    for i, point in enumerate(result):
        source = point.payload["metadata"]["source"]
        page = point.payload["metadata"]["page"]
        company = point.payload["metadata"]["company"].upper()
        
        clean_text = " ".join(point.payload["page_content"].split())
        
        print(f"\n[MATCH {i+1}] COMPANY: {company} | FILE: {source} | PAGE: {page}")
        print(f" Relevance Alignment Score: {point.score:.4f}")
        print(f" Content Snippet: {clean_text[:160]}...")
        print("-" * 60)
