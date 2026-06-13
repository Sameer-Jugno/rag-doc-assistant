from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams


client = QdrantClient(url="http://localhost:6333")


def init_vector_db(collection_name) : 
    
    existing_collections = [collection.name   for collection in client.get_collections().collections ]

    if  collection_name not in existing_collections : 
        print(f"creating a collection of name : {collection_name}...")
        client.create_collection(
            collection_name=collection_name, 
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"creation of collection with name : {collection_name} is successfull...")
    else : 
        print(f"collection of name : {collection_name} already exist...")
        print(f"returning collection of name : {collection_name}...")
        collection_response  = client.get_collections()
        print("Database Response:", collection_response)
        return collection_response


    # This calls the running Docker database API to fetch existing vector tables
    # collections_response = client.get_collections()
    
if __name__ == "__main__" :
    collection_name = "financial_reports" 
    init_vector_db(collection_name) 