from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3") 

def get_embeddings(text: list) -> list:
    encoding = model.encode(text).tolist()  
    return encoding

if __name__ == "__main__":
    text = ["Tesla 10-K report summary", "NVIDIA AI chips sales"]
    get_embeddings(text)
