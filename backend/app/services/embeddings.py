from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3") 

def get_embeddings(text) :
    encoding = model.encode(text).tolist()  
    print(type(encoding))
    print(encoding)
if __name__ == "__main__" : 
    testText = ["Tesla 10-K report summary", "NVIDIA AI chips sales"]
    get_embeddings(testText)