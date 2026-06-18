from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from app.services.llm import generate_response
from dotenv import load_dotenv

load_dotenv() 

app = FastAPI(
    title="Financial RAG Assistant API",
    description="Production web endpoints keeping document embedding models warm in memory."
)

class QueryRequest(BaseModel): 
    question: str 
    top_k: int = 3

@app.get("/")
def home(): 
    return {"status": "healthy", "message": "Home page of rag-doc-assistant"}

@app.post("/api/query")
def query_response(request: QueryRequest): 
    result = generate_response(query=request.question, top_k=request.top_k)
    return {
        "answer": result
    }
