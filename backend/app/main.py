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

    try : 
        result = generate_response(query=request.question, top_k=request.top_k)
        return {
            "answer": result
        }
    except Exception as e:
        # Pulls the raw python crash string and sends it over the network to your screen
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ INTERNAL SERVER ERROR TRACE:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Backend Crash Reason: {str(e)}")
