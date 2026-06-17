import os
from google import genai
from dotenv import load_dotenv
from hybrid_retriever import computing_rrf

load_dotenv()

client = genai.Client()

def generate_response(query: str, top_k: int = 3):
    results = computing_rrf(query, top_k)
    
    context_chunks = []
    for content, data in results:
        context_chunks.append(content)
        
    context_str = "\n---\n".join(context_chunks)
    
    prompt = (
        f"You are an expert financial analyst. Answer the user question based strictly on the provided context. "
        f"If the answer cannot be found in the context, state that you do not have enough information.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}"
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return response.text

if __name__ == "__main__":
    query_text = "What is the net income or cloud revenue growth?"
    try:
        answer = generate_response(query_text, top_k=3)
        print("\nGEMINI GENERATED RESPONSE:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
    except Exception as e:
        print("\n❌ Error connecting to Gemini API:", e)
