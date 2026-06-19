import os
import streamlit as st
import pypdf
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Page Layout & Branding ---
st.set_page_config(page_title="Corporate Financial RAG Assistant", layout="wide")
st.title("🧠 Corporate Financial 10-K RAG Assistant")
st.markdown("Querying live across **Qdrant Cloud AWS Cluster** and **Google Gemini 2.5 Flash**.")

# --- Cache Heavy ML Models & Clients Warm ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")

@st.cache_resource
def get_qdrant_client():
    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    cloud_api_key = os.getenv("QDRANT_API_KEY")
    if cloud_url and cloud_api_key:
        clean_url = cloud_url if cloud_url.startswith("http") else f"https://{cloud_url}"
        return QdrantClient(url=clean_url, api_key=cloud_api_key, timeout=60, prefer_grpc=False, check_compatibility=False)
    return QdrantClient(url="http://127.0.0.1:6333")

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = load_embedding_model()
qdrant_client = get_qdrant_client()
gemini_client = get_gemini_client()

# --- Core RAG Execution Gateway ---
def query_rag_pipeline(question_str: str, k_val: int) -> str:
    try:
        # 1. Compute text embedding vector coordinates on-the-fly
        query_vector = model.encode([question_str])[0].tolist()
        
        # 2. Retrieve winning semantic chunks straight from Qdrant Cloud across the web
        search_results = qdrant_client.search(
            collection_name="financial_reports",
            query_vector=query_vector,
            limit=k_val
        )
        
        if not search_results:
            return "⚠️ No relevant background context blocks discovered inside Qdrant Cloud."
            
        # 3. Assemble context blocks cleanly
        context_blocks = []
        for point in search_results:
            text_snippet = point.payload.get("page_content", "")
            meta = point.payload.get("metadata", {})
            company = meta.get("company", "Unknown").upper()
            page = meta.get("page", "?")
            context_blocks.append(f"[Source: {company} | Page: {page}]\n{text_snippet}")
            
        unified_context = "\n\n---\n\n".join(context_blocks)
        
        # 4. Enforce strict professional context constraint grounding prompt rules
        system_instruction = (
            "You are an expert financial analyst. Answer the user's question using ONLY the provided document context chunks. "
            "If the context does not contain the answer, reply exactly with: 'I do not have enough information to answer this question.' "
            "Do not hallucinate, do not make assumptions, and do not use outside knowledge."
        )
        
        user_prompt = f"Context:\n{unified_context}\n\nQuestion: {question_str}"
        
        # 5. Invoke Gemini 2.5 Flash for grounded generation
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0  # Lock temperature to 0 for absolute factual precision
            )
        )
        return response.text
        
    except Exception as e:
        return f"❌ Pipeline Execution Error: {str(e)}"

# --- Chat Interface Session Architecture ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Context Match Count (Top K)", min_value=1, max_value=10, value=5)
    if st.button("🧹 Clear Chat Memory"):
        st.session_state.chat_history = []
        st.rerun()

# Draw previous conversation tracks
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["text"])

# Handle live inputs
if user_query := st.chat_input("Ask a financial analysis question (e.g., What is Tesla's automotive revenue?)..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing cloud financial vectors..."):
            ans = query_rag_pipeline(user_query, top_k)
            st.write(ans)
    st.session_state.chat_history.append({"role": "assistant", "text": ans})
