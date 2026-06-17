import streamlit as st
import requests
import os                        # New import
from dotenv import load_dotenv 
load_dotenv()

# 1. Page Configuration and Layout
st.set_page_config(
    page_title="Financial 10-K RAG Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Financial 10-K RAG AI Assistant")
st.caption("Chat live with Apple, Microsoft, NVIDIA, and Google 10-K corporate financial reports.")

# 2. Sidebar Configuration Controls
with st.sidebar:
    st.header("⚙️ Retrieval Configurations")
    st.write("Tune how many relevant document chunks are fed into the LLM context brain.")
    
    # User slider to dynamically adjust the top_k parameter
    top_k = st.slider(
        label="Select Top-K Chunks Density",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    
    st.divider()
    st.markdown("### 🟢 System Status")
    st.success("Hybrid Retriever Ready")
    st.success("Gemini API Connected")

# 3. Initialize Persistent Conversation Memory State
# This ensures past text logs do not wipe out when the script refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Render Historical Chat Elements From Memory Log
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Network Route Gateway Hook to FastAPI Backend
def query_rag_backend(question_str: str, k_val: int) -> str:
    base_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    backend_url = f"{base_url}/api/query"

    payload = {
        "question": question_str,
        "top_k": k_val
    }
    try:
        # Add this single line right here to debug the active URL path
        print(f"📡 DEBUG: Streamlit is actively hitting: '{backend_url}'")
        
        response = requests.post(backend_url, json=payload)
        if response.status_code == 200:
            # Extract the raw answer string from your backend server JSON packet
            return response.json().get("answer", "⚠️ Error: Invalid response layout.")
        else:
            return f"❌ Server Error: Received HTTP status code {response.status_code}."
    except requests.exceptions.ConnectionError:
        return "❌ Connection Failure: Unable to reach your FastAPI backend. Ensure uvicorn is running on port 8000!"

# 6. Live Chat Input and Generation Workflow
if user_query := st.chat_input("Ask a financial analysis question (e.g., What is NVIDIA's revenue growth?)..."):
    
    # A. Display user question instantly on screen and log to memory
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # B. Trigger server request pipeline with loading visual spinner
    with st.chat_message("assistant"):
        with st.spinner("🔍 Reviewing 10-K vectors and compiling financial analysis..."):
            ai_response = query_rag_backend(user_query, top_k)
            st.markdown(ai_response)
            
    # C. Lock generated answer to memory logs
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
