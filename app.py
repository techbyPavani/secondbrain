import streamlit as st
import os
from dotenv import load_dotenv
from backend.rag import brain
from backend.ingest import ingest_file, process_web
import time

# Page Config
st.set_page_config(
    page_title="Second Brain | Enterprise Knowledge Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Environment
load_dotenv()

# CSS Customization
st.markdown("""
<style>
    /* Import technical font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

    /* Sidebar Technical Styling */
    [data-testid="stSidebar"] {
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.9rem;
    }

    /* Chat Messages - Enterprise Dark */
    .stChatMessage {
        border: 1px solid #2d3748;
        background-color: #1a1c24; /* Consistent dark card background */
        border-radius: 4px;
        padding: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
        margin-bottom: 1rem;
    }
    
    /* User Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 3px solid #ff0055; /* Cyberpunk Pink/Red accent */
    }
    /* AI Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid #00f2ff; /* Cyan accent */
    }

    /* Buttons: Strong & Flat */
    .stButton button {
        background-color: #3182ce;
        color: white;
        border-radius: 4px;
        border: none;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.8rem;
    }
    .stButton button:hover {
        background-color: #2b6cb0;
    }

    /* Input Fields Border */
    .stTextInput input, .stChatInput textarea {
        border: 1px solid #4a5568 !important;
    }

    /* Expander & Metrics */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem !important;
        border-radius: 4px;
    }
    
    /* Metrics - Fixed Size */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00f2ff; /* Cyan for tech feel */
        font-size: 1.1rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #718096;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    
    /* Status Container */
    [data-testid="stStatusWidget"] {
        background-color: #1a1c24;
        border: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 Second Brain")
    st.caption("ENTERPRISE KNOWLEDGE RETRIEVAL SYSTEM")

with col2:
    if os.getenv("GOOGLE_API_KEY"):
        st.success("● SYSTEM ONLINE")
    else:
        st.error("● OFFLINE (Key Missing)")

st.divider()

# Sidebar - Control Panel
with st.sidebar:
    st.header("CONTROL PANEL")
    
    st.subheader("Ingestion Pipeline")
    
    with st.container():
        ingest_type = st.radio(
            "DATA SOURCE", 
            ["📄 Document (PDF)", "🎙️ Audio (MP3/WAV)", "🌐 Web Resource"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if "Document" in ingest_type or "Audio" in ingest_type:
            uploaded_file = st.file_uploader(
                "Upload Artifact", 
                type=["pdf"] if "Document" in ingest_type else ["mp3", "wav", "m4a"],
                label_visibility="collapsed"
            )
            if uploaded_file and st.button("INITIATE INGESTION", use_container_width=True):
                with st.status("Processing Data Stream...", expanded=True) as status:
                    st.write("Analyizing file structure...")
                    time.sleep(1)
                    file_type = "pdf" if "Document" in ingest_type else "audio"
                    success, msg = ingest_file(uploaded_file, file_type)
                    if success:
                        status.update(label="Ingestion Complete", state="complete", expanded=False)
                        st.success(f"INDEXED: {msg}")
                    else:
                        status.update(label="Ingestion Failed", state="error")
                        st.error(f"ERROR: {msg}")
                        
        elif "Web" in ingest_type:
            url = st.text_input("Resource URL", placeholder="https://...")
            if url and st.button("FETCH & INDEX", use_container_width=True):
                with st.status("Crawling Target...", expanded=True) as status:
                    st.write("Resolving Host...")
                    success, msg = process_web(url)
                    if success:
                        status.update(label="Indexing Complete", state="complete", expanded=False)
                        st.success(f"INDEXED: {msg}")
                    else:
                        status.update(label="Crawling Failed", state="error")
                        st.error(f"ERROR: {msg}")

    st.markdown("---")
    st.subheader("SYSTEM METRICS")
    col_a, col_b = st.columns(2)
    col_a.metric("Model", "Gemini 2.0")
    col_b.metric("Vector DB", "Chroma")
    st.caption("v1.2.0-stable | Latency: ~800ms")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there is context saved with this message, show it
        if "context" in message:
            with st.expander("🔍  VERIFIED CONTEXT SOURCES"):
                st.markdown(message["context"])

# Chat Input
if prompt := st.chat_input("Input query parameter..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call Backend
        try:
            start_time = time.time()
            
            with st.status("EXECUTING NEURAL SEARCH PROTOCOL...", expanded=False) as status:
                st.write("▶ Initializing Query Vector Encoding...")
                time.sleep(0.3) # Simulated micro-latency for effect
                
                st.write("▶ Searching Vector Space (Limit: 3 Nodes)...")
                # Query returns (generator, context_str)
                response_stream, context_used = brain.query(prompt)
                
                st.write("▶ Retrieved Context Blocks.")
                st.write("▶ Hydrating Generative Model (Gemini 2.0)...")
                status.update(label="SYSTEM READY", state="complete", expanded=False)
            
            for chunk in response_stream:
                # Handle error messages (strings)
                if isinstance(chunk, str):
                    full_response += chunk
                # Handle new SDK stream objects
                elif hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            end_time = time.time()
            elapsed = f"{end_time - start_time:.4f}s"
            
            # Show Context Expander for transparency
            if context_used:
                with st.expander(f"🔍 SOURCE TELEMETRY [Latency: {elapsed}]"):
                    st.markdown(context_used)
            else:
                st.caption(f"⚡ Latency: {elapsed} | No Context Retrieved")
            
            # Save interactions to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "context": context_used,
                "latency": elapsed
            })
            
        except Exception as e:
            st.error(f"CRITICAL ERROR: {e}")
            full_response = f"System Failure: {e}"
            st.session_state.messages.append({"role": "assistant", "content": full_response})
