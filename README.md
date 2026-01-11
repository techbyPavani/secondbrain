# 🧠 Second Brain AI Companion

**Second Brain** is a personal AI assistant capable of ingesting your documents, audio files, and web links to answer questions based on your personal knowledge base. It uses a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, context-aware answers.

## ✨ Key Features

*   **Multi-Modal Ingestion**:
    *   **📄 Documents**: Upload PDFs to index text content instantly.
    *   **🎙️ Audio**: Upload MP3/M4A/WAV files. The system uses **Google Gemini 1.5 Flash** (via the new `google-genai` SDK) to transcribe audio into text for searching.
    *   **🌐 Web**: Paste URLs (e.g., Wikipedia, Blogs) to scrape and index content.
*   **Smart Memory**:
    *   **Vector Search**: Uses **HuggingFace `all-MiniLM-L6-v2`** embeddings (runs locally, free, no limits) to find semantically relevant info.
    *   **Temporal Awareness**: Knows when a document was added and understands time-relative queries (e.g., "What did I add last week?").
*   **Robust Architecture**:
    *   **Fallback Mode**: If the LLM (Gemini) hits a Rate Limit or goes down, the system automatically falls back to displaying the raw retrieved context, ensuring you *always* get an answer.
    *   **Graceful Error Handling**: API errors are suppressed in the UI to keep the chat experience clean.
*   **Modern UI**: Built with **Streamlit**, featuring a responsive chat interface, sidebar controls, and real-time streaming tokens.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (Python)
*   **LLM**: Google Gemini 2.0 Flash (via `google-genai`)
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) via Hugging Face
*   **Vector Store**: ChromaDB (Local Persistent Storage)
*   **Ingestion**: `pypdf`, `beautifulsoup4`, `ffmpeg` (for audio processing support)

## 🚀 Getting Started

### Prerequisites
*   Python 3.10 or higher
*   A Google Gemini API Key (Get one [here](https://aistudio.google.com/app/apikey))

### Installation

1.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.



