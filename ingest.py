import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from google import genai
from .rag import brain, GOOGLE_API_KEY

def process_pdf(file_path):
    """Extracts text from PDF."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        from datetime import datetime
        brain.add_document(text, metadata={
            "source": os.path.basename(file_path), 
            "type": "pdf",
            "created_at": datetime.now().isoformat()
        })
        return True, "PDF Processed"
    except Exception as e:
        return False, str(e)

def process_web(url):
    """Extracts text from Web URL with headers and cleaning."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'meta']):
            tag.decompose()
            
        # Get clean text
        text = soup.get_text(separator='\n', strip=True)
        
        if not text or len(text) < 50:
            return False, "Page content too short or empty."

        from datetime import datetime
        brain.add_document(text, metadata={
            "source": url, 
            "type": "web", 
            "created_at": datetime.now().isoformat()
        })
        return True, "Web Page Processed"
    except Exception as e:
        return False, str(e)

def process_audio(file_path):
    """
    Transcribes audio using Gemini 1.5 Flash (New SDK).
    """
    try:
        if not GOOGLE_API_KEY:
            return False, "API Key Missing"
            
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Upload
        print(f"Uploading {file_path}...")
        file_ref = client.files.upload(file=file_path)
        
        # Transcribe
        print("Transcribing...")
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=["Generate a detailed transcript of this audio.", file_ref]
        )
        
        transcript = response.text
        
        from datetime import datetime
        brain.add_document(transcript, metadata={
            "source": os.path.basename(file_path), 
            "type": "audio",
            "created_at": datetime.now().isoformat()
        })
        return True, "Audio Transcribed & Indexed"
    except Exception as e:
        return False, f"Audio Error: {str(e)}"

def ingest_file(uploaded_file, file_type):
    # Save temp file
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    result = (False, "Unknown Type")
    if file_type == "pdf":
        result = process_pdf(temp_path)
    elif file_type == "audio":
        result = process_audio(temp_path)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return result
