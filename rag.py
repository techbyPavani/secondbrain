import os
import re
import chromadb
from google import genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class SecondBrain:
    def __init__(self):
        # Initialize ChromaDB (persistent)
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        
        # Use HuggingFace Embeddings (Local, Free, Unlimited)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="second_brain_memory",
            embedding_function=self.embedding_fn
        )
        
        # Gemini Client
        if GOOGLE_API_KEY:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            print("Warning: GOOGLE_API_KEY not found.")
            self.client = None

        # Preload Local Fallback Model at startup for instant responses
        print("Initializing local fallback model (Flan-T5-base, ~900MB)...")
        print("This happens once at startup. Please wait 2-3 minutes...")
        try:
            from transformers import pipeline
            self.local_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1  # CPU
            )
            print("✓ Local fallback model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not preload local model: {e}")
            print("Model will be loaded on first use instead.")
            self.local_model = None

    def _smart_chunk(self, text: str, chunk_size: int = 1000, overlap: int = 100):
        """
        Splits text respecting sentence boundaries and paragraphs.
        """
        if not text:
            return []
            
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        final_chunks = []
        
        parts = text.split("\n\n")
        current_chunk = ""
        
        for part in parts:
            if not part.strip(): continue
            
            if len(current_chunk) + len(part) < chunk_size:
                current_chunk += part + "\n\n"
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                current_chunk = part + "\n\n"
                
                while len(current_chunk) > chunk_size:
                    split_idx = -1
                    for sep in separators[1:]:
                         match = current_chunk[:chunk_size].rfind(sep)
                         if match != -1:
                             split_idx = match + len(sep)
                             break
                    
                    if split_idx == -1: split_idx = chunk_size
                    
                    final_chunks.append(current_chunk[:split_idx].strip())
                    current_chunk = current_chunk[split_idx:]
                    
        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return [c for c in final_chunks if c.strip()]

    def add_document(self, text: str, metadata: dict):
        """
        Adds a document to the vector store with timestamp.
        """
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()

        chunks = self._smart_chunk(text)
        if not chunks:
            print("Warning: No valid chunks found to ingest.")
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata.copy() for _ in chunks]
        
        for i, meta in enumerate(metadatas):
            meta["chunk_index"] = i
        
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, user_query: str):
        """
        Queries the Second Brain.
        FALLBACK: If Gemini fails, returns the raw context.
        """
        results = self.collection.query(
            query_texts=[user_query],
            n_results=3
        )
        
        context = ""
        if results['documents']:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            for i in range(len(docs)):
                meta = metas[i]
                timestamp = meta.get('created_at', 'Unknown')[:10]
                source = meta.get('source', 'Unknown')
                
                context += f"- [{timestamp}] ({source}): {docs[i]}\n"
        
        current_time = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""You are a helpful Second Brain AI assistant. Today is {current_time}.

Your task: Answer the user's question based ONLY on the context provided below. 
- Provide a clear, concise, and well-structured answer
- Synthesize and summarize the information - don't just repeat raw context
- If the information is not in the context, say "I don't have that information in my memory"

Context from knowledge base:
{context}

User Question: {user_query}

Your Answer:"""

        # Generator wrapper to catch errors during streaming
        def stream_generator():
            if not self.client:
                 yield f"**⚠️ API Key Missing. Context:**\n\n{context}"
                 return
            
            try:
                # Try to get streaming response from LLM
                stream = self.client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                for chunk in stream:
                    yield chunk
                
            except Exception as e:
                # FALLBACK: Use Local CPU Model
                print(f"Primary LLM Error: {e}. Switching to Local Fallback.") 
                yield from self._local_fallback(context, user_query)

        return stream_generator(), context

    def _local_fallback(self, context: str, user_query: str):
        """
        Local CPU-based fallback using transformers (no API calls).
        Uses Flan-T5 for question answering.
        """
        if not context:
            yield "I couldn't find any relevant documents in your Second Brain."
            return
        
        try:
            # Model is already loaded at startup
            if self.local_model is None:
                yield "**Error: Local model failed to load at startup. Please restart the app.**\n\n"
                yield f"Raw context:\n{context[:500]}..."
                return
            
            yield "**(Local Fallback Model Active)**\n\n"
            
            # Extract document content - handle multi-line documents
            doc_texts = []
            print(f"DEBUG: Context length: {len(context)} chars")
            print(f"DEBUG: First 300 chars of context: {repr(context[:300])}")
            
            lines = context.split('\n')
            current_doc = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if this is a new document marker: "- [date] (source):"
                if line_stripped.startswith('-') and re.search(r'\([^)]+\)\s*:', line_stripped):
                    # Save previous document if exists
                    if current_doc:
                        combined = ' '.join(current_doc)
                        # Clean up multiple spaces
                        combined = re.sub(r'\s+', ' ', combined).strip()
                        if len(combined) > 20:
                            doc_texts.append(combined)
                            print(f"DEBUG: Saved doc ({len(combined)} chars): {combined[:100]}...")
                    
                    # Start new document - extract content after ":"
                    match = re.search(r'\)\s*:\s*(.+)', line_stripped)
                    if match:
                        content = match.group(1).strip()
                        current_doc = [content] if content else []
                    else:
                        current_doc = []
                else:
                    # Continue current document
                    if line_stripped:
                        current_doc.append(line_stripped)
            
            # Don't forget the last document
            if current_doc:
                combined = ' '.join(current_doc)
                combined = re.sub(r'\s+', ' ', combined).strip()
                if len(combined) > 20:
                    doc_texts.append(combined)
                    print(f"DEBUG: Saved last doc ({len(combined)} chars): {combined[:100]}...")
            
            print(f"DEBUG: Total extracted documents: {len(doc_texts)}")

            
            if not doc_texts:
                yield "I found documents but couldn't extract their content properly.\n\n"
                yield f"Raw context:\n{context[:300]}..."
                return
            
            # Combine all document texts
            combined_context = ' '.join(doc_texts)
            
            # Limit context length for the model - shorter is better for T5
            max_context_length = 300  # Reduced from 512
            if len(combined_context) > max_context_length:
                combined_context = combined_context[:max_context_length]
            
            # Craft a very explicit prompt to prevent copy-paste
            prompt = f"""Based on the information below, answer this question in your own words. Do not copy the text directly.

Information: {combined_context}

Question: {user_query}

Write a clear answer using different words than the information above:"""
            
            # Generate answer with proper parameters
            result = self.local_model(
                prompt,
                max_new_tokens=80,  # Reduced to encourage concise answers
                min_new_tokens=15,
                do_sample=True,  # Enable sampling for variety
                temperature=0.8,  # Higher temperature for more creative responses
                top_p=0.9,
                repetition_penalty=1.2  # Penalize repetition
            )
            
            answer = result[0]['generated_text'].strip()
            
            # Validate the answer - check if it's just copying context
            # If more than 70% of words match the context, it's likely a copy
            answer_words = set(answer.lower().split())
            context_words = set(combined_context.lower().split())
            overlap = len(answer_words & context_words) / max(len(answer_words), 1)
            
            if overlap > 0.7 or len(answer) < 20:
                # Model is copying - provide a manual summary instead
                yield "**Summary from your documents:**\n\n"
                # Extract key phrases
                sentences = combined_context.split('.')[:3]
                for i, sent in enumerate(sentences, 1):
                    if sent.strip():
                        yield f"{i}. {sent.strip()}.\n\n"
            else:
                # Good answer - use it
                yield f"{answer}\n\n"
                yield f"*Source: {len(doc_texts)} document(s) from your Second Brain*"
            
        except Exception as local_e:
            # Final fallback: Smart context extraction
            print(f"Local model error: {local_e}")
            yield "**Based on your documents:**\n\n"
            
            # Extract and show key information
            lines = context.strip().split('\n')
            shown = 0
            for line in lines:
                if shown >= 3:
                    break
                if line.strip() and ']:' in line:
                    parts = line.split(']:', 1)
                    if len(parts) == 2:
                        content = parts[1].strip()
                        if content:
                            shown += 1
                            yield f"{shown}. {content[:250]}...\n\n"

brain = SecondBrain()
