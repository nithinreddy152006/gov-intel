import os
import glob
import asyncio
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.readers.file import PyMuPDFReader

# Load environment variables
load_dotenv()

async def main():
    # 1. Configuration: Environment variables
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == "your_google_api_key_here":
        print("Error: Valid GOOGLE_API_KEY environment variable is not set in .env")
        return

    # Check for data directory earlier
    data_dir = "data/policies"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
         print(f"Exception/Warning: No PDF files found in '{data_dir}'. Please add your policy PDFs and run again.")
         return

    # Configure LlamaIndex's global settings with confirmed available models
    try:
        # Using Google GenAI SDK integration with models confirmed in list_models.py
        Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=google_api_key)
        Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001", api_key=google_api_key)
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    except Exception as e:
        print(f"Error initializing Google GenAI models: {e}")
        return

    # 2. Vector Store Setup
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./vector_store")
    os.makedirs(persist_dir, exist_ok=True)
    
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("gov_policies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Document Loading
    print(f"Found {len(pdf_files)} PDF files. Starting loading process...")
    
    loader = PyMuPDFReader()
    all_documents = []
    
    for file_path in pdf_files:
        print(f"Loading {file_path}...")
        try:
            documents = loader.load_data(file_path=file_path)
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_documents:
        print("Error: No documents were successfully loaded. Exiting.")
        return

    print(f"Loaded {len(all_documents)} document pages. Starting chunking and indexing...")

    # 4 & 5. Chunking Strategy & Indexing
    index = VectorStoreIndex.from_documents(
        all_documents, 
        storage_context=storage_context,
        show_progress=True
    )
    
    # 6. Execution output
    print(f"\n--- SUCCESS ---")
    print(f"Successfully processed, chunked, and indexed {len(pdf_files)} PDF documents.")
    print(f"Database Collection Name: 'gov_policies'")

if __name__ == "__main__":
    asyncio.run(main())
