import os
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import QueryBundle

# Load environment variables
load_dotenv()

# Global setup flags to ensure one-time initialization
_initialized = False
_index = None
_reranker = None

def initialize_engine():
    global _initialized, _index, _reranker
    if _initialized:
        return

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == "your_google_api_key_here":
        raise ValueError("Valid GOOGLE_API_KEY environment variable is not set in .env")

    # 1. Global Setup (Using confirmed models)
    Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=google_api_key)
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001", api_key=google_api_key)

    # 2. Reranking
    _reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-v2-m3")

    # 3. DB Connection
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./vector_store")
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_collection("gov_policies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    _index = VectorStoreIndex.from_vector_store(vector_store)
    _initialized = True

async def similarity_search(query_text: str, top_k: int = 5):
    initialize_engine()
    retriever = _index.as_retriever(similarity_top_k=top_k * 2) 
    candidate_nodes = await retriever.aretrieve(query_text)
    reranked_nodes = _reranker.postprocess_nodes(
        candidate_nodes, 
        query_bundle=QueryBundle(query_text)
    )
    final_nodes = reranked_nodes[:top_k]
    
    results = []
    for node_with_score in final_nodes:
        node = node_with_score.node
        metadata = node.metadata
        results.append({
            "text": node.get_content(),
            "score": node_with_score.score,
            "file_name": metadata.get("file_name", "Unknown File"),
            "page_number": metadata.get("page_label", "Unknown Page"),
        })
    return results

async def rag_chat(query_text: str):
    initialize_engine()
    query_engine = _index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[_reranker]
    )
    response = await query_engine.aquery(query_text)
    sources = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        metadata = node.metadata
        sources.append({
            "extract": node.get_content()[:200] + "...", 
            "file_name": metadata.get("file_name", "Unknown File"),
            "page_number": metadata.get("page_label", "Unknown Page")
        })
    return {
        "answer": str(response),
        "sources": sources
    }
