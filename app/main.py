from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.engine.retriever import similarity_search, rag_chat

# 1. FastAPI Setup
app = FastAPI(
    title="Higher Education Policy RAG API",
    description="Professional prototype for searching and answering questions based on MoE policies.",
    version="0.1.0",
)

# Enable CORS for future frontend connection points
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Expand to Streamlit/React origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Pydantic Models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Gov-Intel Policy Engine Online"}

# 3. Endpoints

@app.post("/api/search")
async def api_search(request: SearchRequest):
    try:
        results = await similarity_search(request.query, request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    try:
        response = await rag_chat(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
