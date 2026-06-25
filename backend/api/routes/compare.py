"""
backend/api/routes/compare.py
Mode C — Draft Policy Comparison (Similarity Search bypass LLM).
Upload a draft PDF → system finds conflicts/overlaps in knowledge base →
returns similarity scores and an AI summary of findings.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.core.logging_config import get_logger
from backend.etl.parser import parse_pdf
from backend.etl.chunker import chunk_pages
from backend.models.schemas import CompareResponse, SimilarityMatch
from backend.rag.retriever import compare_document
from backend.rag.llm import generate_compare_summary

log = get_logger(__name__)
router = APIRouter()

UPLOAD_TMP_DIR = Path("data/uploads")


@router.post("/compare", response_model=CompareResponse)
async def api_compare(file: UploadFile = File(...)):
    """
    Mode C: Draft Policy Comparison.

    Upload a draft PDF. The system will:
    1. Parse the draft into chunks
    2. Search the knowledge base for similar content (no LLM)
    3. Score and classify matches (duplicate / conflict / related)
    4. Generate a concise AI summary of findings

    Returns deterministic similarity scores + optional conflict analysis.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_id = str(uuid.uuid4())
    save_path = UPLOAD_TMP_DIR / f"draft_{tmp_id}_{file.filename}"

    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    try:
        # 1. Parse draft
        pages = parse_pdf(save_path)

        # 2. Chunk draft (don't add to Qdrant)
        doc_id = f"draft_{tmp_id}"
        draft_chunks = chunk_pages(
            pages, doc_id=doc_id, file_name=file.filename
        )

        if not draft_chunks:
            raise HTTPException(status_code=422, detail="Could not extract text from the draft PDF.")

        # 3. Compare against knowledge base
        matches = compare_document(draft_chunks, top_k=5)

        if not matches:
            return CompareResponse(
                draft_name=file.filename,
                max_similarity=0.0,
                matches=[],
                summary="No similar documents were found in the knowledge base.",
            )

        # 4. Classify matches
        similarity_matches = []
        for m in matches:
            score = m.get("rerank_score", m.get("score", 0.0))
            if score >= 0.85:
                overlap_type = "duplicate"
            elif score >= 0.65:
                overlap_type = "conflict"
            else:
                overlap_type = "related"

            similarity_matches.append(
                SimilarityMatch(
                    file_name=m.get("file_name", "Unknown"),
                    page_number=str(m.get("page_number", "?")),
                    similarity_score=round(score, 4),
                    overlap_type=overlap_type,
                    excerpt=m.get("text", "")[:300] + "…",
                )
            )

        max_similarity = max(m.similarity_score for m in similarity_matches)

        # 5. Generate AI summary (uses first draft chunk as representative excerpt)
        summary = generate_compare_summary(
            draft_excerpt=draft_chunks[0]["text"][:800],
            matches=matches[:5],
        )

        return CompareResponse(
            draft_name=file.filename,
            max_similarity=max_similarity,
            matches=similarity_matches,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Compare failed for %s: %s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")
    finally:
        # Cleanup temp file
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass
