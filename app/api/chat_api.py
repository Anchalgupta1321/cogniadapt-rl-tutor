from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ContentChunks, SourceDocuments
from app.services.vector_dedup import get_embedding, cosine_similarity
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

router = APIRouter(prefix="/chat", tags=["Source Grounded Tutor Chat"])

class TutorChatRequest(BaseModel):
    student_id: str
    query: str
    source_id: Optional[str] = None

class TutorCitation(BaseModel):
    chunk_id: str
    citation_label: str # e.g. "Page 4", "Slide 12", "Timestamp 00:02:15"
    excerpt: str
    relevance_score: float

class TutorChatResponse(BaseModel):
    query: str
    is_grounded: bool
    answer: str
    citations: List[TutorCitation]
    refusal_reason: Optional[str] = None

@router.post("/tutor", response_model=TutorChatResponse)
def grounded_tutor_chat(req: TutorChatRequest, db: Session = Depends(get_db)):
    """
    Source-Grounded AI Tutor Chat.
    1. Retrieves relevant course material chunks via dense vector similarity.
    2. Enforces refusal guardrail for out-of-material / off-topic queries.
    3. Generates grounded explanations with exact inline origin citations ([Page X], [Slide Y], [Timestamp MM:SS]).
    """
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # 1. Fetch chunks (optionally filter by source_id)
    db_query = db.query(ContentChunks)
    if req.source_id:
        db_query = db_query.filter(ContentChunks.source_id == req.source_id)
    chunks = db_query.all()

    if not chunks:
        return TutorChatResponse(
            query=query_text,
            is_grounded=False,
            answer="⚠️ **OUT-OF-MATERIAL REFUSAL**: The provided course materials do not cover this query. Please ingest a PDF, slide deck, or video transcript first.",
            citations=[],
            refusal_reason="No knowledge base available."
        )

    # 2. Compute query embedding & retrieve top matching chunks
    query_vec = get_embedding(query_text)
    chunk_scores = []
    for c in chunks:
        c_vec = get_embedding(c.chunk_text)
        sim = cosine_similarity(query_vec, c_vec)
        chunk_scores.append((c, sim))

    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, sim in chunk_scores[:3] if sim > 0.25]
    max_sim = chunk_scores[0][1] if chunk_scores else 0.0

    # 3. Refusal Guardrail check (Threshold: 0.30 similarity)
    if max_sim < 0.30 or not top_chunks:
        return TutorChatResponse(
            query=query_text,
            is_grounded=False,
            answer="⚠️ **OUT-OF-MATERIAL REFUSAL**: The uploaded course materials do not contain sufficient information to answer this question. To prevent hallucination, outside knowledge is separated from source-backed content.",
            citations=[],
            refusal_reason="Out-of-material query below relevance confidence threshold."
        )

    # 4. Construct Citations and Grounded Answer
    citations = []
    excerpts_str = []
    for idx, c in enumerate(top_chunks):
        # Format citation label based on available metadata
        if c.page_number:
            label = f"Page {c.page_number}"
        elif c.slide_number:
            label = f"Slide {c.slide_number}"
        elif c.video_timestamp:
            label = f"Timestamp {c.video_timestamp}"
        else:
            label = f"Chunk {c.id.split('_')[-1]}"

        citations.append(TutorCitation(
            chunk_id=c.id,
            citation_label=label,
            excerpt=c.chunk_text[:150] + "...",
            relevance_score=round(float(chunk_scores[idx][1]), 3)
        ))
        excerpts_str.append(f"[{label}]: \"{c.chunk_text}\"")

    best_chunk = top_chunks[0]
    best_label = citations[0].citation_label

    grounded_answer = (
        f"Based strictly on your course materials [{best_label}]:\n\n"
        f"{best_chunk.chunk_text}\n\n"
        f"*(Cited from {best_label}, Topic: {best_chunk.topic or 'General'})*"
    )

    return TutorChatResponse(
        query=query_text,
        is_grounded=True,
        answer=grounded_answer,
        citations=citations
    )
