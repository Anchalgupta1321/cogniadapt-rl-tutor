from fastapi import APIRouter, HTTPException
from app.bkt.bkt_engine import bkt_engine
from pydantic import BaseModel
from typing import Dict, Any, Optional

router = APIRouter(prefix="/bkt", tags=["Bayesian Knowledge Tracing"])

class BKTUpdateRequest(BaseModel):
    student_id: str
    topic: str
    is_correct: bool

@router.get("/mastery/{student_id}")
def get_student_mastery(student_id: str):
    """Returns Bayesian Knowledge Tracing topic mastery probabilities and radar chart dataset."""
    return bkt_engine.get_student_mastery(student_id)

@router.post("/update")
def update_bkt_knowledge(req: BKTUpdateRequest):
    """Executes explicit Bayesian Knowledge Tracing update step for a student and topic."""
    next_p_know = bkt_engine.update_knowledge(
        student_id=req.student_id,
        topic=req.topic,
        is_correct=req.is_correct
    )
    return {
        "student_id": req.student_id,
        "topic": req.topic,
        "updated_p_know": next_p_know,
        "mastery_percent": int(next_p_know * 100)
    }
