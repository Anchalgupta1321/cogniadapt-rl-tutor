from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import StudentAnswers, Questions, ContentChunks
from app.bkt.bkt_engine import bkt_engine
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/revision", tags=["Post-Quiz Report & Flashcards"])

class DiagnosticReportRequest(BaseModel):
    student_id: str

class Flashcard(BaseModel):
    id: str
    topic: str
    concept_title: str
    front_prompt: str
    back_explanation: str
    source_citation: str

@router.post("/report")
def generate_post_quiz_report(req: DiagnosticReportRequest, db: Session = Depends(get_db)):
    """
    Generates a Post-Assessment Performance & Diagnostic Misconception Report.
    Identifies weak topics (via BKT), accuracy rates, and likely misconceptions.
    """
    answers = db.query(StudentAnswers).filter(StudentAnswers.student_id == req.student_id).all()
    if not answers:
        return {
            "student_id": req.student_id,
            "status": "No submission history available yet. Please complete a quiz first.",
            "overall_accuracy": "0%",
            "weak_topics": [],
            "strong_topics": [],
            "likely_misconceptions": []
        }

    total_attempts = len(answers)
    correct_count = sum(1 for a in answers if a.is_correct)
    accuracy_pct = round((correct_count / float(total_attempts)) * 100, 1)

    # Fetch BKT mastery profile
    bkt_profile = bkt_engine.get_student_mastery(req.student_id)
    breakdown = bkt_profile.get("mastery_breakdown", {})

    weak_topics = []
    strong_topics = []
    misconceptions = []

    for topic, stats in breakdown.items():
        p_k = stats["p_know"]
        if p_k < 0.60:
            weak_topics.append({
                "topic": topic,
                "mastery_percent": stats["mastery_percent"],
                "status": stats["status"]
            })
            misconceptions.append(f"Gaps identified in core concepts of '{topic}'. Recommend reviewing flashcards and cited course notes.")
        else:
            strong_topics.append({
                "topic": topic,
                "mastery_percent": stats["mastery_percent"],
                "status": stats["status"]
            })

    return {
        "student_id": req.student_id,
        "total_quizzes_attempted": total_attempts,
        "overall_accuracy": f"{accuracy_pct}%",
        "bkt_average_mastery": f"{bkt_profile['overall_mastery_avg']}%",
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "likely_misconceptions": misconceptions,
        "recommended_action": "Study targeted flashcards for weak topics to improve Bayesian mastery probability."
    }

@router.post("/flashcards", response_model=List[Flashcard])
def generate_targeted_flashcards(req: DiagnosticReportRequest, db: Session = Depends(get_db)):
    """
    Generates targeted study revision flashcards prioritized for the student's weakest BKT topics.
    """
    bkt_profile = bkt_engine.get_student_mastery(req.student_id)
    breakdown = bkt_profile.get("mastery_breakdown", {})
    
    # Sort topics by lowest BKT p_know
    sorted_topics = sorted(breakdown.items(), key=lambda x: x[1]["p_know"])
    target_topics = [t for t, stats in sorted_topics[:3]]

    flashcards = []
    card_id = 1
    for topic in target_topics:
        # Query chunks for this topic
        chunks = db.query(ContentChunks).filter(ContentChunks.topic.ilike(f"%{topic}%")).limit(2).all()
        if not chunks:
            chunks = db.query(ContentChunks).limit(2).all()

        for c in chunks:
            label = f"Page {c.page_number}" if c.page_number else (f"Slide {c.slide_number}" if c.slide_number else f"Chunk {c.id}")
            flashcards.append(Flashcard(
                id=f"FC-{card_id:03d}",
                topic=topic,
                concept_title=f"Core Concept: {topic}",
                front_prompt=f"Explain the primary mechanism discussed in {topic} [{label}].",
                back_explanation=f"Key Insight: {c.chunk_text[:200]}...",
                source_citation=f"Cited from {label}"
            ))
            card_id += 1

    return flashcards
