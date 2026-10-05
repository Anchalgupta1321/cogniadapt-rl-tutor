from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import StudentAnswerSubmit, AnswerResponse
from app.models import Questions, StudentAnswers
from app.services.adaptive_engine import adjust_difficulty, rl_engine

router = APIRouter()

@router.post("/submit-answer", response_model=AnswerResponse)
def submit_answer(ans: StudentAnswerSubmit, db: Session = Depends(get_db)):
    """Submit an answer to a question. Grading algorithm calculates the RL adaptive learning curve."""
    question = db.query(Questions).filter(Questions.id == ans.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
        
    is_correct = (str(ans.selected_answer).strip().lower() == str(question.answer).strip().lower())
    
    record = StudentAnswers(
        student_id=ans.student_id,
        question_id=ans.question_id,
        selected_answer=ans.selected_answer,
        is_correct=is_correct
    )
    db.add(record)
    db.commit()
    
    # Query chunk to extract topic for Bayesian Knowledge Tracing
    from app.models import ContentChunks
    from app.bkt.bkt_engine import bkt_engine
    
    chunk = db.query(ContentChunks).filter(ContentChunks.id == question.source_chunk_id).first()
    topic = chunk.topic if (chunk and chunk.topic) else "General"

    # Trigger Bayesian Knowledge Tracing (BKT) Update Step!
    bkt_engine.update_knowledge(
        student_id=ans.student_id,
        topic=topic,
        is_correct=is_correct
    )
    
    # Adapt using Reinforcement Learning Q-Learning Engine!
    next_diff = adjust_difficulty(
        current_difficulty=question.difficulty,
        correct=is_correct,
        student_id=ans.student_id,
        db=db
    )
    
    return AnswerResponse(
        is_correct=is_correct,
        correct_answer=question.answer if not is_correct else "",
        original_difficulty=question.difficulty,
        next_recommended_difficulty=next_diff,
        source_chunk_id=question.source_chunk_id
    )

@router.get("/rl/stats")
def get_rl_stats():
    """Returns RL Adaptive Engine statistics, iteration count, and Q-table snapshot."""
    return {
        "engine": "Q-Learning RL Adaptive Engine",
        "total_iterations": rl_engine.total_iterations,
        "epsilon": rl_engine.epsilon,
        "alpha": rl_engine.alpha,
        "gamma": rl_engine.gamma,
        "states_tracked": len(rl_engine.q_table),
        "q_table": rl_engine.q_table
    }

