from sqlalchemy.orm import Session
from app import models
from app.rl.rlaif_evaluator import rlaif_evaluator
import json
import uuid

from app.services.vector_dedup import is_semantic_duplicate

def save_generated_questions(db: Session, questions_data: list):
    """Saves generated questions into database after Dense Vector Semantic Deduplication and RLAIF reward validation."""
    stored = []
    for q in questions_data:
        # 1. Dense Vector Semantic Deduplication check
        is_dup, sim_score, matched_q = is_semantic_duplicate(db, q['question'], threshold=0.82)
        if is_dup:
            print(f"[Vector Dedup] Skipped semantic duplicate (Similarity: {sim_score}): '{q['question']}' matched with '{matched_q}'")
            continue

        # 2. RLAIF Reward Model Filtering
        eval_result = rlaif_evaluator.evaluate_question(q)
        if not eval_result["accepted"]:
            print(f"[RLAIF] Question rejected due to low reward score ({eval_result['composite_reward']}): '{q.get('question')}'")
            continue
            
        options = q.get('options', [])
        question = models.Questions(
            id=str(uuid.uuid4()),
            source_chunk_id=q['source_chunk_id'],
            type=q.get('type', 'MCQ'),
            question=q['question'],
            options=json.dumps(options) if isinstance(options, list) else str(options),
            answer=str(q.get('answer', '')),
            difficulty=q.get('difficulty', 'medium')
        )
        db.add(question)
        stored.append(question)
        
    db.commit()
    return stored

