import numpy as np
from typing import Tuple, Optional, List
from sqlalchemy.orm import Session
from app.models import Questions

_sentence_model = None
_model_loaded = False

def get_sentence_model():
    """Lazy load SentenceTransformer model."""
    global _sentence_model, _model_loaded
    if not _model_loaded:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            _model_loaded = True
            print("[Vector Dedup] Successfully loaded SentenceTransformer ('all-MiniLM-L6-v2') model.")
        except Exception as e:
            print(f"[Vector Dedup] SentenceTransformer load warning, using TF-IDF fallback: {e}")
            _model_loaded = True
            _sentence_model = None
    return _sentence_model

def get_embedding(text: str) -> np.ndarray:
    """Computes dense 384-d vector embedding for text."""
    model = get_sentence_model()
    if model is not None:
        vec = model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    # Fallback bag-of-words character n-gram embedding
    text_clean = text.lower().strip()
    char_ngrams = [text_clean[i:i+3] for i in range(len(text_clean)-2)]
    vocab = set(char_ngrams)
    if not vocab:
        return np.zeros(64, dtype=np.float32)
    
    vec = np.zeros(128, dtype=np.float32)
    for ng in char_ngrams:
        idx = hash(ng) % 128
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two normalized vectors."""
    if vec1 is None or vec2 is None or len(vec1) != len(vec2):
        return 0.0
    dot_product = np.dot(vec1, vec2)
    return float(dot_product)

def is_semantic_duplicate(
    db: Session,
    new_question_text: str,
    threshold: float = 0.82
) -> Tuple[bool, float, Optional[str]]:
    """
    Checks if new_question_text is semantically duplicate with any existing question in DB.
    Returns (is_duplicate, max_similarity_score, matching_question_text).
    """
    all_questions = db.query(Questions.question).all()
    if not all_questions:
        return False, 0.0, None

    new_vec = get_embedding(new_question_text)
    max_sim = 0.0
    matched_text = None

    for (existing_text,) in all_questions:
        # Fast string check
        if new_question_text.strip().lower() == existing_text.strip().lower():
            return True, 1.0, existing_text

        # Vector semantic similarity check
        existing_vec = get_embedding(existing_text)
        sim = cosine_similarity(new_vec, existing_vec)
        if sim > max_sim:
            max_sim = sim
            matched_text = existing_text

    is_dup = max_sim >= threshold
    return is_dup, round(max_sim, 4), matched_text
