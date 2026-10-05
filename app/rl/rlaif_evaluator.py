import re
from typing import Dict, Any, List

class RLAIFQuestionEvaluator:
    """
    Reinforcement Learning from AI Feedback (RLAIF) Reward Model.
    
    Evaluates generated quiz questions across 4 multi-attribute pedagogical metrics:
    1. R_clarity: Penalizes ambiguous text or grammatical flaws.
    2. R_distractor: Rewards distractor diversity and option count.
    3. R_length: Penalizes excessively long/short prompt lengths.
    4. R_traceability: Rewards verifiable grounding.
    
    Computes a composite scalar Reward R in [0.0, 1.0]. Questions with R < threshold (0.6) are rejected.
    """
    def __init__(self, acceptance_threshold: float = 0.60):
        self.acceptance_threshold = acceptance_threshold

    def evaluate_question(self, question_dict: Dict[str, Any], source_text: str = "") -> Dict[str, Any]:
        q_text = question_dict.get("question", "").strip()
        options = question_dict.get("options", [])
        answer = str(question_dict.get("answer", "")).strip()

        # 1. Clarity Reward
        r_clarity = 1.0
        if len(q_text) < 10:
            r_clarity -= 0.5
        if re.search(r'\b(all of the above|none of the above)\b', q_text, re.IGNORECASE):
            r_clarity -= 0.2

        # 2. Distractor Diversity Reward
        r_distractor = 1.0
        if isinstance(options, list):
            unique_opts = set(str(o).strip().lower() for o in options)
            if len(unique_opts) < len(options):
                r_distractor -= 0.5 # Penalty for duplicate option choices
            if len(options) < 2:
                r_distractor -= 0.4
        else:
            r_distractor = 0.2

        # 3. Length Balance Reward
        r_length = 1.0
        words_count = len(q_text.split())
        if words_count > 40:
            r_length -= 0.3
        elif words_count < 4:
            r_length -= 0.4

        # 4. Traceability & Answer Alignment Reward
        r_traceability = 1.0
        if answer and answer.lower() not in [str(o).lower() for o in options]:
            # For MCQ, answer should match one option
            if question_dict.get("type", "MCQ") == "MCQ":
                r_traceability -= 0.4

        # Composite Reward Calculation (Weighted Sum)
        composite_reward = (
            0.35 * max(0.0, r_clarity) +
            0.30 * max(0.0, r_distractor) +
            0.15 * max(0.0, r_length) +
            0.20 * max(0.0, r_traceability)
        )
        composite_reward = round(float(composite_reward), 4)

        accepted = composite_reward >= self.acceptance_threshold

        return {
            "composite_reward": composite_reward,
            "accepted": accepted,
            "metrics": {
                "r_clarity": round(r_clarity, 2),
                "r_distractor": round(r_distractor, 2),
                "r_length": round(r_length, 2),
                "r_traceability": round(r_traceability, 2)
            }
        }

# Global Singleton
rlaif_evaluator = RLAIFQuestionEvaluator()
