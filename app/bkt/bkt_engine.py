import os
import json
from typing import Dict, Any, List, Optional

class BayesianKnowledgeTracer:
    """
    Bayesian Knowledge Tracing (BKT) Engine.
    Models student hidden knowledge state P(L_t) across academic topics using Hidden Markov Model (HMM) parameters.
    
    Parameters per topic:
      - p_init (P(L0)): Prior probability student knows concept (default 0.20)
      - p_transit (P(T)): Probability of learning after attempt (default 0.15)
      - p_guess (P(G)): Probability of guessing correctly without knowing (default 0.20)
      - p_slip (P(S)): Probability of slipping / error despite knowing (default 0.10)
    """
    def __init__(
        self,
        storage_path: str = "data/bkt_states.json",
        p_init: float = 0.20,
        p_transit: float = 0.15,
        p_guess: float = 0.20,
        p_slip: float = 0.10
    ):
        self.storage_path = storage_path
        self.p_init = p_init
        self.p_transit = p_transit
        self.p_guess = p_guess
        self.p_slip = p_slip
        
        # Dict[student_id, Dict[topic, {"p_know": float, "attempts": int, "correct": int}]]
        self.states: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.load_states()

    def load_states(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.states = json.load(f)
            except Exception as e:
                print(f"[BKT] Error loading states: {e}")

    def save_states(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.states, f, indent=2)
        except Exception as e:
            print(f"[BKT] Error saving states: {e}")

    def update_knowledge(self, student_id: str, topic: str, is_correct: bool) -> float:
        """
        Executes BKT Bayesian posterior & transition update steps for a given student and topic.
        Returns the updated P(L_{t+1}) mastery probability.
        """
        student_id = str(student_id).strip()
        topic = str(topic).strip().capitalize() if topic else "General"

        if student_id not in self.states:
            self.states[student_id] = {}

        if topic not in self.states[student_id]:
            self.states[student_id][topic] = {
                "p_know": self.p_init,
                "attempts": 0,
                "correct": 0
            }

        curr_p_know = self.states[student_id][topic]["p_know"]

        # 1. Observation Update (Bayes Rule)
        if is_correct:
            p_obs = (curr_p_know * (1.0 - self.p_slip)) / (
                curr_p_know * (1.0 - self.p_slip) + (1.0 - curr_p_know) * self.p_guess
            )
        else:
            p_obs = (curr_p_know * self.p_slip) / (
                curr_p_know * self.p_slip + (1.0 - curr_p_know) * (1.0 - self.p_guess)
            )

        # 2. Transition Update (Learning step)
        next_p_know = p_obs + (1.0 - p_obs) * self.p_transit
        next_p_know = min(max(float(next_p_know), 0.001), 0.999)

        # Update persistent metrics
        self.states[student_id][topic]["p_know"] = round(next_p_know, 4)
        self.states[student_id][topic]["attempts"] += 1
        if is_correct:
            self.states[student_id][topic]["correct"] += 1

        self.save_states()
        return round(next_p_know, 4)

    def get_student_mastery(self, student_id: str) -> Dict[str, Any]:
        """Returns BKT topic mastery breakdown for a student."""
        student_id = str(student_id).strip()
        student_data = self.states.get(student_id, {})
        
        # Default topic set if student is cold
        if not student_data:
            default_topics = ["Science", "Math", "Shapes", "Vocabulary", "General"]
            breakdown = {}
            for t in default_topics:
                breakdown[t] = {
                    "p_know": self.p_init,
                    "mastery_percent": int(self.p_init * 100),
                    "attempts": 0,
                    "status": "Developing"
                }
            return {
                "student_id": student_id,
                "overall_mastery_avg": int(self.p_init * 100),
                "topics_count": len(default_topics),
                "mastery_breakdown": breakdown
            }

        breakdown = {}
        p_know_sum = 0.0
        for topic, info in student_data.items():
            p_k = info["p_know"]
            p_know_sum += p_k
            status = "Mastered" if p_k >= 0.85 else ("Proficient" if p_k >= 0.60 else "Developing")
            breakdown[topic] = {
                "p_know": p_k,
                "mastery_percent": int(p_k * 100),
                "attempts": info["attempts"],
                "status": status
            }

        avg_mastery = int((p_know_sum / len(student_data)) * 100) if student_data else 20
        return {
            "student_id": student_id,
            "overall_mastery_avg": avg_mastery,
            "topics_count": len(student_data),
            "mastery_breakdown": breakdown
        }

# Global Singleton Instance
bkt_engine = BayesianKnowledgeTracer()
