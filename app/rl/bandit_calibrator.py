import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple

class ThompsonSamplingCalibrator:
    """
    Contextual Multi-Armed Bandit using Bayesian Beta-Binomial Distributions 
    (Thompson Sampling) for real-time Question Difficulty Calibration.
    
    Instead of trusting static LLM labels, the calibrator models the true probability P(success)
    of each question using Beta(alpha, beta) priors.
    """
    def __init__(self, storage_path: str = "data/bandit_params.json"):
        self.storage_path = storage_path
        # Dict[question_id, {"alpha": float, "beta": float, "attempts": int}]
        self.params: Dict[str, Dict[str, Any]] = {}
        self.load_params()

    def load_params(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.params = json.load(f)
            except Exception as e:
                print(f"[Bandit] Failed to load calibration state: {e}")

    def save_params(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.params, f, indent=2)
        except Exception as e:
            print(f"[Bandit] Failed to save calibration state: {e}")

    def _get_initial_prior(self, static_difficulty: str) -> Tuple[float, float]:
        """Initial Beta prior parameters based on static difficulty assignment."""
        diff = static_difficulty.lower()
        if diff == "easy":
            return (4.0, 1.0) # High success expectation (~80%)
        elif diff == "hard":
            return (1.0, 4.0) # Low success expectation (~20%)
        else:
            return (2.5, 2.5) # Medium success expectation (~50%)

    def record_attempt(self, question_id: str, is_correct: bool, static_difficulty: str = "medium"):
        """Bayesian update step: Beta(alpha + 1, beta) if correct, Beta(alpha, beta + 1) if incorrect."""
        if question_id not in self.params:
            alpha_0, beta_0 = self._get_initial_prior(static_difficulty)
            self.params[question_id] = {
                "alpha": alpha_0,
                "beta": beta_0,
                "attempts": 0,
                "initial_label": static_difficulty
            }

        if is_correct:
            self.params[question_id]["alpha"] += 1.0
        else:
            self.params[question_id]["beta"] += 1.0

        self.params[question_id]["attempts"] += 1
        self.save_params()

    def sample_success_probability(self, question_id: str, static_difficulty: str = "medium") -> float:
        """Draws a sample from the Beta distribution via Thompson Sampling."""
        if question_id not in self.params:
            alpha_0, beta_0 = self._get_initial_prior(static_difficulty)
            return float(np.random.beta(alpha_0, beta_0))

        item = self.params[question_id]
        return float(np.random.beta(item["alpha"], item["beta"]))

    def get_calibrated_difficulty(self, question_id: str, static_difficulty: str = "medium") -> str:
        """
        Determines calibrated difficulty based on Bayesian expectation E[P(success)] = alpha / (alpha + beta).
        """
        if question_id not in self.params:
            return static_difficulty

        item = self.params[question_id]
        expected_success = item["alpha"] / (item["alpha"] + item["beta"])

        if expected_success > 0.75:
            return "easy"
        elif expected_success < 0.40:
            return "hard"
        else:
            return "medium"

    def get_stats(self) -> Dict[str, Any]:
        """Returns calibration statistics."""
        summary = []
        for q_id, item in list(self.params.items())[:10]:
            e_success = item["alpha"] / (item["alpha"] + item["beta"])
            summary.append({
                "question_id": q_id,
                "attempts": item["attempts"],
                "expected_success_rate": round(e_success, 3),
                "calibrated_difficulty": self.get_calibrated_difficulty(q_id, item["initial_label"]),
                "initial_label": item["initial_label"]
            })
        return {
            "total_items_calibrated": len(self.params),
            "sample_calibrations": summary
        }

# Global Singleton
bandit_calibrator = ThompsonSamplingCalibrator()
