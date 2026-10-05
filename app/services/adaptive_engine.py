import os
import json
import random
from typing import Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

class RLAdaptiveEngine:
    """
    Reinforcement Learning (Q-Learning) Adaptive Engine for Student Difficulty Recommendation.
    
    State: (current_difficulty_idx, accuracy_bucket, streak_bucket)
    Action: 0 = recommend 'easy', 1 = recommend 'medium', 2 = recommend 'hard'
    Reward: Evaluated based on Zone of Proximal Development (ZPD) target engagement (~70-80% accuracy target).
    """

    def __init__(self, storage_path: str = "data/rl_q_table.json", alpha: float = 0.15, gamma: float = 0.9, epsilon: float = 0.1):
        self.storage_path = storage_path
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy exploration rate
        self.total_iterations = 0
        self.q_table: Dict[str, list] = {}
        self._initialize_and_load()

    def _default_q_values(self, diff_idx: int, acc_bucket: int, streak_bucket: int) -> list:
        """Domain-informed Q-value initialization for warm-start cold states."""
        # Actions: [Q(easy), Q(medium), Q(hard)]
        if acc_bucket == 2 or streak_bucket == 1: # High accuracy or hot streak
            target = min(diff_idx + 1, 2)
        elif acc_bucket == 0 or streak_bucket == -1: # Low accuracy or cold streak
            target = max(diff_idx - 1, 0)
        else:
            target = diff_idx

        q_vals = [0.0, 0.0, 0.0]
        q_vals[target] = 1.0
        return q_vals

    def _initialize_and_load(self):
        """Loads Q-table from disk or initializes default state space."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.q_table = data.get("q_table", {})
                    self.total_iterations = data.get("total_iterations", 0)
                    return
            except Exception as e:
                print(f"[RL Engine] Error loading Q-table, resetting to default: {e}")

        # Build state space default initialization
        self.q_table = {}
        for diff_idx in range(3):
            for acc_bucket in range(3): # 0: low (<0.4), 1: med (0.4-0.75), 2: high (>=0.75)
                for streak_bucket in [-1, 0, 1]: # -1: cold streak, 0: neutral, 1: hot streak
                    state_key = f"{diff_idx}_{acc_bucket}_{streak_bucket}"
                    self.q_table[state_key] = self._default_q_values(diff_idx, acc_bucket, streak_bucket)
        self.save_q_table()

    def save_q_table(self):
        """Persists Q-table state to disk."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump({
                    "total_iterations": self.total_iterations,
                    "q_table": self.q_table
                }, f, indent=2)
        except Exception as e:
            print(f"[RL Engine] Failed to save Q-table: {e}")

    def get_state_key(self, difficulty: str, recent_accuracy: float, streak: int) -> str:
        """Map raw student stats into discrete RL state space key."""
        try:
            diff_idx = DIFFICULTY_LEVELS.index(str(difficulty).lower())
        except ValueError:
            diff_idx = 1 # Fallback medium

        if recent_accuracy < 0.4:
            acc_bucket = 0
        elif recent_accuracy < 0.75:
            acc_bucket = 1
        else:
            acc_bucket = 2

        if streak <= -2:
            streak_bucket = -1
        elif streak >= 2:
            streak_bucket = 1
        else:
            streak_bucket = 0

        return f"{diff_idx}_{acc_bucket}_{streak_bucket}"

    def compute_reward(self, current_diff: str, is_correct: bool, streak: int, recent_acc: float) -> float:
        """
        ZPD (Zone of Proximal Development) Reward Function.
        Encourages challenge alignment and penalizes boredom/frustration.
        """
        reward = 1.0 if is_correct else -0.5

        # Bonus for mastering hard difficulty
        if is_correct and current_diff.lower() == "hard":
            reward += 0.8

        # Frustration penalty (failing hard questions on cold streak)
        if not is_correct and current_diff.lower() == "hard" and streak <= -2:
            reward -= 1.0

        # Boredom penalty (correct on easy questions when student accuracy is high)
        if is_correct and current_diff.lower() == "easy" and recent_acc >= 0.75:
            reward -= 0.5

        return reward

    def select_action(self, state_key: str) -> int:
        """Epsilon-greedy policy for action selection."""
        if state_key not in self.q_table:
            diff_idx, acc_b, streak_b = [int(x) for x in state_key.split("_")]
            self.q_table[state_key] = self._default_q_values(diff_idx, acc_b, streak_b)

        if random.random() < self.epsilon:
            return random.choice([0, 1, 2]) # Explore
        
        # Exploit: Action with maximum Q-value
        q_values = self.q_table[state_key]
        max_v = max(q_values)
        best_actions = [i for i, v in enumerate(q_values) if v == max_v]
        return random.choice(best_actions)

    def update_q_value(self, state_key: str, action_idx: int, reward: float, next_state_key: str):
        """Bellman Q-Learning update equation."""
        if next_state_key not in self.q_table:
            diff_idx, acc_b, streak_b = [int(x) for x in next_state_key.split("_")]
            self.q_table[next_state_key] = self._default_q_values(diff_idx, acc_b, streak_b)

        current_q = self.q_table[state_key][action_idx]
        max_next_q = max(self.q_table[next_state_key])

        # TD update equation
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = round(new_q, 4)
        self.total_iterations += 1
        self.save_q_table()

    def process_student_feedback(
        self,
        student_id: str,
        current_difficulty: str,
        is_correct: bool,
        db: Optional[Session] = None
    ) -> str:
        """
        Executes an RL update step given student response history and returns the recommended next difficulty level.
        """
        recent_acc = 0.5
        streak = 0

        # Query student history if DB session is available
        if db and student_id:
            from app.models import StudentAnswers
            answers = db.query(StudentAnswers).filter(StudentAnswers.student_id == student_id).all()
            if answers:
                total = len(answers)
                corrects = sum(1 for a in answers if a.is_correct)
                recent_acc = corrects / float(total)

                # Compute streak
                recent_answers = answers[-5:]
                streak = 0
                for a in reversed(recent_answers):
                    if a.is_correct:
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                    else:
                        if streak <= 0:
                            streak -= 1
                        else:
                            break

        # Calculate PyTorch DQN state vectors
        from app.rl.dqn_agent import dqn_agent
        from app.rl.bandit_calibrator import bandit_calibrator
        
        diff_idx = DIFFICULTY_LEVELS.index(current_difficulty.lower()) if current_difficulty.lower() in DIFFICULTY_LEVELS else 1
        state_vec = dqn_agent.get_state_vector(diff_idx, recent_acc, streak, attempts_count=5)

        # State before action evaluation
        state_key = self.get_state_key(current_difficulty, recent_acc, streak)
        reward = self.compute_reward(current_difficulty, is_correct, streak, recent_acc)
        
        # Action chosen by ensemble policy (DQN + Tabular Q-Learning)
        dqn_action = dqn_agent.select_action(state_vec)
        q_action = self.select_action(state_key)
        
        # Ensemble decision: prefer DQN action if trained, else tabular action
        action_idx = dqn_action if dqn_agent.total_steps > 20 else q_action
        next_difficulty = DIFFICULTY_LEVELS[action_idx]

        # Calculate next state after action applied
        next_acc = (recent_acc * 4 + (1.0 if is_correct else 0.0)) / 5.0
        next_streak = streak + (1 if is_correct else -1)
        next_state_key = self.get_state_key(next_difficulty, next_acc, next_streak)
        next_diff_idx = DIFFICULTY_LEVELS.index(next_difficulty)
        next_state_vec = dqn_agent.get_state_vector(next_diff_idx, next_acc, next_streak, attempts_count=6)

        # 1. Update Tabular Q-table
        self.update_q_value(state_key, action_idx, reward, next_state_key)

        # 2. Update PyTorch DQN Agent (Replay Buffer + Gradient Step)
        dqn_agent.memory.push(state_vec, action_idx, reward, next_state_vec, False)
        dqn_agent.update_policy(batch_size=8)

        return next_difficulty

# Singleton Instance
rl_engine = RLAdaptiveEngine()

def adjust_difficulty(
    current_difficulty: str,
    correct: bool,
    student_id: Optional[str] = None,
    db: Optional[Session] = None
) -> str:
    """
    Adjusts difficulty level using the Reinforcement Learning Adaptive Engine.
    Maintains backward compatibility with legacy calls while leveraging Q-learning updates.
    """
    return rl_engine.process_student_feedback(
        student_id=student_id or "default_student",
        current_difficulty=current_difficulty,
        is_correct=correct,
        db=db
    )

