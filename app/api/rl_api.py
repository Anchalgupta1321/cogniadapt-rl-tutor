from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.rl.dqn_agent import dqn_agent
from app.rl.bandit_calibrator import bandit_calibrator
from app.rl.rlaif_evaluator import rlaif_evaluator
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/rl", tags=["Reinforcement Learning Control Center"])

class RLAIFEvalRequest(BaseModel):
    question: str
    options: List[str]
    answer: str
    type: Optional[str] = "MCQ"

@router.get("/dqn/diagnostics")
def get_dqn_diagnostics():
    """Returns PyTorch Deep Q-Network (DQN) policy diagnostics & replay buffer stats."""
    return dqn_agent.get_diagnostics()

@router.post("/dqn/train")
def train_dqn_step(steps: int = 10, batch_size: int = 16):
    """Triggers PyTorch gradient descent optimization steps over the replay buffer."""
    losses = []
    for _ in range(steps):
        loss = dqn_agent.update_policy(batch_size=batch_size)
        losses.append(round(loss, 6))
    return {
        "message": f"Executed {steps} PyTorch optimization steps on ReplayBuffer.",
        "losses": losses,
        "current_total_steps": dqn_agent.total_steps
    }

@router.get("/bandit/calibration")
def get_bandit_calibrations():
    """Returns Bayesian Thompson Sampling item response calibration parameters."""
    return bandit_calibrator.get_stats()

@router.post("/rlaif/evaluate")
def evaluate_question_reward(req: RLAIFEvalRequest):
    """Evaluates a question using the RLAIF Multi-Attribute Reward Model."""
    res = rlaif_evaluator.evaluate_question(req.dict())
    return res
