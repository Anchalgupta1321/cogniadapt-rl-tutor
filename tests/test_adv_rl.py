import numpy as np
import torch
from fastapi.testclient import TestClient
from app.main import app
from app.rl.dqn_agent import dqn_agent, DQNAgent
from app.rl.bandit_calibrator import bandit_calibrator
from app.rl.rlaif_evaluator import rlaif_evaluator

client = TestClient(app)

def test_dqn_agent_tensor_shapes():
    agent = DQNAgent()
    s_vec = agent.get_state_vector(current_diff_idx=1, recent_acc=0.8, streak=2, attempts_count=10)
    assert s_vec.shape == (8,)
    
    action = agent.select_action(s_vec, evaluate=True)
    assert action in [0, 1, 2]

def test_dqn_replay_buffer():
    agent = DQNAgent()
    s1 = np.ones(8, dtype=np.float32)
    s2 = np.zeros(8, dtype=np.float32)
    agent.memory.push(s1, 1, 1.0, s2, False)
    assert len(agent.memory) >= 1

def test_bandit_calibrator():
    bandit_calibrator.record_attempt("q_test_123", is_correct=True, static_difficulty="medium")
    bandit_calibrator.record_attempt("q_test_123", is_correct=True, static_difficulty="medium")
    bandit_calibrator.record_attempt("q_test_123", is_correct=True, static_difficulty="medium")
    diff = bandit_calibrator.get_calibrated_difficulty("q_test_123", "medium")
    assert diff in ["easy", "medium", "hard"]

def test_rlaif_evaluator():
    good_question = {
        "question": "What is the boiling point of water at standard atmospheric pressure?",
        "options": ["50 C", "100 C", "150 C", "200 C"],
        "answer": "100 C",
        "type": "MCQ"
    }
    res = rlaif_evaluator.evaluate_question(good_question)
    assert res["composite_reward"] > 0.5
    assert res["accepted"] is True

def test_rl_api_endpoints():
    r1 = client.get("/rl/dqn/diagnostics")
    assert r1.status_code == 200
    assert "algorithm" in r1.json()

    r2 = client.get("/rl/bandit/calibration")
    assert r2.status_code == 200
    assert "total_items_calibrated" in r2.json()

    r3 = client.post("/rl/dqn/train?steps=2&batch_size=4")
    assert r3.status_code == 200

if __name__ == "__main__":
    test_dqn_agent_tensor_shapes()
    test_dqn_replay_buffer()
    test_bandit_calibrator()
    test_rlaif_evaluator()
    test_rl_api_endpoints()
    print("ALL ADVANCED RL ALGORITHM TESTS PASSED SUCCESSFULLY!")
