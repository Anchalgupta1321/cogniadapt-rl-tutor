import pytest
from app.services.adaptive_engine import RLAdaptiveEngine, adjust_difficulty

def test_rl_engine_initialization():
    engine = RLAdaptiveEngine(storage_path="data/test_q_table.json")
    assert engine.total_iterations >= 0
    assert len(engine.q_table) > 0

def test_state_key_mapping():
    engine = RLAdaptiveEngine(storage_path="data/test_q_table.json")
    key = engine.get_state_key("medium", recent_accuracy=0.8, streak=2)
    assert key == "1_2_1"

def test_reward_computation():
    engine = RLAdaptiveEngine(storage_path="data/test_q_table.json")
    # Correct answer on hard with high accuracy
    r1 = engine.compute_reward("hard", True, streak=3, recent_acc=0.9)
    assert r1 > 1.0

    # Wrong answer on hard with cold streak
    r2 = engine.compute_reward("hard", False, streak=-3, recent_acc=0.2)
    assert r2 < -1.0

def test_adjust_difficulty_feedback():
    next_diff = adjust_difficulty("medium", True, student_id="test_student_1")
    assert next_diff in ["easy", "medium", "hard"]

if __name__ == "__main__":
    test_rl_engine_initialization()
    test_state_key_mapping()
    test_reward_computation()
    test_adjust_difficulty_feedback()
    print("ALL RL TESTS PASSED SUCCESSFULLY!")
