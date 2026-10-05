from fastapi.testclient import TestClient
from app.main import app
from eval.evaluate_ragas import RAGEvaluationFramework
from eval.simulate_students import StudentSimulator

client = TestClient(app)

def test_grounded_tutor_chat_and_refusal():
    # Test grounded chat endpoint
    r1 = client.post("/chat/tutor", json={
        "student_id": "HACKATHON_TEST_STUDENT",
        "query": "What is the primary objective of this lesson?"
    })
    assert r1.status_code == 200
    data1 = r1.json()
    assert "is_grounded" in data1

    # Test out-of-material refusal guardrail
    r2 = client.post("/chat/tutor", json={
        "student_id": "HACKATHON_TEST_STUDENT",
        "query": "How to construct a quantum teleportation device using general relativity?"
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["is_grounded"] is False
    assert "OUT-OF-MATERIAL REFUSAL" in data2["answer"]

def test_post_quiz_report_and_flashcards():
    r1 = client.post("/revision/report", json={"student_id": "S001-ALPHA"})
    assert r1.status_code == 200
    assert "overall_accuracy" in r1.json()

    r2 = client.post("/revision/flashcards", json={"student_id": "S001-ALPHA"})
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)

def test_ragas_evaluation_benchmark():
    evaluator = RAGEvaluationFramework()
    dummy = [{
        "query": "What is photosynthesis?",
        "retrieved_contexts": ["Photosynthesis turns light into energy."],
        "generated_answer": "Photosynthesis turns light into energy.",
        "ground_truth": "Light energy conversion."
    }]
    report = evaluator.run_benchmark_suite(dummy)
    assert report["summary_metrics"]["faithfulness"] > 0.0

def test_student_simulation_benchmark():
    sim = StudentSimulator()
    rep = sim.run_simulation_experiment(num_students=2, sessions_per_student=3)
    assert "question_repetition_rate" in rep
    assert "average_mastery_gain" in rep

if __name__ == "__main__":
    test_grounded_tutor_chat_and_refusal()
    test_post_quiz_report_and_flashcards()
    test_ragas_evaluation_benchmark()
    test_student_simulation_benchmark()
    print("ALL HACKATHON TRACK D INTEGRATION TESTS PASSED SUCCESSFULLY!")
