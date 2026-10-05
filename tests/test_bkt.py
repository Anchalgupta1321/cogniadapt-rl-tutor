from fastapi.testclient import TestClient
from app.main import app
from app.bkt.bkt_engine import bkt_engine, BayesianKnowledgeTracer

client = TestClient(app)

import uuid

def test_bkt_mathematical_updates():
    bkt = BayesianKnowledgeTracer(storage_path="data/test_bkt_states.json")
    student_id = f"test_student_{uuid.uuid4()}"
    
    # Initial state
    p0 = bkt.p_init
    
    # Correct response should increase P(L)
    p1 = bkt.update_knowledge(student_id, "Science", is_correct=True)
    assert p1 > p0
    
    # Second correct response should further increase P(L)
    p2 = bkt.update_knowledge(student_id, "Science", is_correct=True)
    assert p2 > p1

def test_bkt_mastery_breakdown():
    res = bkt_engine.get_student_mastery("test_student_bkt")
    assert "mastery_breakdown" in res
    assert "overall_mastery_avg" in res

def test_bkt_api_endpoints():
    response = client.get("/bkt/mastery/S001-ALPHA")
    assert response.status_code == 200
    data = response.json()
    assert "mastery_breakdown" in data

    up_resp = client.post("/bkt/update", json={
        "student_id": "S001-ALPHA",
        "topic": "Math",
        "is_correct": True
    })
    assert up_resp.status_code == 200
    assert "updated_p_know" in up_resp.json()

if __name__ == "__main__":
    test_bkt_mathematical_updates()
    test_bkt_mastery_breakdown()
    test_bkt_api_endpoints()
    print("ALL BKT TESTS PASSED SUCCESSFULLY!")
