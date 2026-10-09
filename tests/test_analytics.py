from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_classroom_heatmap_endpoint():
    response = client.get("/analytics/classroom-heatmap")
    assert response.status_code == 200
    data = response.json()
    assert "class_name" in data
    assert "students" in data
    assert len(data["students"]) > 0
    assert "weakest_topic" in data

def test_remedial_plan_endpoint():
    req_body = {"student_id": "S001-ALPHA", "weak_topic": "Deep Q-Learning"}
    response = client.post("/analytics/remedial-plan", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "S001-ALPHA"
    assert "5-Minute" in data["remedial_title"]
    assert len(data["practice_questions"]) == 3
