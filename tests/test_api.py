from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [200, 307]


def test_rl_stats_endpoint():
    response = client.get("/rl/stats")
    assert response.status_code == 200
    data = response.json()
    assert "engine" in data
    assert "q_table" in data
    assert data["engine"] == "Q-Learning RL Adaptive Engine"

def test_reset_db_endpoint():
    response = client.delete("/reset-db")
    assert response.status_code == 200
    assert response.json()["message"] == "Database wiped successfully."

def test_vision_chat_endpoint():
    files = {"file": ("test_diagram.png", b"fake_png_bytes", "image/png")}
    data = {"query": "Analyze photosynthesis diagram", "student_id": "S001-ALPHA"}
    response = client.post("/chat/vision", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["is_grounded"] is True
    assert "Multimodal Vision OCR" in res["answer"] or "Diagram" in res["answer"]

