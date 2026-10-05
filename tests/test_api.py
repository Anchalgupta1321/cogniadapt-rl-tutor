from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to CogniAdapt AI Engine"

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
