from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_gamification_profile_endpoint():
    response = client.get("/gamification/profile?student_id=S001-ALPHA")
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "S001-ALPHA"
    assert "xp_points" in data
    assert "daily_streak_days" in data
    assert len(data["unlocked_badges"]) > 0

def test_peer_leaderboard_endpoint():
    response = client.get("/gamification/leaderboard")
    assert response.status_code == 200
    data = response.json()
    assert "top_rankings" in data
    assert len(data["top_rankings"]) > 0
    assert data["top_rankings"][0]["rank"] == 1

def test_claim_xp_endpoint():
    req_body = {"student_id": "S001-ALPHA", "action_type": "complete_quiz"}
    response = client.post("/gamification/claim-xp", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "S001-ALPHA"
    assert data["xp_gained"] > 0
    assert data["total_xp"] > 0
