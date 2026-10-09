from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/gamification", tags=["Gamification & Leaderboard"])

class Badge(BaseModel):
    id: str
    title: str
    description: str
    icon: str
    unlocked: bool
    unlocked_at: Optional[str] = None

class StudentGamificationProfile(BaseModel):
    student_id: str
    student_name: str
    xp_points: int
    level_title: str
    daily_streak_days: int
    quizzes_completed: int
    unlocked_badges: List[Badge]

class LeaderboardEntry(BaseModel):
    rank: int
    student_handle: str
    xp_points: int
    daily_streak: int
    mastery_velocity: str # e.g. "+14% / week"
    badges_count: int
    is_current_user: bool

class LeaderboardResponse(BaseModel):
    cohort_name: str
    top_rankings: List[LeaderboardEntry]

class ClaimXPRequest(BaseModel):
    student_id: str
    action_type: str # "answer_correct", "complete_quiz", "master_topic"

class ClaimXPResponse(BaseModel):
    student_id: str
    xp_gained: int
    total_xp: int
    new_level: str
    badge_unlocked: Optional[str] = None

# In-memory Gamification State Store for Demo
USER_PROFILES = {
    "S001-ALPHA": {
        "xp": 1250,
        "streak": 5,
        "quizzes": 14,
        "badges": ["DQN Scholar", "IRT Pioneer", "Streak Master"]
    }
}

ALL_BADGES = [
    Badge(
        id="BADGE-01",
        title="DQN Scholar",
        description="Achieved > 85% BKT mastery in Deep Q-Learning & RL policy estimation.",
        icon="🧠",
        unlocked=True,
        unlocked_at="2026-10-08"
    ),
    Badge(
        id="BADGE-02",
        title="Master of Biology",
        description="Achieved > 85% BKT mastery in Photosynthesis & Cellular Energy.",
        icon="🌱",
        unlocked=True,
        unlocked_at="2026-10-07"
    ),
    Badge(
        id="BADGE-03",
        title="IRT Pioneer",
        description="Completed 10 adaptive questions with Item Response Theory difficulty scaling.",
        icon="🎯",
        unlocked=True,
        unlocked_at="2026-10-06"
    ),
    Badge(
        id="BADGE-04",
        title="Streak Master",
        description="Maintained a 5-day consecutive daily practice streak.",
        icon="🔥",
        unlocked=True,
        unlocked_at="2026-10-09"
    ),
    Badge(
        id="BADGE-05",
        title="RAGAS Champion",
        description="Achieved 100% grounded source accuracy without off-topic refusals.",
        icon="🛡️",
        unlocked=False,
        unlocked_at=None
    )
]

@router.get("/profile", response_model=StudentGamificationProfile)
def get_student_gamification_profile(student_id: str = "S001-ALPHA"):
    """
    Returns student XP points, daily login streak, level title, and unlocked achievement badges.
    """
    prof = USER_PROFILES.get(student_id, {"xp": 500, "streak": 3, "quizzes": 5, "badges": ["IRT Pioneer"]})
    xp = prof["xp"]

    if xp >= 2000:
        level = "Level 7 - AI Grandmaster"
    elif xp >= 1200:
        level = "Level 5 - Deep RL Scholar"
    elif xp >= 800:
        level = "Level 3 - Adaptive Learner"
    else:
        level = "Level 1 - Apprentice"

    unlocked_list = []
    for b in ALL_BADGES:
        b_copy = b.copy()
        if b.title in prof["badges"]:
            b_copy.unlocked = True
        else:
            b_copy.unlocked = False
            b_copy.unlocked_at = None
        unlocked_list.append(b_copy)

    return StudentGamificationProfile(
        student_id=student_id,
        student_name="Alex Thompson",
        xp_points=xp,
        level_title=level,
        daily_streak_days=prof["streak"],
        quizzes_completed=prof["quizzes"],
        unlocked_badges=unlocked_list
    )

@router.get("/leaderboard", response_model=LeaderboardResponse)
def get_peer_leaderboard():
    """
    Returns anonymous cohort leaderboard comparing XP points and learning velocity.
    """
    rankings = [
        LeaderboardEntry(rank=1, student_handle="Brianna-Beta-88", xp_points=1850, daily_streak=12, mastery_velocity="+22% / wk", badges_count=5, is_current_user=False),
        LeaderboardEntry(rank=2, student_handle="Alex-Alpha-01 (You)", xp_points=1250, daily_streak=5, mastery_velocity="+18% / wk", badges_count=4, is_current_user=True),
        LeaderboardEntry(rank=3, student_handle="Devon-Delta-42", xp_points=1100, daily_streak=7, mastery_velocity="+15% / wk", badges_count=3, is_current_user=False),
        LeaderboardEntry(rank=4, student_handle="Carlos-Gamma-19", xp_points=750, daily_streak=3, mastery_velocity="+9% / wk", badges_count=2, is_current_user=False),
        LeaderboardEntry(rank=5, student_handle="Elena-Epsilon-05", xp_points=620, daily_streak=2, mastery_velocity="+6% / wk", badges_count=2, is_current_user=False),
    ]

    return LeaderboardResponse(
        cohort_name="Grade 10 - CogniAdapt Cohort A",
        top_rankings=rankings
    )

@router.post("/claim-xp", response_model=ClaimXPResponse)
def claim_xp_points(req: ClaimXPRequest):
    """
    Awards XP points for completing quiz questions or mastering topics.
    """
    prof = USER_PROFILES.setdefault(req.student_id, {"xp": 500, "streak": 3, "quizzes": 5, "badges": ["IRT Pioneer"]})

    gained = 50
    badge_unlocked = None

    if req.action_type == "complete_quiz":
        gained = 150
        prof["quizzes"] += 1
    elif req.action_type == "master_topic":
        gained = 250
        if "DQN Scholar" not in prof["badges"]:
            prof["badges"].append("DQN Scholar")
            badge_unlocked = "DQN Scholar"

    prof["xp"] += gained
    total_xp = prof["xp"]

    if total_xp >= 2000:
        level = "Level 7 - AI Grandmaster"
    elif total_xp >= 1200:
        level = "Level 5 - Deep RL Scholar"
    elif total_xp >= 800:
        level = "Level 3 - Adaptive Learner"
    else:
        level = "Level 1 - Apprentice"

    return ClaimXPResponse(
        student_id=req.student_id,
        xp_gained=gained,
        total_xp=total_xp,
        new_level=level,
        badge_unlocked=badge_unlocked
    )
