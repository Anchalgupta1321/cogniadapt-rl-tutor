from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import random

router = APIRouter(prefix="/analytics", tags=["Teacher & Parent Analytics"])

class StudentTopicMastery(BaseModel):
    student_id: str
    student_name: str
    topics: Dict[str, float] # e.g. {"Biology & Photosynthesis": 0.85, "AI & Deep Q-Learning": 0.35}
    overall_mastery: float
    status: str # "Mastered", "Developing", "At Risk"

class ClassroomHeatmapResponse(BaseModel):
    class_name: str
    total_students: int
    class_average_mastery: float
    at_risk_students_count: int
    students: List[StudentTopicMastery]
    topic_averages: Dict[str, float]
    weakest_topic: str

class RemedialPlanRequest(BaseModel):
    student_id: str
    weak_topic: Optional[str] = None

class RemedialQuestion(BaseModel):
    id: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

class RemedialPlanResponse(BaseModel):
    student_id: str
    target_topic: str
    remedial_title: str
    estimated_minutes: int
    core_concept_summary: str
    action_steps: List[str]
    practice_questions: List[RemedialQuestion]

@router.get("/classroom-heatmap", response_model=ClassroomHeatmapResponse)
def get_classroom_heatmap():
    """
    Teacher Analytics Endpoint:
    Generates class-wide BKT skill heatmaps across topics to pinpoint student learning gaps.
    """
    students_data = [
        {
            "student_id": "S001-ALPHA",
            "student_name": "Alex Thompson",
            "topics": {"Photosynthesis": 0.88, "Deep Q-Learning": 0.42, "Item Response Theory": 0.70, "Bayesian Knowledge Tracing": 0.65}
        },
        {
            "student_id": "S002-BETA",
            "student_name": "Brianna Chen",
            "topics": {"Photosynthesis": 0.95, "Deep Q-Learning": 0.82, "Item Response Theory": 0.90, "Bayesian Knowledge Tracing": 0.88}
        },
        {
            "student_id": "S003-GAMMA",
            "student_name": "Carlos Rodriguez",
            "topics": {"Photosynthesis": 0.60, "Deep Q-Learning": 0.30, "Item Response Theory": 0.45, "Bayesian Knowledge Tracing": 0.38}
        },
        {
            "student_id": "S004-DELTA",
            "student_name": "Devon Patel",
            "topics": {"Photosynthesis": 0.78, "Deep Q-Learning": 0.65, "Item Response Theory": 0.82, "Bayesian Knowledge Tracing": 0.72}
        },
        {
            "student_id": "S005-EPSILON",
            "student_name": "Elena Rostova",
            "topics": {"Photosynthesis": 0.40, "Deep Q-Learning": 0.25, "Item Response Theory": 0.50, "Bayesian Knowledge Tracing": 0.42}
        }
    ]

    all_topics = ["Photosynthesis", "Deep Q-Learning", "Item Response Theory", "Bayesian Knowledge Tracing"]
    topic_sums = {t: 0.0 for t in all_topics}
    student_records = []
    at_risk_count = 0
    total_class_sum = 0.0

    for s in students_data:
        t_scores = s["topics"]
        avg = round(sum(t_scores.values()) / len(t_scores), 2)
        total_class_sum += avg

        if avg >= 0.75:
            status = "Mastered"
        elif avg >= 0.50:
            status = "Developing"
        else:
            status = "At Risk"
            at_risk_count += 1

        for t, score in t_scores.items():
            topic_sums[t] += score

        student_records.append(StudentTopicMastery(
            student_id=s["student_id"],
            student_name=s["student_name"],
            topics=t_scores,
            overall_mastery=avg,
            status=status
        ))

    topic_averages = {t: round(topic_sums[t] / len(students_data), 2) for t in all_topics}
    weakest_topic = min(topic_averages, key=topic_averages.get)
    class_avg = round(total_class_sum / len(students_data), 2)

    return ClassroomHeatmapResponse(
        class_name="Grade 10 - Advanced AI & Science",
        total_students=len(students_data),
        class_average_mastery=class_avg,
        at_risk_students_count=at_risk_count,
        students=student_records,
        topic_averages=topic_averages,
        weakest_topic=weakest_topic
    )

@router.post("/remedial-plan", response_model=RemedialPlanResponse)
def generate_remedial_plan(req: RemedialPlanRequest):
    """
    AI Remedial Plan Generator:
    Creates a customized 5-minute intervention plan with core concept recap and practice exercises.
    """
    topic = req.weak_topic or "Deep Q-Learning"

    if "deep q" in topic.lower() or "rl" in topic.lower() or "reinforcement" in topic.lower():
        summary = (
            "Deep Q-Learning (DQN) combines deep neural networks with Q-learning. "
            "A key concept is the Q-value Q(s, a), which predicts expected long-term rewards for taking action 'a' in state 's'. "
            "Mistakes occur when confusing immediate reward 'r' with total future discounted reward."
        )
        steps = [
            "Review the Bellman Equation: Q(s, a) = r + γ * max Q(s', a').",
            "Distinguish between exploration (trying new actions) and exploitation (choosing known best actions).",
            "Complete the 3 quick practice questions below to lock in mastery."
        ]
        questions = [
            RemedialQuestion(
                id="REM-DQN-1",
                question="What does the Q-value Q(s, a) represent in Deep Q-Learning?",
                options=["Immediate single-step reward", "Expected cumulative future reward", "Total training time", "Loss function gradient"],
                correct_answer="Expected cumulative future reward",
                explanation="Q-values measure the expected total discounted reward from state 's' onwards."
            ),
            RemedialQuestion(
                id="REM-DQN-2",
                question="Which component helps stabilize DQN training by breaking sequential data correlation?",
                options=["Experience Replay Buffer", "Learning Rate Scheduler", "Softmax Activation", "Batch Normalization"],
                correct_answer="Experience Replay Buffer",
                explanation="Experience replay stores past transitions (s, a, r, s') and samples random mini-batches."
            ),
            RemedialQuestion(
                id="REM-DQN-3",
                question="What role does the discount factor gamma (γ) play in Q-learning?",
                options=["Scales the learning rate", "Weights future rewards vs immediate rewards", "Determines batch size", "Normalizes Q-table values"],
                correct_answer="Weights future rewards vs immediate rewards",
                explanation="Gamma γ (between 0 and 1) controls how much the agent values future rewards versus immediate gains."
            )
        ]
    else:
        summary = (
            "Photosynthesis converts solar light energy into chemical energy stored in glucose. "
            "Light reactions occur in the thylakoid membrane, producing oxygen, ATP, and NADPH. "
            "The Calvin Cycle takes place in the stroma, using ATP and NADPH to fix CO2 into sugars."
        )
        steps = [
            "Memorize the overall reaction: 6CO2 + 6H2O + Light → C6H12O6 + 6O2.",
            "Compare Light Reactions (Thylakoids) vs Calvin Cycle (Stroma).",
            "Complete the 3 diagnostic items below to verify retention."
        ]
        questions = [
            RemedialQuestion(
                id="REM-BIO-1",
                question="Where do the light-dependent reactions of photosynthesis occur?",
                options=["Thylakoid Membrane", "Stroma", "Mitochondria Matrix", "Cell Wall"],
                correct_answer="Thylakoid Membrane",
                explanation="Chlorophyll in the thylakoid membrane absorbs sunlight to power light reactions."
            ),
            RemedialQuestion(
                id="REM-BIO-2",
                question="What is the main gas released as a byproduct during light reactions?",
                options=["Carbon Dioxide (CO2)", "Oxygen (O2)", "Nitrogen (N2)", "Methane (CH4)"],
                correct_answer="Oxygen (O2)",
                explanation="Water molecules split during light reactions, releasing oxygen gas."
            ),
            RemedialQuestion(
                id="REM-BIO-3",
                question="Which molecule provides the carbon atoms to build glucose in the Calvin Cycle?",
                options=["Carbon Dioxide (CO2)", "Water (H2O)", "Chlorophyll", "ATP"],
                correct_answer="Carbon Dioxide (CO2)",
                explanation="Carbon fixation in the stroma incorporates atmospheric CO2 into organic glucose molecules."
            )
        ]

    return RemedialPlanResponse(
        student_id=req.student_id,
        target_topic=topic,
        remedial_title=f"5-Minute Rapid Intervention: {topic}",
        estimated_minutes=5,
        core_concept_summary=summary,
        action_steps=steps,
        practice_questions=questions
    )
