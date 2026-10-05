import os
import json
from openai import OpenAI
from typing import List

api_key = os.getenv("LLM_API_KEY")
is_groq = False

if api_key and (api_key.startswith("your_") or api_key == "AIzaSyBfs3TiEsQdqFpJAvOiFvoe880nM51PwXM"):
    api_key = None

if api_key and api_key.startswith("gsk_"):
    is_groq = True
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model_name = "llama-3.3-70b-versatile"
elif api_key:
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini"
else:
    client = None
    model_name = "mock"

def is_mock_mode():
    return client is None

def generate_questions_from_chunk(chunk_text: str, chunk_id: str) -> List[dict]:
    if is_mock_mode():
        return generate_mock_questions(chunk_text, chunk_id)
        
    prompt = f"""
You are an educational quiz generator.

Generate 3 quiz questions from the following content.

Content:
{chunk_text}

Rules:
- Include MCQ, True/False, Fill in the blank
- Provide answers
- Assign difficulty level (easy, medium, hard)
- Return ONLY a JSON list of dictionaries.

Example output:
[
  {{
    "question": "How many sides does a triangle have?",
    "type": "MCQ",
    "options": ["2","3","4","5"],
    "answer": "3",
    "difficulty": "easy"
  }}
]
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You output JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        # Clean markdown if present
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        questions = json.loads(content.strip())
        
        # Enforce list type
        if isinstance(questions, dict) and "questions" in questions:
            questions = questions["questions"]
            
        # Add source_chunk_id tracing
        for q in questions:
            q["source_chunk_id"] = chunk_id
            
        return questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def generate_mock_questions(chunk_text: str, chunk_id: str) -> List[dict]:
    """Generates realistic mock questions when LLM is offline, avoiding technical IDs in text."""
    return [
        {
            "question": "Based on the educational content provided, what is the primary objective of this lesson?",
            "type": "MCQ",
            "options": ["Conceptual Understanding", "Skill Memorization", "Historical Recounting", "Logical Prototyping"],
            "answer": "Conceptual Understanding",
            "difficulty": "easy",
            "source_chunk_id": chunk_id
        },
        {
            "question": "Does the passage imply that students should apply these concepts in a practical laboratory setting?",
            "type": "True/False",
            "options": ["True", "False"],
            "answer": "True",
            "difficulty": "medium",
            "source_chunk_id": chunk_id
        },
        {
            "question": "How does the author characterize the relationship between the different variables discussed in the text?",
            "type": "MCQ",
            "options": ["Direct proportionality", "Inverse correlation", "Random independence", "Mutual exclusivity"],
            "answer": "Direct proportionality",
            "difficulty": "hard",
            "source_chunk_id": chunk_id
        }
    ]
