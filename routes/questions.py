from fastapi import APIRouter, Depends
from models.question import add_question, get_questions_by_user

router = APIRouter()

@router.post("/questions")
def create_question(user_id: int, question: str, options: str, correct_answer: str):
    """Create a new question."""
    add_question(user_id, question, options, correct_answer)
    return {"message": "Question created successfully"}

@router.get("/questions/{user_id}")
def get_user_questions(user_id: int):
    """Get all questions for a specific user."""
    questions = get_questions_by_user(user_id)
    return {"questions": [dict(q) for q in questions]}
