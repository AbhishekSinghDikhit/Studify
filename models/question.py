from pydantic import BaseModel

class Question(BaseModel):
    user_id: str
    topic: str
    question_type: str
    difficulty: str
    total_marks: int
    marks_per_question: int
    questions: list
