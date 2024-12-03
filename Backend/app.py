from fastapi import FastAPI, Form, Request, UploadFile, File, Body, Depends
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import logging
from google.generativeai import configure, GenerativeModel
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional
from PyPDF2 import PdfReader
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from utils.firebase_config import verify_token

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Serve static files from the frontend directory
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

configure(api_key="AIzaSyDGKtZ-K_xXzQMNsZdWIslYuiGFxE1CXG8")
model = GenerativeModel("gemini-1.5-flash")


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def get_current_user(request: Request):
    token = request.cookies.get("token")  # Get token from cookies
    if token:
        user = verify_token(token)
        if user:
            return user
    return None


@app.get("/")
async def home(request: Request, user=Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/auth")  # Redirect if unauthenticated
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

# Route: Auth Page (Sign-In/Sign-Up)
@app.get("/auth")
async def auth_page(request: Request):
    return templates.TemplateResponse("auth.html", {"request": request})


@app.get("/question-generator")
async def question_generator(request: Request, user=Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/auth")  # Redirect to sign-in page if unauthenticated
    return templates.TemplateResponse("question-generator.html", {"request": request, "user": user})



# Other existing routes from your app.py can stay here, e.g., /analyze, /verify, etc.
@app.post("/analyze")
async def analyze(
    pdf_file: UploadFile = File(...),
    topic: str = Form(...),
    difficulty: str = Form(...),
    question_type: str = Form(...),
    total_marks: int = Form(...),
    marks_per_question: Optional[int] = Form(None),
):
    try:
        # Placeholder for PDF text extraction and question generation
        content = "Extracted text from PDF"  # Simulate extracted text
        num_questions = total_marks // (marks_per_question or 1)
        questions = []

        if question_type == "mcq":
            for i in range(num_questions):
                questions.append({
                    "question": f"Sample MCQ question {i + 1} on {topic} ({difficulty}).",
                })
        elif question_type == "theory":
            for i in range(num_questions):
                questions.append({
                    "question": f"Sample theory question {i + 1} on {topic} ({difficulty}).",
                })

        return {"questions": questions}
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
