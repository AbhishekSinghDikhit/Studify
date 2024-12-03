from fastapi import FastAPI, Form, Request, UploadFile, File, Body, Depends
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from google.generativeai import configure, GenerativeModel
from PyPDF2 import PdfReader
import logging
import os
import aiofiles
import uvicorn
from utils.firebase_config import verify_token

# Logging configuration
logging.basicConfig(level=logging.INFO)

# FastAPI initialization
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and template directories
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Generative model initialization
configure(api_key="AIzaSyDGKtZ-K_xXzQMNsZdWIslYuiGFxE1CXG8")
model = GenerativeModel("gemini-1.5-flash")

# Embedding model initialization
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Utility to verify the current user
def get_current_user(request: Request):
    token = request.cookies.get("token")
    if token:
        user = verify_token(token)
        if user:
            return user
    return None

# --- Routes ---

@app.get("/")
async def home(request: Request, user=Depends(get_current_user)):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/auth")
async def auth_page(request: Request):
    """Authentication page."""
    return templates.TemplateResponse("auth.html", {"request": request})

@app.get("/question-generator")
async def question_generator(request: Request, user=Depends(get_current_user)):
    """Question Generator page."""
    if not user:
        return RedirectResponse(url="/auth")
    return templates.TemplateResponse("question-generator.html", {"request": request, "user": user})

@app.get("/questions")
async def questions_page(request: Request, user=Depends(get_current_user)):
    """Display questions."""
    if not user:
        return RedirectResponse(url="/auth")
    return templates.TemplateResponse("questions.html", {"request": request, "user": user})

@app.post("/upload")
async def upload(pdf_file: UploadFile = File(...)):
    """Handle PDF uploads."""
    if not pdf_file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    base_folder = "frontend/static/docs/"
    os.makedirs(base_folder, exist_ok=True)
    file_path = os.path.join(base_folder, pdf_file.filename)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await pdf_file.read())
    return {"message": "File uploaded successfully", "file_path": file_path}

def file_processing(file_path: str):
    """Process PDF files into chunks."""
    reader = PdfReader(file_path)
    text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text().strip())
    splitter = CharacterTextSplitter(separator=" ", chunk_size=800, chunk_overlap=100)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text_content) if chunk.strip()]

@app.post("/analyze")
async def analyze(
    pdf_file: UploadFile = File(...),
    topic: str = Form(...),
    difficulty: str = Form(...),
    question_type: str = Form(...),
    total_marks: int = Form(...),
    marks_per_question: int = Form(None),
):
    """Analyze PDF and generate questions."""
    try:
        # Save uploaded file temporarily
        file_path = f"uploads/{pdf_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await pdf_file.read())
        
        documents = file_processing(file_path)
        num_questions = total_marks // (marks_per_question or 1)
        questions = []

        if question_type.lower() == "mcq":
            for doc in documents[:num_questions]:
                prompt = f"Generate a {marks_per_question}-mark MCQ on '{topic}' at '{difficulty}' difficulty:\n{doc.page_content[:800]}"
                response = model.generate_content(prompt)
                questions.append(response.text.strip())
        elif question_type.lower() == "theory":
            for doc in documents[:num_questions]:
                prompt = f"Generate a theory question on '{topic}' at '{difficulty}' difficulty:\n{doc.page_content[:800]}"
                response = model.generate_content(prompt)
                questions.append(response.text.strip())

        return {"questions": questions}

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return {"error": str(e)}

# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
