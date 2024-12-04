from fastapi import FastAPI, Form, Request, UploadFile, File, Body, Depends
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
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
import re
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
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

@app.get("/questionGenerator")
async def question_generator(request: Request):
    return templates.TemplateResponse("question_generator.html", {"request": request})

@app.get("/pdf-summarizer")
async def pdf_summarizer_page(request):
    return templates.TemplateResponse("pdf-summarizer.html", {"request": request})

@app.post("/summarize-pdf")
async def summarize_pdf(pdfFile: UploadFile = File(...)):
    try:
        # Step 1: Read the PDF file and extract text
        pdf_reader = PdfReader(pdfFile.file)
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

        # Step 2: Clean the text
        extracted_text = re.sub(r'\s+', ' ', extracted_text.strip())

        # Step 3: Generate summary using the generative model
        prompt = (
            "Summarize the following content in a concise and clear manner: "
            + extracted_text
        )
        summary = model.generate(prompt, max_tokens=150)

        # Step 4: Return the summary
        return JSONResponse(content={"summary": summary}, status_code=200)

    except Exception as e:
        print("Error:", e)
        return JSONResponse(content={"error": "Failed to summarize the PDF"}, status_code=500)

@app.get("/questions")
async def questions_page(request: Request, user=Depends(get_current_user)):
    """Display questions."""
    if not user:
        return RedirectResponse(url="/auth")
    return templates.TemplateResponse("questions.html", {"request": request, "user": user})

# Function to process the uploaded PDF
def file_processing(file_path):
    """Load and split PDF content into chunks."""
    try:
        reader = PdfReader(file_path)
        text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text().strip())

        if not text_content.strip():
            raise ValueError("The uploaded PDF contains no extractable text.")

        splitter = CharacterTextSplitter(separator=" ", chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text_content)

        documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
        if not documents:
            raise ValueError("No valid document chunks found.")
        logging.info(f"Extracted {len(documents)} document chunks from the PDF.")
        return documents

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise

# Function to generate questions based on the text, topic, and difficulty
def generate_questions(text: str, topic: str, difficulty: str) -> str:
    try:
        if not text.strip():
            raise ValueError("Empty input text provided for question generation.")
        
        prompt = f"Generate a question on the topic '{topic}' with {difficulty} difficulty:\n\n{text[:800]}"
        logging.info(f"Prompt sent: {prompt}")
        response = model.generate_content(prompt)
        logging.info(f"Response received: {response.text if response else 'No response'}")

        # Check for a valid response
        if response and hasattr(response, "text") and response.text.strip():
            return response.text.strip()
        else:
            raise ValueError("Empty or invalid response from the Gemini model.")
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        raise
    except Exception as e:
        logging.error("Ensure the API key is set and valid.")
        raise RuntimeError("Failed to generate questions due to an internal error.") from e

# Function to handle the generation of questions from uploaded files
@app.post("/generate-questions")
async def generate_questions_from_pdf(
    pdf_file: UploadFile = File(...),
    topic: str = Form(...),
    difficulty: str = Form(...),
    num_questions: int = Form(10),
):
    """Generate questions from an uploaded PDF based on topic and difficulty."""
    if not pdf_file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are allowed."}

    base_folder = "static/docs/"
    os.makedirs(base_folder, exist_ok=True)
    pdf_filename = os.path.join(base_folder, pdf_file.filename)

    try:
        # Save the uploaded file temporarily
        async with aiofiles.open(pdf_filename, "wb") as f:
            content = await pdf_file.read()
            await f.write(content)

        # Process the file
        documents = file_processing(pdf_filename)

        # Generate questions from each document chunk
        questions = []
        for doc in documents[:num_questions]:  # Limit to specified number of questions
            question = generate_questions(doc.page_content, topic, difficulty)
            questions.append({"question": question})

        return {"questions": questions}
    
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        return {"error": f"Failed to generate questions: {e}"}

@app.get("/questions")
async def questions_page(request: Request, questions: List[str] = None):
    """Render a page to display questions."""
    return templates.TemplateResponse("questions.html", {"request": request, "questions": questions or []})

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
