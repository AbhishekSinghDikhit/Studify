from fastapi import FastAPI, Form, Request, UploadFile, File, Body
from pydantic import BaseModel, Field
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from google.generativeai import configure, GenerativeModel
from typing import List, Optional
from PyPDF2 import PdfReader
import os
import uvicorn
import aiofiles
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



# Initialize Gemini client
configure(api_key="AIzaSyDGKtZ-K_xXzQMNsZdWIslYuiGFxE1CXG8")
model = GenerativeModel("gemini-1.5-flash")


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

class TheoryQuestionRequest(BaseModel):
    totalMarks: int
    marksDistribution: list

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

             
def generate_questions(text: str,  topic: str, difficulty: str) -> str:
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

def llm_pipeline(file_path: str):
    documents = file_processing(file_path)

    if not documents:
        logging.error("No valid documents found after processing the file.")
        raise ValueError("Unable to process the file into readable chunks.")

    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    questions = []
    for doc in documents:
        if len(questions) >= 10:  # Limit to 10 questions
            break
        try:
            question = generate_questions(doc.page_content)
            questions.append(question)
        except Exception as e:
            logging.warning(f"Skipping question generation for a chunk due to error: {e}")
            
    if not questions:
        logging.error("Failed to generate any questions from the document.")
        raise ValueError("No questions could be generated from the document content.")

    return retriever, questions


@app.get("/")
async def index(request: Request):
    """Render the main index.html."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/questions")
async def questions_page(request: Request, questions: List[str] = None):
    """Render a page to display questions."""
    return templates.TemplateResponse("questions.html", {"request": request, "questions": questions or []})

@app.post("/analyze-theory")
async def analyze_theory(request: TheoryQuestionRequest):
    total_marks = request.totalMarks
    marks_distribution = request.marksDistribution

    questions = []
    word_limits = {2: 60, 4: 150, 8: 250}  

    for marks in marks_distribution:
        word_limit = word_limits.get(marks, 60)
        prompt = f"Generate a {marks}-mark theory question with a word limit of {word_limit} words."
        try:
            response = model.generate_content(prompt)
            questions.append({"question": response.text.strip(), "marks": marks})
        except Exception as e:
            logging.error(f"Error generating theory question: {e}")
            questions.append({"error": f"Could not generate question for {marks} marks."})

    return {"questions": questions}


class MCQQuestionRequest(BaseModel):
    totalMarks: int
    marksPerQuestion: int
    numberOfQuestions: int

@app.post("/analyze-mcq")
async def analyze_mcq(request: MCQQuestionRequest):
    total_marks = request.totalMarks
    marks_per_question = request.marksPerQuestion
    num_questions = request.numberOfQuestions

    questions = []
    for _ in range(num_questions):
        prompt = f"Generate a multiple-choice question worth {marks_per_question} marks."
        try:
            response = model.generate_content(prompt)
            questions.append({"question": response.text.strip(), "marks": marks_per_question})
        except Exception as e:
            logging.error(f"Error generating MCQ: {e}")
            questions.append({"error": f"Could not generate MCQ."})

    return {"questions": questions}


def generate_mcq_question(marks: int):
    # Mocking MCQ generation logic here. Replace with actual model call.
    return f"Sample MCQ question with {marks} marks"

@app.post("/analyze-manual")
async def analyze_manual(content: str):
    """Generate questions from manually entered content."""
    try:
        splitter = CharacterTextSplitter(separator=" ", chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(content)

        questions = []
        for chunk in chunks[:10]:  # Limit to 10 questions
            prompt = f"Generate a question from the following text:\n\n{chunk}"
            try:
                response = model.generate_content(prompt)
                questions.append(response.text.strip())
            except Exception as e:
                logging.error(f"Error generating question for chunk: {e}")

    except Exception as e:
        logging.error(f"Error during manual analysis: {e}")
        return {"error": f"Analysis failed: {e}"}

    return {"questions": questions}

@app.post("/upload")
async def upload(pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are allowed."}

    base_folder = "static/docs/"
    os.makedirs(base_folder, exist_ok=True)
    pdf_filename = os.path.join(base_folder, pdf_file.filename)

    try:
        async with aiofiles.open(pdf_filename, "wb") as f:
            content = await pdf_file.read()
            await f.write(content)
        return {"msg": "success", "pdf_filename": pdf_filename}
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return {"error": f"Failed to save file: {e}"}

class AnalyzeRequest(BaseModel):
    pdf_file: Optional[bytes] = Field(None, description="PDF file to analyze")
    question_type: str = Field(..., description="Type of question")
    total_marks: int = Field(..., description="Total marks for the questions")
    marks_per_question: int = Field(..., description="Marks per question")

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available."}

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
        # Save the uploaded file temporarily
        temp_dir = "uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, pdf_file.filename)
        with open(file_path, "wb") as f:
            f.write(await pdf_file.read())
        if not pdf_file:
            return {"error": "No file uploaded."}

        # Process the file
        documents = file_processing(file_path)

        num_questions = total_marks // marks_per_question if marks_per_question else 1
        questions = []

        if question_type.lower() == "mcq":
            # Generate MCQs
            for i in range(min(num_questions, len(documents))):
                prompt = f"""
                    Generate a {marks_per_question}-mark multiple-choice question on the topic '{topic}' at '{difficulty}' difficulty level. 
                    Provide 4 answer options, one of which is correct. Clearly indicate the correct answer.
                """
                try:
                    response = model.generate_content(prompt)
                    question_text = response.text.strip()
                    
                    # Extract question, options, and correct answer
                    question_lines = question_text.split('\n')
                    question = question_lines[0]
                    options = question_lines[1:5]
                    correct_answer = [line for line in options if "(Correct)" in line][0]

                    questions.append({
                        "question": question,
                        "options": [option.replace("(Correct)", "").strip() for option in options],
                        "correct_answer": correct_answer.replace("(Correct)", "").strip(),
                        "marks": marks_per_question
                    })
                except Exception as e:
                    logging.warning(f"Error generating MCQ for document chunk: {e}")
        elif question_type.lower() == "theory":
            # Placeholder for theory question generation (details can be added in the next prompt)
            for doc in documents[:num_questions]:
                prompt = f"Generate a theory question on the topic '{topic}' at '{difficulty}' difficulty level."
                try:
                    response = model.generate_content(prompt)
                    questions.append({
                        "question": response.text.strip(),
                        "marks": total_marks // len(documents)  # Split marks equally
                    })
                except Exception as e:
                    logging.warning(f"Error generating theory question for document chunk: {e}")

        return {"questions": questions}

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return {"error": str(e)}

@app.post("/verify")
async def verify(
    pdf_filename: str = Form(...),
    user_answers: List[str] = Body(...),
    question_type: str = Form(...)  # "mcq" or "theory"
):
    """Verify the user's answers against the generated questions."""
    base_folder = "static/docs/"
    file_path = os.path.join(base_folder, pdf_filename)

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    logging.info(f"Verifying answers for file: {file_path}")
    qa_chain, questions = llm_pipeline(file_path)

    if len(user_answers) != len(questions):
        return {"error": "Number of answers doesn't match the number of questions."}

    results = []
    for question, user_answer in zip(questions, user_answers):
        correct_answer = qa_chain.run(question)
        similarity_score = embeddings.similarity(user_answer, correct_answer)
        is_correct = similarity_score > 0.75  # Consider making this threshold configurable

        result = {
            "question": question,
            "user_answer": user_answer,
            "is_correct": is_correct,
            "correct_answer": correct_answer if not is_correct else None,
        }
        if question_type == "mcq" and not is_correct:
            result["correct_answer"] = correct_answer.split('\n')[-1]  # Extract the correct option

        results.append(result)

    return {"results": results}

# Entry point for Uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)