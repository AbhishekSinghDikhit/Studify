from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from google.generativeai import configure, GenerativeModel
from typing import List
from PyPDF2 import PdfReader
import os
import uvicorn
import aiofiles
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# model_name = "google/flan-t5-small"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# Initialize Gemini client
configure(api_key="AIzaSyDGKtZ-K_xXzQMNsZdWIslYuiGFxE1CXG8")
model = GenerativeModel("gemini-1.5-flash")
    
# # Hugging Face models
# text_generator = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=-1  # Use CPU; change to 0 for GPU if available
# )
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def file_processing(file_path):
    """Load and split PDF content into chunks."""
    try:
        # Open the PDF file and extract text
        reader = PdfReader(file_path)
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text()

        if not text_content.strip():
            raise ValueError("The uploaded PDF contains no extractable text.")

        # Split text into chunks
        splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text_content)

        documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
        if not documents:
            raise ValueError("No valid document chunks found.")

        return documents
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise

def generate_questions(text):
    """Generate questions using a text generation model."""
    prompt = f"Generate a question from the following text:\n\n{text[:800]}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error during question generation: {e}")
        raise

def llm_pipeline(file_path):
    """Generate questions and prepare the retrieval chain."""
    documents = file_processing(file_path)
    if not documents:
        logging.error("No documents created from the PDF.")
        raise ValueError("No documents created from the PDF content.")

    # Assuming Gemini supports embeddings and vectorization
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    questions = []
    for doc in documents:
        if len(questions) >= 10:
            break
        try:
            question = generate_questions(doc.page_content)
            questions.append(question)
        except Exception as e:
            logging.error(f"Skipping a document due to error: {e}")

    if not questions:
        logging.error("No questions could be generated.")
        raise ValueError("Failed to generate questions from the PDF.")

    return retriever, questions

@app.get("/")
async def index(request: Request):
    """Render the main index.html."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(pdf_file: UploadFile = File(...), filename: str = Form(...)):
    """Handle PDF file upload."""
    if not pdf_file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are allowed."}

    base_folder = "static/docs/"
    os.makedirs(base_folder, exist_ok=True)
    pdf_filename = os.path.join(base_folder, filename)

    try:
        # Save the uploaded file
        async with aiofiles.open(pdf_filename, "wb") as f:
            content = await pdf_file.read()
            await f.write(content)
        logging.info(f"Uploaded file saved to: {pdf_filename}")
        return jsonable_encoder({"msg": "success", "pdf_filename": pdf_filename})
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return {"error": f"Failed to save file: {e}"}

@app.post("/analyze")
async def analyze(pdf_filename: str = Form(...)):
    """Analyze the uploaded PDF and generate questions."""
    file_path = pdf_filename

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        documents = file_processing(file_path)
        questions = []
        for doc in documents:
            if len(questions) >= 2:  # Limit to 10 questions
                break
            question = generate_questions(doc.page_content)
            questions.append(question)
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return {"error": f"Analysis failed: {e}"}

    return {"questions": questions}

@app.post("/verify")
async def verify(pdf_filename: str = Form(...), user_answers: List[str] = Form(...)):
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
        results.append(result)

    return {"results": results}

# Entry point for Uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)