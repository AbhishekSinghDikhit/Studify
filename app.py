from fastapi import FastAPI, Form, Request, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Hugging Face models
text_generator = pipeline("text-generation", model="gpt2", device=-1)  # Lightweight GPT-2
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def file_processing(file_path):
    """Load and split PDF content into chunks."""
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_content = ""
    for page in data:
        text_content += page.page_content

    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text_content)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def generate_questions(text):
    """Generate questions using a text generation model."""
    prompt = f"""
    Based on the text below, generate questions:
    {text}
    """
    result = text_generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"].strip()

def llm_pipeline(file_path):
    """Generate questions and prepare the retrieval chain."""
    documents = file_processing(file_path)
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    retrieval_qa_chain = RetrievalQA.from_chain_type(retriever=retriever, chain_type="stuff")
    questions = [generate_questions(doc.page_content) for doc in documents]
    return retrieval_qa_chain, questions

@app.get("/")
async def index(request: Request):
    """Render the main index.html."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(pdf_file: bytes = File(...), filename: str = Form(...)):
    """Handle PDF file upload."""
    base_folder = "static/docs/"
    os.makedirs(base_folder, exist_ok=True)
    pdf_filename = os.path.join(base_folder, filename)

    try:
        # Save the uploaded file
        async with aiofiles.open(pdf_filename, "wb") as f:
            await f.write(pdf_file)
        logging.info(f"Uploaded file saved to: {pdf_filename}")
        return jsonable_encoder({"msg": "success", "pdf_filename": pdf_filename})
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return {"error": f"Failed to save file: {e}"}

@app.post("/analyze")
async def analyze(pdf_filename: str = Form(...)):
    """Analyze the uploaded PDF and generate questions."""
    base_folder = "static/docs/"
    file_path = os.path.join(base_folder, pdf_filename)

    # Validate file existence
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}

    try:
        logging.info(f"Analyzing file: {file_path}")
        qa_chain, questions = llm_pipeline(file_path)
        logging.info(f"Generated {len(questions)} questions.")  # Log number of questions generated
        if not questions:
            return {"error": "No questions could be generated from the PDF content."}
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
        is_correct = similarity_score > 0.75

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
