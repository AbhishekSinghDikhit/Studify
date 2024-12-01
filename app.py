from fastapi import FastAPI, Form, Request, UploadFile, File
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
from langchain.llms import HuggingFaceHub
from typing import List
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

# Hugging Face models
text_generator = pipeline("text-generation", model="gpt2", device=-1)  # Lightweight GPT-2
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def file_processing(file_path):
    """Load and split PDF content into chunks."""
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_content = ""
        text_content = "".join([page.page_content for page in data])

        if not text_content.strip():
            logging.error("No text content found in the PDF.")
            raise ValueError("The uploaded PDF contains no text.")

        splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text_content)

        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        if not valid_chunks:
            logging.error("No valid text chunks found.")
            raise ValueError("Failed to split text into valid chunks.")

        documents = [Document(page_content=chunk) for chunk in valid_chunks]
        logging.info(f"Generated {len(documents)} valid document chunks.")
        return documents
    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        raise

def generate_questions(text):
    """Generate questions using a text generation model."""
    if not text.strip():
        logging.error("Empty text provided for question generation.")
        raise ValueError("Cannot generate questions from empty text.")
    
    max_input_tokens = 800  # Reserve space for model output
    truncated_text = text[:max_input_tokens]

    prompt = f"""
    {truncated_text}
    """
    try:
        result = text_generator(prompt, max_new_tokens=100, num_return_sequences=1)
        return result[0]["generated_text"].strip()
    except Exception as e:
        logging.error(f"Error during question generation: {e}")
        raise

def llm_pipeline(file_path):
    """Generate questions and prepare the retrieval chain."""
    documents = file_processing(file_path)
    if not documents:
        logging.error("No documents created from the PDF.")
        raise ValueError("No documents created from the PDF content.")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    huggingfacehub_api_token = "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"  

    llm = HuggingFaceHub(
        repo_id="gpt2",
        model_kwargs={"temperature": 0.5, "max_length": 100},
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
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

    return retrieval_qa_chain, questions

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
    # base_folder = "static/docs/"
    file_path = pdf_filename

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