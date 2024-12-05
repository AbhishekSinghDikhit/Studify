from gridfs import GridFS
from datetime import datetime


# Initialize GridFS for storing files
fs = GridFS(db)

def upload_pdf(user_id, pdf_file):
    # Save the PDF to GridFS
    pdf_id = fs.put(pdf_file, filename=pdf_file.filename, user_id=user_id, upload_date=datetime.now())
    
    # Update the user's uploaded_pdfs collection with the file info
    db.users.update_one(
        {"_id": user_id},
        {"$push": {"uploaded_pdfs": {"file_id": pdf_id, "file_name": pdf_file.filename, "upload_date": datetime.now()}}}
    )
    return pdf_id