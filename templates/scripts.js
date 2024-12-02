const uploadBtn = document.getElementById('upload-btn');
const pdfFileInput = document.getElementById('pdf-file');
const questionsSection = document.getElementById('questions-section');
const resultsSection = document.getElementById('results-section');
const questionsList = document.getElementById('questions-list');
const resultsList = document.getElementById('results-list');
const answersForm = document.getElementById('answers-form');

let pdfFilename = ''; // Store the uploaded PDF filename
let questions = []; // Store the generated questions

// Handle PDF upload and question generation
uploadBtn.addEventListener('click', async () => {
    const file = pdfFileInput.files[0];
    if (!file) {
        alert('Please upload a PDF file.');
        return;
    }

    // Show the uploading message
    document.getElementById('uploading-message').style.display = 'block';

    const formData = new FormData();
    formData.append('pdf_file', file); // Ensure the key matches the backend
    formData.append('filename', file.name); // Include filename

    try {
        // Upload PDF to backend
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        // Check if upload was successful
        if (!uploadResponse.ok) {
            const error = await uploadResponse.json();
            alert(`Error: ${error.detail}`);
            return;
        }

        // Get the PDF filename from the backend response
        const uploadResult = await uploadResponse.json();
        pdfFilename = uploadResult.pdf_filename; // Ensure filename is correctly set
        console.log(`File uploaded: ${pdfFilename}`);
        
        // Hide the uploading message
        document.getElementById('uploading-message').style.display = 'none';
        
        // Show the processing message
        document.getElementById('processing-message').style.display = 'block';

        // Generate questions from the uploaded PDF
        const analyzeResponse = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ pdf_filename: pdfFilename })
        });

        // Handle analyze response
        const analyzeResult = await analyzeResponse.json();

        // Check for errors in the analysis response
        if (analyzeResult.error) {
            alert(analyzeResult.error);
            return;
        }

        // Set the questions list
        questions = analyzeResult.questions;

        // Show the questions section
        questionsSection.style.display = 'block';

        // Hide the processing message
        document.getElementById('processing-message').style.display = 'none';

        // Populate questions in the frontend
        questionsList.innerHTML = '';
        questions.forEach((question, index) => {
            questionsList.innerHTML += `
                <div class="mb-3">
                    <p><strong>Q${index + 1}:</strong> ${question}</p>
                    <textarea class="form-control" id="answer-${index}" placeholder="Enter your answer"></textarea>
                </div>
            `;
        });

    } catch (error) {
        console.error('Upload failed:', error);
        alert('An error occurred during file upload or processing.');
    }
});

// Handle answer submission and verification
answersForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const userAnswers = questions.map((_, index) => {
        return document.getElementById(`answer-${index}`).value;
    });

    // Verify the answers
    const verifyResponse = await fetch('/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
            pdf_filename: pdfFilename,
            user_answers: JSON.stringify(userAnswers)
        })
    });

    // Handle verification response
    const verifyResult = await verifyResponse.json();

    // Display the results
    resultsList.innerHTML = '';
    verifyResult.results.forEach((result, index) => {
        resultsList.innerHTML += `
            <div class="mb-3">
                <p><strong>Q${index + 1}:</strong> ${result.question}</p>
                <p><strong>Your Answer:</strong> ${result.user_answer}</p>
                <p><strong>Result:</strong> ${result.is_correct ? 'Correct' : 'Incorrect'}</p>
                ${!result.is_correct ? `<p><strong>Correct Answer:</strong> ${result.correct_answer}</p>` : ''}
            </div>
        `;
    });

    // Show the results section
    resultsSection.style.display = 'block';
});