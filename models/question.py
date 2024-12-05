from config.sqliteConfig import get_db

def add_question(user_id, question, options, correct_answer):
    """Adds a question to the database."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO questions (user_id, question, options, correct_answer)
    VALUES (?, ?, ?, ?);
    """, (user_id, question, options, correct_answer))

    conn.commit()
    conn.close()

def get_questions_by_user(user_id):
    """Fetches all questions for a specific user."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM questions WHERE user_id = ? ORDER BY created_at DESC;
    """, (user_id,))
    questions = cursor.fetchall()
    conn.close()

    return questions
