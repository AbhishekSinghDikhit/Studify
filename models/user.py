from config.sqliteConfig import get_db

def create_user(google_id, name, email, picture):
    """Creates a new user in the database."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO users (google_id, name, email, picture)
    VALUES (?, ?, ?, ?);
    """, (google_id, name, email, picture))

    conn.commit()
    conn.close()

def get_user_by_google_id(google_id):
    """Fetches user by Google ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM users WHERE google_id = ?;
    """, (google_id,))
    user = cursor.fetchone()
    conn.close()

    return user
