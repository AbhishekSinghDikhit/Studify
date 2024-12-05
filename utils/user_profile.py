import sqlite3
from utils.database import get_db_connection

def save_user_data(name, email):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)", (name, email)
            )
    except sqlite3.IntegrityError as e:
        print(f"Error saving user data: {e}")
    finally:
        conn.close()
