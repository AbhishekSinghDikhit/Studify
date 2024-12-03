from pymongo import MongoClient
import os

# MongoDB connection setup
def get_database():
    client = MongoClient(os.getenv("MONGODB_URI"))  # Use environment variable for connection string
    return client['eduai-database']

def get_user_collection():
    db = get_database()
    return db['users']  # MongoDB collection for users

def get_questions_collection():
    db = get_database()
    return db['questions']  # MongoDB collection for questions
