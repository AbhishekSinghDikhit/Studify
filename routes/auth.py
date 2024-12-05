from fastapi import APIRouter, Depends
from utils.firebase_config import verify_token
from models.user import create_user, get_user_by_google_id

router = APIRouter()

@router.post("/auth/login")
def login(id_token: str):
    """Login route to verify token and fetch/create user."""
    decoded_token = verify_token(id_token)
    if decoded_token:
        google_id = decoded_token["uid"]
        name = decoded_token["name"]
        email = decoded_token["email"]
        picture = decoded_token["picture"]

        # Add user to database
        create_user(google_id, name, email, picture)

        # Fetch user
        user = get_user_by_google_id(google_id)
        return {"message": "User logged in successfully", "user": dict(user)}
    else:
        return {"error": "Invalid ID token"}
