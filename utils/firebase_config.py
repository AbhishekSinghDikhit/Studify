import firebase_admin
from firebase_admin import credentials, auth
import logging
import time
import jwt

# Initialize Firebase Admin SDK
cred = credentials.Certificate("D:\Abhishek\Projects\Studify\config\eduai-17e96-firebase-adminsdk-mue14-f016774857.json")
firebase_admin.initialize_app(cred)

def verify_token(id_token):
    try:
        decoded_token = jwt.decode(
            id_token,
            key="AIzaSyA0jZcDLS9b31fVf5Wb3yk4XNBaC9lVYkw",  # Replace with your actual key
            algorithms=["RS256"],
            options={"verify_exp": True, "verify_nbf": True},
            leeway=60  # Add 60 seconds of leeway
        )
        return decoded_token
    except jwt.ExpiredSignatureError:
        logging.error("Token expired")
    except jwt.InvalidTokenError as e:
        logging.error(f"Error verifying token: {e}")
    return None
