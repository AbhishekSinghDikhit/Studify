import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
cred = credentials.Certificate("D:\Question-_Paper_gen\config\eduai-17e96-firebase-adminsdk-mue14-f016774857.json")
firebase_admin.initialize_app(cred)

def verify_token(id_token: str):
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Error verifying token: {e}")
        return None
