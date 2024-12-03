from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from utils.firebase_config import verify_token
from fastapi.responses import RedirectResponse

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency to get current user from Firebase
def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# Middleware to check if the user is logged in and redirect if necessary
@app.get("/question-generator")
async def question_generator(request: Request, user: dict = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/sign-in")
    return templates.TemplateResponse("question-generator.html", {"request": request})
