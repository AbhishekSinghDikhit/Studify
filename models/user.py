from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    profile_picture_url: Optional[str] = None
