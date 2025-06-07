# app/schemas.py
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr # Use EmailStr for basic email validation
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatCreate(BaseModel):
    user_id: int
    question: str
    answer: str

# You might add more schemas for your other routes (speech, resume) as needed
