# app/routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import crud
from app.auth import auth_utils
from app.database.models import User, ChatHistory # Import models for type hinting


router = APIRouter()

# Dependency to get a database session
# This is usually defined in db.py or main.py; importing from crud is also fine if defined there.
from app.database.db import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic model for saving chat data
class ChatCreate(BaseModel):
    user_id: int
    question: str
    answer: str

@router.get("/history/{user_id}", response_model=list[dict])
def get_user_chat_history(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth_utils.get_current_user) # Protect this endpoint
):
    """
    Retrieves the chat history for a specific user.
    Requires authentication. The authenticated user must match the user_id in the path.
    """
    # Security check: Ensure the requesting user is authorized to view this history
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this user's chat history"
        )
    
    chat_history = crud.get_chat_history_for_user(db, user_id=user_id)
    # Convert ChatHistory objects to dictionaries for JSON serialization
    return [
        {
            "id": chat.id,
            "user_id": chat.user_id,
            "question": chat.question,
            "answer": chat.answer,
            "timestamp": chat.timestamp.isoformat() # Convert datetime to ISO format string
        }
        for chat in chat_history
    ]

@router.post("/save")
def save_chat_entry(
    chat: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth_utils.get_current_user) # Protect this endpoint
):
    """
    Saves a new chat entry for a user.
    Requires authentication. The authenticated user must match the user_id in the request body.
    """
    # Security check: Ensure the requesting user is authorized to save chat for this user_id
    if current_user.id != chat.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to save chat for this user"
        )

    db_chat = crud.save_chat(
        db, 
        user_id=chat.user_id, 
        question=chat.question, 
        answer=chat.answer
    )
    return {"message": "Chat saved successfully", "chat_id": db_chat.id}

