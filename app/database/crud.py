# app/database/crud.py
# DB logic: create user, get user, save chat, get chat history
from sqlalchemy.orm import Session
from app.database.models import User, ChatHistory
from app.auth.auth_utils import hash_password # Ensure this import is correct

# Dependency to get a database session (can also be defined here or in db.py)
# For consistency, I'll define it here if it's not strictly in db.py for some reason.
# If you prefer it only in db.py or main.py, remove this `get_db` definition.
from app.database.db import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_username(db: Session, username: str):
    """Retrieves a user by their username."""
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, username: str, email: str, password: str):
    """Creates a new user and hashes their password."""
    db_user = User(username=username, email=email, hashed_password=hash_password(password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def save_chat(db: Session, user_id: int, question: str, answer: str):
    """Saves a chat history entry for a specific user."""
    chat = ChatHistory(user_id=user_id, question=question, answer=answer)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

def get_chat_history_for_user(db: Session, user_id: int):
    """Retrieves all chat history entries for a given user, ordered by timestamp."""
    # Ensure to order by timestamp for a chronological display
    return db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.timestamp).all()
