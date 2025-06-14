# app/database/crud.py
from sqlalchemy.orm import Session
from app.database.models import User, ChatHistory, BlacklistedToken # <<< Import BlacklistedToken
from app.auth.auth_utils import hash_password # Ensure this import is correct
from datetime import datetime, timezone # Import datetime, timezone for UTC comparison

# Dependency to get a database session (can also be defined here or in db.py)
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

def save_chat(db: Session, user_id: int, question: str, answer: str): # <<< Added chat_type
    """Saves a chat history entry for a specific user."""
    # Ensure ChatHistory model has 'chat_type' if you want to save it
    # If not, remove chat_type from here or add it to ChatHistory model.
    chat = ChatHistory(user_id=user_id, question=question, answer=answer)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

def get_chat_history_for_user(db: Session, user_id: int):
    """Retrieves all chat history entries for a given user, ordered by timestamp."""
    return db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.timestamp).all()


# <<< NEW FUNCTIONS FOR TOKEN BLACKLISTING

def add_token_to_blacklist(db: Session, jti: str, token_type: str, expires_at: datetime):
    """Adds a token's JTI to the blacklist."""
    blacklisted_entry = BlacklistedToken(
        jti=jti,
        token_type=token_type,
        expires_at=expires_at
    )
    db.add(blacklisted_entry)
    db.commit()
    db.refresh(blacklisted_entry)
    return blacklisted_entry

def is_token_blacklisted(db: Session, jti: str) -> bool:
    """Checks if a token's JTI is in the blacklist and is not expired."""
    # Filter by jti and ensure the blacklist entry itself hasn't expired (optional cleanup logic)
    # The primary check is just by JTI existence. Expiration is handled by JWT itself.
    # We remove entries from the blacklist once their original token has expired.
    return db.query(BlacklistedToken).filter(
        BlacklistedToken.jti == jti,
        BlacklistedToken.expires_at > datetime.now(timezone.utc) # Only consider active blacklist entries
    ).first() is not None

def remove_expired_blacklisted_tokens(db: Session):
    """Removes blacklisted tokens that have already expired."""
    db.query(BlacklistedToken).filter(
        BlacklistedToken.expires_at <= datetime.now(timezone.utc)
    ).delete()
    db.commit()