# app/database/models.py
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean 
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.db import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    chats = relationship("ChatHistory", back_populates="user")
    # New: Relationship to blacklisted tokens 
    # blacklisted_tokens = relationship("BlacklistedToken", back_populates="user")


class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(String)
    answer = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="chats")

# <<< NEW MODEL: BlacklistedToken
class BlacklistedToken(Base):
    __tablename__ = "blacklisted_tokens"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String, unique=True, index=True, nullable=False) # JWT ID claim
    token_type = Column(String, nullable=False) # e.g., "refresh", "access" (though typically only refresh are blacklisted)
    expires_at = Column(DateTime, nullable=False) # When the original token would have expired
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Optional: Link to user for logging/management (uncomment if you add relationship in User model)
    # user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    # user = relationship("User", back_populates="blacklisted_tokens")