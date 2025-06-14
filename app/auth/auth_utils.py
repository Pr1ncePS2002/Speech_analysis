# app/auth/auth_utils.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import uuid # <<< NEW IMPORT: For generating unique JWT IDs (jti)

# Import models and crud for get_current_user and blacklisting
from app.database import crud
from app.database.models import User # Ensure User model is imported if used in get_current_user
from app.database.db import SessionLocal 

# --- Dependency to get a database session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# <<< IMPORTANT: CHANGE THIS TO A STRONG, RANDOM KEY IN PRODUCTION!
SECRET_KEY = "our-secret-key"
ALGORITHM = "HS256" 

ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Short-lived access token
REFRESH_TOKEN_EXPIRE_DAYS = 7    # Longer-lived refresh token

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login") 

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed: str):
    return pwd_context.verify(plain_password, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a JWT access token.
    data: Dictionary to encode into the token (e.g., {"sub": username}).
    expires_delta: Optional timedelta for token expiration.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Generate a unique JWT ID (jti)
    jti = str(uuid.uuid4()) # <<< ADDED jti
    
    to_encode.update({"exp": expire, "token_type": "access", "jti": jti}) 
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Creates a JWT refresh token.
    data: Dictionary to encode into the token (e.g., {"sub": username}).
    expires_delta: Optional timedelta for token expiration.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
    # Generate a unique JWT ID (jti)
    jti = str(uuid.uuid4()) # <<< ADDED jti
    
    to_encode.update({"exp": expire, "token_type": "refresh", "jti": jti}) 
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    """Decodes a JWT token. Returns payload or None if decoding fails."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Dependency function to get the current authenticated user from a JWT token.
    Raises HTTPException if the token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("token_type")
        jti: str = payload.get("jti") # <<< Get jti

        if username is None or token_type != "access" or jti is None: 
            raise credentials_exception
        
        # <<< NEW CHECK: Is this access token blacklisted? (Less common for access tokens due to short expiry)
        # However, for immediate logout effect on all tokens, this is useful.
        if crud.is_token_blacklisted(db, jti):
            raise credentials_exception # Token is blacklisted

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = crud.get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user