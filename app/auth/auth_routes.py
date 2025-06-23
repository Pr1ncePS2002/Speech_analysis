# app/auth/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.database.db import SessionLocal
from app.database import crud
from app.auth import auth_utils
from app.schemas import UserCreate
from typing import Annotated 
from datetime import datetime, timezone # For blacklisting expiry

from fastapi import Depends
from app.auth.auth_utils import get_current_user
from app.database.models import User

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/signup")
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_username(db, user_data.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    user = crud.create_user(db, user_data.username, user_data.email, user_data.password)
    return {"message": "User created", "user_id": user.id}

@router.post("/login")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, form_data.username)
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Access token and refresh token now include 'jti'
    access_token = auth_utils.create_access_token(data={"sub": user.username})
    refresh_token = auth_utils.create_refresh_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
        "user_id": user.id
    }

@router.post("/refresh")
def refresh_access_token(
    refresh_token: Annotated[str, Depends(auth_utils.oauth2_scheme)], 
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = auth_utils.decode_token(refresh_token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    token_type: str = payload.get("token_type")
    old_refresh_jti: str = payload.get("jti") # <<< Get JTI of the old refresh token
    old_refresh_exp: datetime = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc) # <<< Get expiry of old refresh token

    if username is None or token_type != "refresh" or old_refresh_jti is None: 
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token type or missing subject/jti",
        )
    
    # <<< NEW CHECK: Is the old refresh token already blacklisted?
    if crud.is_token_blacklisted(db, old_refresh_jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token already used or revoked.",
        )

    # 1. (Optional) Check if the user is still active in the database:
    user = crud.get_user_by_username(db, username=username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    # 2. Blacklist the old refresh token
    crud.add_token_to_blacklist(db, old_refresh_jti, "refresh", old_refresh_exp)

    # 3. Issue a new access token AND a new refresh token (token rotation)
    new_access_token = auth_utils.create_access_token(data={"sub": username})
    new_refresh_token = auth_utils.create_refresh_token(data={"sub": username}) # <<< New refresh token

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "refresh_token": new_refresh_token # <<< Return new refresh token
    }

# <<< NEW ENDPOINT: Logout
@router.post("/logout")
def logout(
    token: Annotated[str, Depends(auth_utils.oauth2_scheme)], # Can be access or refresh token
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token for logout",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = auth_utils.decode_token(token)
    if payload is None:
        raise credentials_exception
    
    jti: str = payload.get("jti")
    token_type: str = payload.get("token_type")
    token_exp: datetime = datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)

    if jti is None or token_type is None:
        raise credentials_exception # Token doesn't have required JTI or type

    # Blacklist the token based on its type
    # Usually, you'd only blacklist refresh tokens for a full session logout
    # or both if you want to invalidate even current access token for all devices.
    # For a full logout, blacklisting the refresh token is key.
    crud.add_token_to_blacklist(db, jti, token_type, token_exp)
    
    return {"message": f"{token_type.capitalize()} token successfully blacklisted."}

# (Optional) Endpoint to clean up expired blacklisted tokens
@router.post("/clean-blacklist")
def clean_blacklist_endpoint(db: Session = Depends(get_db)):
    """
    Endpoint to manually trigger cleanup of expired blacklisted tokens.
    In a real application, this would be a scheduled background task.
    """
    crud.remove_expired_blacklisted_tokens(db)
    return {"message": "Expired blacklisted tokens cleaned up."}

@router.get("/me")
def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email
    }