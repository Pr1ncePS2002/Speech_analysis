# app/auth/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.database.db import SessionLocal
from app.database import crud
from app.auth import auth_utils
from app.schemas import UserCreate # <<< NEW IMPORT

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/signup")
# Change the parameter to accept UserCreate model as a request body
def signup(user_data: UserCreate, db: Session = Depends(get_db)): # <<< CHANGED
    if crud.get_user_by_username(db, user_data.username): # Access fields via user_data
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Pass individual fields from user_data to crud.create_user
    user = crud.create_user(db, user_data.username, user_data.email, user_data.password) # <<< CHANGED
    return {"message": "User created", "user_id": user.id}

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, form_data.username)
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth_utils.create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}
