# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import speech_routes, resume_routes, chat_routes # Import chat_routes
from app.auth import auth_routes
from app.database.db import Base, engine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Speech Analysis API")

# Create database tables on startup
@app.on_event("startup")
def startup():
    """Ensures all database tables are created when the application starts."""
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created/checked.")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # IMPORTANT: In production, change this to your specific frontend domain(s)!
    allow_credentials=True,
    allow_methods=["*"],   # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],   # Allows all headers in the request
)

# Include routers
app.include_router(auth_routes.router, prefix="/api/auth")
app.include_router(speech_routes.router, prefix="/api/speech")
app.include_router(resume_routes.router, prefix="/api/resume")
app.include_router(chat_routes.router, prefix="/api/chat") # Include the new chat routes

@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": "Speech Analysis API is running"}

