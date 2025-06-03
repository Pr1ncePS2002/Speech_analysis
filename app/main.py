from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import speech_routes
import logging
from app.routes import resume_routes 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Speech Analysis API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(speech_routes.router, prefix="/api/speech")
app.include_router(resume_routes.router, prefix="/api/resume")
@app.get("/")
async def root():
    return {"message": "Speech Analysis API is running"}