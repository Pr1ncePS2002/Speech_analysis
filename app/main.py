from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import speech

app = FastAPI()

# Allow frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(speech.router, prefix="/api/speech")
