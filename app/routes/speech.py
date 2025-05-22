from fastapi import APIRouter, UploadFile, File
import shutil
import os
from app.services.whisper_service import transcribe_audio

router = APIRouter()

@router.post("/speech/upload")
async def upload_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = transcribe_audio(temp_file_path)

    # Cleanup
    os.remove(temp_file_path)

    return result
