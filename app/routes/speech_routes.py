from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from app.services.whisper_service import transcribe_audio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
            raise HTTPException(status_code=400, detail="Only WAV, MP3, and M4A files are allowed")

        temp_file_path = f"temp_{file.filename}"
        
        # Async file handling
        with open(temp_file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(content)

        logger.info(f"Processing file: {file.filename}")
        result = transcribe_audio(temp_file_path)

        # Cleanup
        os.remove(temp_file_path)
        
        return {"status": "success", "transcript": result}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))