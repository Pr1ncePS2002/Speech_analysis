from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.whisper_service import transcribe_audio
from app.utils.file_utils import save_temp_file, remove_file

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio_route(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are supported.")

    temp_filename = await save_temp_file(file)

    try:
        result = transcribe_audio(temp_filename)
        return {"text": result["text"]}
    finally:
        remove_file(temp_filename)
