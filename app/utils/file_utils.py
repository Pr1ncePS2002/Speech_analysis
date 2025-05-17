import uuid
import os
from fastapi import UploadFile

async def save_temp_file(file: UploadFile) -> str:
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())
    return filename

def remove_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
