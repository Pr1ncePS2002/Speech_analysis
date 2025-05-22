import whisper
import os

# Load the Whisper model
model = whisper.load_model("medium")

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe audio using Whisper Medium model.
    :param file_path: Path to the uploaded audio file
    :return: Dictionary with transcription result
    """
    if not os.path.exists(file_path):
        return {"error": "File not found."}

    print(f"Transcribing {file_path} using Whisper Medium on CPU...")

    result = model.transcribe(file_path)
    return {"text": result["text"]}
