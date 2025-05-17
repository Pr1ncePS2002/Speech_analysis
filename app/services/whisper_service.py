import whisper
#logic to transcribe audio to text using Whisper
# Load the whisper model 
model = whisper.load_model("base")

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribes audio using OpenAI's Whisper.
    Returns transcription text and segments.
    """
    result = model.transcribe(file_path, language="en", verbose=False)
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", [])
    }
