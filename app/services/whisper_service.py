import whisper
import os
# from pydub import AudioSegment
# Load the Whisper model
model = whisper.load_model("small")

# def convert_to_wav(input_file):
#     # Load the audio file
#     audio = AudioSegment.from_file(input_file)
    
#     # Export as WAV
#     audio.export("output.wav", format="wav", parameters=["-ac", "2", "-ar", "44100"])
    
#     # Return the output file path
#     return "output.wav"

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe audio using Whisper Medium model.
    :param file_path: Path to the uploaded audio file
    :return: Dictionary with transcription result
    """
    if not os.path.exists(file_path):
        return {"error": "File not found."}

    print(f"Transcribing {file_path} using Whisper Medium on CPU...")
    
    # wav_file = convert_to_wav(file_path)
    result = model.transcribe(file_path)
    return {"text": result["text"]}
