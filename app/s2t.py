import gradio as gr
from faster_whisper import WhisperModel
import os

# Load the model
model = WhisperModel("small", compute_type="float16")  # use "base" for faster response // use "float16" if on GPU

def transcribe(audio_path):
    print("Received audio path:", audio_path)

    if not audio_path or not os.path.isfile(audio_path):
        return "⚠️ Please provide a valid audio file."

    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcription = " ".join(segment.text for segment in segments)
        return f"🗣 Detected language: {info.language}\n\n📝 Transcription:\n{transcription.strip()}"
    
    except Exception as e:
        return f"❌ Error during transcription: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""## 🎤 Fast Whisper ASR Gradio Web UI  
    Upload or record audio and press Transcribe.""")

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Audio Input"
    )

    submit_btn = gr.Button("Transcribe")

    output_text = gr.Textbox(label="Transcription Result", lines=6)

    submit_btn.click(fn=transcribe, inputs=audio_input, outputs=output_text)

demo.launch()
