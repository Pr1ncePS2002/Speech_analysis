import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
API_ENDPOINT = "http://localhost:8000/api/speech/upload"
ALLOWED_FILE_TYPES = ["wav", "mp3", "m4a"]

# Initialize Streamlit
st.set_page_config(page_title="Speech Assistant", layout="wide")
st.title("🎙️ Speech Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "Bot", "content": "Hi! I'm your Speech Assistant. How can I help you today?"}
    ]
if "chatbot_chain" not in st.session_state:
    st.session_state.chatbot_chain = None

# --- Chatbot Functions ---
def initialize_chatbot():
    if st.session_state.chatbot_chain is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)

        prompt = ChatPromptTemplate.from_template(
            """You are an expert communication coach analyzing speech patterns:
            
            History: {chat_history}
            Current: {message}

            Provide analysis covering:
            1. Grammar/Fillers
            2. Pauses/Flow
            3. Vocabulary
            4. Fluency Score (1-10)
            5. Improvement Suggestions"""
        )

        memory = ConversationBufferWindowMemory(
            input_key="message",
            memory_key="chat_history",
            k=10
        )

        st.session_state.chatbot_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
    return st.session_state.chatbot_chain

def chat_with_bot(message):
    if isinstance(message, dict) and 'text' in message:
        message = message['text']
    
    if not isinstance(message, str):
        message = str(message)
    
    chain = initialize_chatbot()  # ✅ GET the chain here
    return chain.run(message=message)



# --- Audio Processing ---
def process_audio(uploaded_file):
    try:
        with st.spinner("🔊 Processing audio..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(API_ENDPOINT, files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data["transcript"]
            else:
                st.error(f"❌ Backend error {response.status_code}: {response.text}")
                return None
    except Exception as e:
        st.error(f"🚨 Connection error: {str(e)}")
        return None

# --- UI Components ---
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader(
        "Drag file here",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=False,
        help="Max 200MB, WAV/MP3/M4A only"
    )
    st.header("Record Audio")
    audio = st.audio_input("Record a voice message")
    if audio:
            transcript = process_audio(audio)
            if transcript:
                st.session_state.messages.append({"role": "User", "content": transcript})
                with st.spinner("🤖 Analyzing..."):
                    response = chat_with_bot(transcript)
                    st.session_state.messages.append({"role": "Bot", "content": response})

    if uploaded_file:
        transcript = process_audio(uploaded_file)
        if transcript:
            st.session_state.messages.append({"role": "User", "content": transcript})
            with st.spinner("🤖 Analyzing..."):
                response = chat_with_bot(transcript)
                st.session_state.messages.append({"role": "Bot", "content": response})
    

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your speech analysis..."):
    st.session_state.messages.append({"role": "User", "content": prompt})
    with st.chat_message("User"):
        st.markdown(prompt)
    
    with st.chat_message("Bot"):
        with st.spinner("Thinking..."):
            response = chat_with_bot(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "Bot", "content": response})