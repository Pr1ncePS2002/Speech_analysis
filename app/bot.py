import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
API_ENDPOINT = "http://localhost:8000/api/speech/upload"
RESUME_ENDPOINT = "http://localhost:8000/api/resume/upload"
ALLOWED_FILE_TYPES = ["wav", "mp3", "m4a"]

# Page setup
st.set_page_config(page_title="Speech Assistant", layout="wide")

# Header
st.markdown("## 🎙️ Speech Analysis Assistant")
st.markdown("Enhance your speaking skills, upload resumes, and prepare for interviews with AI.")

# Session State
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
        memory = ConversationBufferWindowMemory(input_key="message", memory_key="chat_history", k=10)
        st.session_state.chatbot_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return st.session_state.chatbot_chain

def chat_with_bot(message):
    if isinstance(message, dict) and 'text' in message:
        message = message['text']
    return initialize_chatbot().run(message=str(message))


# --- Audio Processing ---
def process_audio(uploaded_file):
    try:
        with st.spinner("🔊 Processing audio..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(API_ENDPOINT, files=files, timeout=30)
            if response.status_code == 200:
                return response.json()["transcript"]["text"]
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"🚨 Connection error: {e}")


# --- Sidebar ---

with st.sidebar:
    st.header("📄 Upload Resume")

    # Resume Upload
    resume_file = st.file_uploader("Upload Resume here", type=["pdf", "docx"])
    if resume_file:
        with st.spinner("📄 Parsing resume..."):
            try:
                files = {"file": (resume_file.name, resume_file.getvalue())}
                response = requests.post(RESUME_ENDPOINT, files=files, timeout=20)
                if response.status_code == 200:
                    parsed = response.json().get("parsed", {})
                    st.session_state.skills = parsed.get("skills", [])
                    st.session_state.role = parsed.get("roles", [None])[0]
                    st.success("✅ Resume parsed!")
                else:
                    st.error("❌ Resume parsing failed.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Audio Upload (Collapsed by default)
    with st.expander("🎙️ Audio Input", expanded=False):
        uploaded_file = st.file_uploader("Upload Audio", type=ALLOWED_FILE_TYPES)
        audio = st.audio_input("Or Record Audio")



# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "🧠 Interview Questions", "🎭 Mock Interview"])

# --- Tab 1: Chat ---
with tab1:
    st.subheader("💬 Chat-based Speech Feedback")

    if audio or uploaded_file:
        input_file = audio if audio else uploaded_file
        transcript = process_audio(input_file)
        if transcript:
            st.session_state.messages.append({"role": "User", "content": transcript})
            with st.spinner("🤖 Analyzing..."):
                response = chat_with_bot(transcript)
                st.session_state.messages.append({"role": "Bot", "content": response})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your speech or resume..."):
        st.session_state.messages.append({"role": "User", "content": prompt})
        with st.chat_message("User"):
            st.markdown(prompt)
        with st.chat_message("Bot"):
            with st.spinner("Thinking..."):
                response = chat_with_bot(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "Bot", "content": response})


# --- Tab 2: Interview Questions ---
with tab2:
    st.subheader("🧠 Auto-Generated Interview Questions")

    skills = st.session_state.get("skills", [])
    role = st.session_state.get("role", "")

    if not skills or not role:
        st.warning("⚠️ Please upload your resume first.")
    else:
        st.markdown(f"**Role**: {role}")
        st.markdown("**Skills:** " + ", ".join(skills))
        st.info("ℹ️ Question generation using Gemini coming soon!")


# --- Tab 3: Mock Interview ---
with tab3:
    st.subheader("🎭 Mock Interview Practice")

    skills = st.session_state.get("skills", [])
    role = st.session_state.get("role", "")

    if not skills or not role:
        st.warning("⚠️ Please upload your resume first.")
    else:
        st.markdown(f"**Role**: {role}")
        st.markdown("**Skills:**")
        st.code(", ".join(skills))
        st.info("ℹ️ Gemini-powered mock interview will be available soon!")
