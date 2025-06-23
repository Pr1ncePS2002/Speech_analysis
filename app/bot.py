import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import os
import requests
from dotenv import load_dotenv
import traceback
import logging
from typing import List, Dict, Any # Import Dict and Any for broader type hinting
import json # Import the json module
from services.whisper_service import transcribe_audio
from utils.bot_utils import chat_with_speech_bot, chat_with_interview_bot, chat_with_interviewer
import time
from services.resume_parser import parse_resume, parse_entire_resume


#  allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# For local testing, it's usually http://127.0.0.1:8000
FASTAPI_BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = f"{FASTAPI_BASE_URL}/api/speech/upload"
RESUME_ENDPOINT = f"{FASTAPI_BASE_URL}/api/resume/upload"
CHAT_HISTORY_FETCH_ENDPOINT = f"{FASTAPI_BASE_URL}/api/chat/history" # Will append user_id
CHAT_SAVE_ENDPOINT = f"{FASTAPI_BASE_URL}/api/chat/save"
LOGIN_ENDPOINT = f"{FASTAPI_BASE_URL}/api/auth/login"
SIGNUP_ENDPOINT = f"{FASTAPI_BASE_URL}/api/auth/signup"

ALLOWED_FILE_TYPES = ["wav", "mp3", "m4a"]

# Page setup
st.set_page_config(page_title="Speech Assistant", layout="wide")

# Header (will only show if logged in, or can be a general welcome)
st.markdown("## 🎙️ Speech Analysis Assistant")
st.markdown("Enhance your speaking skills, upload resumes, and prepare for interviews with AI.")

# --- Session State Initialization ---
# Existing session states
if "speech_chain" not in st.session_state:
    st.session_state.speech_chain = None
if "interview_qna_chain" not in st.session_state:
    st.session_state.interview_qna_chain = None
if "interviewer_chain" not in st.session_state:
    st.session_state.interviewer_chain = None
if "speech_messages" not in st.session_state:
    st.session_state.speech_messages = [
        {"role": "Bot", "content": "Hi! I'm your Speech Assistant. How can I help you today?"}
    ]
if "interview_qna_messages" not in st.session_state:
    st.session_state.interview_qna_messages = []
if "interviewer_messages" not in st.session_state:
    st.session_state.interviewer_messages = []
if "skills" not in st.session_state:
    st.session_state.skills = []
if "role" not in st.session_state:
    st.session_state.role = ""
if "entiredata" not in st.session_state:
    st.session_state.entiredata = {}

# New session states for authentication and chat history
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'user_chat_history' not in st.session_state: # Renamed to avoid conflict with existing 'chat_history' in chains
    st.session_state.user_chat_history = []


# --- API Functions for Auth & Chat History ---

def signup_user(username: str, email: str, password: str) -> Dict[str, Any] | None:
    """Sends a signup request to the FastAPI backend."""
    response = None # Initialize response to None
    try:
        response = requests.post(
            SIGNUP_ENDPOINT,
            json={"username": username, "email": email, "password": password}
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        # Catch any request-related errors (connection issues, HTTP errors)
        st.error(f"Signup error: {e}")
        if response is not None and response.content: # Check if response object exists and has content
            try:
                error_detail = response.json().get("detail", "Unknown error from server")
                st.error(f"Detail: {error_detail}")
                return {"detail": error_detail} # Return the detail so your Streamlit app can use it
            except json.JSONDecodeError:
                st.error(f"Detail: {response.text}") # Fallback if response content isn't JSON
                return {"detail": response.text}
        else:
            st.error("No response received from server or empty response.")
        return None # Return None if no structured error detail could be extracted

def login_user(username: str, password: str) -> Dict[str, Any] | None:
    """Sends a login request to the FastAPI backend."""
    try:
        # FastAPI's OAuth2PasswordRequestForm expects form-data, not JSON for /login
        response = requests.post(
            LOGIN_ENDPOINT,
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Login error: {e}")
        if response is not None:
            st.error(f"Detail: {response.text}")
        return None

def fetch_user_chat_history(user_id: int, access_token: str) -> List[Dict[str, Any]] | None:
    """Fetches chat history for the logged-in user."""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{CHAT_HISTORY_FETCH_ENDPOINT}/{user_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching chat history: {e}")
        if response is not None and response.status_code in [401, 403]:
            st.warning("Your session has expired or is invalid. Please log in again.")
            # Clear session state for re-login
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.access_token = None
            st.rerun() # Rerun to display login page
        return None

def logout():
    """Resets session state to log out the user."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.access_token = None
    st.session_state.user_chat_history = []
    # Reset chat chains/messages if desired, to start fresh
    st.session_state.speech_messages = [{"role": "Bot", "content": "Hi! I'm your Speech Assistant. How can I help you today?"}]
    st.session_state.interview_qna_messages = []
    st.session_state.interviewer_messages = []
    st.session_state.speech_chain = None
    st.session_state.interview_qna_chain = None
    st.session_state.interviewer_chain = None
    st.success("You have been logged out.")
    st.rerun() # Rerun the app to show login/signup

# --- Audio Processing ---
def process_audio(uploaded_file: Any) -> str | None:
    try:
        with st.spinner("🔊 Processing audio..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            headers = {}
            if st.session_state.logged_in and st.session_state.access_token:
                headers["Authorization"] = f"Bearer {st.session_state.access_token}"
            
            response = requests.post(API_ENDPOINT, files=files, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()["transcript"]["text"]
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"🚨 Connection error: {e}")
        st.text(traceback.format_exc())
    return None

# --- Resume Processing ---
def process_resume(resume_file: Any) -> Dict[str, Any] | None:
    try:
        with st.spinner("📄 Parsing resume..."):
            files = {"file": (resume_file.name, resume_file.getvalue())}
            headers = {}
            if st.session_state.logged_in and st.session_state.access_token:
                headers["Authorization"] = f"Bearer {st.session_state.access_token}"

            response = requests.post(RESUME_ENDPOINT, files=files, headers=headers, timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ Resume parsing failed. Status: {response.status_code}, Detail: {response.text}")
    except Exception as e:
        st.error(f"❌ Error during resume processing: {e}")
        st.text(traceback.format_exc())
    return None

# --- UI Functions for Login/Signup ---

def show_login_signup_form():
    """Displays the login and signup forms."""
    st.title("Welcome to the Speech Analysis App")
    st.markdown("Please log in or sign up to access the features.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.header("Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username_input")
            login_password = st.text_input("Password", type="password", key="login_password_input")
            submitted_login = st.form_submit_button("Login")

            if submitted_login:
                if login_username and login_password:
                    with st.spinner("Logging in..."):
                        login_result = login_user(login_username, login_password)
                        if login_result and "access_token" in login_result:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.user_id = login_result.get("user_id")
                            st.session_state.access_token = login_result["access_token"]
                            st.success(f"Logged in as {login_username}!")
                            st.rerun() # Rerun to display authenticated content
                        else:
                            st.error("Login failed. Please check your username and password.")
                else:
                    st.error("Please enter both username and password.")

    with signup_tab:
        st.header("Sign Up")
        with st.form("signup_form"):
            signup_username = st.text_input("Username", key="signup_username_input")
            signup_email = st.text_input("Email", key="signup_email_input")
            signup_password = st.text_input("Password", type="password", key="signup_password_input")
            submitted_signup = st.form_submit_button("Sign Up")

            if submitted_signup:
                if signup_username and signup_email and signup_password:
                    with st.spinner("Signing up..."):
                        signup_result = signup_user(signup_username, signup_email, signup_password)
                        if signup_result and signup_result.get("message") == "User created":
                            st.success("Account created successfully! Please log in.")
                            # Optionally switch to login tab or pre-fill username
                        else:
                            # This block will now correctly use the 'detail' from the returned dictionary
                            st.error(signup_result.get("detail", "Signup failed. Try a different username or email."))
                else:
                    st.error("Please fill in all fields.")


def show_authenticated_content():
    """Contains all the main application features accessible after login."""

    # Sidebar
    with st.sidebar:
        st.header(f"Hello, {st.session_state.username}!")
        st.button("Logout", on_click=logout)

        st.header("📄 Upload Resume")
        resume_file = st.file_uploader("Upload Resume here", type=["pdf", "docx"])

        if resume_file and "resume_parsed" not in st.session_state:
            file_content = resume_file.read()

            # Run both parsers and store in session_state
            parsed = parse_resume(file_content, resume_file.name)
            entire = parse_entire_resume(file_content, resume_file.name)

            if parsed.get("success") and entire.get("success"):
                # Cache both
                st.session_state.resume_parsed = parsed
                st.session_state.resume_entire = entire

                # Store specific fields for easy access
                st.session_state.skills = parsed.get("skills", [])
                roles = parsed.get("roles", [])
                st.session_state.role = roles[0] if roles else ""
                st.session_state.entiredata = entire

                st.success("✅ Resume parsed!")
            else:
                st.error("❌ Resume parsing failed.")

        # Optional: Display resume data if available
        # if "resume_parsed" in st.session_state:
        #     with st.expander("🧠 Resume Summary"):
        #         st.write("**Skills:**", st.session_state.skills)
        #         st.write("**Role:**", st.session_state.role)
        #         st.write("**Email:**", st.session_state.entiredata.get("metadata", {}).get("email"))
        #         st.write("**Phone:**", st.session_state.entiredata.get("metadata", {}).get("phone"))

        with st.expander("🎙️ Audio Input", expanded=False):
            uploaded_file = st.file_uploader("Upload Audio", type=ALLOWED_FILE_TYPES)



    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Assistant", "🧠 Interview Questions", "🎭 Mock Interview", "📝 My Chats"])

    # --- Tab 1: Chat Assistant ---
    with tab1:
        st.subheader("💬 Chat-based Speech Feedback")
        if uploaded_file:
            transcript = process_audio(uploaded_file)
            if transcript:
                st.session_state.speech_messages.append({"role": "User", "content": transcript})
                with st.spinner("🤖 Analyzing..."):
                    response = chat_with_speech_bot(transcript) # This now includes saving
                st.session_state.speech_messages.append({"role": "Bot", "content": response})
            else:
                st.warning("⚠️ Could not extract text from the audio. Please try a clearer recording.")

        for msg in st.session_state.speech_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your speech or general queries..."):
            st.session_state.speech_messages.append({"role": "User", "content": prompt})
            with st.chat_message("User"):
                st.markdown(prompt)
            with st.chat_message("Bot"):
                with st.spinner("Thinking..."):
                    response = chat_with_speech_bot(prompt) # This now includes saving
                st.session_state.speech_messages.append({"role": "Bot", "content": response})

    # --- Tab 2: Interview Questions ---
    with tab2:
        st.subheader("🧠 Auto-Generated Interview Questions")
        skills = st.session_state.get("skills", [])
        role = st.session_state.get("role", "")

        has_specific_context = bool(role or skills)

        if not st.session_state.interview_qna_messages:
            with st.spinner("🤖 Generating initial question..."):
                response = chat_with_interview_bot("", role, skills) # This now includes saving
            st.session_state.interview_qna_messages.append({"role": "Bot", "content": response})
        elif not has_specific_context and len(st.session_state.interview_qna_messages) == 1:
            pass # Prevents re-warning for general questions

        if not has_specific_context and len(st.session_state.interview_qna_messages) <=1 :
            st.warning("⚠️ No specific role or skills found in the uploaded resume. Generating general interview questions.")

        for msg in st.session_state.interview_qna_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Speech-to-text button
        audio_value = st.audio_input("🎤 Record a voice message (max 30 seconds)", key="audio_input_1")

        if audio_value:
            st.audio(audio_value)

            # Unique filename
            timestamp = int(time.time())
            audio_path = Path(f"audio_{timestamp}.wav")

            try:
                # Save uploaded audio
                audio_bytes = audio_value.read()
                if len(audio_bytes) == 0:
                    st.warning("Empty audio recording — please try again.")
                    st.stop()

                audio_path.write_bytes(audio_bytes)

                # Transcribe using Whisper
                with st.spinner():
                    result = transcribe_audio(str(audio_path))
                    user_input = result.get("text", "").strip()

                if not user_input or len(user_input) < 2:
                    st.warning("Could not transcribe audio — please try again.")
                    st.stop()

                # Store chat history
                if "interview_qna_messages" not in st.session_state:
                    st.session_state.interview_qna_messages = []

                st.session_state.interview_qna_messages.append({"role": "User", "content": user_input})
                with st.chat_message("User"):
                    st.markdown(user_input)

                # Bot response
                with st.chat_message("Bot"):
                    with st.spinner("Thinking..."):
                        response = chat_with_interview_bot(user_input, role, skills)
                        st.markdown(response)

                st.session_state.interview_qna_messages.append({"role": "Bot", "content": response})
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                st.stop()

            finally:
                if audio_path.exists():
                    try:
                        audio_path.unlink()
                    except Exception:
                        pass  # Safe fail


        if prompt := st.chat_input("Type your answer here...", key="interview_qna_input"):
            st.session_state.interview_qna_messages.append({"role": "User", "content": prompt})
            with st.chat_message("User"):
                st.markdown(prompt)
            with st.chat_message("Bot"):
                with st.spinner("Thinking..."):
                    response = chat_with_interview_bot(prompt, role, skills) # This now includes saving
                st.session_state.interview_qna_messages.append({"role": "Bot", "content": response})

    # --- Tab 3: Mock Interview ---
    with tab3:
        st.subheader("🎭 Mock Interview Practice")
        entire_data = st.session_state.get("entiredata", {})

        # Step 1: Ensure messages list is initialized
        if "interviewer_messages" not in st.session_state:
            st.session_state.interviewer_messages = []

        # Step 2: Require resume
        if not entire_data:
            st.warning("⚠️ Please upload your resume first.")
            st.stop()

        # Step 3: Generate initial interviewer question
        if not st.session_state.interviewer_messages:
            with st.spinner("🤖 Generating initial question..."):
                response = chat_with_interviewer("", entire_data)
            st.session_state.interviewer_messages.append({"role": "Bot", "content": response})

        # Step 4: Render chat history
        for msg in st.session_state.interviewer_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Step 5: Handle audio input
        audio_value = st.audio_input("🎤 Record a voice message (max 30 seconds)", key="audio_input_2")
        if audio_value:
            st.audio(audio_value)
            import time
            from pathlib import Path

            timestamp = int(time.time())
            audio_path = Path(f"audio_{timestamp}.wav")

            try:
                audio_bytes = audio_value.read()
                if len(audio_bytes) == 0:
                    st.warning("Empty audio recording — please try again.")
                    st.stop()

                audio_path.write_bytes(audio_bytes)

                # Transcribe
                with st.spinner("🧠 Transcribing audio..."):
                    result = transcribe_audio(str(audio_path))
                    user_input = result.get("text", "").strip()

                if not user_input or len(user_input) < 2:
                    st.warning("Could not transcribe audio — please try again.")
                    st.stop()

                # Show user's transcribed message
                st.session_state.interviewer_messages.append({"role": "User", "content": user_input})
                with st.chat_message("User"):
                    st.markdown(user_input)

                # Get interviewer response
                with st.chat_message("Bot"):
                    with st.spinner("Thinking..."):
                        response = chat_with_interviewer(user_input, entire_data)
                    st.markdown(response)

                st.session_state.interviewer_messages.append({"role": "Bot", "content": response})

            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                if audio_path.exists():
                    try:
                        audio_path.unlink()
                    except Exception:
                        pass

        # Step 6: Handle text input
        if prompt := st.chat_input("Type your reply here...", key="mock_interview_input"):
            st.session_state.interviewer_messages.append({"role": "User", "content": prompt})
            with st.chat_message("User"):
                st.markdown(prompt)
            with st.chat_message("Bot"):
                with st.spinner("Thinking..."):
                    response = chat_with_interviewer(prompt, entire_data)
                st.markdown(response)
            st.session_state.interviewer_messages.append({"role": "Bot", "content": response})


    # --- Tab 4: My Chats ---
    with tab4:
        st.subheader("📝 Your Previous Chat History")
        
        # Only fetch history if user is logged in and history not loaded yet
        if st.session_state.logged_in and not st.session_state.user_chat_history:
            with st.spinner("⏳ Loading your chat history..."):
                history = fetch_user_chat_history(st.session_state.user_id, st.session_state.access_token)
                if history:
                    # Sort by timestamp to ensure chronological order
                    st.session_state.user_chat_history = sorted(history, key=lambda x: x.get('timestamp', ''))
                    st.success("Chat history loaded.")
                else:
                    st.info("No previous chat history found for your account.")
        
        if st.session_state.user_chat_history:
            for chat_entry in st.session_state.user_chat_history:
                st.markdown(f"**Q:** {chat_entry.get('question', 'N/A')}")
                st.markdown(f"**A:** {chat_entry.get('answer', 'N/A')}")
                # Optional: Display timestamp
                # st.caption(f"_{chat_entry.get('timestamp', 'N/A')}_")
                st.markdown("---") # Separator for readability
        elif st.session_state.logged_in:
            st.info("Start chatting in other tabs to save your history here!")
        else:
            st.warning("Please log in to view your chat history.")


# --- Main Application Logic (Conditional Rendering) ---
if st.session_state.logged_in:
    show_authenticated_content()
else:
    show_login_signup_form()

