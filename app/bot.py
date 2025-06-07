import sys
from pathlib import Path
# Add project root to Python path
# Ensure this path is correct for your project structure
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain # Already imported
from langchain_core.documents import Document # Already imported
import os
import requests
from dotenv import load_dotenv
import traceback
import logging
from typing import List, Dict, Any # Import Dict and Any for broader type hinting
import json # <<< ADDED: Import the json module

# NEW IMPORTS FOR RAG (Assuming these paths are correct)
from utils.rag_utils import load_and_embed_docs # Your internal docs RAG utility
from services.rag_service import build_vector_store # Your resume RAG utility

# Add this line to allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Configuration ---
# IMPORTANT: Replace with the actual URL of your deployed FastAPI backend
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

def save_new_chat_entry(user_id: int, question: str, answer: str, access_token: str) -> Dict[str, Any] | None:
    """Saves a new chat entry for the logged-in user to the backend."""
    if not st.session_state.logged_in:
        logger.info("Not logged in, skipping chat save.")
        return None
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.post(
            CHAT_SAVE_ENDPOINT,
            json={"user_id": user_id, "question": question, "answer": answer},
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error saving chat: {e}")
        if response is not None:
            st.error(f"Detail: {response.text}")
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


# --- Chatbot Initialization Functions ---
def initialize_speech_chain():
    if st.session_state.speech_chain is None:
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
        st.session_state.speech_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return st.session_state.speech_chain

# You will need your GOOGLE_API_KEY here for embeddings for the temp resume vector store
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this is accessible

def initialize_interview_qna_chain():
    logger.info("Attempting to initialize interview_qna_chain.")
    if st.session_state.interview_qna_chain is None:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)
            logger.info("LLM for interview_qna_chain initialized.")

            prompt_template_str = """You are an AI mentor preparing a candidate for job interviews.

Use the following retrieved context to answer the question:
{context}

**Your primary instruction is as follows:**

IF the candidate's LAST message indicates they "don't know", are "not sure", "don't remember", or express similar uncertainty like "idk","i dont know","no idea" about the PREVIOUS question:
- IMMEDIATELY provide a **'Sample Answer:'** or **'Explanation:'** for the *previous question*.
- Follow this with constructive feedback: "Here's some feedback on that topic:"
- THEN, ask a NEW, relevant interview question.

OTHERWISE (if the candidate provides an answer):
- Provide short, constructive feedback on their answer: what was good, what can be improved, and what was missing.
- THEN, ask a NEW, relevant interview question.

Constraints for all questions and feedback:
- Ask only one interview question at a time.
- Use a balanced mix of technical, behavioral, and situational questions.
- Start with basic or moderate questions and gradually increase difficulty.
- Do not simulate an interview or play a character beyond being an AI mentor.

History: {chat_history}
Current: {question}
"""
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            logger.info("Prompt template for interview_qna_chain created.")

            memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
            logger.info("Memory for interview_qna_chain initialized.")

            logger.info("Calling load_and_embed_docs() for interview_qna_chain...")
            retriever = load_and_embed_docs()
            logger.info("load_and_embed_docs() for interview_qna_chain completed. Retriever obtained.")

            st.session_state.interview_qna_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=False,
                verbose=True,
            )
            logger.info("interview_qna_chain initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing interview_qna_chain: {e}", exc_info=True)
            st.error(f"Failed to initialize Interview Q&A Chain. Error: {e}")
            st.session_state.interview_qna_chain = None
    return st.session_state.interview_qna_chain

def get_mock_interview_context(
    user_query: str,
    full_resume_text: str,
    skills: List[str],
    roles: List[str]
) -> str:
    """
    Generates a combined context string for the mock interview.
    Prioritizes resume content and then internal documents.
    """
    context_parts = []

    # 1. Directly inject the full formatted resume text into the context
    context_parts.append(f"### Candidate's Full Resume Text:\n{full_resume_text}")

    # 2. Retrieve from the uploaded resume (create temporary FAISS for it)
    try:
        resume_doc = Document(page_content=full_resume_text, metadata={"source": "uploaded_resume"})
        resume_vector_store = build_vector_store([resume_doc], desc="temporary_resume_vs")
        resume_retriever = resume_vector_store.as_retriever(search_kwargs={"k": 2})

        resume_retrieval_query = user_query if user_query else "key skills, experience, and projects from this resume"
        retrieved_resume_docs = resume_retriever.get_relevant_documents(resume_retrieval_query)
        
        if retrieved_resume_docs:
            context_parts.append("\n### Relevant Snippets from Resume (RAG):\n" + 
                                 "\n".join([doc.page_content for doc in retrieved_resume_docs]))
    except Exception as e:
        logger.error(f"Error creating/retrieving from temporary resume vector store: {e}")

    # 3. Retrieve from internal documents (your app/data PDFs)
    try:
        internal_docs_retriever = load_and_embed_docs()

        internal_queries = [user_query]
        if skills:
            internal_queries.append(f"{user_query} related to {', '.join(skills)}")
        if roles:
            internal_queries.append(f"{user_query} for a {', '.join(roles)} role")
        
        internal_retrieved_text = []
        for q in internal_queries:
            retrieved_internal_docs = internal_docs_retriever.get_relevant_documents(q)
            internal_retrieved_text.extend([doc.page_content for doc in retrieved_internal_docs])
        
        unique_internal_context = list(set(internal_retrieved_text))
        if unique_internal_context:
            context_parts.append("\n### Relevant Snippets from Internal Documents (RAG):\n" + 
                                 "\n".join(unique_internal_context[:3]))
    except Exception as e:
        logger.error(f"Error retrieving from internal documents: {e}")

    final_context = "\n\n".join(context_parts)
    return final_context[:10000] # Adjust based on Gemini-Pro's actual context window


def initialize_interviewer_chain():
    logger.info("Attempting to initialize interviewer_chain.")
    if st.session_state.interviewer_chain is None:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)
            logger.info("LLM for interviewer_chain initialized.")

            prompt_template_str = """You are an experienced, professional, and analytical hiring bot designed to conduct structured interviews for job candidates. Your task is to ask relevant, insightful, and role-specific questions based on the candidate’s resume and the provided context.

---
**Combined Context:**
{context}

---
Instructions:
- **Prioritize information from the "Combined Context" section** when formulating questions and providing feedback.
- Ask only one question at a time.
- Use a balanced mix of technical, behavioral, and situational questions.
- Tailor questions specifically based on the candidate's background and the job role/skills extracted from their resume.
- Start with basic or moderate questions and gradually increase difficulty.
- Your first response should be a brief introduction, then ask the first question.
- Do not simulate an interview or play a character beyond being an AI mentor.

**Example Questions:**
- "Your resume mentions experience in [Specific Skill/Technology]. Can you elaborate on a project where you heavily utilized this?"
- "Based on your experience at [Company Name], can you describe a time you faced a significant technical challenge and how you overcame it?"
- "Given your [Role] experience, how would you approach [Hypothetical Scenario relevant to the role]?"

---
Chat History: {chat_history}
User Input: {question}
"""
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            logger.info("Prompt template for interviewer_chain created.")

            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                input_key="question",
                return_messages=True
            )
            logger.info("Memory for interviewer_chain initialized.")

            st.session_state.interviewer_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            logger.info("interviewer_chain (LLMChain) initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing interviewer_chain: {e}", exc_info=True)
            st.error(f"Failed to initialize Mock Interview Chain. Error: {e}")
            st.session_state.interviewer_chain = None
    return st.session_state.interviewer_chain


# --- Chain Interaction Wrappers (Modified to save to backend) ---
def chat_with_speech_bot(message: str) -> str:
    response = initialize_speech_chain().run(message=message)
    if st.session_state.logged_in:
        save_new_chat_entry(
            user_id=st.session_state.user_id,
            question=message,
            answer=response,
            access_token=st.session_state.access_token
        )
    return response

def chat_with_interview_bot(message: str, roles: List[str], skills: List[str]) -> str:
    chain = initialize_interview_qna_chain()

    context_prefix = ""
    if roles and isinstance(roles, list) and roles[0]:
        context_prefix += f"Based on the provided role: {roles[0]}."
    if skills:
        if context_prefix:
            context_prefix += " And "
        context_prefix += f"Focus on these skills: {', '.join(skills)}."

    if not context_prefix:
        context_prefix = "Generate general interview questions that are commonly asked in various job roles."
        logger.info("No specific roles/skills found. Generating general interview questions.")
    else:
        logger.info(f"Generating questions based on: {context_prefix}")

    full_question_for_llm = f"{context_prefix}\n\nUser input: {message}" if message else context_prefix

    response = chain.run({
        "question": full_question_for_llm,
        "chat_history": st.session_state.interview_qna_messages,
    })
    if st.session_state.logged_in:
        save_new_chat_entry(
            user_id=st.session_state.user_id,
            question=message,
            answer=response,
            access_token=st.session_state.access_token
        )
    return response

# --- NEW HELPER FUNCTION TO FORMAT RESUME DATA ---
def format_resume_data_for_llm(entire_data: Dict[str, Any]) -> str:
    """
    Formats the parsed resume data into a readable string for the LLM.
    Focuses on key sections to keep it concise and relevant.
    """
    formatted_text = []

    if entire_data.get("name"):
        formatted_text.append(f"Name: {entire_data['name']}")
    if entire_data.get("email"):
        formatted_text.append(f"Email: {entire_data['email']}")
    if entire_data.get("phone"):
        formatted_text.append(f"Phone: {entire_data['phone']}")
    if entire_data.get("linkedin"):
        formatted_text.append(f"LinkedIn: {entire_data['linkedin']}")
    if entire_data.get("objective"):
        formatted_text.append(f"\nObjective: {entire_data['objective']}")

    if entire_data.get("experience"):
        formatted_text.append("\nExperience:")
        for exp in entire_data["experience"]:
            title = exp.get("title", "N/A")
            company = exp.get("company", "N/A")
            years = exp.get("years", "N/A")
            description = exp.get("description", "").strip()
            formatted_text.append(f"- {title} at {company} ({years})")
            if description:
                formatted_text.append(f"  Description: {description}")

    if entire_data.get("education"):
        formatted_text.append("\nEducation:")
        for edu in entire_data["education"]:
            degree = edu.get("degree", "N/A")
            university = edu.get("university", "N/A")
            years = edu.get("years", "N/A")
            formatted_text.append(f"- {degree} from {university} ({years})")

    if entire_data.get("skills"):
        formatted_text.append(f"\nSkills: {', '.join(entire_data['skills'])}")

    if entire_data.get("projects"):
        formatted_text.append("\nProjects:")
        for proj in entire_data["projects"]:
            name = proj.get("name", "N/A")
            description = proj.get("description", "").strip()
            formatted_text.append(f"- {name}")
            if description:
                formatted_text.append(f"  Description: {description}")

    return "\n".join(formatted_text)


def chat_with_interviewer(message: str, entire_data: Dict[str, Any]) -> str:
    chain = initialize_interviewer_chain()
    
    full_resume_text = entire_data.get("full_text", format_resume_data_for_llm(entire_data))
    
    extracted_data = entire_data.get("extracted_data", {})
    skills = extracted_data.get("skills", entire_data.get("skills", []))
    roles = extracted_data.get("roles", entire_data.get("roles", []))
    roles = [r for r in roles if r]

    combined_context = get_mock_interview_context(
        user_query=message,
        full_resume_text=full_resume_text,
        skills=skills,
        roles=roles
    )
    
    response = chain.run({
        "question": message,
        "context": combined_context,
        "chat_history": st.session_state.interviewer_messages
    })
    if st.session_state.logged_in:
        save_new_chat_entry(
            user_id=st.session_state.user_id,
            question=message,
            answer=response,
            access_token=st.session_state.access_token
        )
    return response


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
        if resume_file:
            parsed_data = process_resume(resume_file)
            if parsed_data:
                st.session_state.skills = parsed_data.get("parsed", {}).get("skills", [])
                roles = parsed_data.get("parsed", {}).get("roles", [])
                st.session_state.role = roles[0] if roles else ""
                st.session_state.entiredata = parsed_data.get("entire_data", {})
                st.success("✅ Resume parsed!")
            else:
                st.error("❌ Resume parsing failed.")

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

        if not entire_data:
            st.warning("⚠️ Please upload your resume first.")
        else:
            if not st.session_state.interviewer_messages:
                with st.spinner("🤖 Generating initial question..."):
                    response = chat_with_interviewer("", entire_data) # This now includes saving
                st.session_state.interviewer_messages.append({"role": "Bot", "content": response})

            for msg in st.session_state.interviewer_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Type your reply here...", key="mock_interview_input"):
                st.session_state.interviewer_messages.append({"role": "User", "content": prompt})
                with st.chat_message("User"):
                    st.markdown(prompt)
                with st.chat_message("Bot"):
                    with st.spinner("Thinking..."):
                        response = chat_with_interviewer(prompt, entire_data) # This now includes saving
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

