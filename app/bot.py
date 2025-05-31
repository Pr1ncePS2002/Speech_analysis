# speech_assistant_app.py

import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from dotenv import load_dotenv
import traceback

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

# Session State Initialization
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

def initialize_interview_qna_chain():
    if st.session_state.interview_qna_chain is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)
        prompt = ChatPromptTemplate.from_template(
"""You are an AI mentor preparing a candidate for job interviews.

Based on the provided roles: {roles} and skills: {skills}.

**Your primary instruction is as follows:**

IF the candidate's LAST message indicates they "don't know", are "not sure", "don't remember", or express similar uncertainty about the PREVIOUS question:
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
Current: {message}
"""
)

        memory = ConversationBufferWindowMemory(input_key="message", memory_key="chat_history", k=10)
        st.session_state.interview_qna_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return st.session_state.interview_qna_chain

def initialize_interviewer_chain():
    if st.session_state.interviewer_chain is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)
        prompt = ChatPromptTemplate.from_template(
            """You are an experienced, professional, and analytical hiring bot designed to conduct structured interviews for job candidates. Your task is to ask relevant, insightful, and role-specific questions based on the candidate’s parsed resume.

Instructions:
Review the Resume: {resume_text}:

Analyze the candidate’s work experience, education, skills, certifications, and projects.

Identify key strengths, potential gaps, and areas requiring clarification.

Interview Approach:
Ask one question at a time.

Behavioral Questions: Ask about past experiences (e.g., "Can you describe a challenge you faced in [Job Role] and how you resolved it?").

Technical/Skill-Based Questions: Probe expertise (e.g., "Explain how you applied [Skill] in [Project/Job]").

Situational Questions: Gauge problem-solving (e.g., "How would you handle [Scenario] in this role?").

Culture Fit: Assess alignment with company values (if provided).

Tailoring Questions:

Prioritize questions based on the job description (if provided).

For senior roles: Focus on leadership, strategy, and impact.

For junior roles: Emphasize learning agility and foundational skills.

Tone & Style:

Professional, respectful, and engaging.

Mix open-ended and follow-up questions (e.g., "Why did you choose this approach?").

Output Format:

Begin with a brief introduction: "Thank you for your time. I’ll ask questions about your background and skills."

Group questions by theme (Experience, Skills, Behavior).

Example:

"Your resume shows [X] experience at [Company]. Can you walk me through a key achievement there?"

"How does your [Skill/Certification] prepare you for this role?"

Avoid:

Repetitive or overly generic questions.

Assumptions beyond the resume data.

Example Output (for a Software Engineer):

"You worked on [Project Y] using Python. What was your most complex contribution?"

"Your resume mentions leading a team. How did you handle conflicts or deadlines?"

"How would you improve [Process Z] based on your past experience?"

Final Step: After drafting questions, verify they align with the resume details and role requirements.
"""
        )
        memory = ConversationBufferWindowMemory(input_key="message", memory_key="chat_history", k=10)
        st.session_state.interviewer_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return st.session_state.interviewer_chain

# --- Chain Interaction Wrappers ---
def chat_with_speech_bot(message):
    return initialize_speech_chain().run(message=str(message))

def chat_with_interview_bot(message, roles, skills):
    return initialize_interview_qna_chain().run({"skills": str(skills), "roles": str(roles), "message": str(message)})

def chat_with_interviewer(message, entire_data):
    return initialize_interviewer_chain().run({"resume_text": str(entire_data), "message": str(message)})

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
        st.text(traceback.format_exc())

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Upload Resume")
    resume_file = st.file_uploader("Upload Resume here", type=["pdf", "docx"])
    if resume_file:
        with st.spinner("📄 Parsing resume..."):
            try:
                files = {"file": (resume_file.name, resume_file.getvalue())}
                response = requests.post(RESUME_ENDPOINT, files=files, timeout=20)
                if response.status_code == 200:
                    parsed = response.json().get("parsed", {})
                    parsed_full_resume_data = response.json().get("entire_data", {})
                    st.session_state.skills = parsed.get("skills", [])
                    roles = parsed.get("roles", [])
                    st.session_state.role = roles[0] if roles else "Software Developer Engineer"
                    st.session_state.entiredata = parsed_full_resume_data
                    st.success("✅ Resume parsed!")
                else:
                    st.error("❌ Resume parsing failed.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.text(traceback.format_exc())

    with st.expander("🎙️ Audio Input", expanded=False):
        uploaded_file = st.file_uploader("Upload Audio", type=ALLOWED_FILE_TYPES)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "🧠 Interview Questions", "🎭 Mock Interview"])

# --- Tab 1: Chat ---
with tab1:
    st.subheader("💬 Chat-based Speech Feedback")
    if uploaded_file:
        transcript = process_audio(uploaded_file)
        if transcript:
            st.session_state.speech_messages.append({"role": "User", "content": transcript})
            with st.spinner("🤖 Analyzing..."):
                response = chat_with_speech_bot(transcript)
                st.session_state.speech_messages.append({"role": "Bot", "content": response})
        else:
            st.warning("⚠️ Could not extract text from the audio. Please try a clearer recording.")

    for msg in st.session_state.speech_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your speech or resume..."):
        st.session_state.speech_messages.append({"role": "User", "content": prompt})
        with st.chat_message("User"):
            st.markdown(prompt)
        with st.chat_message("Bot"):
            with st.spinner("Thinking..."):
                response = chat_with_speech_bot(prompt)
                st.markdown(response)
        st.session_state.speech_messages.append({"role": "Bot", "content": response})

# --- Tab 2: Interview Questions ---
with tab2:
    st.subheader("🧠 Auto-Generated Interview Questions")
    skills = st.session_state.get("skills", [])
    role = st.session_state.get("role", "")

    if not skills or not role:
        st.warning("⚠️ Please upload your resume.")
    else:
        if not st.session_state.interview_qna_messages:
            with st.spinner("🤖 Generating initial question..."):
                response = chat_with_interview_bot("", role, skills)
            st.session_state.interview_qna_messages.append({"role": "Bot", "content": response})

        for msg in st.session_state.interview_qna_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Type your answer here..."):
            st.session_state.interview_qna_messages.append({"role": "User", "content": prompt})
            with st.chat_message("User"):
                st.markdown(prompt)
            with st.chat_message("Bot"):
                with st.spinner("Thinking..."):
                    response = chat_with_interview_bot(prompt, role, skills)
                    st.markdown(response)
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
                response = chat_with_interviewer("", entire_data)
            st.session_state.interviewer_messages.append({"role": "Bot", "content": response})

        for msg in st.session_state.interviewer_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Type your reply here..."):
            st.session_state.interviewer_messages.append({"role": "User", "content": prompt})
            with st.chat_message("User"):
                st.markdown(prompt)
            with st.chat_message("Bot"):
                with st.spinner("Thinking..."):
                    response = chat_with_interviewer(prompt, entire_data)
                    st.markdown(response)
            st.session_state.interviewer_messages.append({"role": "Bot", "content": response})