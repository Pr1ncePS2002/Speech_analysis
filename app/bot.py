import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
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
import logging
from typing import List # Import List for type hinting

# NEW IMPORTS FOR RAG
from langchain_core.documents import Document # For creating document objects for resume
from utils.rag_utils import load_and_embed_docs # Your internal docs RAG utility
from services.rag_service import build_vector_store # Your resume RAG utility (will call this directly)

# Add this line to allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
if "skills" not in st.session_state:
    st.session_state.skills = []
if "role" not in st.session_state:
    st.session_state.role = ""
if "entiredata" not in st.session_state:
    st.session_state.entiredata = {}


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

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

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

# You will need your GOOGLE_API_KEY here for embeddings for the temp resume vector store
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this is accessible

def get_mock_interview_context(
    user_query: str, # The user's input (answer or request for question)
    full_resume_text: str, # The full extracted text of the resume
    skills: List[str],
    roles: List[str]
) -> str:
    """
    Generates a combined context string for the mock interview.
    Prioritizes resume content and then internal documents.
    """
    context_parts = []

    # 1. Directly inject the full formatted resume text into the context
    # This is often the most effective way to ensure the LLM "sees" the resume.
    context_parts.append(f"### Candidate's Full Resume Text:\n{full_resume_text}")

    # 2. Retrieve from the uploaded resume (create temporary FAISS for it)
    try:
        # Create a Document object from the full resume text
        resume_doc = Document(page_content=full_resume_text, metadata={"source": "uploaded_resume"})
        # Build a temporary FAISS vector store just for this resume
        # This uses the build_vector_store from rag_service.py
        resume_vector_store = build_vector_store([resume_doc], desc="temporary_resume_vs")
        resume_retriever = resume_vector_store.as_retriever(search_kwargs={"k": 2})

        # Query the resume to get relevant snippets based on the user's input or general relevance
        # Use a general query if user_query is empty (e.g., initial question generation)
        resume_retrieval_query = user_query if user_query else "key skills, experience, and projects from this resume"
        retrieved_resume_docs = resume_retriever.get_relevant_documents(resume_retrieval_query)
        
        if retrieved_resume_docs:
            context_parts.append("\n### Relevant Snippets from Resume (RAG):\n" + 
                                 "\n".join([doc.page_content for doc in retrieved_resume_docs]))
    except Exception as e:
        logger.error(f"Error creating/retrieving from temporary resume vector store: {e}")
        # If there's an error, we just proceed without RAG from resume, relying on direct injection


    # 3. Retrieve from internal documents (your app/data PDFs)
    # Use the cached internal document retriever from rag_utils
    try:
        internal_docs_retriever = load_and_embed_docs() # This loads/creates the FAISS for internal docs

        # Craft contextual queries for internal docs
        internal_queries = [user_query] # Start with the direct user query
        if skills:
            internal_queries.append(f"{user_query} related to {', '.join(skills)}")
        if roles:
            internal_queries.append(f"{user_query} for a {', '.join(roles)} role")
        
        internal_retrieved_text = []
        for q in internal_queries:
            # Get documents from the internal RAG
            retrieved_internal_docs = internal_docs_retriever.get_relevant_documents(q)
            internal_retrieved_text.extend([doc.page_content for doc in retrieved_internal_docs])
        
        # Deduplicate and add to context
        unique_internal_context = list(set(internal_retrieved_text))
        if unique_internal_context:
            context_parts.append("\n### Relevant Snippets from Internal Documents (RAG):\n" + 
                                 "\n".join(unique_internal_context[:3])) # Limit to avoid too much context
    except Exception as e:
        logger.error(f"Error retrieving from internal documents: {e}")


    final_context = "\n\n".join(context_parts)
    # Ensure the combined context doesn't exceed LLM's token limit (e.g., 8000 characters or more intelligently by tokens)
    return final_context[:10000] # Adjust based on Gemini-Pro's actual context window


def initialize_interviewer_chain():
    logger.info("Attempting to initialize interviewer_chain.")
    if st.session_state.interviewer_chain is None:
        try:
            # Revert to gemini-2.0-flash-lite or try another available model
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)
            logger.info("LLM for interviewer_chain initialized.")

            # --- UPDATED PROMPT TEMPLATE ---
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
                input_key="question", # The main input for the chain
                return_messages=True
            )
            logger.info("Memory for interviewer_chain initialized.")

            # --- CHANGE FROM ConversationalRetrievalChain TO LLMChain ---
            # We are now manually managing the context, so LLMChain is more appropriate.
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


# --- Chain Interaction Wrappers ---
def chat_with_speech_bot(message):
    return initialize_speech_chain().run(message=str(message))

def chat_with_interview_bot(message, roles, skills):
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

    return chain.run({
        "question": full_question_for_llm,
        "chat_history": st.session_state.interview_qna_messages,
    })

# --- NEW HELPER FUNCTION TO FORMAT RESUME DATA ---
def format_resume_data_for_llm(entire_data: dict) -> str:
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

    # Add other relevant sections if your parser returns them (e.g., awards, certifications)
    # Example:
    # if entire_data.get("certifications"):
    #     formatted_text.append(f"\nCertifications: {', '.join(entire_data['certifications'])}")

    return "\n".join(formatted_text)


def chat_with_interviewer(message, entire_data):
    chain = initialize_interviewer_chain()
    
    # Extract relevant data from entire_data for context generation
    # Ensure your backend's `entire_data` structure matches this
    full_resume_text = entire_data.get("full_text", format_resume_data_for_llm(entire_data))
    
    # Attempt to get skills and roles from 'extracted_data' first, then fallback to top-level
    extracted_data = entire_data.get("extracted_data", {})
    skills = extracted_data.get("skills", entire_data.get("skills", []))
    roles = extracted_data.get("roles", entire_data.get("roles", []))
    roles = [r for r in roles if r] # Filter out empty strings if any


    # Generate the combined context string
    combined_context = get_mock_interview_context(
        user_query=message,
        full_resume_text=full_resume_text,
        skills=skills,
        roles=roles
    )
    
    # Now, pass this combined_context directly to the LLMChain
    return chain.run({
        "question": message,
        "context": combined_context, # THIS IS THE KEY CHANGE
        "chat_history": st.session_state.interviewer_messages
    })


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
                    st.session_state.role = roles[0] if roles else ""
                    st.session_state.entiredata = parsed_full_resume_data # Store the full data
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
            st.session_state.speech_messages.append({"role": "Bot", "content": response})

# --- Tab 2: Interview Questions ---
with tab2:
    st.subheader("🧠 Auto-Generated Interview Questions")
    skills = st.session_state.get("skills", [])
    role = st.session_state.get("role", "")

    has_specific_context = bool(role or skills)

    if not st.session_state.interview_qna_messages:
        with st.spinner("🤖 Generating initial question..."):
            response = chat_with_interview_bot("", role, skills)
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
                response = chat_with_interview_bot(prompt, role, skills)
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
                # Pass the *formatted* resume data
                response = chat_with_interviewer("", entire_data)
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
                    # Pass the *formatted* resume data
                    response = chat_with_interviewer(prompt, entire_data)
                st.session_state.interviewer_messages.append({"role": "Bot", "content": response})