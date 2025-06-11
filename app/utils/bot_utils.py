import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain 
import logging
import os
import requests
from typing import List, Dict, Any # Import Dict and Any for broader type hinting
from dotenv import load_dotenv
from utils.rag_utils import load_and_embed_docs #internal docs RAG utility
from services.rag_service import build_vector_store # resume RAG utility

FASTAPI_BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = f"{FASTAPI_BASE_URL}/api/speech/upload"
RESUME_ENDPOINT = f"{FASTAPI_BASE_URL}/api/resume/upload"
CHAT_HISTORY_FETCH_ENDPOINT = f"{FASTAPI_BASE_URL}/api/chat/history" # Will append user_id
CHAT_SAVE_ENDPOINT = f"{FASTAPI_BASE_URL}/api/chat/save"

#  allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# You will need your GOOGLE_API_KEY here for embeddings for the temp resume vector store
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this is accessible

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

def speech_to_text():
    st.audio_input()
    
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