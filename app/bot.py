import streamlit as st
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up Streamlit page configuration
st.set_page_config(page_title="Speech Assistant", layout="wide")

# Initialize session state for chat history and chain
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "Bot", "content": "Hi! I'm your Speech Assistant. How can I help you today?"}
    ]
if "chatbot_chain" not in st.session_state:
    st.session_state.chatbot_chain = None

# Function to initialize the chatbot
def initialize_chatbot():
    if st.session_state.chatbot_chain is None:
        # Initialize LangChain compatible Google Generative AI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.7)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert communication coach and speech analyst. Analyze the following transcribed speech:
            
            **Conversation History:**
            {chat_history}

            **User's Current Message:**
            {message}

            Provide detailed analysis covering:
            1. Grammatical Errors
            2. Filler Words
            3. Pauses and Flow
            4. Vocabulary Usage
            5. Fluency Score (0-10)
            6. Improvement Suggestions
            """
        )

        # Configure memory with window for last 10 interactions
        memory = ConversationBufferWindowMemory(
            input_key="message",
            memory_key="chat_history",
            k=10
        )

        # Create chain with memory
        st.session_state.chatbot_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
    return st.session_state.chatbot_chain

# Function to interact with the chatbot
def chat_with_bot(message):
    chain = initialize_chatbot()
    return chain.run(message=message)

# Chat UI
st.title("Speech Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**{message['role']}:** {message['content']}")

# Text input handling
input_text = st.chat_input("What do you need help with?")
if input_text:
    # Add user message to history
    st.session_state.messages.append({"role": "User", "content": input_text})
    
    # Generate and display response
    with st.chat_message("User"):
        st.markdown(f"**User:** {input_text}")
    
    with st.chat_message("Bot"):
        with st.spinner("Analyzing your speech..."):
            response = chat_with_bot(input_text)
            st.markdown(f"**Bot:** {response}")
    
    st.session_state.messages.append({"role": "Bot", "content": response})