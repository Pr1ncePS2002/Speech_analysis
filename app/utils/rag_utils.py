import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import unicodedata
import logging
from dotenv import load_dotenv
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Define the path where you want to save/load your FAISS index
FAISS_INDEX_PATH = "app/faiss_index" # Choose a good path

def clean_text_for_embedding(text: str) -> str:
    normalized_text = unicodedata.normalize('NFKC', text)
    cleaned_chars = []
    for char in normalized_text:
        try:
            char.encode('utf-8')
            cleaned_chars.append(char)
        except UnicodeEncodeError:
            cleaned_chars.append(' ')
    return "".join(cleaned_chars)

@st.cache_resource
def load_and_embed_docs(pdf_dir="app/data/") -> FAISS:
    logger.info(f"Attempting to load and embed documents from {pdf_dir}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set. Cannot initialize embeddings.")
        raise ValueError("GOOGLE_API_KEY is not set. Cannot initialize embeddings.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # --- Check if FAISS index already exists on disk ---
    if os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        logger.info(f"Loading FAISS vectorstore from disk: {FAISS_INDEX_PATH}")
        try:
            retriever = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 3})
            logger.info("FAISS vectorstore loaded successfully from disk.")
            return retriever
        except Exception as e:
            logger.error(f"Error loading FAISS index from disk: {e}. Rebuilding index.", exc_info=True)
            # If loading fails, proceed to build from scratch

    logger.info("FAISS index not found on disk or failed to load. Building from documents.")

    try:
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_files = [fn for fn in os.listdir(pdf_dir) if fn.endswith(".pdf")]

        if not pdf_files:
            logger.warning(f"No PDFs found in {pdf_dir}. Returning a retriever with a dummy document.")
            dummy_doc = Document(page_content="This is a placeholder document for an empty vector store. Please upload PDFs to 'app/data/'.")
            return FAISS.from_documents([dummy_doc], embeddings).as_retriever(search_kwargs={"k": 3})

        loaders = [PyPDFLoader(os.path.join(pdf_dir, fn)) for fn in pdf_files]
        pages = []
        for loader in loaders:
            file_path = loader.file_path
            logger.info(f"Loading pages from {file_path}")
            try:
                loaded_pages = loader.load()
                for page in loaded_pages:
                    page.page_content = clean_text_for_embedding(page.page_content)
                pages.extend(loaded_pages)
                logger.info(f"Successfully loaded {len(loaded_pages)} pages from {file_path}")
            except Exception as e:
                logger.error(f"Error loading pages from {file_path}: {e}", exc_info=True)
                continue

        if not pages:
            logger.warning("No readable content could be loaded from any PDFs. Returning a retriever with a dummy document.")
            dummy_doc = Document(page_content="No readable content found in PDFs. Please check your PDF files.")
            return FAISS.from_documents([dummy_doc], embeddings).as_retriever(search_kwargs={"k": 3})

        logger.info(f"Total {len(pages)} pages loaded and cleaned across all PDFs.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        logger.info(f"Split into {len(docs)} chunks.")

        logger.info("Creating FAISS vectorstore from documents (this involves API calls for embeddings)...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        logger.info("FAISS vectorstore created successfully.")

        # --- Save the FAISS index to disk ---
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"FAISS vectorstore saved to disk: {FAISS_INDEX_PATH}")

        return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        logger.error(f"Critical error in load_and_embed_docs: {str(e)}", exc_info=True)
        raise