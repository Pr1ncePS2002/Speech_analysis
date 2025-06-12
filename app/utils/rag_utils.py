import os
import logging
import unicodedata
import nltk
from dotenv import load_dotenv
from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
nltk.download('punkt')

FAISS_INDEX_PATH = "app/faiss_index"

# --- Util: Clean Unicode ---
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

# --- Util: Semantic Chunker using SentenceTransformer ---
def sentence_transformer_chunker(text: str, chunk_size: int = 3) -> List[str]:
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# --- Main Function ---
@st.cache_resource
def load_and_embed_docs(pdf_dir="app/data/") -> FAISS:
    logger.info(f"Attempting to load and embed documents from {pdf_dir}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set.")
        raise ValueError("GOOGLE_API_KEY is not set.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Load existing FAISS index
    if os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        try:
            logger.info(f"Loading FAISS vectorstore from {FAISS_INDEX_PATH}")
            retriever = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 3})
            return retriever
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}. Rebuilding...", exc_info=True)

    try:
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

        if not pdf_files:
            logger.warning("No PDFs found.")
            dummy = Document(page_content="This is a placeholder document. Upload PDFs to 'app/data/'.")
            return FAISS.from_documents([dummy], embeddings).as_retriever(search_kwargs={"k": 3})

        pages = []
        for fn in pdf_files:
            loader = PyPDFLoader(os.path.join(pdf_dir, fn))
            logger.info(f"Loading {fn}")
            try:
                loaded = loader.load()
                for page in loaded:
                    page.page_content = clean_text_for_embedding(page.page_content)
                pages.extend(loaded)
            except Exception as e:
                logger.error(f"Error loading {fn}: {e}", exc_info=True)

        if not pages:
            logger.warning("No readable content found.")
            dummy = Document(page_content="No readable content found in PDFs.")
            return FAISS.from_documents([dummy], embeddings).as_retriever(search_kwargs={"k": 3})

        # Semantic chunking
        chunks = []
        for doc in pages:
            doc_chunks = sentence_transformer_chunker(doc.page_content)
            for chunk in doc_chunks:
                chunks.append(Document(page_content=chunk))

        logger.info(f"Created {len(chunks)} semantic chunks.")

        # Embedding and saving
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved at {FAISS_INDEX_PATH}")

        return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
