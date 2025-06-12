# app/services/rag_service.py

import os
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from tempfile import NamedTemporaryFile
import unicodedata
import logging
from .resume_parser import parse_entire_resume  # Import the function

# 🔹 NEW: Sentence tokenizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


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


def load_internal_docs(data_folder="app/data") -> List[Document]:
    all_docs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, filename))
            pages = loader.load()
            for page in pages:
                page.page_content = clean_text_for_embedding(page.page_content)
            all_docs.extend(pages)
    return all_docs


def load_uploaded_resume(uploaded_file_bytes: bytes) -> List[Document]:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file_bytes)
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    os.remove(temp_pdf_path)

    for doc in docs:
        doc.page_content = clean_text_for_embedding(doc.page_content)

    return docs


# 🔹 NEW: Semantic chunking function (replaces SemanticChunker)
def semantic_chunk(documents: List[Document], chunk_size: int = 3) -> List[Document]:
    chunks = []
    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i:i + chunk_size])
            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
    return chunks


def build_vector_store(documents: List[Document], desc: str = "documents") -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    logger.info(f"Chunking documents for {desc}")
    chunks = semantic_chunk(documents, chunk_size=3)  # 🔹 Using custom chunking
    logger.info(f"Creating vector store for {desc} with {len(chunks)} chunks.")
    return FAISS.from_documents(chunks, embedding=embeddings)


def generate_interview_with_resume(uploaded_resume_bytes: bytes, filename: str, query: str) -> str:
    try:
        internal_docs = load_internal_docs()
        resume_docs = load_uploaded_resume(uploaded_resume_bytes)

        # 1. Parse the entire resume
        resume_data = parse_entire_resume(uploaded_resume_bytes, filename)

        if not resume_data.get("success"):
            return f"Error parsing resume: {resume_data.get('error', 'Unknown error')}"

        skills = resume_data["extracted_data"].get("skills", [])
        roles = resume_data["extracted_data"].get("roles", [])
        full_resume_text = resume_data.get("full_text", "")

        # 2. Build separate vector stores
        resume_vector_store = build_vector_store(resume_docs, "resume")
        internal_vector_store = build_vector_store(internal_docs, "internal docs")

        # 3. Initial query of resume
        resume_retriever = resume_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        resume_context = resume_retriever.get_relevant_documents(query)
        resume_context_text = "\n".join([doc.page_content for doc in resume_context])

        # 4. Generate contextual queries based on resume
        contextual_queries = []
        if skills:
            contextual_queries.append(f"{query} related to {', '.join(skills)}")
        if roles:
            contextual_queries.append(f"{query} for a {', '.join(roles)}")
        contextual_queries.append(f"{query} given the following resume context: {full_resume_text}")

        # 5. Query internal docs with contextual queries
        internal_context_docs = []
        for q in contextual_queries:
            internal_context_docs.extend(internal_vector_store.similarity_search(q, k=2))
        internal_context_text = "\n".join([doc.page_content for doc in internal_context_docs])

        # 6. Combine and prioritize context
        combined_context = f"Resume Context:\n{resume_context_text}\n\nInternal Document Context:\n{internal_context_text}"

        # 7. Use ChatGoogleGenerativeAI for LLM
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=internal_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        )

        result = qa_chain({"query": f"{query}\nContext:{combined_context}"})
        return result["result"]

    except Exception as e:
        logger.error(f"Error in generate_interview_with_resume: {e}", exc_info=True)
        return f"An error occurred: {e}"
