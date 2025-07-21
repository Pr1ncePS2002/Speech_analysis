
# 📄 Real-time Fluency Analysis Application for Speech

This project implements a real-time fluency analysis application for speech. It leverages various technologies for speech-to-text, natural language processing, resume parsing, and interview preparation.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Installation and Setup](#installation-and-setup)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to provide a comprehensive platform for speech analysis, resume parsing, and interview preparation. It offers features such as:

- Real-time speech transcription and analysis.
- Resume parsing to extract key information.
- Interview question generation based on resumes and internal documents.
- Mock interview simulation with feedback.
- Chat history management.
- User authentication and secure access.

## Key Features

- **Speech Analysis:** Upload audio files (wav, mp3, m4a) for transcription and fluency analysis.
- **Resume Parsing:** Upload PDF or DOCX resumes for skill and role extraction.
- **Interview Preparation:** Generate interview questions based on resumes and provide feedback.
- **Mock Interviews:** Simulate mock interviews with tailored questions and feedback.
- **User Authentication:** Secure user accounts with signup, login, and logout functionality.
- **Chat History:** Store and display user chat history.

## Technology Stack

- **Backend:**
  - Python
  - FastAPI (Web Framework)
  - SQLAlchemy (ORM)
  - PostgreSQL (Database)
  - Alembic (Database Migrations)
  - Uvicorn (ASGI Server)
  - OpenAI Whisper (Speech-to-Text)
  - Google Generative AI (LLM)
  - Langchain (LLM Framework)
  - Streamlit (Frontend)
  - JWT (JSON Web Tokens)
  - Passlib (Password Hashing)
  - python-jose (JWT Operations)
  - pdfplumber, docx2txt, pypdf, unstructured[all-docs], pdfminer.six, pypdfium2 (Document Processing)
  - FAISS (Vector Store)
  - sentence-transformers (Sentence Embeddings)
  - NLTK (Natural Language Toolkit)

- **Frontend:**
  - Streamlit


## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    - On Linux/macOS:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up the database:**
    - Configure your PostgreSQL database connection details in the appropriate configuration files (e.g., `alembic.ini`, environment variables).
    - Run database migrations:
      ```bash
      alembic upgrade head
      ```

6.  **Set up Environment Variables**
    - Set up the necessary environment variables, including:
        - `DATABASE_URL` (PostgreSQL connection string)
        - `SECRET_KEY` (for JWT signing - use a strong, random key in production)
        - `GOOGLE_API_KEY` (for Google Generative AI)
        - Any other API keys or configurations required by your project.



