import io
import docx2txt
import pdfplumber
from typing import Dict, Union


def normalize_text(text: str) -> str:
    """Normalize text for better keyword matching"""
    text = text.lower()
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    return ' '.join(text.split())  # Remove extra whitespace


def extract_skills_and_roles(text: str) -> Dict[str, list]:
    """Enhanced keyword matching with normalization"""
    normalized_text = normalize_text(text)

    skills_keywords = [
        "python", "java", "sql", "docker", "kubernetes", "machine learning",
        "data analysis", "fastapi", "react", "cloud", "aws", "git", "linux"
    ]

    roles_keywords = [
        "data analyst", "software engineer", "backend developer",
        "ml engineer", "developer", "engineer"
    ]

    found_skills = list({skill for skill in skills_keywords if skill in normalized_text})
    found_roles = list({role for role in roles_keywords if role in normalized_text})

    return {
        "skills": found_skills,
        "roles": found_roles if found_roles else ["Software Professional"]
    }


def extract_text_from_resume(content: bytes, filename: str) -> str:
    """Extract raw text from resume"""
    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    elif filename.endswith(".docx"):
        return docx2txt.process(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")


def parse_resume(content: bytes, filename: str) -> Dict[str, Union[dict, str]]:
    """Parse resume to extract skills and roles"""
    try:
        if not content:
            return {"error": "Empty file content"}

        text = extract_text_from_resume(content, filename)

        if not text.strip():
            return {"error": "Could not extract text from file"}

        return extract_skills_and_roles(text)

    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}


def parse_entire_resume(content: bytes, filename: str) -> Dict[str, Union[str, dict]]:
    """Parse the entire resume and return both text and extracted data"""
    try:
        if not content:
            return {"error": "Empty file content"}

        text = extract_text_from_resume(content, filename)

        if not text.strip():
            return {"error": "Could not extract text from file"}

        return {
            "full_text": text,

        }

    except Exception as e:
        return {"error": f"Full resume parsing failed: {str(e)}"}
