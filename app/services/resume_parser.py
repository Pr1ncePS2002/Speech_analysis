import io
import docx2txt
import pdfplumber
from typing import Dict, Union

def extract_skills_and_roles(text: str) -> Dict[str, list]:
    """Enhanced keyword matching with normalization"""
    text = text.lower()
    
    skills_keywords = [
        "python", "java", "sql", "docker", "kubernetes", "machine learning",
        "data analysis", "fastapi", "react", "cloud", "aws", "git", "linux"
    ]
    
    roles_keywords = [
        "data analyst", "software engineer", "backend developer", 
        "ml engineer", "developer", "engineer"
    ]

    found_skills = list(set([skill for skill in skills_keywords if skill in text]))
    found_roles = list(set([role for role in roles_keywords if role in text]))
    
    return {
        "skills": found_skills,
        "roles": found_roles if found_roles else ["Software Professional"]  # Default role
    }

def parse_resume(content: bytes, filename: str) -> Dict[str, Union[dict, str]]:
    try:
        if not content:
            return {"error": "Empty file content"}
            
        text = ""
        if filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif filename.endswith(".docx"):
            text = docx2txt.process(io.BytesIO(content))
        else:
            return {"error": "Unsupported file format. Please upload PDF or DOCX"}
            
        if not text.strip():
            return {"error": "Could not extract text from file"}
            
        return extract_skills_and_roles(text)
        
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}

def normalize_text(text: str) -> str:
    """Normalize text for better keyword matching"""
    text = text.lower()
    # Remove special characters
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    return ' '.join(text.split())  # Remove extra whitespace