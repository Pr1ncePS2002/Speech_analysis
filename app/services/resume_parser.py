# services/resume_parser.py

import io
import os
import re
import docx2txt
import pdfplumber
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path
import unicodedata # Added import for clean_text_for_embedding

#unstructured lib import
from unstructured.partition.image import partition_image
from unstructured.partition.pdf import partition_pdf
import tempfile

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Copy the clean_text_for_embedding function here as well ---
def clean_text_for_embedding(text: str) -> str:
    """
    Cleans text to ensure it can be encoded with UTF-8,
    removing or replacing problematic characters.
    """
    normalized_text = unicodedata.normalize('NFKC', text)
    cleaned_chars = []
    for char in normalized_text:
        try:
            char.encode('utf-8')
            cleaned_chars.append(char)
        except UnicodeEncodeError:
            cleaned_chars.append(' ') # Replace with space or ''
    return "".join(cleaned_chars)
# --- End of copied function ---

def normalize_text(text: str) -> str:
    """Normalize text for better keyword matching"""
    try:
        text = text.lower()
        text = re.sub(r'[^\w\s\+#\-/]', ' ', text)
        return ' '.join(text.split())
    except Exception as e:
        logger.warning(f"Text normalization failed: {str(e)}")
        return text.lower() if text else ""

def load_keywords() -> Dict[str, List[str]]:
    # ... (your existing keywords list) ...
    return {
        "skills": [
            # Programming Languages
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", 
            "ruby", "php", "swift", "kotlin", "scala", "r", "dart", "perl",

            # Web Development
            "html", "css", "sass", "less", "react", "angular", "vue", "node.js", 
            "express", "django", "flask", "spring", "laravel", "asp.net", "graphql",

            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "cassandra", 
            "dynamodb", "firebase", "neo4j", "elasticsearch",

            # DevOps & Cloud
            "docker", "kubernetes", "terraform", "ansible", "jenkins", "gitlab", 
            "github actions", "aws", "azure", "gcp", "ibm cloud", "serverless",
            "ci/cd", "helm", "prometheus", "grafana",

            # Data Science & AI
            "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "keras", 
            "opencv", "nltk", "spacy", "hadoop", "spark", "kafka", "airflow",
            "tableau", "power bi", "matplotlib", "seaborn",

            # Mobile Development
            "android", "ios", "react native", "flutter", "xamarin", "ionic",

            # Cybersecurity
            "cybersecurity", "penetration testing", "ethical hacking", "siem", 
            "nmap", "metasploit", "burp suite", "owasp", "soc", "pki", "vpn",

            # Networking
            "tcp/ip", "dns", "dhcp", "vlan", "ospf", "bgp", "mpls", "sdn",

            # Other Technologies
            "blockchain", "solidity", "smart contracts", "arduino", "raspberry pi",
            "iot", "computer vision", "nlp", "quantum computing"
        ],
        "roles": [
            # Software Development
            "software engineer", "backend developer", "frontend developer", 
            "full stack developer", "web developer", "mobile developer",
            "embedded systems engineer", "game developer",

            # Data & AI
            "data scientist", "machine learning engineer", "ai engineer", 
            "data analyst", "data engineer", "business intelligence analyst",
            "research scientist", "quantitative analyst",

            # DevOps & Cloud
            "devops engineer", "site reliability engineer", "cloud engineer",
            "cloud architect", "platform engineer", "release engineer",

            # Cybersecurity
            "security engineer", "cybersecurity analyst", "penetration tester",
            "security consultant", "information security officer",

            # Networking & Systems
            "network engineer", "systems administrator", "network administrator",
            "it support specialist", "database administrator",

            # Management & Architecture
            "technical lead", "engineering manager", "cto", 
            "solutions architect", "system architect",

            # QA & Testing
            "qa engineer", "test engineer", "automation engineer",
            "performance engineer", "quality analyst",

            # Other IT Roles
            "technical writer", "it consultant", "scrum master", 
            "product owner", "it project manager"
        ]
    }

def extract_skills_and_roles(text: str) -> Dict[str, List[str]]:
    """Enhanced keyword matching with comprehensive IT vocabulary"""
    try:
        normalized_text = normalize_text(text)
        keywords = load_keywords()

        found_skills = []
        found_roles = []

        for skill in keywords["skills"]:
            if re.search(rf'\b{re.escape(skill)}\b', normalized_text):
                found_skills.append(skill)

        for role in keywords["roles"]:
            if re.search(rf'\b{re.escape(role)}\b', normalized_text):
                found_roles.append(role)

        found_skills = list(dict.fromkeys(found_skills))
        found_roles = list(dict.fromkeys(found_roles))

        return {
            "skills": found_skills,
            "roles": found_roles if found_roles else ["IT Professional"]
        }
    except Exception as e:
        logger.error(f"Keyword extraction failed: {str(e)}")
        return {"skills": [], "roles": ["IT Professional"]}

def extract_text_from_resume(content: bytes, filename: str) -> Optional[str]:
    """Robust text extraction from various resume formats"""
    try:
        file_ext = Path(filename).suffix.lower()

        if file_ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            elements = partition_pdf(filename=tmp_path, languages=["eng"])
            os.remove(tmp_path)  # Clean up temporary file

            text = ""
            for element in elements:
                text += element.text + "\n"

            # Fallback using pdfplumber if partitioned text is empty
            if not text.strip():
                logger.warning("PDF text extraction failed, trying pdfplumber fallback")
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    text = "\n".join(
                        str(page.chars) if hasattr(page, 'chars') else ""
                        for page in pdf.pages
                    )

            return text

        elif file_ext == ".docx":
            return docx2txt.process(io.BytesIO(content))

        elif file_ext in (".txt", ".rtf"):
            return content.decode('utf-8', errors='ignore')
        
        elif file_ext in (".jpg", ".jpeg"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            elements = partition_image(filename=tmp_path)
            os.remove(tmp_path)

            text = ""
            for element in elements:
                text += element.text

            return text

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        return None

def parse_resume(content: bytes, filename: str) -> Dict[str, Union[dict, str]]:
    """Robust resume parsing with comprehensive error handling"""
    try:
        if not content:
            logger.warning("Received empty file content")
            return {"error": "Empty file content"}

        if not filename:
            return {"error": "No filename provided"}

        text = extract_text_from_resume(content, filename)

        if not text or not text.strip():
            return {"error": "Could not extract text from file"}

        # Apply cleaning here before further processing
        text = clean_text_for_embedding(text)

        result = extract_skills_and_roles(text)
        result["success"] = True
        return result

    except Exception as e:
        logger.error(f"Resume parsing failed: {str(e)}")
        return {"error": f"Parsing failed: {str(e)}", "success": False}

def parse_entire_resume(content: bytes, filename: str) -> Dict[str, Union[str, dict]]:
    """Parse the entire resume with metadata and extracted data"""
    try:
        if not content:
            return {"error": "Empty file content", "success": False}

        text = extract_text_from_resume(content, filename)

        if not text or not text.strip():
            return {"error": "Could not extract text from file", "success": False}

        # Apply cleaning here before further processing and storing
        clean_full_text = clean_text_for_embedding(text)

        extracted_data = extract_skills_and_roles(clean_full_text)

        # Basic metadata extraction
        email = re.search(r'[\w\.-]+@[\w\.-]+', clean_full_text)
        phone = re.search(r'(\+\d{1,3})?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', clean_full_text)

        return {
            "success": True,
            "full_text": clean_full_text[:5000] + "..." if len(clean_full_text) > 5000 else clean_full_text,  # Limit size
            "metadata": {
                "email": email.group(0) if email else None,
                "phone": phone.group(0) if phone else None,
            },
            "extracted_data": extracted_data
        }

    except Exception as e:
        logger.error(f"Full resume parsing failed: {str(e)}")
        return {"error": f"Full parsing failed: {str(e)}", "success": False}
    
