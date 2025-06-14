import streamlit as st
from unstructured.partition.pdf import partition_pdf
import tempfile
import re
from typing import List, Dict
import unicodedata

def custom_resume_splitter(text: str) -> Dict[str, str]:
    """
    Splits resume text into chunks based on given keywords (section headers).
    
    Args:
        text (str): The entire resume text.
        keywords (List[str]): Section headers to split on (case-insensitive).
    
    Returns:
        Dict[str, str]: A dictionary with section names as keys and their content as values.
    """
    keywords = ["SUMMARY", "EDUCATION", "PROJECTS", "RELEVANT SKILLS", "SKILLS", "CERTIFICATIONS", "ADDITIONAL INFORMATION"]
    
    # Normalize text
    text = text.replace('\n', '\n\n')  # Separate lines visually for regex reliability

    # Create regex pattern for section titles
    pattern = r'(?i)^(' + '|'.join(re.escape(k) for k in keywords) + r')\s*$'
    
    # Find all matches
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))

    #Extracting name
    name = ''
    for i in range(50):
        if text[i] == "\n":
            break
        name += text[i]
    
    # Extract sections
    chunks = {}
    chunks['Name'] = name
    
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = match.group(1).strip().title()
        chunks[section] = text[start:end].strip()
    
    return chunks
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

st.title("📄 Resume Text Extractor with Unstructured")

file = st.file_uploader("Upload Resume here", type=["pdf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    elements = partition_pdf(filename=tmp_path, languages=["eng"])

    text = ""
    for element in elements:
        text += element.text + "\n"

    st.subheader("📃 Extracted Text:")
    st.markdown(text)

    st.subheader("📃 Chunks:")
    st.markdown(custom_resume_splitter(clean_text_for_embedding(text)))
