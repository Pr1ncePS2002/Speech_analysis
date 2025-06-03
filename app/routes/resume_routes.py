from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
import os
from pathlib import Path
from typing import Dict, Any
from app.services.resume_parser import parse_resume, parse_entire_resume


router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing resume: {file.filename}")
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
            
        parsed_data = parse_resume(content, file.filename)
        entire_data = parse_entire_resume(content, file.filename)
        
        if "error" in parsed_data:
            raise HTTPException(status_code=422, detail=parsed_data["error"])
            
        return {
            "status": "success",
            "filename": file.filename,
            "parsed": parsed_data,
            "entire_data": entire_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/test")
async def test_parser() -> Dict[str, Any]:
    """
    Test endpoint for resume parsing with a built-in sample resume.
    Returns:
        Dict: Parsed resume data or error message
    """
    try:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        
        # Path to sample resume (create a 'sample_resumes' folder in your routes directory)
        sample_path = current_dir / "sample_resumes" / "sample_resume.pdf"
        
        # Check if sample exists
        if not sample_path.exists():
            logger.error("Sample resume not found at: %s", sample_path)
            raise HTTPException(
                status_code=404,
                detail="Sample resume file not found. Please add a PDF at: " + str(sample_path)
            )
            
        with open(sample_path, "rb") as f:
            content = f.read()
            
        parsed_data = parse_resume(content, "sample_resume.pdf")
        
        if "error" in parsed_data:
            logger.error("Test parsing failed: %s", parsed_data["error"])
            raise HTTPException(
                status_code=422,
                detail=f"Test parsing failed: {parsed_data['error']}"
            )
            
        return {
            "status": "test_success",
            "filename": "sample_resume.pdf",
            "parsed": parsed_data
        }
        
    except Exception as e:
        logger.exception("Test endpoint failed unexpectedly")
        raise HTTPException(
            status_code=500,
            detail=f"Test failed unexpectedly: {str(e)}"
        )