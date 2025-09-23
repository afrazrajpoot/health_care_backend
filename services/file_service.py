import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import logging

from config.settings import CONFIG
from services.document_converter import DocumentConverter

logger = logging.getLogger("document_ai")

class FileService:
    """Service for handling file operations"""
    
    @staticmethod
    def validate_file(file: UploadFile, max_size: int) -> bytes:
        """Validate uploaded file"""
        if not file:
            raise ValueError("No file uploaded")
        
        # Check file extension
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if not DocumentConverter.is_supported_format(file.filename):
                supported_formats = DocumentConverter.get_supported_formats()
                raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {', '.join(sorted(supported_formats))}")
        
        # Read file content
        content = file.file.read()
        file.file.seek(0)  # Reset file pointer
        
        if len(content) > max_size:
            raise ValueError(f"File too large. Maximum size is {max_size / (1024*1024)}MB")
        
        return content
    
    @staticmethod
    def save_temp_file(content: bytes, filename: str) -> str:
        """Save file content to temporary file"""
        try:
            # Create temporary file
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                dir=CONFIG["upload_dir"]
            ) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            logger.info(f"💾 Temporary file created: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"❌ Error creating temporary file: {str(e)}")
            raise
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up temp file {file_path}: {str(e)}")
    
    @staticmethod
    def get_file_info(file: UploadFile, content: bytes) -> dict:
        """Get file information"""
        return {
            "originalName": file.filename,
            "size": len(content),
            "mimeType": file.content_type or "application/octet-stream"
        }