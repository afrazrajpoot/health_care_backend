import os
from google.cloud import storage
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile
import tempfile
from datetime import datetime, timedelta
from config.settings import CONFIG
from services.document_converter import DocumentConverter
import logging

logger = logging.getLogger("document_ai")

class FileService:
    """Service for handling file operations with Google Cloud Storage"""

    def __init__(self):
        self.bucket_name = CONFIG.get("gcs_bucket_name", "hiregenix")
        if not self.bucket_name:
            raise ValueError("GCS bucket name not configured")
        
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
            if not self.bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist, attempting to create")
                self.bucket = self.storage_client.create_bucket(self.bucket_name)
                logger.info(f"✅ Created new bucket: {self.bucket_name}")
            else:
                logger.info(f"✅ Using existing bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"❌ Error initializing GCS client: {str(e)}")
            raise

    def download_from_gcs(self, blob_path: str) -> bytes:
        """Download file content from Google Cloud Storage"""
        try:
            blob = self.bucket.blob(blob_path)
            content = blob.download_as_bytes()
            logger.info(f"✅ Downloaded file from GCS: {blob_path} ({len(content)} bytes)")
            return content
        except Exception as e:
            logger.error(f"❌ Error downloading from GCS: {str(e)}")
            raise

    @staticmethod
    def validate_file(file: UploadFile, max_size: int) -> bytes:
        """Validate uploaded file"""
        if not file:
            raise ValueError("No file uploaded")
        
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if not DocumentConverter.is_supported_format(file.filename):
                supported_formats = DocumentConverter.get_supported_formats()
                raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {', '.join(sorted(supported_formats))}")
        
        content = file.file.read()
        file.file.seek(0)
        
        if len(content) > max_size:
            raise ValueError(f"File too large. Maximum size is {max_size / (1024*1024)}MB")
        
        return content
    
    def save_to_gcs(self, content: bytes, filename: str, content_type: str = 'application/octet-stream', folder: str = "uploads") -> Tuple[str, str]:
        """Upload file content to Google Cloud Storage and return the signed URL and blob path"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            original_name = Path(filename).stem
            extension = Path(filename).suffix
            unique_filename = f"{original_name}_{timestamp}{extension}"
            
            destination_blob_name = f"{folder}/{unique_filename}"
            
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_string(content, content_type=content_type)
            
            logger.info(f"✅ Uploaded file to GCS: {destination_blob_name} (Content-Type: {content_type})")
            
            # Generate signed URL for preview/download (browsers will preview if content-type supports it, e.g., PDF, images)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(days=7),
                method="GET",
                # Optional: Force inline disposition for better preview support
                response_disposition="inline"
            )
            
            logger.info(f"📎 Signed GCS URL: {signed_url}")
            
            return signed_url, destination_blob_name
            
        except Exception as e:
            logger.error(f"❌ Error uploading to GCS: {str(e)}")
            raise
    
    def save_temp_file(self, content: bytes, filename: str) -> str:
        """Save file content to temporary local file for processing"""
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"doc_ai_{timestamp}_{filename}"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(content)
            
            logger.info(f"📁 Created temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"❌ Error creating temporary file: {str(e)}")
            raise
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary local file"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up temporary file {file_path}: {str(e)}")
    
    def get_file_info(self, file: UploadFile, content: bytes, gcs_url: str = None) -> dict:
        """Get file information"""
        return {
            "originalName": file.filename,
            "size": len(content),
            "mimeType": file.content_type or "application/octet-stream",
            "gcsUrl": gcs_url
        }
    
    def delete_from_gcs(self, blob_path: str) -> bool:
        """Delete file from Google Cloud Storage"""
        try:
            blob = self.bucket.blob(blob_path)
            blob.delete()
            logger.info(f"🗑️ Deleted file from GCS: {blob_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting file from GCS: {str(e)}")
            return False