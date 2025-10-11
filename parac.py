# controllers/document_controller.py (updated: check for existing document before GCS upload and handle like ignored/required field issue)

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query
from typing import Dict, List, Any
from datetime import datetime
import traceback

from models.schemas import ExtractionResult
from services.file_service import FileService
from config.settings import CONFIG
from utils.logger import logger
from services.report_analyzer import ReportAnalyzer
from services.database_service import get_database_service
from utils.celery_task import finalize_document_task
from utils.socket_manager import sio
from pathlib import Path
from fastapi import Path as FastAPIPath
from services.document_ai_service import get_document_ai_processor
from services.document_converter import DocumentConverter
import os
from urllib.parse import urlparse
router = APIRouter()

def extract_blob_path_from_gcs_url(gcs_url: str) -> str:
    """Extract blob path from signed GCS URL."""
    try:
        parsed = urlparse(gcs_url)
        full_path = parsed.path.lstrip('/')
        parts = full_path.split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GCS URL format")
        blob_name = '/'.join(parts[1:])
        if '?' in blob_name:
            blob_name = blob_name.split('?')[0]
        return blob_name
    except Exception as e:
        logger.error(f"❌ Failed to extract blob path from {gcs_url}: {str(e)}")
        raise ValueError(f"Invalid GCS URL: {str(e)}")


@router.post("/extract-documents", response_model=Dict[str, Any])
async def extract_documents(
    documents: List[UploadFile] = File(...),
    physicianId: str = Query(None, description="Optional physician ID for associating documents"),
    userId: str = Query(None, description="Optional user ID for associating documents")
):
    """
    Upload multiple documents: parse/validate synchronously, then queue for finalization.
    All documents with extracted text are sent to webhook for processing.
    """
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    file_service = FileService()
    task_ids = []
    ignored_filenames = []
    successful_uploads = []  # Track only successful uploads for cleanup
    
    try:
        logger.info(f"\n🔄 === NEW MULTI-DOCUMENT PROCESSING REQUEST ({len(documents)} files) ===\n")
        if physicianId:
            logger.info(f"👨‍⚕️ Physician ID provided: {physicianId}")
        
        for document in documents:
            document_start_time = datetime.now()
            content = await document.read()
            file_size = len(content)
            blob_path = None
            temp_path = None  # Initialize for finally block
            
            try:
                file_service.validate_file(document, CONFIG["max_file_size"])
                logger.info(f"📁 Starting processing for file: {document.filename}")
                logger.info(f"📏 File size: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
                logger.info(f"📋 MIME type: {document.content_type}")
                
                # Save to temp for processing
                temp_path = file_service.save_temp_file(content, document.filename)
                was_converted = False
                converted_path = None
                processing_path = temp_path
                
                # Conversion
                try:
                    if DocumentConverter.needs_conversion(temp_path):
                        logger.info(f"🔄 Converting file: {Path(temp_path).suffix}")
                        converted_path, was_converted = DocumentConverter.convert_document(temp_path, target_format="pdf")
                        processing_path = converted_path
                        logger.info(f"✅ Converted to: {processing_path}")
                    else:
                        logger.info(f"✅ Direct support for: {Path(temp_path).suffix}")
                except Exception as convert_exc:
                    logger.error(f"