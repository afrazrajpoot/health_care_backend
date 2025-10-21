# API Routes (full relevant extract_documents endpoint with added debug prints)
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from typing import Dict, Any, List
from datetime import datetime
import traceback
from pathlib import Path
from utils.celery_task import process_batch_documents
from services.progress_service import progress_service  # ENSURE THIS IMPORT IS PRESENT
# ... (other imports: FileService, get_database_service, compute_file_hash, DocumentConverter, get_document_ai_processor, ExtractionResult, ReportAnalyzer, CONFIG, logger)

router = APIRouter()
