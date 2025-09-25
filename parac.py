from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import asyncio
import traceback
import requests
import json

from models.schemas import ExtractionResult
from services.document_ai_service import get_document_ai_processor
from services.file_service import FileService
from services.document_converter import DocumentConverter
from config.settings import CONFIG
from utils.logger import logger
from services.report_analyzer import ReportAnalyzer
from services.database_service import get_database_service

from config.celery_config import app as celery_app

router = APIRouter()

# Webhook URL for saving results (can be configured in settings)
WEBHOOK_URL = CONFIG.get("webhook_url", "http://localhost:8000/api/save-result")

# Celery task for processing a single document (without database operations)
@celery_app.task(bind=True, name='process_document_task', max_retries=3, retry_backoff=True)
def process_document_task(self, gcs_url: str, original_filename: str, mime_type: str, file_size: int, blob_path: str, webhook_url: str = None):
    """
    Celery task to process a single document and call webhook to save results.
    """
    start_time = datetime.now()
    file_service = FileService()
    processor = get_document_ai_processor()
    
    # Initialize result with proper values to avoid Pydantic errors
    result = ExtractionResult(
        text="",
        pages=0,
        entities=[],
        tables=[],
        formFields=[],
        confidence=0.0,
        success=False,
        gcs_file_link=gcs_url
    )
    
    temp_path = None
    converted_path = None
    was_converted = False
    content = None
    
    try:
        logger.info(f"\n🔄 === ASYNC DOCUMENT PROCESSING STARTED (Task ID: {self.request.id}) ===\n")
        logger.info(f"📁 Original filename: {original_filename}")
        logger.info(f"📏 File size: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
        logger.info(f"📋 MIME type: {mime_type}")
        logger.info(f"☁️ GCS URL: {gcs_url}")
        
        # Download from GCS to temp for processing
        try:
            content = file_service.download_from_gcs(blob_path)
            logger.info("✅ File downloaded from GCS successfully")
        except Exception as gcs_error:
            logger.error(f"❌ GCS download failed: {str(gcs_error)}")
            raise
        
        # Save to temporary local file for processing
        temp_path = file_service.save_temp_file(content, original_filename)
        
        # Check if file needs conversion
        if DocumentConverter.needs_conversion(temp_path):
            logger.info(f"🔄 File requires conversion: {Path(temp_path).suffix}")
            try:
                converted_path, was_converted = DocumentConverter.convert_document(temp_path, target_format="pdf")
                processing_path = converted_path
                logger.info(f"✅ File converted successfully: {processing_path}")
            except Exception as convert_error:
                logger.error(f"❌ File conversion failed: {str(convert_error)}")
                raise
        else:
            processing_path = temp_path
            logger.info(f"✅ File format supported directly: {Path(temp_path).suffix}")
        
        # Process document with Document AI
        logger.info("🔍 Processing document with Document AI...")
        try:
            document_result = processor.process_document(processing_path)
            
            # Update the result object with extracted data
            result.text = document_result.text
            result.pages = document_result.pages
            result.entities = document_result.entities
            result.tables = document_result.tables
            result.formFields = document_result.formFields
            result.confidence = document_result.confidence
            result.success = document_result.success
            
            logger.info("✅ Document AI processing completed")
            
        except Exception as dai_error:
            logger.error(f"❌ Document AI processing failed: {str(dai_error)}")
            raise
        
        # Add file info with GCS URL
        result.fileInfo = {
            "originalName": original_filename,
            "size": file_size,
            "mimeType": mime_type or "application/octet-stream",
            "gcsUrl": gcs_url
        }
        
        # Comprehensive analysis with GPT-4o and document type detection
        if result.text:
            logger.info("🤖 Starting comprehensive document analysis...")
            try:
                analyzer = ReportAnalyzer()
                
                # Quick document type detection
                detected_type = analyzer.detect_document_type_preview(result.text)
                logger.info(f"🔍 Detected document type: {detected_type}")
                
                # Comprehensive analysis
                comprehensive_analysis = analyzer.analyze_document(result.text)
                result.comprehensive_analysis = comprehensive_analysis
                
                # Enhanced summary with document type context
                if comprehensive_analysis and comprehensive_analysis.summary:
                    summary_parts = comprehensive_analysis.summary
                    result.summary = " | ".join(summary_parts)
                    
                    logger.info(f"📋 Document Analysis Summary:")
                    logger.info(f"   📄 Type: {detected_type}")
                    logger.info(f"   👤 Patient: {comprehensive_analysis.report_json.patient_name or 'Unknown'}")
                    logger.info(f"   📑 Title: {comprehensive_analysis.report_json.report_title or 'Untitled'}")
                    logger.info(f"   📝 Summary: {len(summary_parts)} key points extracted")
                    
                    if comprehensive_analysis.work_status_alert:
                        logger.info(f"   🚨 Alerts: {len(comprehensive_analysis.work_status_alert)} generated")
                else:
                    result.summary = f"Document Type: {detected_type} - Analysis completed successfully"
                
                logger.info("✅ Comprehensive analysis completed")
                
            except Exception as e:
                logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
                analyzer = ReportAnalyzer()
                try:
                    detected_type = analyzer.detect_document_type_preview(result.text)
                    result.summary = f"Document Type: {detected_type} - Processing completed with limited analysis due to: {str(e)}"
                    logger.info(f"🔄 Fallback: Document type detected as {detected_type}")
                except Exception as fallback_e:
                    result.summary = f"Document processed successfully but analysis encountered errors: {str(fallback_e)}"
                result.comprehensive_analysis = None
        else:
            logger.warning("⚠️ No text extracted from document")
            result.summary = "Document processed but no readable text content was extracted"
            result.comprehensive_analysis = None
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"⏱️ Total processing time: {processing_time:.0f}ms")
        
        # Prepare the final result
        final_result = {
            "status": "success",
            "result": result.dict(),
            "processing_time_ms": int(processing_time),
            "filename": original_filename,
            "gcs_url": gcs_url,
            "task_id": self.request.id,
            "processed_at": datetime.now().isoformat()
        }
        
        # Call webhook to save results to database
        if webhook_url:
            try:
                logger.info(f"🌐 Calling webhook to save results: {webhook_url}")
                response = requests.post(
                    webhook_url,
                    json=final_result,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    webhook_result = response.json()
                    final_result["document_id"] = webhook_result.get("document_id")
                    final_result["webhook_status"] = "success"
                    logger.info(f"✅ Webhook call successful, document ID: {webhook_result.get('document_id')}")
                else:
                    final_result["webhook_status"] = f"failed_{response.status_code}"
                    logger.warning(f"⚠️ Webhook call failed with status {response.status_code}")
                    
            except Exception as webhook_error:
                final_result["webhook_status"] = "error"
                final_result["webhook_error"] = str(webhook_error)
                logger.error(f"❌ Webhook call failed: {str(webhook_error)}")
        else:
            final_result["webhook_status"] = "not_called"
            logger.info("ℹ️ No webhook URL provided, skipping database save")
        
        logger.info("✅ === ASYNC PROCESSING COMPLETED ===\n")
        
        return final_result
    
    except Exception as e:
        logger.error(f"❌ Error in document processing (Task ID: {self.request.id}): {str(e)}")
        # Clean up GCS file on error
        try:
            file_service.delete_from_gcs(blob_path)
        except:
            pass
        raise self.retry(exc=e, countdown=60)
    
    finally:
        # Clean up temporary files
        file_service.cleanup_temp_file(temp_path)
        if was_converted and converted_path:
            DocumentConverter.cleanup_converted_file(converted_path, was_converted)

# Webhook endpoint to save results to database
@router.post("/save-result")
async def save_result_webhook(result_data: Dict[str, Any]):
    """
    Webhook endpoint to save processed results to database.
    Called by Celery task after processing is complete.
    """
    try:
        logger.info(f"💾 Webhook received result data for saving to database")
        
        # Extract data from webhook payload
        extraction_result_dict = result_data.get("result", {})
        filename = result_data.get("filename", "unknown")
        file_size = result_data.get("result", {}).get("fileInfo", {}).get("size", 0)
        mime_type = result_data.get("result", {}).get("fileInfo", {}).get("mimeType", "application/octet-stream")
        processing_time_ms = result_data.get("processing_time_ms", 0)
        gcs_url = result_data.get("gcs_url", "")
        task_id = result_data.get("task_id", "")
        
        # Convert dict back to ExtractionResult model
        extraction_result = ExtractionResult(**extraction_result_dict)
        
        # Get last changes from analysis if available
        last_changes = None
        if extraction_result.comprehensive_analysis:
            patient_name = extraction_result.comprehensive_analysis.report_json.patient_name
            if patient_name:
                db_service = await get_database_service()
                previous_document = await db_service.get_last_document_for_patient(patient_name)
                if previous_document:
                    previous_summary = previous_document.get('summary', [])
                    analyzer = ReportAnalyzer()
                    current_summary = extraction_result.comprehensive_analysis.summary or []
                    last_changes = analyzer.compare_summaries(previous_summary, current_summary)
                else:
                    last_changes = "this patient is new"
        
        # Save to database
        db_service = await get_database_service()
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=filename,
            file_size=file_size,
            mime_type=mime_type,
            processing_time_ms=processing_time_ms,
            gcs_file_link=gcs_url,
            last_changes=last_changes
        )
        
        logger.info(f"✅ Result saved to database via webhook, document ID: {document_id}")
        
        return {
            "status": "success",
            "document_id": document_id,
            "task_id": task_id,
            "saved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error saving result via webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save result: {str(e)}")

@router.post("/extract-documents", response_model=Dict[str, List[str]])
async def extract_documents(
    documents: List[UploadFile] = File(...),
    save_to_db: bool = True  # Option to enable/disable database saving
):
    """
    Upload multiple documents and queue them for asynchronous processing with Celery.
    Optionally save results to database via webhook.
    """
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    file_service = FileService()
    task_ids = []
    uploaded_files = []
    
    # Determine webhook URL
    webhook_url = f"http://localhost:{CONFIG.get('port', 8000)}/api/save-result" if save_to_db else None
    
    try:
        logger.info(f"\n🔄 === NEW MULTI-DOCUMENT PROCESSING REQUEST ({len(documents)} files) ===\n")
        logger.info(f"💾 Database saving: {'ENABLED' if save_to_db else 'DISABLED'}")
        
        for document in documents:
            content = await document.read()
            file_service.validate_file(document, CONFIG["max_file_size"])
            
            logger.info(f"📁 Processing file: {document.filename}")
            logger.info(f"📏 File size: {len(content)} bytes ({len(content)/(1024*1024):.2f} MB)")
            logger.info(f"📋 MIME type: {document.content_type}")
            
            # Save to Google Cloud Storage
            logger.info("☁️ Uploading file to Google Cloud Storage...")
            gcs_url, blob_path = file_service.save_to_gcs(content, document.filename)
            logger.info(f"✅ Uploaded file to GCS: {gcs_url}")
            uploaded_files.append(blob_path)
            
            # Enqueue Celery task with webhook URL
            task = process_document_task.delay(
                gcs_url, 
                document.filename, 
                document.content_type, 
                len(content),
                blob_path,
                webhook_url  # Pass webhook URL to task
            )
            task_ids.append(task.id)
            logger.info(f"🚀 Task queued: {task.id}")
            if webhook_url:
                logger.info(f"🌐 Webhook configured for database saving")
        
        logger.info("✅ === ALL FILES QUEUED FOR PROCESSING ===\n")
        
        return {
            "task_ids": task_ids,
            "save_to_db": save_to_db,
            "webhook_url": webhook_url,
            "message": f"Documents queued for processing. Database saving: {'enabled' if save_to_db else 'disabled'}"
        }
    
    except ValueError as ve:
        logger.error(f"❌ Validation error: {str(ve)}")
        for path in uploaded_files:
            file_service.delete_from_gcs(path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Error in multi-document queuing: {str(e)}")
        for path in uploaded_files:
            file_service.delete_from_gcs(path)
        raise HTTPException(status_code=500, detail=f"Queuing failed: {str(e)}")

# Task result endpoint
@router.get("/task-result/{task_id}", response_model=Dict[str, Any])
async def get_task_result(task_id: str):
    """
    Retrieve the result of a Celery task by its ID.
    """
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return {"task_id": task_id, "status": "pending"}
    elif task.state == 'SUCCESS':
        return {"task_id": task_id, "status": "success", "result": task.result}
    elif task.state == 'FAILURE':
        return {"task_id": task_id, "status": "failed", "error": str(task.result)}
    else:
        return {"task_id": task_id, "status": task.state}

# Keep your original synchronous endpoint
@router.post("/extract-documents", response_model=Dict[str, List[str]])
async def extract_documents(
    documents: List[UploadFile] = File(...)
):
    """
    Upload multiple documents and queue them for asynchronous processing with Celery.
    Returns task IDs for tracking results.
    """
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    file_service = FileService()
    task_ids = []
    uploaded_files = []
    
    try:
        logger.info(f"\n🔄 === NEW MULTI-DOCUMENT PROCESSING REQUEST ({len(documents)} files) ===\n")
        
        for document in documents:
            content = await document.read()
            file_service.validate_file(document, CONFIG["max_file_size"])
            
            logger.info(f"📁 Processing file: {document.filename}")
            logger.info(f"📏 File size: {len(content)} bytes ({len(content)/(1024*1024):.2f} MB)")
            logger.info(f"📋 MIME type: {document.content_type}")
            
            # Save to Google Cloud Storage
            logger.info("☁️ Uploading file to Google Cloud Storage...")
            gcs_url, blob_path = file_service.save_to_gcs(content, document.filename)
            logger.info(f"✅ Uploaded file to GCS: {gcs_url}")
            logger.info(f"📎 Signed GCS URL: {gcs_url}")
            uploaded_files.append(blob_path)
            
            # Enqueue Celery task
            logger.debug(f"DEBUG: Queuing task for {document.filename}")
            task = process_document_task.delay(
                gcs_url, 
                document.filename, 
                document.content_type, 
                len(content),
                blob_path
            )
            task_ids.append(task.id)
            logger.info(f"🚀 Task queued: {task.id}")
            logger.debug(f"DEBUG: Task queued successfully: {task.id}")
        
        logger.info("✅ === ALL FILES QUEUED FOR PROCESSING ===\n")
        
        return {"task_ids": task_ids}
    
    except ValueError as ve:
        logger.error(f"❌ Validation error: {str(ve)}")
        logger.debug(f"DEBUG: Validation error traceback: {traceback.format_exc()}")
        for path in uploaded_files:
            file_service.delete_from_gcs(path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Error in multi-document queuing: {str(e)}")
        logger.debug(f"DEBUG: Queuing error traceback: {traceback.format_exc()}")
        for path in uploaded_files:
            file_service.delete_from_gcs(path)
        raise HTTPException(status_code=500, detail=f"Queuing failed: {str(e)}")
