from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
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

# Webhook endpoint to save document analysis to database
@router.post("/webhook/save-document")
async def save_document_webhook(request: Request):
    """
    Webhook endpoint to save document analysis to the database and compute last_changes.
    """
    try:
        data = await request.json()
        logger.info(f"📥 Webhook received for document save: {data.get('document_id', 'unknown')}")

        # Validate required fields
        if not data.get("result") or not data.get("filename") or not data.get("gcs_url"):
            raise HTTPException(status_code=400, detail="Missing required fields in webhook payload")

        # Convert result dict to ExtractionResult
        result_data = data["result"]
        result = ExtractionResult(
            text=result_data.get("text", ""),
            pages=result_data.get("pages", 0),
            entities=result_data.get("entities", []),
            tables=result_data.get("tables", []),
            formFields=result_data.get("formFields", []),
            confidence=result_data.get("confidence", 0.0),
            success=result_data.get("success", False),
            gcs_file_link=result_data.get("gcs_file_link", data["gcs_url"]),
            fileInfo=result_data.get("fileInfo", {}),
            summary=result_data.get("summary", ""),
            comprehensive_analysis=result_data.get("comprehensive_analysis"),
            document_id=result_data.get("document_id", f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )

        # Compute last_changes
        last_changes = None
        if result.text and result.comprehensive_analysis:
            try:
                patient_name = result.comprehensive_analysis.report_json.patient_name if result.comprehensive_analysis.report_json else None
                if patient_name:
                    db_service = await get_database_service()
                    previous_document = await db_service.get_last_document_for_patient(patient_name)
                    if previous_document:
                        previous_summary = previous_document.get('summary', [])
                        analyzer = ReportAnalyzer()
                        last_changes = analyzer.compare_summaries(previous_summary, result.comprehensive_analysis.summary)
                        logger.info(f"🔄 Generated last_changes based on previous summary for patient: {patient_name}")
                    else:
                        last_changes = "this patient is new"
                        logger.info(f"✅ This is a new patient: {patient_name}")
                else:
                    logger.warning("⚠️ No patient name extracted for last_changes comparison")
            except Exception as e:
                logger.error(f"❌ Failed to compute last_changes: {str(e)}")
                last_changes = f"last_changes computation failed: {str(e)}"

        # Save to database
        db_service = await get_database_service()
        document_id = await db_service.save_document_analysis(
            extraction_result=result,
            file_name=data["filename"],
            file_size=data.get("file_size", 0),
            mime_type=data.get("mime_type", "application/octet-stream"),
            processing_time_ms=data.get("processing_time_ms", 0),
            gcs_file_link=data["gcs_url"],
            last_changes=last_changes
        )

        logger.info(f"💾 Document saved via webhook with ID: {document_id}")
        return {"status": "success", "document_id": document_id}

    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

# Celery task for processing a single document
@celery_app.task(bind=True, name='process_document_task', max_retries=3, retry_backoff=True)
def process_document_task(self, gcs_url: str, original_filename: str, mime_type: str, file_size: int, blob_path: str):
    """
    Celery task to process a single document and trigger webhook for database save.
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
        logger.info(f"📍 Blob path: {blob_path}")
        
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
                
                # Log patient info (no database lookup in task)
                patient_name = None
                if comprehensive_analysis and comprehensive_analysis.report_json:
                    patient_name = comprehensive_analysis.report_json.patient_name
                    logger.info(f"👤 Patient identified: {patient_name}")
                else:
                    logger.warning("⚠️ No patient name extracted")
                
                # Enhanced summary with document type context
                if comprehensive_analysis and comprehensive_analysis.summary:
                    summary_parts = comprehensive_analysis.summary
                    result.summary = " | ".join(summary_parts)
                    
                    logger.info(f"📋 Document Analysis Summary:")
                    logger.info(f"   📄 Type: {detected_type}")
                    logger.info(f"   👤 Patient: {patient_name or 'Unknown'}")
                    logger.info(f"   📑 Title: {comprehensive_analysis.report_json.report_title or 'Untitled'}")
                    logger.info(f"   📝 Summary: {len(summary_parts)} key points extracted")
                    
                    if comprehensive_analysis.work_status_alert:
                        logger.info(f"   🚨 Alerts: {len(comprehensive_analysis.work_status_alert)} generated")
                else:
                    result.summary = f"Document Type: {detected_type} - Analysis completed successfully"
                
                logger.info("✅ Comprehensive analysis completed")
                
            except Exception as e:
                logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
                # Fallback analysis
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
        
        # Assign a temporary document ID
        result.document_id = f"celery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare payload for webhook (remove last_changes)
        webhook_payload = {
            "result": result.dict(),
            "filename": original_filename,
            "file_size": file_size,
            "mime_type": mime_type or "application/octet-stream",
            "processing_time_ms": int(processing_time),
            "gcs_url": gcs_url,
            "document_id": result.document_id
        }
        
        # Call webhook to save to database synchronously
        webhook_url = CONFIG.get("api_base_url", "http://localhost:8000") + "/api/webhook/save-document"
        logger.info(f"🌐 Calling webhook: {webhook_url}")
        try:
            response = requests.post(webhook_url, json=webhook_payload, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                result.document_id = response_data.get("document_id", result.document_id)
                logger.info(f"✅ Webhook called successfully, document ID: {result.document_id}")
            else:
                logger.error(f"❌ Webhook call failed with status {response.status_code}: {response.text}")
                result.database_error = f"Webhook call failed with status {response.status_code}: {response.text}"
                result.document_id = f"webhook_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception as webhook_error:
            logger.error(f"❌ Webhook call failed: {str(webhook_error)}")
            result.database_error = str(webhook_error)
            result.document_id = f"webhook_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("✅ === ASYNC PROCESSING COMPLETED ===\n")
        
        return {
            "status": "success",
            "result": result.dict(),
            "processing_time_ms": int(processing_time),
            "filename": original_filename,
            "gcs_url": gcs_url,
            "document_id": result.document_id
        }
    
    except Exception as e:
        logger.error(f"❌ Error in document processing (Task ID: {self.request.id}): {str(e)}")
        # Skip GCS deletion during retries to preserve file for next attempt
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying task (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=60)
        else:
            # Only delete GCS file after all retries are exhausted
            try:
                file_service.delete_from_gcs(blob_path)
                logger.info(f"🗑️ Deleted file from GCS: {blob_path}")
            except:
                logger.warning(f"⚠️ Failed to delete GCS file: {blob_path}")
        raise
    
    finally:
        # Clean up temporary files
        file_service.cleanup_temp_file(temp_path)
        if was_converted and converted_path:
            DocumentConverter.cleanup_converted_file(converted_path, was_converted)

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
            logger.info(f"📍 Blob path: {blob_path}")
            logger.info(f"📎 Signed GCS URL: {gcs_url}")
            uploaded_files.append(blob_path)
            
            # Enqueue Celery task
            logger.debug(f"DEBUG: Queuing task for {document.filename}")
            task = process_document_task.delay(
                gcs_url=gcs_url,
                original_filename=document.filename,
                mime_type=document.content_type,
                file_size=len(content),
                blob_path=blob_path
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
            try:
                file_service.delete_from_gcs(path)
                logger.info(f"🗑️ Deleted file from GCS due to error: {path}")
            except:
                logger.warning(f"⚠️ Failed to delete GCS file: {path}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Error in multi-document queuing: {str(e)}")
        logger.debug(f"DEBUG: Queuing error traceback: {traceback.format_exc()}")
        for path in uploaded_files:
            try:
                file_service.delete_from_gcs(path)
                logger.info(f"🗑️ Deleted file from GCS due to error: {path}")
            except:
                logger.warning(f"⚠️ Failed to delete GCS file: {path}")
        raise HTTPException(status_code=500, detail=f"Queuing failed: {str(e)}")