# utils/celery_task.py (final: no emits, no RedisManager—emits now only in webhook)
from config.celery_config import app as celery_app  # Reuse the existing Celery app config
from datetime import datetime
from pathlib import Path
import traceback

from models.schemas import ExtractionResult
from services.document_ai_service import get_document_ai_processor
from services.file_service import FileService
from services.document_converter import DocumentConverter
from config.settings import CONFIG
from utils.logger import logger
from services.report_analyzer import ReportAnalyzer

# Celery task for processing a single document
@celery_app.task(bind=True, name='process_document_task', max_retries=3, retry_backoff=True)
def process_document_task(self, gcs_url: str, original_filename: str, mime_type: str, file_size: int, blob_path: str, physician_id: str = None, user_id: str = None):
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
        if physician_id:
            logger.info(f"👨‍⚕️ Physician ID: {physician_id}")
        
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
        
        # Basic analysis for task completion
        if result.text:
            logger.info("📝 Performing basic document analysis...")
            try:
                analyzer = ReportAnalyzer()
                
                # Quick document type detection
                detected_type = analyzer.detect_document_type_preview(result.text)
                logger.info(f"🔍 Detected document type: {detected_type}")
                
                # Basic summary
                result.summary = f"Document Type: {detected_type} - Processing completed successfully"
                
                logger.info("✅ Basic analysis completed")
                
            except Exception as e:
                logger.error(f"❌ Basic analysis failed: {str(e)}")
                result.summary = f"Document processed successfully but analysis encountered errors: {str(e)}"
        else:
            logger.warning("⚠️ No text extracted from document")
            result.summary = "Document processed but no readable text content was extracted"
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"⏱️ Total processing time: {processing_time:.0f}ms")
        
        # Assign a temporary document ID
        result.document_id = f"celery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare payload for webhook
        webhook_payload = {
            "result": result.dict(),
            "filename": original_filename,
            "file_size": file_size,
            "mime_type": mime_type or "application/octet-stream",
            "processing_time_ms": int(processing_time),
            "gcs_url": gcs_url,
            "document_id": result.document_id,
            "physician_id": physician_id , # Pass physician_id to webhook
            "user_id": user_id  # Pass user_id to webhook (if needed)
        }
        
        # Call webhook to save to database synchronously
        webhook_url = CONFIG.get("api_base_url", "http://localhost:8000") + "/api/webhook/save-document"
        logger.info(f"🌐 Calling webhook: {webhook_url}")
        import requests  # Import here if not already available in task context
        try:
            response = requests.post(webhook_url, json=webhook_payload, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") == "ignored":
                    logger.warning(f"⚠️ Document ignored: {response_data.get('reason', '')} for {original_filename}")
                    result.document_id = f"ignored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    result.database_error = response_data.get('reason', 'Ignored due to missing info')
                else:
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
            "document_id": result.document_id,
            "physician_id": physician_id
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