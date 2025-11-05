# services/document_extractor_service.py (updated: pass mode to process_single_document and include in webhook_payload)
import traceback
from datetime import datetime
from typing import Dict, Any, List
from fastapi import UploadFile


from services.file_service import FileService
from services.database_service import get_database_service
from services.document_ai_service import get_document_ai_processor
from services.document_converter import DocumentConverter
from services.report_analyzer import ReportAnalyzer
from models.schemas import ExtractionResult
from utils.celery_task import process_batch_documents
from services.progress_service import progress_service
from utils.socket_manager import sio
from utils.logger import logger
from config.settings import CONFIG

class DocumentExtractorService:
    """
    Service for extracting and processing multiple documents.
    Handles validation, conversion, AI processing, analysis, GCS upload, and batch queuing.
    """

    def __init__(self):
        self.file_service = FileService()
        self.db_service = None  # Will be initialized asynchronously

    async def initialize_db(self):
        """Initialize database service."""
        if self.db_service is None:
            self.db_service = await get_database_service()

    async def process_single_document(
        self,
        document: UploadFile,
        physician_id: str = None,
        user_id: str = None,
        mode: str = None
    ) -> Dict[str, Any]:
        """
        Process a single document: validate, convert, analyze, upload to GCS, and prepare webhook payload.
        Returns payload if successful, or raises exception / returns None on failure.
        """
        await self.initialize_db()
        document_start_time = datetime.now()
        content = await document.read()
        file_size = len(content)
        blob_path = None
        temp_path = None
        was_converted = False
        converted_path = None
        print(document,'curretn document')
        try:
            # Validate file
            self.file_service.validate_file(document, CONFIG["max_file_size"])
            logger.info(f"📁 Starting processing for file: {document.filename}")
            logger.info(f"📏 File size: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
            logger.info(f"📋 MIME type: {document.content_type}")

            # Early check for existing document: only skip if both filename and physicianId match
            file_hash = self._compute_file_hash(content)
            # Find document with same filename and physicianId
            existing_doc = await self.db_service.prisma.document.find_first(
                where={
                    "physicianId": physician_id,
                    "originalName": document.filename,
                }
            )

           # In your process_single_document method, update the skipped document section:

            if existing_doc:
                logger.warning(f"⚠️ Document already exists for this physician: {document.filename}")
                # Emit skipped event with proper structure
                emit_data = {
                    'document_id': existing_doc.id or 'unknown',
                    'filename': document.filename,
                    'status': 'skipped',
                    'reason': 'Document already processed',
                    'user_id': user_id,
                    'blob_path': blob_path,
                    'physician_id': physician_id,
                    'message': f'Document "{document.filename}" was already processed and will be skipped'
                }
                
                # Use task_id format for consistency
                if user_id:
                    await sio.emit('document_skipped', emit_data, room=f"user_{user_id}")
                else:
                    await sio.emit('document_skipped', emit_data)
                
                return {
                    "success": False,
                    "reason": "Document already processed",
                    "filename": document.filename
                }

            # Save to temp
            temp_path = self.file_service.save_temp_file(content, document.filename)
            processing_path = temp_path

            # Conversion
            if DocumentConverter.needs_conversion(temp_path):
                logger.info(f"🔄 Converting file: {temp_path}")
                converted_path, was_converted = DocumentConverter.convert_document(temp_path, target_format="pdf")
                processing_path = converted_path
                logger.info(f"✅ Converted to: {processing_path}")
            else:
                logger.info(f"✅ Direct support for: {temp_path}")

            # Document AI Processing
            processor = get_document_ai_processor()
            document_result = processor.process_document(processing_path)
           
            # Build ExtractionResult
            result = ExtractionResult(
                text=document_result.text,
                pages=document_result.pages,
                entities=document_result.entities,
                tables=document_result.tables,
                formFields=document_result.formFields,
                confidence=document_result.confidence,
                success=document_result.success,
                gcs_file_link="",
                fileInfo={
                    "originalName": document.filename,
                    "size": file_size,
                    "mimeType": document.content_type or "application/octet-stream",
                    "gcsUrl": ""
                },
                summary="",
                comprehensive_analysis=None,
                document_id=f"endpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            processing_time = (datetime.now() - document_start_time).total_seconds() * 1000
            logger.info(f"⏱️ Document AI time for {document.filename}: {processing_time:.0f}ms")

            # Check if text was extracted
            if not result.text:
                reason = "No text extracted from document"
                logger.warning(f"⚠️ {reason}")
                return {
                    "success": False,
                    "reason": reason,
                    "filename": document.filename
                }

            # # Analysis
            # analyzer = ReportAnalyzer()
            # try:
            #     detected_type = analyzer.detect_document_type_preview(result.text)
            #     logger.info(f"🔍 Detected type: {detected_type}")
            # except AttributeError:
            #     logger.warning("⚠️ Document type detection unavailable—skipping")
            #     detected_type = "unknown"
            # result.summary = f"Document Type: {detected_type} - Processed successfully"

            logger.info(f"✅ Document analysis completed for {document.filename}")

            # Upload to GCS
            gcs_url, blob_path = self.file_service.save_to_gcs(content, document.filename)
            logger.info(f"✅ GCS upload: {gcs_url} | Blob: {blob_path}")

            # Update result
            result.gcs_file_link = gcs_url
            result.fileInfo["gcsUrl"] = gcs_url

            # Prepare webhook payload
            webhook_payload = {
                "result": result.dict(),
                "filename": document.filename,
                "file_size": file_size,
                "mime_type": document.content_type or "application/octet-stream",
                "processing_time_ms": int(processing_time),
                "gcs_url": gcs_url,
                "blob_path": blob_path,
                "file_hash": file_hash,
                "document_id": result.document_id,
                "physician_id": physician_id,
                "user_id": user_id,
                "mode": mode  # Include the mode in the webhook payload
            }

            return {
                "success": True,
                "payload": webhook_payload,
                "filename": document.filename
            }

        except Exception as proc_exc:
            logger.error(f"❌ Unexpected processing error for {document.filename}: {str(proc_exc)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "reason": f"Processing failed: {str(proc_exc)}",
                "filename": document.filename
            }
        finally:
            # Cleanup
            if temp_path:
                self.file_service.cleanup_temp_file(temp_path)
            if was_converted and converted_path:
                DocumentConverter.cleanup_converted_file(converted_path, was_converted)

    def _compute_file_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of file content."""
        import hashlib
        return hashlib.sha256(content).hexdigest()

    async def process_documents_batch(
        self,
        documents: List[UploadFile],
        physician_id: str = None,
        user_id: str = None,
        mode: str = None
    ) -> Dict[str, Any]:
        """
        Process a batch of documents and return payloads and ignored files.
        """
        await self.initialize_db()
        payloads = []
        ignored = []

        api_start_msg = f"\n🔄 === NEW MULTI-DOCUMENT PROCESSING REQUEST ({len(documents)} files) ===\n"
        logger.info(api_start_msg)
        if physician_id:
            logger.info(f"👨‍⚕️ Physician ID provided: {physician_id}")

        for document in documents:
            result = await self.process_single_document(document, physician_id, user_id, mode)  # Pass mode to single document processing
            if result["success"]:
                payloads.append(result["payload"])
            else:
                ignored.append({
                    "filename": result["filename"],
                    "reason": result["reason"]
                })

        preprocess_msg = f"✅ Preprocessing complete: {len(payloads)} ready for batch, {len(ignored)} ignored"
        logger.info(preprocess_msg)

        return {
            "payloads": payloads,
            "ignored": ignored
        }

    async def queue_batch_and_track_progress(
        self,
        payloads: List[Dict[str, Any]],
        user_id: str = None
    ) -> str:
        """
        Queue the batch for processing and initialize progress tracking.
        Handles both batches and single documents (as a batch of 1).
        Returns task_id.
        """
        if not payloads:
            return None

        # Enqueue batch task
        task = process_batch_documents.delay(payloads)
        task_id = task.id

        # Initialize progress tracking
        filenames = [p['filename'] for p in payloads]
        progress_service.initialize_task_progress(
            task_id=task_id,
            total_steps=len(payloads),
            filenames=filenames,
            user_id=user_id
        )

        queued_msg = f"🚀 Batch task queued for {len(payloads)} docs: {task_id}"
        logger.info(queued_msg)

        return task_id

    async def cleanup_on_error(self, successful_uploads: List[str]):
        """Cleanup GCS uploads on global error."""
        for path in successful_uploads:
            try:
                self.file_service.delete_from_gcs(path)
                logger.info(f"🗑️ Cleanup GCS (successful): {path}")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup failed: {path} - {str(e)}")