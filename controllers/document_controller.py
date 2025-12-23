from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
from services.file_service import FileService
from utils.logger import logger
from services.database_service import get_database_service
from utils.socket_manager import sio
from pathlib import Path
from fastapi import Path as FastAPIPath
from services.document_converter import DocumentConverter
from datetime import timezone
from services.webhook_service import WebhookService
from services.document_route_services import DocumentExtractorService
from services.get_document_services import DocumentAggregationService
from utils.document_splitter import get_document_splitter
from utils.page_extractor import get_page_extractor
from utils.document_detector import detect_document_type
from services.report_analyzer import ReportAnalyzer
from services.resoning_agent import EnhancedReportAnalyzer
from services.patient_lookup_service import EnhancedPatientLookup
from services.webhook_service import WebhookService
from models.schemas import ExtractionResult
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
from fastapi import HTTPException, UploadFile, File, Form, Query
from prisma import Prisma
import hashlib
from fastapi.responses import Response
from services.document_converter import DocumentConverter
import urllib.parse
from helpers.helpers import check_subscription

PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-gcp-project-id')

# Initialize router
router = APIRouter()
# Create Prisma client instance
db = Prisma()

def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

@router.post("/extract-documents", response_model=Dict[str, Any])
async def extract_documents(
    # subscription_data: dict = Depends(check_subscription),  # 🔒 Temporarily disabled
    documents: List[UploadFile] = File(...),
    mode: str = Form(..., description="Mode: wc for Workers' Comp or gm for General Medicine"),
    physicianId: str = Query(None, description="Physician ID for subscription check"),
    userId: str = Query(None, description="Optional user ID for associating documents"),
):
    """
    Upload multiple documents: parse/validate synchronously, then queue for finalization.
    Subscription check temporarily disabled for testing.
    🆕 NOW WITH INSTANT UPLOAD PROGRESS: Client can poll /api/agent/progress/{upload_task_id} immediately!
    """
    # 🆕 INSTANT PROGRESS: Generate upload task ID and initialize progress BEFORE reading files
    import uuid
    from services.progress_service import progress_service
    
    upload_task_id = f"upload_{uuid.uuid4().hex[:12]}"
    filenames = [doc.filename for doc in documents]
    
    # Initialize upload progress (client can start polling NOW)
    progress_service.initialize_task_progress(
        task_id=upload_task_id,
        total_steps=len(documents),
        filenames=filenames,
        user_id=userId
    )
    # 🆕 Start at 10% (upload phase: 10-30%)
    progress_service.update_status(upload_task_id, "uploading", f"Receiving {len(documents)} file(s)...", progress=10)
    
    try:
        # 🧩 Temporary mock data (bypasses subscription)
        subscription_data = {
            "subscription": {"id": "temp-subscription-id"},
            "document_count": len(documents),
            "physician_id": physicianId or "temp-physician-id",
            "remaining_parses": 9999  # unlimited for now
        }

        # Extract mock subscription info
        subscription = subscription_data["subscription"]
        document_count = subscription_data["document_count"]
        physician_id = subscription_data["physician_id"]
        remaining_parses = subscription_data["remaining_parses"]

        print(f"🎯 Starting document processing for physician: {physician_id}")
        print(f"📊 Initial - Documents: {document_count}, Remaining parses: {remaining_parses}")

        service = DocumentExtractorService()
        successful_uploads = []

        # 🆕 Update progress: Files received, starting validation (15%)
        progress_service.update_status(upload_task_id, "validating", "Files received, validating...", progress=15)
        
        # Process batch (now with progress callback)
        batch_result = await service.process_documents_batch_with_progress(
            documents, physicianId, userId, mode, upload_task_id, progress_service
        )
        payloads = batch_result["payloads"]
        ignored = batch_result["ignored"]

        successful_uploads = [p["blob_path"] for p in payloads if p.get("blob_path")]

        # ⚠️ Skip subscription decrement (since validation disabled)
        print("⚠️ Subscription decrement skipped (testing mode)")

        # Queue batch if payloads exist
        task_id = await service.queue_batch_and_track_progress(payloads, userId)
        
        # 🆕 Mark upload phase complete at 30% (not "completed" status, just upload done)
        progress_service.update_status(upload_task_id, "upload_complete", f"Upload complete. Starting AI processing for {len(payloads)} documents...", progress=30, completed=True)

        print("✅ === END MULTI-DOCUMENT REQUEST (Subscription Disabled) ===")

        return {
            "upload_task_id": upload_task_id,  # 🆕 For polling upload progress
            "task_id": task_id,  # Processing task ID (existing)
            "payload_count": len(payloads),
            "ignored": ignored,
            "ignored_count": len(ignored),
            "remaining_parses": remaining_parses
        }

    except Exception as global_exc:
        global_error = f"❌ Global error in extract-documents: {str(global_exc)}"
        logger.error(global_error)
        logger.debug(f"Traceback: {traceback.format_exc()}")

        if 'service' in locals():
            await service.cleanup_on_error(successful_uploads)

        raise HTTPException(status_code=500, detail=f"Global processing failed: {str(global_exc)}")

@router.post("/update-fail-document")
async def update_fail_document(request: Request):
    try:
        data = await request.json()
        logger.info(f"📥 Update fail document request for ID: {data.get('fail_doc_id', 'unknown')}")

        fail_doc_id = data.get("fail_doc_id")
        if not fail_doc_id:
            raise HTTPException(status_code=400, detail="Missing required fail_doc_id in request payload")

        updated_fields = {
            "document_text": data.get("document_text"),
            "dob": data.get("dob"),
            "doi": data.get("doi"),
            "claim_number": data.get("claim_number"),
            "patient_name": data.get("patient_name")
        }

        db_service = await get_database_service()
        fail_doc = await db_service.get_fail_doc_by_id(fail_doc_id)

        if not fail_doc:
            raise HTTPException(status_code=404, detail="Fail document not found")

        service = WebhookService()
        result = await service.update_fail_document(fail_doc, updated_fields, data.get("user_id"), db_service)

        # Defensive logging: avoid KeyError if result doesn't include expected keys
        if not isinstance(result, dict):
            logger.warning(f"⚠️ update_fail_document returned non-dict result: {result}")
            return result

        doc_id = result.get("document_id") or result.get("id") or result.get("fail_doc_id")
        status = result.get("status") or result.get("result_status") or "unknown"

        if doc_id is None:
            logger.warning(f"⚠️ update_fail_document result missing document id. Full result: {json.dumps(result, default=str)[:1000]}")
        else:
            logger.info(f"💾 Fail document updated and saved via route with ID: {doc_id}, status: {status}")

        return result

    except Exception as e:
        logger.error(f"❌ Update fail document failed: {str(e)}", exc_info=True)

        # On exception, emit error
        if 'data' in locals():
            try:
                await sio.emit('task_error', {
                    'document_id': data.get('fail_doc_id', 'unknown'),
                    'filename': fail_doc.fileName if 'fail_doc' in locals() else 'unknown',
                    'error': str(e),
                    'gcs_url': fail_doc.gcsFileLink if 'fail_doc' in locals() else 'unknown',
                    'physician_id': fail_doc.physicianId if 'fail_doc' in locals() else None,
                    'blob_path': fail_doc.blobPath if 'fail_doc' in locals() else ''
                })
            except:
                pass  # Ignore emit failure during error
        raise HTTPException(status_code=500, detail=f"Update fail document processing failed: {str(e)}")

@router.get('/document')
async def get_document(
    patient_name: str,
    dob: str,
    physicianId: Optional[str] = None,
    claim_number: Optional[str] = None
):
    """
    Get aggregated document for a patient
    Returns a single aggregated document from all patient documents
    """
    try:
        service = DocumentAggregationService()
        response = await service.get_aggregated_document(
            patient_name=patient_name,
            dob=dob,
            physician_id=physicianId,
            claim_number=claim_number
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/proxy-decrypt")
async def proxy_decrypt(request: Request):
    """
    Proxy endpoint to decrypt patient token and return data.
    This can be called from Next.js to avoid CORS issues.
    """
    db_service = await get_database_service()
    try:
        # Parse JSON body from request
        body = await request.json()
        token = body.get("token")
        if not token:
            raise HTTPException(status_code=400, detail="Missing 'token' in request body")
        
        patient_data = db_service.decrypt_patient_token(token)
        return {"success": True, "data": patient_data}
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid token")
    except Exception as e:
        logger.error(f"❌ Decryption error: {str(e)}")
        raise HTTPException(status_code=500, detail="Decryption failed")   
    
@router.get("/fail-docs", response_model=List[Dict[str, Any]])
async def get_fail_documents(
    physicianId: str = Query(..., description="Physician ID to filter failed documents")
):
    """
    Retrieve failed documents for a specific physician.
    """
    try:
        db_service = await get_database_service()
        fail_docs = await db_service.get_fail_docs_by_physician(physicianId)
        return fail_docs
    except Exception as e:
        logger.error(f"❌ Error retrieving fail docs for physician {physicianId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve fail docs: {str(e)}")

 # ✅ pathlib.Path here too
@router.get("/preview/{blob_path:path}")
async def preview_file(blob_path: str = FastAPIPath(..., description="GCS blob path, e.g., uploads/filename.ext")):  # ✅ Use aliased FastAPIPath
    """
    Preview any file from GCS inline in the browser.
    Converts non-renderable files (e.g., DOCX, TXT) to PDF for universal preview.
    """
    if not blob_path.startswith('uploads/'):  # Basic security: only allow uploads folder
        raise HTTPException(status_code=403, detail="Invalid path")
    
    file_service = FileService()
    converter = DocumentConverter()
    
    try:
        content = file_service.download_from_gcs(blob_path)
        mime_type = file_service.get_mime_type(blob_path) if hasattr(file_service, 'get_mime_type') else 'application/octet-stream'
        extension = Path(blob_path).suffix.lower()  # ✅ This uses pathlib.Path correctly
        
        # Supported for direct inline preview
        directly_previewable = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.txt': 'text/plain',  # Text renders in browser
            # Add more: '.bmp': 'image/bmp', etc.
        }
        
        preview_mime = directly_previewable.get(extension)
        if preview_mime:
            # Serve directly with inline disposition
            headers = {"Content-Disposition": f"inline; filename*=UTF-8''{urllib.parse.quote(Path(blob_path).name)}"}  # ✅ pathlib.Path here too
            return Response(content, media_type=preview_mime, headers=headers)
        
        # Convert to PDF for everything else (e.g., DOCX, XLSX)
        logger.info(f"🔄 Converting {blob_path} to PDF for preview...")
        pdf_content = converter.convert_to_pdf(content, blob_path)  # Assumes DocumentConverter has this method; adjust if named differently
        
        headers = {"Content-Disposition": "inline; filename*=UTF-8''preview.pdf"}
        return Response(pdf_content, media_type='application/pdf', headers=headers)
    
    except Exception as e:
        logger.error(f"❌ Preview error for {blob_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Preview failed")

@router.delete("/fail-docs/{doc_id}")
async def delete_fail_document(
    doc_id: str = FastAPIPath(..., description="ID of the failed document to delete"),
    physicianId: str = Query(..., description="Physician ID to authorize deletion")
):
    """
    Delete a failed document: remove from GCS using blob_path and from DB, scoped to physician.
    """
    try:
        db_service = await get_database_service()
        file_service = FileService()
        
        # Fetch the doc to get blob_path and verify
        fail_doc = await db_service.prisma.faildocs.find_unique(
            where={"id": doc_id}
        )
        
        if not fail_doc:
            raise HTTPException(status_code=404, detail="Document not found")
            
        if fail_doc.physicianId != physicianId:
            raise HTTPException(status_code=403, detail="Unauthorized: Access denied")
        
        # Log the blobPath for debugging
        logger.info(f"Debug: blobPath type={type(fail_doc.blobPath)}, value={repr(fail_doc.blobPath)}")
        
        # Safely extract blob_path from GCS
        # Handle None, Ellipsis, or any non-string values
        blob_path = None
        if fail_doc.blobPath and isinstance(fail_doc.blobPath, str) and fail_doc.blobPath != "...":
            blob_path = fail_doc.blobPath
        
        # Delete from GCS (Google Cloud Storage)
        if blob_path:
            try:
                success = file_service.delete_from_gcs(blob_path)
                if success:
                    logger.info(f"✅ Successfully deleted from GCS: {blob_path}")
                else:
                    logger.warning(f"⚠️ GCS delete returned False for {blob_path}, but proceeding with DB delete")
            except Exception as gcs_error:
                logger.error(f"❌ GCS deletion error for {blob_path}: {str(gcs_error)}")
                # Continue with DB deletion even if GCS deletion fails
        else:
            logger.info(f"ℹ️ No valid blob_path found for doc {doc_id}, skipping GCS deletion")
        
        # Delete from Database
        await db_service.delete_fail_doc_by_id(doc_id, physicianId)
        logger.info(f"✅ Successfully deleted from database: {doc_id}")
        
        return {
            "status": "success",
            "message": f"Failed document {doc_id} deleted successfully",
            "doc_id": doc_id,
            "gcs_deleted": bool(blob_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in delete_fail_document for {doc_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.post("/split-and-process-document")
async def split_and_process_document(request: Request):
    """
    Split a multi-report document by page ranges, process each page group, and save as documents.
    Staff provides page ranges (e.g., pages 1-5 for QME, pages 6-10 for PR2) from client side.
    This is optimized - only processes the specified pages, not all text.
    """
    try:
        data = await request.json()
        logger.info("📥 Split and process document request received (page-based)")
        
        mode = data.get("mode", "wc")
        physician_id = data.get("physician_id")
        original_filename = data.get("original_filename", "split_document")
        fail_doc_id = data.get("fail_doc_id")  # Optional: ID of the failed document being split
        blob_path = data.get("blob_path")  # GCS blob path to the original PDF
        page_ranges = data.get("page_ranges")  # List of page ranges: [{"start_page": 1, "end_page": 5, "report_title": "QME"}, ...]
        
        if not physician_id:
            raise HTTPException(status_code=400, detail="Missing physician_id in request")
        
        if not blob_path:
            raise HTTPException(status_code=400, detail="Missing blob_path - need original PDF file to extract pages")
        
        if not page_ranges or not isinstance(page_ranges, list) or len(page_ranges) == 0:
            raise HTTPException(status_code=400, detail="Missing page_ranges - staff must specify which pages belong to each report")
        
        logger.info(f"📄 Processing page-based split request (mode: {mode}, {len(page_ranges)} page groups)")
        logger.info(f"   Page ranges: {page_ranges}")
        
        # Initialize services
        db_service = await get_database_service()
        webhook_service = WebhookService()
        report_analyzer = ReportAnalyzer(mode)
        enhanced_analyzer = EnhancedReportAnalyzer()
        patient_lookup = EnhancedPatientLookup()
        page_extractor = get_page_extractor()
        
        # Use dedicated thread pool for LLM operations
        from concurrent.futures import ThreadPoolExecutor
        llm_executor = ThreadPoolExecutor(max_workers=10)
        loop = asyncio.get_event_loop()
        
        # Step 1: Extract text from specified page ranges (OPTIMIZED - only process specified pages)
        logger.info(f"📄 Extracting text from {len(page_ranges)} page groups...")
        extracted_groups = await loop.run_in_executor(
            llm_executor,
            page_extractor.extract_pages_from_gcs,
            blob_path,
            page_ranges
        )
        
        total_reports = len(extracted_groups)
        logger.info(f"✅ Extracted text from {total_reports} page groups")
        
        # Step 2: Process and save each extracted page group (OPTIMIZED - only process specified pages)
        processed_reports = []
        saved_documents = []
        
        for idx, page_group in enumerate(extracted_groups):
            try:
                report_text = page_group.get("text", "")
                raw_text = page_group.get("raw_text", report_text)
                report_title = page_group.get("report_title", f"Report_{idx + 1}")
                start_page = page_group.get("start_page", 1)
                end_page = page_group.get("end_page", 1)
                
                if not report_text or len(report_text.strip()) < 50:
                    error_msg = page_group.get("error", "Text too short")
                    logger.warning(f"⚠️ Page group {idx + 1} (pages {start_page}-{end_page}) too short or error: {error_msg}")
                    processed_reports.append({
                        "report_index": idx + 1,
                        "report_title": report_title,
                        "start_page": start_page,
                        "end_page": end_page,
                        "error": error_msg,
                        "status": "failed"
                    })
                    continue
                
                logger.info(f"🔍 Processing and saving report {idx + 1}/{total_reports}: {report_title} (pages {start_page}-{end_page})")
                
                # Step 2a: Detect document type
                doc_type_result = detect_document_type(report_text)
                doc_type = doc_type_result.get("doc_type", "Unknown")
                doc_confidence = doc_type_result.get("confidence", 0.0)
                
                logger.info(f"   Document type detected: {doc_type} (confidence: {doc_confidence})")
                
                # Step 2b: Extract using ReportAnalyzer (routes to appropriate extractor)
                # Use raw_text (Document AI summary) if available, otherwise use full text
                report_result = await loop.run_in_executor(
                    llm_executor, report_analyzer.extract_document, report_text, raw_text
                )
                long_summary = report_result.get("long_summary", "")
                short_summary = report_result.get("short_summary", "")
                
                # Step 2c: Use EnhancedReportAnalyzer for full extraction
                analysis_task = loop.run_in_executor(
                    llm_executor,
                    lambda: enhanced_analyzer.extract_document_data_with_reasoning(
                        long_summary, None, None, mode
                    )
                )
                
                summary_task = loop.run_in_executor(
                    llm_executor,
                    lambda: enhanced_analyzer.generate_brief_summary(raw_text if raw_text else report_text, mode)
                )
                
                document_analysis, brief_summary = await asyncio.gather(
                    analysis_task, summary_task
                )
                
                # Step 2d: Prepare processed_data similar to webhook processing
                processed_data = {
                    "document_analysis": document_analysis,
                    "brief_summary": brief_summary,
                    "text_for_analysis": report_text,
                    "raw_text": report_text,
                    "report_analyzer_result": report_result,
                    "patient_name": (
                        document_analysis.patient_name
                        if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"
                        else None
                    ),
                    "claim_number": (
                        document_analysis.claim_number
                        if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
                        else None
                    ),
                    "dob": (
                        document_analysis.dob
                        if hasattr(document_analysis, 'dob') and document_analysis.dob and str(document_analysis.dob).lower() != "not specified"
                        else None
                    ),
                    "physician_id": physician_id,
                    "filename": f"{original_filename}_pages{start_page}-{end_page}_{report_title}",
                    "gcs_url": "",  # No file upload for split documents
                    "blob_path": None,
                    "file_size": len(report_text),
                    "mime_type": "text/plain",
                    "processing_time_ms": 0,
                    "file_hash": None,
                    "mode": mode,
                    "result_data": {
                        "text": report_text,
                        "raw_text": raw_text if raw_text else report_text,
                        "pages": end_page - start_page + 1,
                        "entities": [],
                        "tables": [],
                        "formFields": [],
                        "confidence": 1.0,
                        "success": True,
                        "gcs_file_link": "",
                        "fileInfo": {
                            "originalName": f"{original_filename}_pages{start_page}-{end_page}_{report_title}",
                            "size": len(report_text),
                            "mimeType": "text/plain",
                            "gcsUrl": ""
                        },
                        "document_id": f"split_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx + 1}"
                    }
                }
                
                # Step 2e: Perform patient lookup
                lookup_result = await patient_lookup.perform_patient_lookup(db_service, processed_data)
                
                # Step 2f: Save document
                save_result = await webhook_service.save_document(db_service, processed_data, lookup_result)
                
                document_id = save_result.get("document_id")
                status = save_result.get("status", "unknown")
                
                if document_id:
                    logger.info(f"   ✅ Report {idx + 1} saved as document: {document_id}")
                    saved_documents.append(document_id)
                else:
                    logger.warning(f"   ⚠️ Report {idx + 1} processing completed but not saved")
                
                processed_reports.append({
                    "report_index": idx + 1,
                    "report_title": report_title,
                    "document_type": doc_type,
                    "document_type_confidence": doc_confidence,
                    "long_summary": long_summary,
                    "short_summary": short_summary,
                    "document_id": document_id,
                    "status": status,
                    "patient_name": processed_data.get("patient_name"),
                    "claim_number": processed_data.get("claim_number"),
                    "text_length": len(report_text),
                    "start_page": start_page,
                    "end_page": end_page,
                    "page_count": end_page - start_page + 1
                })
                
                logger.info(f"   ✅ Report {idx + 1} processed and saved successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing report {idx + 1}: {str(e)}", exc_info=True)
                processed_reports.append({
                    "report_index": idx + 1,
                    "report_title": page_group.get("report_title", "Unknown"),
                    "start_page": page_group.get("start_page", 1),
                    "end_page": page_group.get("end_page", 1),
                    "error": str(e),
                    "status": "failed"
                })
        
        # Step 3: Optionally delete the failed document if all reports were saved successfully
        if fail_doc_id and len(saved_documents) == total_reports and all(r.get("document_id") for r in processed_reports):
            try:
                logger.info(f"🗑️ All reports saved successfully, deleting failed document: {fail_doc_id}")
                await db_service.delete_fail_doc_by_id(fail_doc_id, physician_id)
                logger.info(f"✅ Failed document {fail_doc_id} deleted")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete failed document: {str(e)}")
        
        result = {
            "status": "success",
            "total_reports": total_reports,
            "processed_reports": len(processed_reports),
            "saved_documents": len(saved_documents),
            "document_ids": saved_documents,
            "reports": processed_reports
        }
        
        logger.info(f"✅ Page-based split and process completed: {len(processed_reports)} reports processed, {len(saved_documents)} documents saved")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Split and process document failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Split and process failed: {str(e)}")

@router.get("/workflow-stats")
async def get_workflow_stats(
    date: Optional[str] = Query(None, description="Optional date filter in YYYY-MM-DD format")
):
    """
    Fetch workflow statistics for a specific date or today by default.
    Returns stats with labels and values for dashboard visualization.
    Matches Next.js /api/workflow-stats functionality.
    """
    try:
        db_service = await get_database_service()
        
        # Ensure database is connected
        await db_service.connect()
        
        # Determine date range for query
        if date:
            # Parse the provided date
            try:
                filter_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Expected YYYY-MM-DD"
                )
            
            # Set to start of day
            filter_date = filter_date.replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            next_day = filter_date + timedelta(days=1)
        else:
            # Default: get today's stats
            today = datetime.now()
            filter_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            next_day = filter_date + timedelta(days=1)
        
        # Fetch WorkflowStats from database
        workflow_stats = await db_service.prisma.workflowstats.find_first(
            where={
                "date": {
                    "gte": filter_date,
                    "lt": next_day
                }
            },
            order={"createdAt": "desc"}
        )
        
        # If no stats found, return zeros with default structure
        if not workflow_stats:
            stats = {
                "id": None,
                "date": filter_date.isoformat(),
                "referralsProcessed": 0,
                "rfasMonitored": 0,
                "qmeUpcoming": 0,
                "payerDisputes": 0,
                "externalDocs": 0,
                "intakes_created": 0,
                "createdAt": filter_date.isoformat(),
                "updatedAt": filter_date.isoformat()
            }
            has_data = False
        else:
            stats = {
                "id": workflow_stats.id,
                "date": workflow_stats.date.isoformat() if workflow_stats.date else filter_date.isoformat(),
                "referralsProcessed": workflow_stats.referralsProcessed or 0,
                "rfasMonitored": workflow_stats.rfasMonitored or 0,
                "qmeUpcoming": workflow_stats.qmeUpcoming or 0,
                "payerDisputes": workflow_stats.payerDisputes or 0,
                "externalDocs": workflow_stats.externalDocs or 0,
                "intakes_created": workflow_stats.intakes_created or 0,
                "createdAt": workflow_stats.createdAt.isoformat() if workflow_stats.createdAt else filter_date.isoformat(),
                "updatedAt": workflow_stats.updatedAt.isoformat() if workflow_stats.updatedAt else filter_date.isoformat()
            }
            has_data = True
        
        # Format response to match Next.js structure
        response = {
            "success": True,
            "data": {
                "stats": stats,
                "labels": [
                    "Referrals Processed",
                    "RFAs Monitored",
                    "QME Upcoming",
                    "Payer Disputes",
                    "External Docs",
                    "Intakes Created"
                ],
                "vals": [
                    stats["referralsProcessed"],
                    stats["rfasMonitored"],
                    stats["qmeUpcoming"],
                    stats["payerDisputes"],
                    stats["externalDocs"],
                    stats["intakes_created"]
                ],
                "date": stats["date"],
                "hasData": has_data
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching workflow stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/office-pulse")
async def get_tasks_pulse(
    physicianId: str = Query(..., description="Physician ID to filter tasks")
):
    """
    Fetch all tasks with department statistics (pulse) for a specific physician.
    Returns tasks list and aggregated department stats (open, overdue, unclaimed).
    Matches Next.js /api/tasks functionality.
    """
    try:
        db_service = await get_database_service()
        
        # Ensure database is connected
        await db_service.connect()
        
        # Use timezone-aware datetime for comparison
        now = datetime.now(timezone.utc)
        
        # Fetch all tasks with document relation filtered by physicianId
        tasks = await db_service.prisma.task.find_many(
            where={"physicianId": physicianId},
            include={"document": True},
            order={"createdAt": "desc"}
        )
        
        # Aggregate statistics by department
        dept_stats = {}
        
        for task in tasks:
            dept = task.department or "Unknown"
            if dept not in dept_stats:
                dept_stats[dept] = {
                    "open": 0,
                    "overdue": 0,
                    "unclaim": 0
                }
            
            # Count open tasks (status != "Done")
            is_open = task.status != "Done"
            if is_open:
                dept_stats[dept]["open"] += 1
                # Count overdue tasks - handle both timezone-aware and naive datetimes
                if task.dueDate:
                    due_date = task.dueDate
                    # Make due_date timezone-aware if it's naive
                    if due_date.tzinfo is None:
                        due_date = due_date.replace(tzinfo=timezone.utc)
                    if due_date < now:
                        dept_stats[dept]["overdue"] += 1
            
            # Count unclaimed tasks (actions doesn't include "Claimed")
            actions = task.actions or []
            if "Claimed" not in actions:
                dept_stats[dept]["unclaim"] += 1
        
        # Calculate totals
        total_open = sum(stats["open"] for stats in dept_stats.values())
        total_overdue = sum(stats["overdue"] for stats in dept_stats.values())
        total_unclaim = sum(stats["unclaim"] for stats in dept_stats.values())
        
        # Format pulse data
        pulse = {
            "depts": [
                {
                    "department": department,
                    "open": stats["open"],
                    "overdue": stats["overdue"],
                    "unclaimed": stats["unclaim"]
                }
                for department, stats in dept_stats.items()
            ],
            "labels": ["Total Open", "Total Overdue", "Total Unclaimed"],
            "vals": [total_open, total_overdue, total_unclaim]
        }
        
        # Convert tasks to dict format
        tasks_data = [
            {
                "id": task.id,
                "description": task.description,
                "department": task.department,
                "status": task.status,
                "dueDate": task.dueDate.isoformat() if task.dueDate else None,
                "patient": task.patient,
                "actions": task.actions or [],
                "sourceDocument": task.sourceDocument,
                "documentId": task.documentId,
                "physicianId": task.physicianId,
                "createdAt": task.createdAt.isoformat() if task.createdAt else None,
                "updatedAt": task.updatedAt.isoformat() if task.updatedAt else None,
                "document": {
                    "id": task.document.id,
                    "patientName": task.document.patientName,
                    "claimNumber": task.document.claimNumber,
                    "status": task.document.status
                } if task.document else None
            }
            for task in tasks
        ]
        
        return {
            "tasks": tasks_data,
            "pulse": pulse
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching tasks pulse: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")