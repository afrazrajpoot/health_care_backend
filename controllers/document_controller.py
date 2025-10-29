# controllers/document_controller.py (updated: check for existing document before GCS upload and handle like ignored/required field issue)

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query
from typing import Dict, List, Any
from datetime import datetime
import traceback

from services.file_service import FileService

from utils.logger import logger

from services.database_service import DatabaseService, get_database_service

from services.progress_service import progress_service
from utils.socket_manager import sio
from pathlib import Path
from fastapi import Path as FastAPIPath

from services.document_converter import DocumentConverter

from services.webhook_service import WebhookService
from services.document_route_services import DocumentExtractorService
from services.get_document_services import DocumentAggregationService
import os
from fastapi import Form
import json

router = APIRouter()



import os

from google.cloud import dlp_v2
from typing import Dict, Any
from datetime import datetime

PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-gcp-project-id')


@router.post("/webhook/save-document")
async def save_document_webhook(request: Request):
    """
    Main webhook handler: Uses WebhookService to process the request.
    """
    try:
        data = await request.json()
        logger.info(f"📥 Webhook received for document save: {data.get('document_id', 'unknown')}")

        db_service = await get_database_service()
        service = WebhookService()
        result = await service.handle_webhook(data, db_service)

        return result

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}", exc_info=True)

        # Save to FailDocs on general exception
        if 'data' in locals():
            blob_path = data.get("blob_path", "") ,
            physician_id = data.get("physician_id")
            reason = f"Webhook processing failed: {str(e)}"

            # Emit error event
            try:
                await sio.emit('task_error', {
                    'document_id': data.get('document_id', 'unknown'),
                    'filename': data.get('filename', 'unknown'),
                    'error': str(e),
                    'gcs_url': data.get('gcs_url', 'unknown'),
                    'physician_id': physician_id,
                    'blob_path': blob_path
                })
            except:
                pass  # Ignore emit failure

        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")
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
import hashlib

def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()







from fastapi import Depends, HTTPException, UploadFile, File, Form, Query
from typing import Dict, Any, List
from prisma import Prisma
import traceback

# Create Prisma client instance
db = Prisma()

async def check_subscription(
    documents: List[UploadFile] = File(...),
    physicianId: str = Query(None),
    userId: str = Query(None)
):
    """Dependency to check subscription before processing documents"""
    
    try:
        # Ensure database is connected
        if not db.is_connected():
            await db.connect()
            print("✅ Database connected in subscription check")

        # Count documents
        document_count = len(documents)
        print(f"📄 Documents to process: {document_count}")
        
        # Use physicianId for subscription check (as per your schema)
        subscription_id = physicianId
        if not subscription_id:
            raise HTTPException(status_code=400, detail="physicianId is required for subscription check")
        
        print(f"🔍 Checking subscription for physicianId: {subscription_id}")

        # Query DB for active subscription using physicianId
        sub = await db.subscription.find_first(
            where={
                "physicianId": subscription_id,
                "status": "active"
            }
        )
        
        print(f"📊 Database query completed. Found subscription: {sub is not None}")
        
        if not sub:
            print("❌ No active subscription found")
            raise HTTPException(
                status_code=400,
                detail="No active subscription found. Please upgrade your plan."
            )
        
        # Get documentParse from subscription (as per your schema)
        remaining_parses = sub.documentParse
        print(f"📊 Subscription ID: {sub.id}, Remaining parses: {remaining_parses}")
        print(f"📄 Requested documents: {document_count}")
        
        if document_count > remaining_parses:
            print(f"❌ Document count ({document_count}) exceeds remaining parses ({remaining_parses})")
            raise HTTPException(
                status_code=400,
                detail=f"Not enough remaining parses. You requested {document_count} documents, but only {remaining_parses} parses available. Please upgrade your plan."
            )
        
        if remaining_parses <= 0:
            print("❌ Document parse limit exceeded")
            raise HTTPException(
                status_code=400,
                detail="Document parse limit exceeded. Please upgrade your plan."
            )
        
        print("✅ Subscription check passed")
        return {
            "subscription": sub,
            "document_count": document_count,
            "physician_id": subscription_id,
            "remaining_parses": remaining_parses
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Subscription check error: {e}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during subscription check")


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
    """
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

        # Process batch
        batch_result = await service.process_documents_batch(documents, physicianId, userId, mode)
        payloads = batch_result["payloads"]
        ignored = batch_result["ignored"]

        successful_uploads = [p["blob_path"] for p in payloads if p.get("blob_path")]

        # ⚠️ Skip subscription decrement (since validation disabled)
        print("⚠️ Subscription decrement skipped (testing mode)")

        # Queue batch if payloads exist
        task_id = await service.queue_batch_and_track_progress(payloads, userId)

        print("✅ === END MULTI-DOCUMENT REQUEST (Subscription Disabled) ===")

        return {
            "task_id": task_id,
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


@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get current progress for a task"""
    try:
        progress = await progress_service.get_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task progress not found")
        return progress
    except Exception as e:
        error_msg = f"❌ Error getting progress for task {task_id}: {str(e)}"
        logger.error(error_msg)
        print(error_msg)  # DEBUG PRINT
        raise HTTPException(status_code=500, detail=f"Error retrieving progress: {str(e)}")

from typing import Optional

from datetime import datetime

def parse_date(date_str: Optional[str], field_name: str) -> datetime:
    """Parse a date string safely, supporting multiple formats."""
    if not date_str or date_str.strip() == "":
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty or missing")

    # Supported formats
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    # If no format worked
    raise HTTPException(
        status_code=400,
        detail=f"Invalid date format for {field_name}. Expected one of: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY"
    )



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
from fastapi.responses import Response
from services.document_converter import DocumentConverter
import urllib.parse
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


@router.get("/workflow-stats")
async def get_workflow_stats():
    db = DatabaseService()
    await db.prisma.connect()
    stats = await db.prisma.workFlowStats.find_first(order={"date": "desc"})
    await db.prisma.disconnect()
    return stats or {}