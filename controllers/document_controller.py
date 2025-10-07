# controllers/document_controller.py (updated: added socket emit in webhook success)
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
from utils.celery_task import process_document_task
from utils.socket_manager import sio  # ✅ Import sio for emitting events

router = APIRouter()

@router.post("/webhook/save-document")
async def save_document_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"📥 Webhook received for document save: {data.get('document_id', 'unknown')}")

        if not data.get("result") or not data.get("filename") or not data.get("gcs_url"):
            raise HTTPException(status_code=400, detail="Missing required fields in webhook payload")

        result_data = data["result"]
        analyzer = ReportAnalyzer()
        document_analysis = analyzer.extract_document_data(result_data.get("text", ""))
        
        # Check for required fields before proceeding
        required_fields = {
            "patient_name": document_analysis.patient_name,
            "dob": document_analysis.dob,
            "doi": document_analysis.doi
        }
        missing_fields = [k for k, v in required_fields.items() if not v]
        if missing_fields:
            warning_msg = f"Document ignored: this document is invalid for file {data['filename']}"
            logger.warning(f"⚠️ {warning_msg}")
            # Optional: Emit ignored event
            await sio.emit('task_complete', {
                'document_id': data.get('document_id', 'unknown'),
                'filename': data["filename"],
                'status': 'ignored',
                'reason': 'Invalid document',
                'gcs_url': data["gcs_url"],
                'physician_id': data.get("physician_id")
            })
            return {
                "status": "ignored",
                "reason": "Invalid document",
                "filename": data["filename"]
            }
        
        # Generate AI brief summary
        brief_summary = analyzer.generate_brief_summary(result_data.get("text", ""))
        
        try:
            dob = datetime.strptime(document_analysis.dob, "%Y-%m-%d")
        except:
            dob = datetime.now()
            
        try:
            doi = datetime.strptime(document_analysis.doi, "%Y-%m-%d")
        except:
            doi = datetime.now()
        
        # Mock database service - replace with your actual implementation
        db_service = await get_database_service()
        file_exists = await db_service.document_exists(
            data["filename"], 
            data.get("file_size", 0)
        )
        
        if file_exists:
            logger.warning(f"⚠️ Document already exists: {data['filename']}")
            # Optional: Emit skipped event
            await sio.emit('task_complete', {
                'document_id': data.get('document_id', 'unknown'),
                'filename': data["filename"],
                'status': 'skipped',
                'reason': 'Document already processed',
                # 'gcs_url': data["gcs_url"],
                # 'physician_id': data.get("physician_id")
                "user_id": data.get("user_id")  # Pass user_id if available
            })
            return {"status": "skipped", "reason": "Document already processed"}
        
        # Retrieve previous documents
        db_response = await db_service.get_all_unverified_documents(
            document_analysis.patient_name,
            physicianId=data.get("physician_id", None)
        )
        
        # Extract documents list from database response
        previous_documents = db_response.get('documents', []) if db_response else []
        print(previous_documents,'previous documents')
        # Compare with previous documents using LLM
        whats_new_data = analyzer.compare_with_previous_documents(
            document_analysis, 
            previous_documents
        )
        print(whats_new_data,'what new data')
        
        # Check if whats_new_data is valid; if not, ignore the document
        if whats_new_data is None or not isinstance(whats_new_data, dict):
            warning_msg = f"Document ignored: this document is invalid for file {data['filename']}"
            logger.warning(f"⚠️ {warning_msg}")
            # Optional: Emit invalid whats_new event
            await sio.emit('task_complete', {
                'document_id': data.get('document_id', 'unknown'),
                'filename': data["filename"],
                'status': 'ignored',
                'reason': 'Invalid whats_new data',
                # 'gcs_url': data["gcs_url"],
                # 'physician_id': data.get("physician_id"),
                "user_id": data.get("user_id")  # Pass user_id if available
            })
            return {
                "status": "ignored",
                "reason": "Invalid document",
                "filename": data["filename"]
            }
        
        summary_snapshot = {
            "dx": document_analysis.diagnosis,
            "keyConcern": document_analysis.key_concern,
            "nextStep": document_analysis.next_step
        }
        
        adl_data = {
            "adlsAffected": document_analysis.adls_affected,
            "workRestrictions": document_analysis.work_restrictions
        }
        
        summary_text = " | ".join(document_analysis.summary_points) if document_analysis.summary_points else "No summary"
        
        document_summary = {
            "type": document_analysis.document_type,
            "createdAt": datetime.now(),
            "summary": summary_text
        }
        
        # Mock ExtractionResult - replace with your actual implementation
        extraction_result = ExtractionResult(
            text=result_data.get("text", ""),
            pages=result_data.get("pages", 0),
            entities=result_data.get("entities", []),
            tables=result_data.get("tables", []),
            formFields=result_data.get("formFields", []),
            confidence=result_data.get("confidence", 0.0),
            success=result_data.get("success", False),
            gcs_file_link=result_data.get("gcs_file_link", data["gcs_url"]),
            fileInfo=result_data.get("fileInfo", {}),
            summary=summary_text,
            comprehensive_analysis=result_data.get("comprehensive_analysis"),
            document_id=result_data.get("document_id", f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
        
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=data["filename"],
            file_size=data.get("file_size", 0),
            mime_type=data.get("mime_type", "application/octet-stream"),
            processing_time_ms=data.get("processing_time_ms", 0),
            gcs_file_link=data["gcs_url"],
            patient_name=document_analysis.patient_name,
            claim_number=document_analysis.claim_number,
            dob=dob,
            doi=doi,
            status=document_analysis.status,
            brief_summary=brief_summary,  # Pass the AI-generated brief summary
            summary_snapshot=summary_snapshot,
            whats_new=whats_new_data,
            adl_data=adl_data,
            document_summary=document_summary,
            physician_id=data.get("physician_id")  # Pass physician_id if provided in webhook payload
        )

        logger.info(f"💾 Document saved via webhook with ID: {document_id}")
        
        # Emit socket event for task complete (now in main process)
        emit_data = {
            'document_id': document_id,
            'filename': data["filename"],
            'status': 'success',
            # 'gcs_url': data["gcs_url"],
            # 'physician_id': data.get("physician_id")
        }
        # physician_id = data.get("physician_id")
        user_id = data.get("user_id")
        if user_id:
            await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
        else:
            await sio.emit('task_complete', emit_data)  # Broadcast to all
        logger.info(f"📡 Emitted 'task_complete' event from webhook: {emit_data}")
        
        return {"status": "success", "document_id": document_id}

    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}", exc_info=True)
        # Optional: Emit error event on exception
        try:
            await sio.emit('task_error', {
                'document_id': data.get('document_id', 'unknown') if 'data' in locals() else 'unknown',
                'filename': data.get('filename', 'unknown') if 'data' in locals() else 'unknown',
                'error': str(e),
                'gcs_url': data.get('gcs_url', 'unknown') if 'data' in locals() else 'unknown',
                'physician_id': data.get('physician_id', None) if 'data' in locals() else None
            })
        except:
            pass  # Ignore emit failure during webhook error
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@router.post("/extract-documents", response_model=Dict[str, List[str]])
async def extract_documents(
    documents: List[UploadFile] = File(...),
    physicianId: str = Query(None, description="Optional physician ID for associating documents"),
    userId: str = Query(None, description="Optional user ID for associating documents")
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
        if physicianId:
            logger.info(f"👨‍⚕️ Physician ID provided: {physicianId}")
        
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
            
            # Enqueue Celery task with physician_id
            logger.debug(f"DEBUG: Queuing task for {document.filename}")
            task = process_document_task.delay(
                gcs_url=gcs_url,
                original_filename=document.filename,
                mime_type=document.content_type,
                file_size=len(content),
                blob_path=blob_path,
                physician_id=physicianId , # Pass the physicianId to the task
                user_id=userId  # Pass the userId to the task (if needed
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
from typing import Optional


@router.get('/document')
async def get_document(
    patient_name: str,
    dob: str,
    doi: str,
    physicianId: Optional[str] = None,
    claim_number: Optional[str] = None
):
    """
    Get aggregated document for a patient
    Returns a single aggregated document from all patient documents
    """
    try:
        logger.info(f"📄 Fetching aggregated document for patient: {patient_name}")
        
        # Parse date strings
        try:
            dob_date = datetime.strptime(dob, "%Y-%m-%d")
            doi_date = datetime.strptime(doi, "%Y-%m-%d")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
        
        db_service = await get_database_service()
        
        # Get all documents (aggregated structure)
        document_data = await db_service.get_document_by_patient_details(
            patient_name=patient_name,
            physicianId=physicianId,
            # dob=dob_date,
            # doi=doi_date,
            # claim_number=claim_number
        )
        
        if not document_data or document_data["total_documents"] == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No documents found for patient: {patient_name}"
            )
        
        # Format the aggregated response
        response = await format_aggregated_document_response(document_data)
        
        logger.info(f"✅ Returned aggregated document for: {patient_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def format_aggregated_document_response(all_documents_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the aggregated document response from all documents"""
    documents = all_documents_data["documents"]
    
    if not documents:
        return {
            "patient_name": all_documents_data["patient_name"],
            "total_documents": 0,
            "documents": [],
            "is_multiple_documents": False
        }
    
    # Use the latest document for base info
    latest_doc = documents[0]
    base_response = await format_single_document_base(latest_doc)
    
    # Collect all full summarySnapshot objects in an array (latest first)
    summary_snapshots = [doc.get("summarySnapshot") for doc in documents]
    
    # adl from latest document
    adl = await format_adl(latest_doc)
    
    # whats_new from latest document
    whats_new = latest_doc.get("whatsNew", {})
    
    # document_summary: group by type
    grouped_summaries = {}
    grouped_brief_summaries = {}
    for doc in documents:
        # Group brief_summary by type
        doc_summary = doc.get("documentSummary")
        doc_type = doc_summary.get("type", "unknown") if doc_summary else "unknown"
        
        brief_summary = doc.get("briefSummary")
        if brief_summary:
            if doc_type not in grouped_brief_summaries:
                grouped_brief_summaries[doc_type] = []
            grouped_brief_summaries[doc_type].append(brief_summary)
        
        # Group document_summary
        if doc_summary:
            if doc_type not in grouped_summaries:
                grouped_summaries[doc_type] = []
            summary_entry = {
                "date": doc_summary.get("date").isoformat() if doc_summary.get("date") else None,
                "summary": doc_summary.get("summary")
            }
            grouped_summaries[doc_type].append(summary_entry)
    
    base_response.update({
        "summary_snapshots": summary_snapshots,
        "whats_new": whats_new,
        "adl": adl,
        "document_summary": grouped_summaries,
        "brief_summary": grouped_brief_summaries,
        "document_index": 1,
        "is_latest": True
    })
    
    # Wrap in single document structure
    return {
        "patient_name": all_documents_data["patient_name"],
        "total_documents": all_documents_data["total_documents"],
        "documents": [base_response],
        "is_multiple_documents": len(documents) > 1
    }

async def format_single_document_base(document: Dict[str, Any]) -> Dict[str, Any]:
    """Format base info for a single document"""
    return {
        "document_id": document.get("id"),
        "patient_name": document.get("patientName"),
        "dob": document.get("dob").isoformat() if document.get("dob") else None,
        "doi": document.get("doi").isoformat() if document.get("doi") else None,
        "claim_number": document.get("claimNumber"),
        "status": document.get("status"),
        "gcs_file_link": document.get("gcsFileLink"),
        "created_at": document.get("createdAt").isoformat() if document.get("createdAt") else None,
        "updated_at": document.get("updatedAt").isoformat() if document.get("updatedAt") else None,
    }

async def format_adl(document: Dict[str, Any]) -> Dict[str, Any]:
    """Format ADL data"""
    adl_data = document.get("adl")
    if adl_data:
        return {
            "adls_affected": adl_data.get("adlsAffected"),
            "work_restrictions": adl_data.get("workRestrictions")
        }
    return None

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