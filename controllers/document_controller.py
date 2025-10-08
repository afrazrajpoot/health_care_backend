# controllers/document_controller.py (updated: refined invalid message for clarity)
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
        
        # Enhanced check for required fields: treat "Not specified" as missing
        required_fields = {
            "patient_name": document_analysis.patient_name,
            "dob": document_analysis.dob,
            # "doi": document_analysis.doi
        }
        missing_fields = [k for k, v in required_fields.items() if not v or str(v).lower() == "not specified"]
        if missing_fields:
            missing_details = ", ".join(missing_fields)
            user_friendly_msg = f"Invalid document: This file lacks proper patient details ({missing_details}). Please upload a complete medical report with patient name, DOB, and DOI."
            warning_msg = f"Document ignored: {user_friendly_msg} for file {data['filename']}"
            logger.warning(f"⚠️ {warning_msg}")
            # Emit ignored event with user-friendly reason
            await sio.emit('task_complete', {
                'document_id': data.get('document_id', 'unknown'),
                'filename': data["filename"],
                'status': 'ignored',
                'reason': user_friendly_msg,
                'gcs_url': data["gcs_url"],
                'physician_id': data.get("physician_id")
            })
            return {
                "status": "ignored",
                "reason": user_friendly_msg,
                "filename": data["filename"]
            }
        
        # Generate AI brief summary
        brief_summary = analyzer.generate_brief_summary(result_data.get("text", ""))
        
        try:
            dob = datetime.strptime(document_analysis.dob, "%Y-%m-%d")
            rd=datetime.strptime(document_analysis.rd, "%Y-%m-%d")
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
        # print(previous_documents,'previous documents')
        # Compare with previous documents using LLM
        whats_new_data = analyzer.compare_with_previous_documents(
            document_analysis, 
            previous_documents
        )
        # print(whats_new_data,'what new data')
        
        # Ensure whats_new_data is always a dict (empty if invalid) to avoid DB required field error
        if whats_new_data is None or not isinstance(whats_new_data, dict):
            logger.warning(f"⚠️ Invalid whats_new data for {data['filename']}; using empty dict")
            whats_new_data = {}
        
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
            blob_path=data.get("blob_path", ""),
            gcs_file_link=data["gcs_url"],
            patient_name=document_analysis.patient_name,
            claim_number=document_analysis.claim_number,
            dob=dob,
            doi=doi,
            status=document_analysis.status,
            brief_summary=brief_summary,  # Pass the AI-generated brief summary
            summary_snapshot=summary_snapshot,
            whats_new=whats_new_data,  # Now always a dict
            adl_data=adl_data,
            document_summary=document_summary,
            rd=rd,
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

@router.post("/extract-documents", response_model=Dict[str, Any])
async def extract_documents(
    documents: List[UploadFile] = File(...),
    physicianId: str = Query(None, description="Optional physician ID for associating documents"),
    userId: str = Query(None, description="Optional user ID for associating documents")
):
    """
    Upload multiple documents: parse/validate synchronously, upload to GCS if valid, then queue for finalization.
    """
    if not documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    file_service = FileService()
    task_ids = []
    ignored_filenames = []
    uploaded_files = []  # Track for cleanup on global errors
    
    try:
        logger.info(f"\n🔄 === NEW MULTI-DOCUMENT PROCESSING REQUEST ({len(documents)} files) ===\n")
        if physicianId:
            logger.info(f"👨‍⚕️ Physician ID provided: {physicianId}")
        
        for document in documents:
            document_start_time = datetime.now()
            content = await document.read()
            
            try:
                file_service.validate_file(document, CONFIG["max_file_size"])
                logger.info(f"📁 Starting processing for file: {document.filename}")
                logger.info(f"📏 File size: {len(content)} bytes ({len(content)/(1024*1024):.2f} MB)")
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
                    logger.error(f"❌ Conversion failed for {document.filename}: {str(convert_exc)}")
                    raise ValueError(f"File conversion failed: {str(convert_exc)}")
                
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
                    gcs_file_link="",  # Set after upload
                    fileInfo={
                        "originalName": document.filename,
                        "size": len(content),
                        "mimeType": document.content_type or "application/octet-stream",
                        "gcsUrl": ""
                    },
                    summary="",  # Set after validation
                    comprehensive_analysis=None,
                    document_id=f"endpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                processing_time = (datetime.now() - document_start_time).total_seconds() * 1000
                logger.info(f"⏱️ Document AI time for {document.filename}: {processing_time:.0f}ms")

                # Validation
                if not result.text:
                    raise ValueError("No text extracted from document")

                logger.info(f"📝 Validating patient details for {document.filename}...")
                analyzer = ReportAnalyzer()
                
                # Skip or stub document type detection to avoid AttributeError
                try:
                    detected_type = analyzer.detect_document_type_preview(result.text)
                    logger.info(f"🔍 Detected type: {detected_type}")
                except AttributeError:
                    logger.warning("⚠️ Document type detection unavailable—skipping")
                    detected_type = "unknown"
                result.summary = f"Document Type: {detected_type} - Validated successfully"
                
                document_analysis = analyzer.extract_document_data(result.text)
                # Enhanced check for required fields: treat "Not specified" as missing
                required_fields = {
                    "patient_name": document_analysis.patient_name,
                    "dob": document_analysis.dob,
                    # "doi": document_analysis.doi
                }
                missing_fields = [k for k, v in required_fields.items() if not v or str(v).lower() == "not specified"]
                if missing_fields:
                    missing_details = ", ".join(missing_fields)
                    user_friendly_msg = f"Invalid document: This file lacks proper patient details ({missing_details}). Please upload a complete medical report with patient name, DOB, and DOI."
                    error_msg = f"Invalid document: {user_friendly_msg} for {document.filename}"
                    logger.warning(f"⚠️ {error_msg}")
                    ignored_filenames.append({
                        "filename": document.filename,
                        "reason": user_friendly_msg
                    })
                    # Emit ignored
                    emit_data = {
                        'document_id': 'ignored_validation',
                        'filename': document.filename,
                        'status': 'ignored',
                        'reason': user_friendly_msg,
                        'user_id': userId
                    }
                    if userId:
                        await sio.emit('task_complete', emit_data, room=f"user_{userId}")
                    else:
                        await sio.emit('task_complete', emit_data)
                    continue  # Skip to next file

                logger.info(f"✅ Validation passed for {document.filename}")
                
                # Upload to GCS (only if valid)
                try:
                    logger.info("☁️ Uploading to GCS...")
                    gcs_url, blob_path = file_service.save_to_gcs(content, document.filename)
                    logger.info(f"✅ GCS upload: {gcs_url} | Blob: {blob_path}")
                    uploaded_files.append(blob_path)
                    
                    # Update result
                    result.gcs_file_link = gcs_url
                    result.fileInfo["gcsUrl"] = gcs_url
                except Exception as gcs_exc:
                    logger.error(f"❌ GCS upload failed for {document.filename}: {str(gcs_exc)}")
                    raise ValueError(f"GCS upload failed: {str(gcs_exc)}")
                
                # Prepare and queue task
                webhook_payload = {
                    "result": result.dict(),
                    "filename": document.filename,
                    "file_size": len(content),
                    "mime_type": document.content_type or "application/octet-stream",
                    "processing_time_ms": int(processing_time),
                    "gcs_url": gcs_url,
                    "blob_path": blob_path,
                    "document_id": result.document_id,
                    "physician_id": physicianId,
                    "user_id": userId
                }
                logger.info(f"📤 Queuing payload for {document.filename} (size: {len(str(webhook_payload))} chars)")
                
                task = finalize_document_task.delay(webhook_payload)
                task_ids.append(task.id)
                logger.info(f"🚀 Task queued for {document.filename}: {task.id}")
                
            except ValueError as ve:
                logger.error(f"❌ Validation error for {document.filename}: {str(ve)}")
                ignored_filenames.append({
                    "filename": document.filename,
                    "reason": str(ve)
                })
                # Emit error
                emit_data = {
                    'filename': document.filename,
                    'error': str(ve),
                    'user_id': userId
                }
                if userId:
                    await sio.emit('task_error', emit_data, room=f"user_{userId}")
                else:
                    await sio.emit('task_error', emit_data)
            except Exception as proc_exc:
                logger.error(f"❌ Unexpected processing error for {document.filename}: {str(proc_exc)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                ignored_filenames.append({
                    "filename": document.filename,
                    "reason": f"Processing failed: {str(proc_exc)}"
                })
                # Emit error
                emit_data = {
                    'filename': document.filename,
                    'error': str(proc_exc),
                    'user_id': userId
                }
                if userId:
                    await sio.emit('task_error', emit_data, room=f"user_{userId}")
                else:
                    await sio.emit('task_error', emit_data)
            finally:
                # Always cleanup temp files
                file_service.cleanup_temp_file(temp_path)
                if was_converted and converted_path:
                    DocumentConverter.cleanup_converted_file(converted_path, was_converted)
        
        total_ignored = len(ignored_filenames)
        logger.info(f"✅ Processing complete: {len(task_ids)} queued, {total_ignored} ignored")
        logger.info("✅ === END MULTI-DOCUMENT REQUEST ===\n")
        
        return {
            "task_ids": task_ids,
            "ignored": ignored_filenames,
            "ignored_count": total_ignored
        }
    
    except Exception as global_exc:
        logger.error(f"❌ Global error in extract-documents: {str(global_exc)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        # Cleanup any uploaded files
        for path in uploaded_files:
            try:
                file_service.delete_from_gcs(path)
                logger.info(f"🗑️ Cleanup GCS: {path}")
            except:
                logger.warning(f"⚠️ Cleanup failed: {path}")
        raise HTTPException(status_code=500, detail=f"Global processing failed: {str(global_exc)}")
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
            dob=dob_date,
            doi=doi_date,
            claim_number=claim_number
        )
        
        # Get patient quiz data
        quiz_data = await db_service.get_patient_quiz(patient_name, dob, doi)
       
        if not document_data or document_data["total_documents"] == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No documents found for patient: {patient_name}"
            )
        
        # Format the aggregated response
        response = await format_aggregated_document_response(document_data, quiz_data)
        
        logger.info(f"✅ Returned aggregated document for: {patient_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def format_aggregated_document_response(all_documents_data: Dict[str, Any], quiz_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format the aggregated document response from all documents"""
    documents = all_documents_data["documents"]
    
    if not documents:
        return {
            "patient_name": all_documents_data["patient_name"],
            "total_documents": 0,
            "documents": [],
            "patient_quiz": quiz_data,
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
        "patient_quiz": quiz_data,
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
        "blob_path": document.get("blobPath"),  # ✅ Ensure blob_path is included (adjust key if DB uses snake_case like "blob_path")
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