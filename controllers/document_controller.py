# controllers/document_controller.py (updated: check for existing document before GCS upload and handle like ignored/required field issue)

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Query
from typing import Dict, List, Any
from datetime import datetime, timedelta
import traceback
from prisma import Prisma
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
from services.task_creation import TaskCreator

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
        print(document_analysis,'document analysis in webhook')
        # 🔹 AI Task Creation
        task_creator = TaskCreator()
        tasks = task_creator.generate_tasks(document_analysis.dict(), data["filename"])
        
        # Save tasks to DB (map AI task keys to Prisma Task model fields)
        prisma = Prisma()
        await prisma.connect()
        created_tasks = 0
        for task in tasks:
            try:
                # Map snake_case keys to camelCase expected by Prisma schema
                mapped_task = {
                    "description": task.get("description"),
                    "department": task.get("department"),
                    "status": task.get("status", "Pending"),
                    # Prisma Task model expects `dueDate` (DateTime?)
                    "dueDate": None,
                    "patient": task.get("patient", "Unknown"),
                    "actions": task.get("actions") if isinstance(task.get("actions"), list) else [],
                    # sourceDocument field in Prisma
                    "sourceDocument": task.get("source_document") or task.get("sourceDocument") or data.get("filename"),
                    # quickNotes: keep as JSON/dict
                    "quickNotes": task.get("quick_notes") or task.get("quickNotes") or {},
                }

                # Normalize due date if present (accept datetime or ISO string)
                due_raw = task.get("due_date") or task.get("dueDate")
                if due_raw:
                    if isinstance(due_raw, str):
                        try:
                            mapped_task["dueDate"] = datetime.strptime(due_raw, "%Y-%m-%d")
                        except Exception:
                            # best-effort parse, fallback to now + 3 days
                            mapped_task["dueDate"] = datetime.now() + timedelta(days=3)
                    else:
                        # assume it's already a datetime-like object
                        mapped_task["dueDate"] = due_raw

                await prisma.task.create(data=mapped_task)
                created_tasks += 1
            except Exception as task_err:
                logger.error(f"❌ Failed to create task for document {data.get('filename')}: {task_err}", exc_info=True)
                # continue creating other tasks even if one fails
                continue

        logger.info(f"✅ {created_tasks} / {len(tasks)} tasks created for document {data['filename']}")
        # Enhanced check for required fields: treat "Not specified" as missing but don't block processing
        required_fields = {
            "patient_name": document_analysis.patient_name,
            "dob": document_analysis.dob,
            # "doi": document_analysis.doi
        }
        missing_fields = [k for k, v in required_fields.items() if not v or str(v).lower() == "not specified"]
        
        # Extract blob_path from gcs_url (original uploads path)
        blob_path = data.get("blob_path")
        if not blob_path:
            blob_path = extract_blob_path_from_gcs_url(data["gcs_url"])
        
        physician_id = data.get("physician_id")
        db_service = await get_database_service()
        
        # Check for existing document first (this should still block processing)
        file_exists = await db_service.document_exists(
            data["filename"], 
            data.get("file_size", 0)
        )
        
        if file_exists:
            logger.warning(f"⚠️ Document already exists: {data['filename']}")
            
            user_friendly_msg = "Document already processed"
            
            await db_service.save_fail_doc(
                reasson=user_friendly_msg,
                blob_path=blob_path,
                physician_id=physician_id
            )
            
            # Emit skipped event
            await sio.emit('task_complete', {
                'document_id': data.get('document_id', 'unknown'),
                'filename': data["filename"],
                'status': 'skipped',
                'reason': user_friendly_msg,
                "user_id": data.get("user_id"),  # Pass user_id if available
                'blob_path': blob_path,
                'physician_id': physician_id
            })
            return {"status": "skipped", "reason": "Document already processed", "blob_path": blob_path}
        
        # For missing required fields, we'll process but mark as failed in main flow
        has_missing_required_fields = len(missing_fields) > 0
        
        # Generate AI brief summary (do this even for documents with missing fields)
        brief_summary = analyzer.generate_brief_summary(result_data.get("text", ""))
        
        # Handle dates - use "Not specified" string instead of datetime.now() when not available
        dob = document_analysis.dob if document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else "Not specified"
        rd = document_analysis.rd if document_analysis.rd and str(document_analysis.rd).lower() != "not specified" else "Not specified"
        doi = document_analysis.doi if document_analysis.doi and str(document_analysis.doi).lower() != "not specified" else "Not specified"
        
        # For database queries that need datetime, create a fallback (use None or current date as needed)
        dob_for_query = None
        if dob and dob.lower() != "not specified":
            try:
                dob_for_query = datetime.strptime(dob, "%Y-%m-%d")
            except ValueError:
                dob_for_query = None
        
        # Parse rd for DB (reportDate): Handle both MM/DD and YYYY-MM-DD formats, convert to datetime object using 2025 as year if MM/DD
        rd_for_db = None
        if rd.lower() != "not specified":
            try:
                if '/' in rd:  # MM/DD format (e.g., "05/15")
                    month, day = rd.split('/')
                    year = 2025  # Current year from "October 11, 2025"
                    full_date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif '-' in rd:  # YYYY-MM-DD format (e.g., "2025-05-15")
                    full_date_str = rd
                else:
                    raise ValueError("Invalid date format")
                
                rd_for_db = datetime.strptime(full_date_str, "%Y-%m-%d")
                logger.debug(f"Parsed rd '{rd}' to datetime: {rd_for_db}")
            except (ValueError, AttributeError) as parse_err:
                logger.warning(f"Failed to parse rd '{rd}': {parse_err}; using None")
                rd_for_db = None
        
        # Retrieve previous documents (even for documents with missing fields)
        print(physician_id,'physician id in webhook')
       
        claimNUmbers = await db_service.get_patient_claim_numbers(
            document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else "Unknown Patient",
            physicianId=physician_id,
            dob=dob_for_query  # Use the parsed datetime for query, or None
        )
        
        db_response = await db_service.get_all_unverified_documents(
            document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else "Unknown Patient",
            physicianId=physician_id,
            claimNumber=document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else None,
            dob=dob_for_query  # Use the parsed datetime for query, or None
        )
        
        print(claimNUmbers,'claim numbers')
        # Extract documents list from database response
        previous_documents = db_response.get('documents', []) if db_response else []
        
        # Compare with previous documents using LLM (even for documents with missing fields)
        whats_new_data = analyzer.compare_with_previous_documents(
            document_analysis, 
            previous_documents
        )
        
        # Ensure whats_new_data is always a dict (empty if invalid) to avoid DB required field error
        if whats_new_data is None or not isinstance(whats_new_data, dict):
            logger.warning(f"⚠️ Invalid whats_new data for {data['filename']}; using empty dict")
            whats_new_data = {}
        
        # Handle claim number logic (even for documents with missing fields)
        claim_to_use = document_analysis.claim_number
        claim_numbers_list = []  # Initialize here for scope
        if not claim_to_use or str(claim_to_use).lower() == "not specified":
            claim_numbers_list = claimNUmbers.get('claim_numbers', []) if isinstance(claimNUmbers, dict) else claimNUmbers if isinstance(claimNUmbers, list) else []
            # Deduplicate claim numbers to handle cases where duplicates exist
            claim_numbers_list = list(set(claim_numbers_list))
            if len(claim_numbers_list) == 0:
                # First time: OK to proceed with "Not specified" or extracted value
                claim_to_use = document_analysis.claim_number or "Not specified"
                logger.info(f"ℹ️ First document for patient '{document_analysis.patient_name}': Using claim '{claim_to_use}'")
            elif len(claim_numbers_list) == 1:
                # One previous claim (after dedup): use it
                claim_to_use = claim_numbers_list[0]
                logger.info(f"ℹ️ Using single previous claim '{claim_to_use}' for patient '{document_analysis.patient_name}'")
            else:
                # Multiple distinct previous claims, no current: always use UNSPECIFIED and flag as pending
                claim_to_use = "Not specified"
                pending_reason = "Pending claim assignment: Patient has multiple associated claims"
                logger.warning(f"⚠️ {pending_reason} for file {data['filename']}; saving as pending")
        else:
            # If claim is specified, use it directly
            claim_numbers_list = []  # No need for prior claims check
            logger.info(f"ℹ️ Using extracted claim '{claim_to_use}' for patient '{document_analysis.patient_name}'")
        
        # Determine status: prioritize analyzer status, but override to "pending" for multi-claim issues or "failed" for missing fields
        base_status = document_analysis.status
        if len(claim_numbers_list) > 1 and not document_analysis.claim_number:
            document_status = "pending"
        elif has_missing_required_fields:
            document_status = "failed"
        else:
            document_status = base_status

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
        
        # Use "Not specified" values for missing fields but process normally
        patient_name_to_use = document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else "Not specified"
        
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=data["filename"],
            file_size=data.get("file_size", 0),
            mime_type=data.get("mime_type", "application/octet-stream"),
            processing_time_ms=data.get("processing_time_ms", 0),
            blob_path=data.get("blob_path", ""),
            gcs_file_link=data["gcs_url"],
            patient_name=patient_name_to_use,
            claim_number=claim_to_use,
            dob=dob,  # Now using string "Not specified" instead of datetime
            doi=doi,  # Now using string "Not specified" instead of datetime
            status=document_status,  # Now can be "pending" for multi-claim
            brief_summary=brief_summary,
            summary_snapshot=summary_snapshot,
            whats_new=whats_new_data,
            adl_data=adl_data,
            document_summary=document_summary,
            rd=rd_for_db,  # Pass parsed datetime object (or None) instead of string
            physician_id=physician_id
        )

        # Update previous documents with null claim_number to use the current claim_to_use
        # Only if claim_to_use is not "Not specified" or similar placeholder and document is not failed/pending
        if document_status not in ["failed", "pending"] and claim_to_use not in ["Not specified", "UNSPECIFIED", ""]:
            await db_service.update_previous_claim_numbers(
                patient_name=patient_name_to_use,
                dob=dob_for_query,  # Use the parsed datetime for query
                physician_id=physician_id,
                claim_number=claim_to_use
            )
            logger.info(f"🔄 Updated previous documents' claim numbers for patient '{patient_name_to_use}' using claim '{claim_to_use}'")
        elif document_status in ["failed", "pending"]:
            logger.info(f"ℹ️ Skipping previous claim update as document status is {document_status}")

        logger.info(f"💾 Document saved via webhook with ID: {document_id}, status: {document_status}")
        
        # Emit socket event for task complete with appropriate status
        emit_data = {
            'document_id': document_id,
            'filename': data["filename"],
            'status': document_status,  # Now emits "pending" directly
            'missing_fields': missing_fields if has_missing_required_fields else None,
            'pending_reason': pending_reason if document_status == "pending" else None  # Optional: pass reason for UI
        }
        
        user_id = data.get("user_id")
        if user_id:
            await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
        else:
            await sio.emit('task_complete', emit_data)
        
        logger.info(f"📡 Emitted 'task_complete' event from webhook: {emit_data}")
        
        return {
            "status": document_status, 
            "document_id": document_id,
            "missing_fields": missing_fields if has_missing_required_fields else None,
            "pending_reason": pending_reason if document_status == "pending" else None
        }

    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}", exc_info=True)
        
        # Save to FailDocs on general exception (only for true processing errors, not missing field cases)
        blob_path = data.get("blob_path", "") if 'data' in locals() else ""
        if not blob_path and 'data' in locals() and data.get("gcs_url"):
            blob_path = extract_blob_path_from_gcs_url(data["gcs_url"])
        physician_id = data.get("physician_id") if 'data' in locals() else None
        reasson = f"Webhook processing failed: {str(e)}"
        
        if blob_path:
            db_service = await get_database_service()
            await db_service.save_fail_doc(
                reasson=reasson,
                blob_path=blob_path,
                physician_id=physician_id
            )
        
        # Emit error event on exception
        try:
            await sio.emit('task_error', {
                'document_id': data.get('document_id', 'unknown') if 'data' in locals() else 'unknown',
                'filename': data.get('filename', 'unknown') if 'data' in locals() else 'unknown',
                'error': str(e),
                'gcs_url': data.get('gcs_url', 'unknown') if 'data' in locals() else 'unknown',
                'physician_id': physician_id,
                'blob_path': blob_path
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
                    logger.error(f"❌ Conversion failed for {document.filename}: {str(convert_exc)}")
                    
                    # For conversion failure, just log and skip
                    ignored_filenames.append({
                        "filename": document.filename,
                        "reason": f"File conversion failed: {str(convert_exc)}"
                    })
                    continue  # Skip to next file
                
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
                        "size": file_size,
                        "mimeType": document.content_type or "application/octet-stream",
                        "gcsUrl": ""
                    },
                    summary="",  # Set after processing
                    comprehensive_analysis=None,
                    document_id=f"endpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                processing_time = (datetime.now() - document_start_time).total_seconds() * 1000
                logger.info(f"⏱️ Document AI time for {document.filename}: {processing_time:.0f}ms")

                # Only check if text was extracted - no other validation
                if not result.text:
                    reason = "No text extracted from document"
                    logger.warning(f"⚠️ {reason}")
                    
                    ignored_filenames.append({"filename": document.filename, "reason": reason})
                    continue

                logger.info(f"📝 Processing document analysis for {document.filename}...")
                analyzer = ReportAnalyzer()
                
                # Skip or stub document type detection to avoid AttributeError
                try:
                    detected_type = analyzer.detect_document_type_preview(result.text)
                    logger.info(f"🔍 Detected type: {detected_type}")
                except AttributeError:
                    logger.warning("⚠️ Document type detection unavailable—skipping")
                    detected_type = "unknown"
                result.summary = f"Document Type: {detected_type} - Processed successfully"
                
                # Extract document data but DON'T validate any fields
                document_analysis = analyzer.extract_document_data(result.text)
                logger.info(f"✅ Document analysis completed for {document.filename}")
                
                # Upload to GCS for all documents with text
                try:
                    logger.info("☁️ Uploading to GCS...")
                    gcs_url, blob_path = file_service.save_to_gcs(content, document.filename)  # Default folder="uploads"
                    logger.info(f"✅ GCS upload: {gcs_url} | Blob: {blob_path}")
                    successful_uploads.append(blob_path)
                    
                    # Update result
                    result.gcs_file_link = gcs_url
                    result.fileInfo["gcsUrl"] = gcs_url
                except Exception as gcs_exc:
                    logger.error(f"❌ GCS upload failed for {document.filename}: {str(gcs_exc)}")
                    ignored_filenames.append({"filename": document.filename, "reason": f"GCS upload failed: {str(gcs_exc)}"})
                    continue
                
                # Prepare and queue task - ALL documents with text go to webhook
                webhook_payload = {
                    "result": result.dict(),
                    "filename": document.filename,
                    "file_size": file_size,
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
                
            except Exception as proc_exc:
                logger.error(f"❌ Unexpected processing error for {document.filename}: {str(proc_exc)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                ignored_filenames.append({
                    "filename": document.filename,
                    "reason": f"Processing failed: {str(proc_exc)}"
                })
            finally:
                # Always cleanup temp files
                if temp_path:
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
        # Cleanup any successful uploads
        for path in successful_uploads:
            try:
                file_service.delete_from_gcs(path)
                logger.info(f"🗑️ Cleanup GCS (successful): {path}")
            except:
                logger.warning(f"⚠️ Cleanup failed: {path}")
        raise HTTPException(status_code=500, detail=f"Global processing failed: {str(global_exc)}")

from typing import Optional

from datetime import datetime

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
    
    # Define function to parse reportDate for sorting (obey document dates)
    def parse_report_date(doc):
        report_date = doc.get("reportDate")
        if report_date:
            if isinstance(report_date, str):
                return datetime.fromisoformat(report_date.replace('Z', '+00:00'))  # Handle ISO with Z
            elif isinstance(report_date, datetime):
                return report_date
        # Fallback to createdAt if no reportDate
        created_at = doc.get("createdAt")
        if created_at:
            if isinstance(created_at, str):
                return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            elif isinstance(created_at, datetime):
                return created_at
        return datetime.min
    
    # Define function to parse createdAt for chain selection
    def parse_created_at(doc):
        created_at = doc.get("createdAt")
        if created_at:
            if isinstance(created_at, str):
                return datetime.fromisoformat(created_at.replace('Z', '+00:00'))  # Handle ISO with Z
            elif isinstance(created_at, datetime):
                return created_at
        return datetime.min
    
    # Sort documents by reportDate ascending (oldest first) to obey chronological document order
    sorted_documents = sorted(documents, key=parse_report_date)
    
    # Find the latest document based on reportDate
    latest_doc = max(documents, key=parse_report_date)
    
    # Find the last saved document based on createdAt for full whats_new chain
    last_saved_doc = max(documents, key=parse_created_at)
    
    # Use the latest document for base info
    base_response = await format_single_document_base(latest_doc)
    
    # Collect all full summarySnapshot objects in an array (chronological order, but reverse to have latest first)
    summary_snapshots_list = [doc.get("summarySnapshot") for doc in sorted_documents]
    summary_snapshots = summary_snapshots_list[::-1]  # Reverse to put latest first
    
    # adl from latest document
    adl = await format_adl(latest_doc)
    
    # whats_new from the last saved document (most recent comparison, full chain)
    whats_new = last_saved_doc.get("whatsNew", {})
    
    # status from the last saved document
    status = last_saved_doc.get("status")
    
    # document_summary: group by type (as per chronological reportDate order)
    grouped_summaries = {}
    grouped_brief_summaries = {}
    for doc in sorted_documents:
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
        "is_latest": True,
        "status": status  # Override status from last saved doc
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
    # Handle datetime fields safely
    def format_date_field(field_value):
        if field_value is None:
            return None
        if isinstance(field_value, datetime):
            return field_value.isoformat()
        if isinstance(field_value, str):
            # Assume ISO string already
            return field_value
        return None
    
    return {
        "document_id": document.get("id"),
        "patient_name": document.get("patientName"),
        "dob": document.get("dob"),  # String as per schema
        "doi": document.get("doi"),  # String as per schema
        "claim_number": document.get("claimNumber"),
        "status": document.get("status"),
        "gcs_file_link": document.get("gcsFileLink"),
        "blob_path": document.get("blobPath"),  # ✅ Ensure blob_path is included (adjust key if DB uses snake_case like "blob_path")
        "created_at": format_date_field(document.get("createdAt")),
        "updated_at": format_date_field(document.get("updatedAt")),
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