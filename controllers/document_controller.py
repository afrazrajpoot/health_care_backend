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
from services.database_service import DatabaseService, get_database_service
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


import os
import re
from google.cloud import dlp_v2
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta

PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-gcp-project-id')

def deidentify_and_extract_phi(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Uses Google Cloud DLP to inspect for PHI, extract key PHI values, and de-identify the text.
    Returns extracted PHI dict and de-identified text.
    """
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/global"

    # Default PHI structure
    extracted_phi = {
        "patient_name": None,
        "claim_number": None,
        "dates": []
    }

    # PHI info types relevant to healthcare
    phi_info_types = [
        dlp_v2.InfoType(name="PERSON_NAME"),
        dlp_v2.InfoType(name="DATE"),
        dlp_v2.InfoType(name="PHONE_NUMBER"),
        dlp_v2.InfoType(name="EMAIL_ADDRESS"),
        dlp_v2.InfoType(name="US_SOCIAL_SECURITY_NUMBER"),
        dlp_v2.InfoType(name="ADDRESS"),
        dlp_v2.InfoType(name="MEDICAL_RECORD_NUMBER"),
    ]

    inspect_config = dlp_v2.InspectConfig(
        info_types=phi_info_types,
        min_likelihood=dlp_v2.Likelihood.POSSIBLE,
    )

    inspect_request = dlp_v2.InspectContentRequest(
        parent=parent,
        inspect_config=inspect_config,
        item=dlp_v2.ContentItem(value=text),
    )

    try:
        inspect_response = client.inspect_content(request=inspect_request)
    except Exception as e:
        print("❌ Permission denied for DLP API. Make sure the service is enabled and IAM roles are correct.")
        print(e)
        # Return default values so later code won't crash
        return extracted_phi, text
    except Exception as e:
        print("❌ Unexpected error while calling DLP API:", e)
        return extracted_phi, text

    # Extract PHI from findings
    for finding in inspect_response.result.findings:
        info_type = finding.info_type.name
        start = finding.location.content_locations[0].content_location.byte_range.start
        end = finding.location.content_locations[0].content_location.byte_range.end
        value = text[start:end].strip()

        if info_type == "PERSON_NAME" and not extracted_phi["patient_name"]:
            extracted_phi["patient_name"] = value
        elif info_type == "MEDICAL_RECORD_NUMBER" and not extracted_phi["claim_number"]:
            extracted_phi["claim_number"] = value
        elif info_type == "DATE":
            extracted_phi["dates"].append(value)

    # Fallback regex for claim number if not found
    if not extracted_phi["claim_number"]:
        claim_match = re.search(r'claim\s*#?\s*([A-Z0-9-]+)', text, re.I)
        if claim_match:
            extracted_phi["claim_number"] = claim_match.group(1)

    # De-identify the text
    deidentify_config = dlp_v2.DeidentifyConfig(
        info_type_transformations=dlp_v2.InfoTypeTransformations(
            transformations=[
                dlp_v2.InfoTypeTransformations.InfoTypeTransformation(
                    primitive_transformation=dlp_v2.PrimitiveTransformation(
                        replace_with_info_type_config=dlp_v2.ReplaceWithInfoTypeConfig(),
                    ),
                    info_types=phi_info_types,
                )
            ]
        )
    )

    deidentify_request = dlp_v2.DeidentifyContentRequest(
        parent=parent,
        deidentify_config=deidentify_config,
        inspect_config=inspect_config,
        item=dlp_v2.ContentItem(value=text),
    )

    try:
        deidentify_response = client.deidentify_content(request=deidentify_request)
        deidentified_text = deidentify_response.item.value
    except Exception as e:
        print("❌ Error during de-identification:", e)
        deidentified_text = text

    return extracted_phi, deidentified_text


def pseudonymize_structured(analysis: Dict[str, Any]) -> Dict[str, Any]:
    pseudo = analysis.copy() if isinstance(analysis, dict) else analysis.__dict__.copy()
    pseudo["patient_name"] = "[PATIENT]"
    if pseudo.get("dob") and str(pseudo["dob"]).lower() != "not specified":
        pseudo["dob"] = "[DOB]"
    if pseudo.get("rd") and str(pseudo["rd"]).lower() != "not specified":
        pseudo["rd"] = "[REPORT_DATE]"
    if pseudo.get("doi") and str(pseudo["doi"]).lower() != "not specified":
        pseudo["doi"] = "[DOCUMENT_DATE]"
    if pseudo.get("claim_number") and str(pseudo["claim_number"]).lower() != "not specified":
        pseudo["claim_number"] = "[CLAIM_NUMBER]"
    return pseudo



@router.post("/webhook/save-document")
async def save_document_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"📥 Webhook received for document save: {data.get('document_id', 'unknown')}")

        if not data.get("result") or not data.get("filename") or not data.get("gcs_url"):
            raise HTTPException(status_code=400, detail="Missing required fields in webhook payload")

        result_data = data["result"]
        text = result_data.get("text", "")
        
        # HIPAA Compliance: De-identify PHI before LLM processing
        extracted_phi, deidentified_text = deidentify_and_extract_phi(text)
        
        analyzer = ReportAnalyzer()
        document_analysis = analyzer.extract_document_data(result_data.get("text", ""))
        print(document_analysis,'document analysis in webhook')
        
        # Override extracted PHI with real values from DLP
        if extracted_phi["patient_name"]:
            document_analysis.patient_name = extracted_phi["patient_name"]
        if extracted_phi["claim_number"]:
            document_analysis.claim_number = extracted_phi["claim_number"]
        
        # Assign dates: first as DOB, second as RD, third as DOI
        dates = extracted_phi["dates"]
        if len(dates) > 0:
            document_analysis.dob = dates[0]
        if len(dates) > 1:
            document_analysis.rd = dates[1]
        if len(dates) > 2:
            document_analysis.doi = dates[2]
        
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
        
        # Enhanced check for required fields: treat "Not specified" as missing but don't block processing
        required_fields = {
            "patient_name": document_analysis.patient_name,
            "dob": document_analysis.dob,
            # "doi": document_analysis.doi
        }
        missing_fields = [k for k, v in required_fields.items() if not v or str(v).lower() == "not specified"]
        
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
        
        # Parse rd for DB (reportDate): Handle both MM/DD and YYYY-MM-DD formats, convert to datetime object using current year if MM/DD
        rd_for_db = None
        if rd.lower() != "not specified":
            try:
                if '/' in rd:  # MM/DD format (e.g., "05/15")
                    month, day = rd.split('/')
                    year = datetime.now().year  # Use dynamic current year
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
        
        # Retrieve previous claim numbers (always for patient/dob/physician)
        print(physician_id,'physician id in webhook')
       
        patient_name_for_query = document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else "Unknown Patient"
        claimNUmbers = await db_service.get_patient_claim_numbers(
            patient_name_for_query,
            physicianId=physician_id,
            dob=dob_for_query  # Use the parsed datetime for query, or None
        )
        
        # Always fetch all previous unverified documents for the patient (ignore new claim for fetching all previous)
        # 🆕 Updated: Pass claimNumber only if valid; otherwise None to use dob + patient_name
        claim_number_for_query = document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else None
        db_response = await db_service.get_claim_numbers(
            patient_name=patient_name_for_query,
            physicianId=physician_id,
            claimNumber=claim_number_for_query,  # Use valid claim if available; else None to fallback to dob + patient_name
            dob=dob_for_query  # Use the parsed datetime for query, or None
        )
        
        print(claimNUmbers,'claim numbers')
        # Extract documents list from database response
        previous_documents = db_response.get('documents', []) if db_response else []
        
        # Compare with previous documents using LLM (pseudonymize for compliance)
        pseudo_analysis = pseudonymize_structured(document_analysis)
        pseudo_previous = [pseudonymize_structured(doc) for doc in previous_documents]
        whats_new_data = analyzer.compare_with_previous_documents(
            pseudo_analysis, 
            pseudo_previous
        )
        
        # Ensure whats_new_data is always a dict (empty if invalid) to avoid DB required field error
        if whats_new_data is None or not isinstance(whats_new_data, dict):
            logger.warning(f"⚠️ Invalid whats_new data for {data['filename']}; using empty dict")
            whats_new_data = {}
        
        # Simplified claim number logic
        claim_to_use = None
        pending_reason = None
        claim_numbers_list = claimNUmbers.get('claim_numbers', []) if isinstance(claimNUmbers, dict) else claimNUmbers if isinstance(claimNUmbers, list) else []
        # Deduplicate claim numbers to handle cases where duplicates exist
        claim_numbers_list = list(set(claim_numbers_list))
        
        # Filter to valid claims only (exclude 'Not specified' and None)
        valid_claims_list = [c for c in claim_numbers_list if c and str(c).lower() != 'not specified']
        
        has_new_claim = document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
        
        if has_new_claim:
            # If document analysis has claim number, use it
            claim_to_use = document_analysis.claim_number
            logger.info(f"ℹ️ Using extracted claim '{claim_to_use}' for patient '{document_analysis.patient_name}'")
        else:
            # New document has no claim number
            if len(previous_documents) == 0:
                # First time: OK, proceed without claim
                claim_to_use = "Not specified"
                logger.info(f"ℹ️ First document for patient '{document_analysis.patient_name}': No claim specified, proceeding as OK")
            elif len(valid_claims_list) == 0 and len(previous_documents) > 0:
                # Previous documents exist but no valid claim in them: fail (second or later without claim)
                claim_to_use = None
                pending_reason = "No claim number specified and previous documents exist without valid claim"
                logger.warning(f"⚠️ {pending_reason} for file {data['filename']}")
            elif len(valid_claims_list) == 1:
                # One previous valid claim: use it for new document
                claim_to_use = valid_claims_list[0]
                logger.info(f"ℹ️ Using single previous valid claim '{claim_to_use}' for patient '{document_analysis.patient_name}'")
            else:
                # Multiple previous valid claims: fail
                claim_to_use = None
                pending_reason = "No claim number specified and multiple previous valid claims found"
                logger.warning(f"⚠️ {pending_reason} for file {data['filename']}")
        
        # Determine status: fail only for missing fields or claim issues
        base_status = document_analysis.status
        if has_missing_required_fields:
            document_status = "failed"
        elif claim_to_use is None:
            document_status = "failed"
        else:
            document_status = base_status

        # Use "Not specified" values for missing fields but process normally
        patient_name_to_use = document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else "Not specified"
        
        # If claim_to_use is None, set to "Not specified" for saving but status failed
        claim_to_save = claim_to_use if claim_to_use is not None else "Not specified"

        if document_status == "failed":
            # Fail: save to FailDocs, no further processing, no AI tasks
            fail_reason = pending_reason if pending_reason else f"Missing required fields: {', '.join(missing_fields)}"
            logger.warning(f"⚠️ Failing document {data['filename']}: {fail_reason}")
            
            await db_service.save_fail_doc(
                reason=fail_reason,
                db=dob,
                doi=doi,
                claim_number=claim_to_save,
                patient_name=patient_name_to_use,
                document_text=result_data.get("text", ""),
                physician_id=physician_id,
                gcs_file_link=data["gcs_url"],
                file_name=data["filename"],
                file_hash=data.get("file_hash"),
                blob_path=blob_path
            )
            
            # Emit failed event
            emit_data = {
                'document_id': data.get('document_id', 'unknown'),
                'filename': data["filename"],
                'status': 'failed',
                'missing_fields': missing_fields if has_missing_required_fields else None,
                'pending_reason': pending_reason if pending_reason else None
            }
            
            user_id = data.get("user_id")
            if user_id:
                await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
            else:
                await sio.emit('task_complete', emit_data)
            
            logger.info(f"📡 Emitted 'task_complete' failed event from webhook: {emit_data}")
            
            return {
                "status": "failed", 
                "reason": fail_reason,
                "missing_fields": missing_fields if has_missing_required_fields else None,
                "pending_reason": pending_reason if pending_reason else None
            }
        
        # Success: proceed with saving to main document and creating tasks
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
        
        # 🆕 FIRST save the document to get the document_id
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=data["filename"],
            file_size=data.get("file_size", 0),
            mime_type=data.get("mime_type", "application/octet-stream"),
            processing_time_ms=data.get("processing_time_ms", 0),
            blob_path=data.get("blob_path", ""),
            file_hash=data.get("file_hash"),
            gcs_file_link=data["gcs_url"],
            patient_name=patient_name_to_use,
            claim_number=claim_to_save,
            dob=dob,  # Now using string "Not specified" instead of datetime
            doi=doi,  # Now using string "Not specified" instead of datetime
            status=document_status,  # Now "failed" only for specific cases
            brief_summary=brief_summary,
            summary_snapshot=summary_snapshot,
            whats_new=whats_new_data,
            adl_data=adl_data,
            document_summary=document_summary,
            rd=rd_for_db,  # Pass parsed datetime object (or None) instead of string
            physician_id=physician_id
        )

        # 🔹 AI Task Creation - AFTER document is saved so we have document_id
        task_creator = TaskCreator()
        tasks = await task_creator.generate_tasks(document_analysis.dict(), data["filename"])
        
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
                    # 🆕 Save document_id and physician_id - NOW we have the document_id
                    "documentId": document_id,  # Use the actual document_id from the saved document
                    "physicianId": data.get("physician_id"),
                    # 🚫 REMOVED quickNotes entirely
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

        # Prepare dob_str for update
        dob_str = None
        if dob_for_query:
            dob_str = dob_for_query.strftime("%Y-%m-%d")

        # 🔄 Update previous documents' claim numbers (only if new has a valid claim number)
        should_update_previous = (
            document_status not in ["failed"] and  # Skip if failed due to claim issues
            has_new_claim and  # Only update if new doc provided a valid claim
            patient_name_to_use != "Not specified" and  # Avoid updating if patient is unknown
            dob_str is not None  # Ensure we have dob string for query
        )

        if should_update_previous:
            # Fetch previous documents to check for null claims (reuse the earlier query if possible)
            db_response = await db_service.get_all_unverified_documents(
                patient_name=patient_name_to_use,
                physicianId=physician_id,
                claimNumber=None,  # Explicitly fetch all, including null claims
                dob=dob_for_query
            )
            previous_documents = db_response.get('documents', []) if db_response else []
            
            # Check if any previous docs have null/unset claims
            has_null_claim_docs = any(
                doc.get('claim_number') is None or str(doc.get('claim_number', '')).lower() == 'not specified'
                for doc in previous_documents
            )
            
            if has_null_claim_docs:
                await db_service.update_previous_claim_numbers(
                    patient_name=patient_name_to_use,
                    dob=dob_str,  # Send only date string
                    physician_id=physician_id,
                    claim_number=claim_to_use
                )
                logger.info(f"🔄 Updated previous documents' claim numbers for patient '{patient_name_to_use}' using claim '{claim_to_use}'")
            else:
                logger.info(f"ℹ️ No previous documents with null claims for patient '{patient_name_to_use}'; skipping update")
        else:
            logger.info(f"ℹ️ Skipping previous claim update: status={document_status}, has_new_claim={has_new_claim}, patient={patient_name_to_use}, has_dob={dob_str is not None}")

        logger.info(f"💾 Document saved via webhook with ID: {document_id}, status: {document_status}")
        
        # Emit socket event for task complete with appropriate status
        emit_data = {
            'document_id': document_id,
            'filename': data["filename"],
            'status': document_status,
            'missing_fields': missing_fields if has_missing_required_fields else None,
            'pending_reason': pending_reason if pending_reason else None  # Use the claim-related reason if applicable
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
            "pending_reason": pending_reason if pending_reason else None
        }

    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}", exc_info=True)
        
        # Save to FailDocs on general exception (only for true processing errors, not missing field cases)
        blob_path = data.get("blob_path", "") if 'data' in locals() else ""
        if not blob_path and 'data' in locals() and data.get("gcs_url"):
            blob_path = extract_blob_path_from_gcs_url(data["gcs_url"])
        physician_id = data.get("physician_id") if 'data' in locals() else None
        reason = f"Webhook processing failed: {str(e)}"
        
      
        
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
@router.post("/update-fail-document")
async def update_fail_document(request: Request):
    try:
        data = await request.json()
        logger.info(f"📥 Update fail document request for ID: {data.get('fail_doc_id', 'unknown')}")

        fail_doc_id = data.get("fail_doc_id")
        if not fail_doc_id:
            raise HTTPException(status_code=400, detail="Missing required fail_doc_id in request payload")

        updated_document_text = data.get("document_text")
        updated_dob = data.get("dob")
        updated_doi = data.get("doi")
        updated_claim_number = data.get("claim_number")
        updated_patient_name = data.get("patient_name")

        db_service = await get_database_service()
        fail_doc = await db_service.get_fail_doc_by_id(fail_doc_id)

        if not fail_doc:
            raise HTTPException(status_code=404, detail="Fail document not found")

        # Use updated values if provided, otherwise fallback to fail_doc values
        document_text = updated_document_text or fail_doc.documentText
        dob_str = updated_dob or fail_doc.db
        doi = updated_doi or fail_doc.doi
        claim_number = updated_claim_number or fail_doc.claimNumber
        patient_name = updated_patient_name or fail_doc.patientName
        physician_id = fail_doc.physicianId
        filename = fail_doc.fileName
        gcs_url = fail_doc.gcsFileLink
        blob_path = fail_doc.blobPath
        file_hash = fail_doc.fileHash

        # Mock result_data similar to webhook (since no new extraction, use the text directly)
        result_data = {
            "text": document_text,
            "pages": 0,  # Default
            "entities": [],  # Default
            "tables": [],  # Default
            "formFields": [],  # Default
            "confidence": 0.0,  # Default
            "success": False,  # Default
            "gcs_file_link": gcs_url,
            "fileInfo": {},  # Default
            "comprehensive_analysis": None,  # Default
            "document_id": f"update_fail_{fail_doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        analyzer = ReportAnalyzer()
        document_analysis = analyzer.extract_document_data(result_data.get("text", ""))
        print(document_analysis, 'document analysis in update fail route')

        # Check for existing document first (this should still block processing)
        file_exists = await db_service.document_exists(filename, 0)  # file_size default to 0

        if file_exists:
            logger.warning(f"⚠️ Document already exists: {filename}")

            user_friendly_msg = "Document already processed"

            # Emit skipped event
            await sio.emit('task_complete', {
                'document_id': fail_doc_id,
                'filename': filename,
                'status': 'skipped',
                'reason': user_friendly_msg,
                "user_id": data.get("user_id"),  # Pass user_id if available
                'blob_path': blob_path,
                'physician_id': physician_id
            })
            return {"status": "skipped", "reason": "Document already processed", "blob_path": blob_path}

        # Generate AI brief summary (do this even for documents with missing fields)
        brief_summary = analyzer.generate_brief_summary(result_data.get("text", ""))

        # Handle dates - use "Not specified" string instead of datetime.now() when not available
        dob = dob_str if dob_str and str(dob_str).lower() != "not specified" else "Not specified"
        rd = document_analysis.rd if document_analysis.rd and str(document_analysis.rd).lower() != "not specified" else "Not specified"
        doi = doi if doi and str(doi).lower() != "not specified" else "Not specified"

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
                    year = 2025  # Current year from "October 14, 2025"
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

        print(physician_id, 'physician id in update fail route')

        patient_name_for_query = patient_name if patient_name and str(patient_name).lower() != "not specified" else "Unknown Patient"
        claimNUmbers = await db_service.get_patient_claim_numbers(
            patient_name_for_query,
            physicianId=physician_id,
            dob=dob_for_query  # Use the parsed datetime for query, or None
        )

        # Always fetch all previous unverified documents for the patient (ignore new claim for fetching all previous)
        # 🆕 Updated: Pass claimNumber only if valid; otherwise None to use dob + patient_name
        claim_number_for_query = claim_number if claim_number and str(claim_number).lower() != "not specified" else None
        db_response = await db_service.get_all_unverified_documents(
            patient_name=patient_name_for_query,
            physicianId=physician_id,
            claimNumber=claim_number_for_query,  # Use valid claim if available; else None to fallback to dob + patient_name
            dob=dob_for_query  # Use the parsed datetime for query, or None
        )

        print(claimNUmbers, 'claim numbers')
        # Extract documents list from database response
        previous_documents = db_response.get('documents', []) if db_response else []

        # Compare with previous documents using LLM (even for documents with missing fields)
        whats_new_data = analyzer.compare_with_previous_documents(
            document_analysis,
            previous_documents
        )

        # Ensure whats_new_data is always a dict (empty if invalid) to avoid DB required field error
        if whats_new_data is None or not isinstance(whats_new_data, dict):
            logger.warning(f"⚠️ Invalid whats_new data for {filename}; using empty dict")
            whats_new_data = {}

        # Simplified claim number logic
        claim_to_use = None
        pending_reason = None
        claim_numbers_list = claimNUmbers.get('claim_numbers', []) if isinstance(claimNUmbers, dict) else claimNUmbers if isinstance(claimNUmbers, list) else []
        # Deduplicate claim numbers to handle cases where duplicates exist
        claim_numbers_list = list(set(claim_numbers_list))

        # Filter to valid claims only (exclude 'Not specified' and None)
        valid_claims_list = [c for c in claim_numbers_list if c and str(c).lower() != 'not specified']

        has_new_claim = claim_number and str(claim_number).lower() != "not specified"

        if has_new_claim:
            # If document analysis has claim number, use it
            claim_to_use = claim_number
            logger.info(f"ℹ️ Using extracted claim '{claim_to_use}' for patient '{patient_name}'")
        else:
            # New document has no claim number
            if len(previous_documents) == 0:
                # First time: OK, proceed without claim
                claim_to_use = "Not specified"
                logger.info(f"ℹ️ First document for patient '{patient_name}': No claim specified, proceeding as OK")
            elif len(valid_claims_list) == 0 and len(previous_documents) > 0:
                # Previous documents exist but no valid claim in them: fail (second or later without claim)
                claim_to_use = None
                pending_reason = "No claim number specified and previous documents exist without valid claim"
                logger.warning(f"⚠️ {pending_reason} for file {filename}")
            elif len(valid_claims_list) == 1:
                # One previous valid claim: use it for new document
                claim_to_use = valid_claims_list[0]
                logger.info(f"ℹ️ Using single previous valid claim '{claim_to_use}' for patient '{patient_name}'")
            else:
                # Multiple previous valid claims: fail
                claim_to_use = None
                pending_reason = "No claim number specified and multiple previous valid claims found"
                logger.warning(f"⚠️ {pending_reason} for file {filename}")

        # Determine status: always proceed to success unless claim issues (no validation for missing fields)
        base_status = document_analysis.status
        document_status = base_status  # Default to base
        if claim_to_use is None:
            document_status = "failed"

        # Use "Not specified" values for missing fields but process normally
        patient_name_to_use = patient_name if patient_name and str(patient_name).lower() != "not specified" else "Not specified"

        # If claim_to_use is None, set to "Not specified" for saving but status failed
        claim_to_save = claim_to_use if claim_to_use is not None else "Not specified"

        # Early return ONLY on claim-related failures
        if document_status == "failed":
            # Emit failed event
            emit_data = {
                'document_id': fail_doc_id,
                'filename': filename,
                'status': 'failed',
                'pending_reason': pending_reason if pending_reason else None
            }

            user_id = data.get("user_id")
            if user_id:
                await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
            else:
                await sio.emit('task_complete', emit_data)

            logger.info(f"📡 Emitted 'task_complete' failed event from update route: {emit_data}")

            return {
                "status": "failed",
                "reason": pending_reason,
                "pending_reason": pending_reason if pending_reason else None
            }

        # Success: proceed with saving to main document and creating tasks, then delete FailDoc
        # (This now runs always, as long as no claim issues, assuming all data provided in body)
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
            gcs_file_link=result_data.get("gcs_file_link", gcs_url),
            fileInfo=result_data.get("fileInfo", {}),
            summary=summary_text,
            comprehensive_analysis=result_data.get("comprehensive_analysis"),
            document_id=result_data.get("document_id", f"update_fail_{fail_doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )

        # 🆕 FIRST save the document to get the document_id
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=filename,
            file_size=0,  # Not available in fail_doc, set default
            mime_type="application/octet-stream",  # Default
            processing_time_ms=0,  # Default
            blob_path=blob_path,
            file_hash=file_hash,
            gcs_file_link=gcs_url,
            patient_name=patient_name_to_use,
            claim_number=claim_to_save,
            dob=dob,  # Now using string "Not specified" instead of datetime
            doi=doi,  # Now using string "Not specified" instead of datetime
            status=document_status,  # Now "failed" only for specific cases
            brief_summary=brief_summary,
            summary_snapshot=summary_snapshot,
            whats_new=whats_new_data,
            adl_data=adl_data,
            document_summary=document_summary,
            rd=rd_for_db,  # Pass parsed datetime object (or None) instead of string
            physician_id=physician_id
        )

        # 🔹 AI Task Creation - AFTER document is saved so we have document_id
        task_creator = TaskCreator()
        tasks = await task_creator.generate_tasks(document_analysis.dict(), filename)

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
                    "sourceDocument": task.get("source_document") or task.get("sourceDocument") or filename,
                    # 🆕 Save document_id and physician_id - NOW we have the document_id
                    "documentId": document_id,  # Use the actual document_id from the saved document
                    "physicianId": physician_id,
                    # 🚫 REMOVED quickNotes entirely
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
                logger.error(f"❌ Failed to create task for document {filename}: {task_err}", exc_info=True)
                # continue creating other tasks even if one fails
                continue

        logger.info(f"✅ {created_tasks} / {len(tasks)} tasks created for updated document {filename}")

        # Prepare dob_str for update
        dob_str_for_query = None
        if dob_for_query:
            dob_str_for_query = dob_for_query.strftime("%Y-%m-%d")

        # 🔄 Update previous documents' claim numbers (only if new has a valid claim number)
        should_update_previous = (
            document_status not in ["failed"] and  # Skip if failed due to claim issues
            has_new_claim and  # Only update if new doc provided a valid claim
            patient_name_to_use != "Not specified" and  # Avoid updating if patient is unknown
            dob_str_for_query is not None  # Ensure we have dob string for query
        )

        if should_update_previous:
            # Fetch previous documents to check for null claims (reuse the earlier query if possible)
            db_response = await db_service.get_all_unverified_documents(
                patient_name=patient_name_to_use,
                physicianId=physician_id,
                claimNumber=None,  # Explicitly fetch all, including null claims
                dob=dob_for_query
            )
            previous_documents = db_response.get('documents', []) if db_response else []

            # Check if any previous docs have null/unset claims
            has_null_claim_docs = any(
                doc.get('claim_number') is None or str(doc.get('claim_number', '')).lower() == 'not specified'
                for doc in previous_documents
            )

            if has_null_claim_docs:
                await db_service.update_previous_claim_numbers(
                    patient_name=patient_name_to_use,
                    dob=dob_str_for_query,  # Send only date string
                    physician_id=physician_id,
                    claim_number=claim_to_use
                )
                logger.info(f"🔄 Updated previous documents' claim numbers for patient '{patient_name_to_use}' using claim '{claim_to_use}'")
            else:
                logger.info(f"ℹ️ No previous documents with null claims for patient '{patient_name_to_use}'; skipping update")
        else:
            logger.info(f"ℹ️ Skipping previous claim update: status={document_status}, has_new_claim={has_new_claim}, patient={patient_name_to_use}, has_dob={dob_str_for_query is not None}")

        # Delete the FailDoc since successfully processed (no update, just delete)
        await db_service.delete_fail_doc(fail_doc_id)
        logger.info(f"🗑️ Deleted fail doc {fail_doc_id} after successful update")

        logger.info(f"💾 Fail document updated and saved via route with ID: {document_id}, status: {document_status}")

        # Emit socket event for task complete with appropriate status
        emit_data = {
            'document_id': document_id,
            'filename': filename,
            'status': document_status,
            'pending_reason': pending_reason if pending_reason else None  # Use the claim-related reason if applicable
        }

        user_id = data.get("user_id")
        if user_id:
            await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
        else:
            await sio.emit('task_complete', emit_data)

        logger.info(f"📡 Emitted 'task_complete' event from update route: {emit_data}")

        return {
            "status": document_status,
            "document_id": document_id,
            "pending_reason": pending_reason if pending_reason else None
        }

    except Exception as e:
        logger.error(f"❌ Update fail document failed: {str(e)}", exc_info=True)

        # On exception, no update to FailDoc, just emit error
        try:
            await sio.emit('task_error', {
                'document_id': data.get('fail_doc_id', 'unknown') if 'data' in locals() else 'unknown',
                'filename': fail_doc.fileName if 'fail_doc' in locals() else 'unknown',
                'error': str(e),
                'gcs_url': gcs_url if 'gcs_url' in locals() else 'unknown',
                'physician_id': physician_id if 'physician_id' in locals() else None,
                'blob_path': blob_path if 'blob_path' in locals() else ''
            })
        except:
            pass  # Ignore emit failure during error
        raise HTTPException(status_code=500, detail=f"Update fail document processing failed: {str(e)}")

import hashlib

def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
    db_service = await get_database_service()  # Get DB service early for existence checks
    payloads = []  # Collect all valid payloads for batch task
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
            was_converted = False
            converted_path = None
            
            try:
                file_service.validate_file(document, CONFIG["max_file_size"])
                logger.info(f"📁 Starting processing for file: {document.filename}")
                logger.info(f"📏 File size: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
                logger.info(f"📋 MIME type: {document.content_type}")
                
                # Early check for existing document
                file_hash = compute_file_hash(content)

                file_exists = await db_service.document_exists_by_hash(
                file_hash=file_hash,
                user_id=userId,
                physician_id=physicianId
                )
                
                if file_exists:
                    logger.warning(f"⚠️ Document already exists: {document.filename}")
                    
                    # Emit skipped event
                    emit_data = {
                        'document_id': 'unknown',
                        'filename': document.filename,
                        'status': 'skipped',
                        'reason': 'Document already processed',
                        'user_id': userId,
                        'blob_path': blob_path,
                        'physician_id': physicianId
                    }
                    if userId:
                        await sio.emit('task_complete', emit_data, room=f"user_{userId}")
                    else:
                        await sio.emit('task_complete', emit_data)
                    
                    ignored_filenames.append({
                        "filename": document.filename,
                        "reason": "Document already processed"
                    })
                    continue  # Skip to next file
                
                # Save to temp for processing
                temp_path = file_service.save_temp_file(content, document.filename)
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
                
                # Prepare payload - ALL documents with text go to batch
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
                    "physician_id": physicianId,
                    "user_id": userId
                }
                logger.info(f"📤 Adding payload to batch for {document.filename} (size: {len(str(webhook_payload))} chars)")
                
                payloads.append(webhook_payload)
                
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
        logger.info(f"✅ Preprocessing complete: {len(payloads)} ready for batch, {total_ignored} ignored")
        
        if payloads:
            # Enqueue ONE master batch task for sequential processing
            from utils.celery_task import process_batch_documents
            task = process_batch_documents.delay(payloads)
            task_id = task.id
            logger.info(f"🚀 Batch task queued for {len(payloads)} docs: {task_id}")
            
            # Emit initial socket event for batch start
            if userId:
                await sio.emit('batch_started', {
                    'task_id': task_id,
                    'filenames': [p['filename'] for p in payloads],
                    'user_id': userId
                })
            else:
                await sio.emit('batch_started', {
                    'task_id': task_id,
                    'filenames': [p['filename'] for p in payloads]
                })
        else:
            task_id = None
            logger.info("ℹ️ No payloads to queue")
        
        logger.info("✅ === END MULTI-DOCUMENT REQUEST ===\n")
        
        return {
            "task_id": task_id,  # Single batch task ID
            "payload_count": len(payloads),
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
    # doi: str,
    physicianId: Optional[str] = None,
    claim_number: Optional[str] = None
):
    """
    Get aggregated document for a patient
    Returns a single aggregated document from all patient documents
    """
    try:
        logger.info(f"📄 Fetching aggregated document for patient: {patient_name}")

        # ✅ Parse date strings using helper
        dob_date = parse_date(dob, "Date of Birth")
        # doi_date = parse_date(doi, "Date of Injury")

        db_service = await get_database_service()
        
        # Get all documents
        document_data = await db_service.get_document_by_patient_details(
            patient_name=patient_name,
            physicianId=physicianId,
            dob=dob_date,
            # doi=doi_date,
            claim_number=claim_number
        )

        # quiz_data = await db_service.get_patient_quiz(patient_name, dob, doi)
       
        if not document_data or document_data.get("total_documents") == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for patient: {patient_name}"
            )

        # ✅ Fetch tasks (quick notes) for the document IDs, filtered by physicianId if provided
        documents = document_data["documents"]
        document_ids = [doc["id"] for doc in documents]
        tasks = await db_service.get_tasks_by_document_ids(
            document_ids=document_ids,
            physician_id=physicianId  # Optional filter
        )
        # Create a mapping of document_id to quickNotes
        tasks_dict = {task["documentId"]: task["quickNotes"] for task in tasks}
        
        response = await format_aggregated_document_response(
            all_documents_data=document_data, 
            # quiz_data=quiz_data, 
            tasks_dict=tasks_dict
        )
        
        logger.info(f"✅ Returned aggregated document for: {patient_name}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def format_aggregated_document_response(
    all_documents_data: Dict[str, Any], 
    quiz_data: Optional[Dict[str, Any]] = None,
    tasks_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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
    
    # ✅ Collect quick notes snapshots in chronological order, reverse to latest first, FILTER OUT None/null
    quick_notes_snapshots_list = [
        tasks_dict.get(doc["id"]) if tasks_dict else None 
        for doc in sorted_documents
    ]
    # Filter out None/null values
    quick_notes_snapshots_filtered = [note for note in quick_notes_snapshots_list if note is not None]
    quick_notes_snapshots = quick_notes_snapshots_filtered[::-1]
    
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
        "quick_notes_snapshots": quick_notes_snapshots,  # ✅ Now filtered, no nulls
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


@router.get("/workflow-stats")
async def get_workflow_stats():
    db = DatabaseService()
    await db.prisma.connect()
    stats = await db.prisma.workFlowStats.find_first(order={"date": "desc"})
    await db.prisma.disconnect()
    return stats or {}