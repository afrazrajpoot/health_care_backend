


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
from services.report_analyzer import ReportAnalyzer, DocumentAnalysis
from services.database_service import get_database_service

from config.celery_config import app as celery_app

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
            return {"status": "skipped", "reason": "Document already processed"}
        
        # Retrieve previous documents
        db_response = await db_service.get_all_unverified_documents(
            document_analysis.patient_name
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
            document_summary=document_summary
        )

        logger.info(f"💾 Document saved via webhook with ID: {document_id}")
        return {"status": "success", "document_id": document_id}

    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}", exc_info=True)
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

from typing import Optional

@router.get('/document')
async def get_document(
    patient_name: str,
    dob: str,
    doi: str,
    claim_number: Optional[str] = None
):
    """
    Get last two documents for a patient
    Returns multiple documents in structured format
    """
    try:
        logger.info(f"📄 Fetching last 2 documents for patient: {patient_name}")
        
        # Parse date strings
        try:
            dob_date = datetime.strptime(dob, "%Y-%m-%d")
            doi_date = datetime.strptime(doi, "%Y-%m-%d")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
        
        db_service = await get_database_service()
        
        # Get documents (always returns multi-document structure)
        document_data = await db_service.get_document_by_patient_details(
            patient_name=patient_name,
            # dob=dob_date,
            # doi=doi_date,
            # claim_number=claim_number
        )
        
        if not document_data:
            raise HTTPException(
                status_code=404, 
                detail=f"No documents found for patient: {patient_name}"
            )
        
        # Format the response
        response = await format_document_response(document_data)
        
        logger.info(f"✅ Returned {response['total_documents']} documents for: {patient_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def format_document_response(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format the document data response - handles multiple documents"""
    
    # Check if this is a multi-document response
    if "documents" in document_data and "total_documents" in document_data:
        return await format_multiple_documents_response(document_data)
    else:
        # If it's a single document (old format), wrap it in multi-document structure
        return await format_single_document_as_multiple(document_data)

async def format_single_document_as_multiple(document: Dict[str, Any]) -> Dict[str, Any]:
    """Format a single document as a multi-document response"""
    formatted_doc = await format_single_document(document)
    formatted_doc["document_index"] = 1
    formatted_doc["is_latest"] = True
    
    return {
        "patient_name": document.get("patientName"),
        "total_documents": 1,
        "documents": [formatted_doc],
        "is_multiple_documents": False
    }

async def format_multiple_documents_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format response for multiple documents"""
    formatted_documents = []
    
    for doc in response_data["documents"]:
        formatted_doc = await format_single_document(doc)
        formatted_doc["document_index"] = doc.get("document_index")
        formatted_doc["is_latest"] = doc.get("is_latest", False)
        formatted_documents.append(formatted_doc)
    
    return {
        "patient_name": response_data["patient_name"],
        "total_documents": response_data["total_documents"],
        "documents": formatted_documents,
        "is_multiple_documents": True
    }

async def format_single_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """Format a single document"""
    # Base document info
    response = {
        "document_id": document.get("id"),
        "patient_name": document.get("patientName"),
        "dob": document.get("dob").isoformat() if document.get("dob") else None,
        "doi": document.get("doi").isoformat() if document.get("doi") else None,
        "claim_number": document.get("claimNumber"),
        "status": document.get("status"),
        "brief_summary": document.get("briefSummary"),
        "gcs_file_link": document.get("gcsFileLink"),
        "created_at": document.get("createdAt").isoformat() if document.get("createdAt") else None,
        "updated_at": document.get("updatedAt").isoformat() if document.get("updatedAt") else None,
    }
    
    # Add summary snapshot data
    summary_snapshot = document.get("summarySnapshot")
    if summary_snapshot:
        response["summary_snapshot"] = {
            "diagnosis": summary_snapshot.get("dx"),
            "key_concern": summary_snapshot.get("keyConcern"),
            "next_step": summary_snapshot.get("nextStep")
        }
    else:
        response["summary_snapshot"] = None
    
    # Add what's new data
    whats_new = document.get("whatsNew")
    if whats_new:
        response["whats_new"] = {
            "diagnostic": whats_new.get("diagnostic"),
            "qme": whats_new.get("qme"),
            "ur_decision": whats_new.get("urDecision"),
            "legal": whats_new.get("legal")
        }
    else:
        response["whats_new"] = None
    
    # Add ADL data
    adl_data = document.get("adl")
    if adl_data:
        response["adl"] = {
            "adls_affected": adl_data.get("adlsAffected"),
            "work_restrictions": adl_data.get("workRestrictions")
        }
    else:
        response["adl"] = None
    
    # Add document summary
    doc_summary = document.get("documentSummary")
    if doc_summary:
        response["document_summary"] = {
            "type": doc_summary.get("type"),
            "date": doc_summary.get("date").isoformat() if doc_summary.get("date") else None,
            "summary": doc_summary.get("summary")
        }
    else:
        response["document_summary"] = None
    
    return response