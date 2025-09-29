from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from pathlib import Path
import traceback
import requests
import json

PLACEHOLDER_TITLES = {None, "", "untitled", "unknown", "n/a", "na", "tbd", "untitled document"}


def normalize_title(candidate: Any) -> Optional[str]:
    if not isinstance(candidate, str):
        return None
    trimmed = candidate.strip()
    if not trimmed:
        return None
    if trimmed.lower() in PLACEHOLDER_TITLES:
        return None
    return trimmed


def extract_decision_date(text: str) -> Optional[str]:
    match = re.search(r"Decision Date\s*[:\-]?\s*([0-9]{2}[\-/][0-9]{2}[\-/][0-9]{4})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    generic = re.search(r"(0[1-9]|1[0-2])[\-/](0[1-9]|[12][0-9]|3[01])[\-/](20\d{2})", text)
    if generic:
        return generic.group(0)
    return None


def extract_line_by_keywords(lines: List[str], keywords: List[str]) -> Optional[str]:
    for line in lines:
        lower = line.lower()
        if all(keyword in lower for keyword in keywords):
            return line
    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            return line
    return None


def extract_reviewer_line(lines: List[str]) -> Optional[str]:
    credential_pattern = re.compile(r"\b(MD|DO|LVN|RN|NP|PA|Physician|Doctor|Director|Nurse)\b", re.IGNORECASE)
    for line in lines:
        if credential_pattern.search(line):
            return line
    return None


def extract_next_steps_line(lines: List[str]) -> Optional[str]:
    next_step_keywords = ["contact", "questions", "arrange", "reimbursement", "follow", "message", "call"]
    for line in lines:
        if any(keyword in line.lower() for keyword in next_step_keywords):
            return line
    return None


def shorten(text: str, limit: int = 180) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_structured_summary(
    summary_items: Optional[List[str]],
    detected_type: Optional[str],
    analysis,
    document_text: str
) -> List[str]:
    existing = summary_items or []
    label_map: Dict[str, str] = {}
    for item in existing:
        label, detail = split_summary_line(item)
        label_map[label.lower()] = detail

    patient = getattr(analysis.report_json, "patient_name", None) if analysis and analysis.report_json else None
    claim = getattr(analysis.report_json, "claim_no", None) if analysis and analysis.report_json else None
    doc_type = normalize_title(detected_type) or normalize_title(analysis.report_json.report_title if analysis and analysis.report_json else None) or "Medical Document"

    lines = [line.strip() for line in document_text.splitlines() if line.strip()]
    decision_date = extract_decision_date(document_text)
    treatment_detail = label_map.get("treatment decision") or extract_line_by_keywords(lines, ["treatment", "description"])
    if treatment_detail and ":" in treatment_detail:
        treatment_detail = treatment_detail.split(":", 1)[1].strip()
    if not treatment_detail:
        treatment_detail = extract_line_by_keywords(lines, ["approved", "authorization"]) or "Decision noted; specifics not detailed"

    reviewer_detail = label_map.get("reviewer") or extract_reviewer_line(lines) or "Reviewer not listed"
    next_steps_detail = label_map.get("next steps") or extract_next_steps_line(lines) or "None stated"

    overview_detail = label_map.get("clinical overview")
    if not overview_detail or "document" in overview_detail.lower():
        components = [doc_type]
        if decision_date:
            components.append(f"decision dated {decision_date}")
        if patient:
            components.append(f"for {patient}")
        if claim:
            components.append(f"claim {claim}")
        overview_detail = " | ".join(components)

    summary_lines = [
        f"Clinical Overview — {shorten(overview_detail)}",
        f"Treatment Decision — {shorten(treatment_detail)}",
        f"Reviewer — {shorten(reviewer_detail)}",
        f"Next Steps — {shorten(next_steps_detail)}"
    ]

    return summary_lines


def split_summary_line(line: str) -> Tuple[str, str]:
    if "—" in line:
        parts = line.split("—", 1)
    elif ":" in line:
        parts = line.split(":", 1)
    else:
        return "Update", line.strip()
    label = parts[0].strip(" :") or "Update"
    detail = parts[1].strip()
    return label, detail


def build_arrow_last_changes(
    document_type: Optional[str],
    summary_lines: Optional[List[str]],
    analysis,
    raw_changes: Optional[List[str]] = None
) -> List[str]:
    arrow_lines: List[str] = []
    seen: set = set()

    def add_line(label: str, detail: str):
        if not detail:
            return
        formatted = f"{label.strip()} ---> {detail.strip()}"
        if formatted not in seen:
            arrow_lines.append(formatted)
            seen.add(formatted)

    normalized_doc_type = normalize_title(document_type) or document_type
    if normalized_doc_type:
        add_line("Document Type", normalized_doc_type)

    summary_lines = summary_lines or []
    for line in summary_lines:
        label, detail = split_summary_line(line)
        normalized_label = label.lower()
        if normalized_label in {"clinical overview", "overview"}:
            add_line("Overview", detail)
        elif normalized_label in {"treatment decision", "treatment"}:
            add_line("Treatment", detail)
        elif normalized_label in {"reviewer", "physician", "provider"}:
            add_line("Reviewed By", detail)
        elif normalized_label in {"next steps", "next step"}:
            add_line("Next Step", detail)
        else:
            add_line(label.title(), detail)

    raw_changes = raw_changes or []
    for entry in raw_changes:
        cleaned = entry.split(":", 1)[-1].strip() if ":" in entry else entry.strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if any(keyword in lower for keyword in ["approve", "authorization", "medication", "dosage", "treatment"]):
            add_line("Change", cleaned)
        elif any(keyword in lower for keyword in ["review", "exam", "physician", "doctor", "nurse"]):
            add_line("Reviewer", cleaned)
        else:
            add_line("Update", cleaned)

    if analysis and analysis.report_json:
        patient = getattr(analysis.report_json, "patient_name", None)
        claim = getattr(analysis.report_json, "claim_no", None)
        status = getattr(analysis.report_json, "status", None)
        if patient:
            add_line("Patient", patient)
        if claim:
            add_line("Claim", claim)
        if status and status.lower() not in {"", "normal"}:
            add_line("Status", status.title())

    fallback_index = 1
    while len(arrow_lines) < 3:
        add_line("Update", f"No additional detail provided ({fallback_index})")
        fallback_index += 1

    return arrow_lines[:6]

from models.schemas import ExtractionResult
from services.document_ai_service import get_document_ai_processor
from services.file_service import FileService
from services.document_converter import DocumentConverter
from config.settings import CONFIG
from utils.logger import logger
from services.report_analyzer import ReportAnalyzer
from services.database_service import get_database_service

from config.celery_config import app as celery_app

# New deterministic services
from services.rule_engine import RuleEngine
from services.review_service import ReviewService

router = APIRouter()
rule_engine = RuleEngine()
review_service = ReviewService()


@router.post("/webhook/save-document")
async def save_document_webhook(request: Request):
    """
    Webhook endpoint to save document analysis to the database and compute last_changes.
    Now: deterministic alerts/actions are generated by RuleEngine (no AI alerts persisted).
    """
    try:
        data = await request.json()
        logger.info(f"📥 Webhook received for document save: {data}")

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
            comprehensive_analysis=None,
            document_id=result_data.get("document_id", f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )

        logger.info("💾 Preparing document analysis save for webhook payload")
        # If comprehensive_analysis included in payload, rehydrate Pydantic model
        if result_data.get("comprehensive_analysis"):
            try:
                ca = result_data.get("comprehensive_analysis")
                # If LLM returned analysis, we accept it (but we will not persist LLM alerts)
                analyzer = ReportAnalyzer()
                # The analyzer.validate_analysis_data will ignore LLM alerts and return a ComprehensiveAnalysis
                comprehensive_analysis = analyzer.validate_analysis_data(ca)
                result.comprehensive_analysis = comprehensive_analysis
            except Exception as e:
                logger.warning(f"⚠️ Could not parse comprehensive_analysis from payload: {str(e)}")
                result.comprehensive_analysis = None

        # Compute deterministic last_changes, alerts, actions, and review tickets
        last_changes = None
        generated_alerts = []
        generated_actions = []
        review_tickets = []

        if result.comprehensive_analysis:
            try:
                # Fetch database service and previous document for this patient (if available)
                db_service = await get_database_service()
                patient_name = result.comprehensive_analysis.report_json.patient_name if result.comprehensive_analysis.report_json else None

                previous_document = None
                if patient_name:
                    previous_document = await db_service.get_last_document_for_patient(patient_name)

                # Deterministic "What's New"
                prev_summary = previous_document.get("summary", []) if previous_document else []
                curr_summary = result.comprehensive_analysis.summary or []
                last_changes = rule_engine.compute_whats_new(prev_summary, curr_summary)
                logger.info(f"🔄 Generated deterministic last_changes for patient: {patient_name}")
                logger.info(
                    "🧾 last_changes type=%s | preview=%s",
                    type(last_changes),
                    str(last_changes)[:500]
                )

                # Deterministic alerts & actions (no AI-generated alerts persisted)
                generated_alerts = rule_engine.generate_alerts(result.comprehensive_analysis)
                compliance_nudges = rule_engine.generate_compliance_nudges(result.comprehensive_analysis)
                referrals = rule_engine.generate_referrals(result.comprehensive_analysis)
                generated_actions = rule_engine.generate_actions(result.comprehensive_analysis, previous_document)

                # fallback: ensure at least one deterministic artifact exists so nothing falls through
                if not (generated_alerts or compliance_nudges or referrals):
                    now = datetime.utcnow().date().isoformat()
                    generated_alerts.append({
                        "alert_type": "Manual Review",
                        "title": "No deterministic rule matched - manual review required",
                        "date": now,
                        "status": "normal",
                        "source": "rule_engine.fallback_manual_review",
                        "rule_id": "RE_FALLBACK"
                    })
                logger.info(
                    "🚨 Deterministic alerts generated=%d | type=%s",
                    len(generated_alerts),
                    type(generated_alerts)
                )
                logger.info(
                    "🛠️ Deterministic actions generated=%d | type=%s",
                    len(generated_actions),
                    type(generated_actions)
                )

                # Review tickets (confidence gating, missing fields)
                review_tickets = review_service.generate_review_tickets(result, result.comprehensive_analysis)
                if review_tickets:
                    logger.info(f"📥 Generated {len(review_tickets)} review tickets for document")
                logger.info(
                    "🎫 Review tickets type=%s | preview=%s",
                    type(review_tickets),
                    str(review_tickets)[:500]
                )

            except Exception as e:
                logger.error(f"❌ Failed to compute deterministic artifacts: {str(e)}")
                last_changes = f"last_changes computation failed: {str(e)}"
                logger.info(
                    "🧾 last_changes (error path) type=%s | preview=%s",
                    type(last_changes),
                    str(last_changes)[:500]
                )

        else:
            logger.warning("⚠️ No comprehensive_analysis provided in webhook payload")

        logger.info(
            "📦 Payload snapshot before DB save | last_changes=%s | alerts=%s | actions=%s | review_tickets=%s",
            type(last_changes),
            type(generated_alerts),
            type(generated_actions),
            type(review_tickets)
        )

        summary_lines = result.comprehensive_analysis.summary if result.comprehensive_analysis else []
        raw_changes_input: List[str] = []
        if isinstance(last_changes, list):
            raw_changes_input = last_changes
        elif isinstance(last_changes, str) and last_changes:
            raw_changes_input = [last_changes]

        formatted_last_changes = build_arrow_last_changes(
            document_type=data.get("document_type"),
            summary_lines=summary_lines,
            analysis=result.comprehensive_analysis,
            raw_changes=raw_changes_input
        )
        last_changes = formatted_last_changes

        logger.info("➡️ Last changes formatted for storage: %s", last_changes)

        resolved_report_title = None
        if result.comprehensive_analysis and result.comprehensive_analysis.report_json:
            resolved_report_title = normalize_title(result.comprehensive_analysis.report_json.report_title)

        if not resolved_report_title:
            resolved_report_title = normalize_title(data.get("document_type"))

        if not resolved_report_title:
            resolved_report_title = normalize_title(data.get("report_title"))

        if not resolved_report_title and isinstance(result.fileInfo, dict):
            resolved_report_title = normalize_title(result.fileInfo.get("originalName"))

        if not resolved_report_title:
            resolved_report_title = normalize_title(data.get("filename")) or "Unknown Document"

        logger.info(
            "🧾 Resolved report title for persistence: %s | document_type hint: %s",
            resolved_report_title,
            data.get("document_type")
        )

        # Save to database (include deterministic alerts/actions/review tickets)
        db_service = await get_database_service()
        document_id = await db_service.save_document_analysis(
            extraction_result=result,
            report_title=resolved_report_title,
            file_name=data["filename"],
            file_size=data.get("file_size", 0),
            mime_type=data.get("mime_type", "application/octet-stream"),
            processing_time_ms=data.get("processing_time_ms", 0),
            gcs_file_link=data["gcs_url"],
            last_changes=last_changes,
            alerts=generated_alerts,
            compliance_nudges=compliance_nudges,
            referrals=referrals,
            actions=generated_actions,
            review_tickets=review_tickets
        )

        logger.info(f"💾 Document saved via webhook with ID: {document_id}")
        return {
            "status": "success",
            "document_id": document_id,
            "alerts_created": len(generated_alerts),
            "actions_created": len(generated_actions),
            "review_tickets": len(review_tickets)
        }

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
        report_title = original_filename
        detected_type = None
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
                logger.info(f"✅ Comprehensive analysis obtained from LLM assistant: {comprehensive_analysis}")

                candidate_title = None
                if comprehensive_analysis and comprehensive_analysis.report_json:
                    candidate_title = normalize_title(comprehensive_analysis.report_json.report_title)

                if not candidate_title:
                    candidate_title = normalize_title(detected_type) or detected_type

                if not candidate_title:
                    candidate_title = normalize_title(original_filename) or original_filename

                if comprehensive_analysis and comprehensive_analysis.report_json:
                    comprehensive_analysis.report_json.report_title = candidate_title

                report_title = candidate_title or original_filename

                # Log patient info (no database lookup in task)
                patient_name = None
                if comprehensive_analysis and comprehensive_analysis.report_json:
                    patient_name = comprehensive_analysis.report_json.patient_name
                    logger.info(f"👤 Patient identified: {patient_name}")
                else:
                    logger.warning("⚠️ No patient name extracted")

                summary_parts = build_structured_summary(
                    summary_items=comprehensive_analysis.summary,
                    detected_type=detected_type,
                    analysis=comprehensive_analysis,
                    document_text=result.text
                )
                comprehensive_analysis.summary = summary_parts
                result.summary = " | ".join(summary_parts)

                logger.info("📋 Document Analysis Summary:")
                logger.info(f"   📄 Type: {detected_type}")
                logger.info(f"   👤 Patient: {patient_name or 'Unknown'}")
                logger.info(f"   📑 Title: {report_title}")
                logger.info(f"   📝 Summary: {len(summary_parts)} key points extracted")

                if comprehensive_analysis.work_status_alert:
                    logger.info(f"   🚨 Alerts: {len(comprehensive_analysis.work_status_alert)} generated")

                logger.info("✅ Comprehensive analysis completed")

            except Exception as e:
                logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
                analyzer = ReportAnalyzer()
                try:
                    detected_type = analyzer.detect_document_type_preview(result.text)
                    result.summary = f"Document Type: {detected_type} - Processing completed with limited analysis due to: {str(e)}"
                    logger.info(f"🔄 Fallback: Document type detected as {detected_type}")
                    report_title = detected_type or report_title
                except Exception as fallback_e:
                    result.summary = f"Document processed successfully but analysis encountered errors: {str(fallback_e)}"
                    report_title = report_title or original_filename
                result.comprehensive_analysis = None
        else:
            logger.warning("⚠️ No text extracted from document")
            result.summary = "Document processed but no readable text content was extracted"
            result.comprehensive_analysis = None
            report_title = report_title or original_filename
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"⏱️ Total processing time: {processing_time:.0f}ms")
        
        # Assign a temporary document ID
        result.document_id = f"celery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare payload for webhook (remove last_changes)
        webhook_payload = {
            "result": result.dict(),
            "report_title": report_title,
            "document_type": detected_type,
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