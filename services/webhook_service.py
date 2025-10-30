# services/webhook_service.py
from fastapi import HTTPException
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from models.schemas import ExtractionResult
from services.database_service import get_database_service
from services.task_creation import TaskCreator
from services.resoning_agent import EnhancedReportAnalyzer
from utils.socket_manager import sio
from utils.logger import logger
from prisma import Prisma

class WebhookService:
    """
    Service class encapsulating the webhook processing logic.
    Handles document processing, lookup, status determination, and saving.
    """

    async def process_document_data(self, data: dict) -> dict:
        """
        Step 1: Extract, de-identify, analyze, and prepare initial document data.
        Handles PHI de-identification, LLM analysis, date reasoning, and summary generation.
        """
        logger.info(f"📥 Processing document data for: {data.get('document_id', 'unknown')}")

        if not data.get("result") or not data.get("filename") or not data.get("gcs_url"):
            raise HTTPException(status_code=400, detail="Missing required fields in webhook payload")

        result_data = data["result"]
        text = result_data.get("text", "")

        # Extract mode from payload (default to "wc")
        mode = data.get("mode", "wc")
        logger.info(f"📋 Document mode: {mode}")

        # Skip DLP de-identification; use text directly (note: ensure HIPAA compliance via other means)
        deidentified_text = text
        extracted_phi = {
            "patient_name": "",
            "claim_number": "",
            "dates": []
        }

        # Use enhanced analyzer with reasoning
        analyzer = EnhancedReportAnalyzer()
        document_analysis = analyzer.extract_document_data_with_reasoning(deidentified_text)
        logger.info(f"Document analysis (with reasoning): {document_analysis}")

        # Skip overrides with DLP values; rely on analyzer extraction

        # Enhanced date handling: Use reasoned dates as primary (no DLP fallback)
        has_date_reasoning = hasattr(document_analysis, 'date_reasoning') and document_analysis.date_reasoning is not None

        # Log reasoning results
        if has_date_reasoning:
            logger.info(f"🔍 Date reasoning completed:")
            logger.info(f"   - Reasoning: {document_analysis.date_reasoning.reasoning}")
            logger.info(f"   - Confidence: {document_analysis.date_reasoning.confidence_scores}")
            logger.info(f"   - Extracted dates: {document_analysis.date_reasoning.extracted_dates}")
        else:
            logger.info("ℹ️ No date reasoning available in document analysis")

        # Generate AI brief summary (using de-identified text)
        brief_summary = analyzer.generate_brief_summary(deidentified_text)

        # Enhanced date handling for queries
        dob = document_analysis.dob if document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else "Not specified"
        rd = document_analysis.rd if document_analysis.rd and str(document_analysis.rd).lower() != "not specified" else "Not specified"
        doi = document_analysis.doi if document_analysis.doi and str(document_analysis.doi).lower() != "not specified" else "Not specified"

        # Parse DOB for query
        dob_for_query = None
        if dob and dob.lower() != "not specified":
            try:
                dob_for_query = datetime.strptime(dob, "%Y-%m-%d")
            except ValueError:
                if has_date_reasoning:
                    for date_str in document_analysis.date_reasoning.extracted_dates:
                        try:
                            dob_for_query = datetime.strptime(date_str, "%Y-%m-%d")
                            logger.info(f"🔄 Used alternative date from reasoning for DOB: {date_str}")
                            break
                        except ValueError:
                            continue
                if not dob_for_query:
                    dob_for_query = None

        # Parse RD for DB
        rd_for_db = None
        if rd.lower() != "not specified":
            try:
                if '/' in rd:  # MM/DD format
                    month, day = rd.split('/')
                    year = datetime.now().year
                    full_date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif '-' in rd:  # YYYY-MM-DD format
                    full_date_str = rd
                else:
                    raise ValueError("Invalid date format")

                rd_for_db = datetime.strptime(full_date_str, "%Y-%m-%d")
                logger.debug(f"Parsed rd '{rd}' to datetime: {rd_for_db}")
            except (ValueError, AttributeError) as parse_err:
                logger.warning(f"Failed to parse rd '{rd}': {parse_err}; checking reasoning context")
                if has_date_reasoning:
                    for date_str in document_analysis.date_reasoning.extracted_dates:
                        try:
                            rd_for_db = datetime.strptime(date_str, "%Y-%m-%d")
                            logger.info(f"🔄 Used reasoning date for RD fallback: {date_str}")
                            break
                        except ValueError:
                            continue
                if not rd_for_db:
                    rd_for_db = None

        # Prepare initial fields for lookup
        patient_name_for_query = document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else None
        
        # ADD THESE MISSING FIELDS:
        has_patient_name = document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"
        has_claim_number = document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
        claim_number_for_query = document_analysis.claim_number if has_claim_number else None

        return {
            "document_analysis": document_analysis,
            "brief_summary": brief_summary,
            "extracted_phi": extracted_phi,
            "deidentified_text": deidentified_text,
            "has_date_reasoning": has_date_reasoning,
            "date_reasoning_data": {
                "reasoning": document_analysis.date_reasoning.reasoning if has_date_reasoning else "",
                "confidence_scores": document_analysis.date_reasoning.confidence_scores if has_date_reasoning else {},
                "extracted_dates": document_analysis.date_reasoning.extracted_dates if has_date_reasoning else [],
                "date_contexts": document_analysis.date_reasoning.date_contexts if has_date_reasoning else {}
            },
            "dob": dob,
            "rd": rd,
            "doi": doi,
            "dob_for_query": dob_for_query,
            "rd_for_db": rd_for_db,
            "patient_name_for_query": patient_name_for_query,
            "claim_number_for_query": claim_number_for_query,
            "has_claim_number": has_claim_number,
            "has_patient_name": has_patient_name,  # ADD THIS MISSING FIELD
            "physician_id": data.get("physician_id"),
            "blob_path": data.get("blob_path"),
            "filename": data["filename"],
            "gcs_url": data["gcs_url"],
            "file_size": data.get("file_size", 0),
            "mime_type": data.get("mime_type", "application/octet-stream"),
            "processing_time_ms": data.get("processing_time_ms", 0),
            "file_hash": data.get("file_hash"),
            "result_data": result_data,
            "user_id": data.get("user_id"),
            "document_id": data.get("document_id", "unknown"),
            "mode": mode  # Include mode in processed data
        }

    async def perform_patient_lookup(self, db_service, processed_data: dict, physician_id: str) -> dict:
        """
        Step 2: Perform patient/claim lookup, handle conflicts, and override missing fields.
        """
        logger.info(f"🔍 Performing patient lookup for physician_id: {physician_id}")

        lookup_data = await db_service.get_patient_claim_numbers(
            patient_name=processed_data["patient_name_for_query"],
            physicianId=physician_id,
            dob=processed_data["dob_for_query"],
            claim_number=processed_data["claim_number_for_query"]  # Prioritize claim if available
        )
        logger.info(f"Lookup data: {lookup_data}")

        # Check if this is first time document with only claim number (no previous documents found)
        is_first_time_claim_only = (
            processed_data["has_claim_number"]
            and not processed_data["has_patient_name"]
            and lookup_data
            and lookup_data.get("total_documents", 0) == 0
        )

        if is_first_time_claim_only:
            logger.info("✅ First time document with only claim number - no conflicts expected")

        # Check for conflicting claim numbers
        has_conflicting_claims = lookup_data.get("has_conflicting_claims", False) if lookup_data else False
        conflicting_claims_reason = None

        # Ignore conflicts for first-time claim-only documents
        if has_conflicting_claims and is_first_time_claim_only:
            has_conflicting_claims = False
            logger.info("🔄 Ignoring conflicts for first-time claim-only document")

        if has_conflicting_claims and not is_first_time_claim_only:
            conflicting_claims_reason = f"Multiple conflicting claim numbers found: {lookup_data.get('unique_valid_claims', [])}"
            logger.warning(f"⚠️ {conflicting_claims_reason}")

        # Override missing fields only if missing in document analysis; log mismatches if present
        document_analysis = processed_data["document_analysis"]
        if lookup_data and lookup_data.get("total_documents", 0) > 0 and (not has_conflicting_claims or is_first_time_claim_only):
            fetched_patient_name = lookup_data.get("patient_name")
            fetched_dob = lookup_data.get("dob")
            fetched_doi = lookup_data.get("doi")
            fetched_claim_number = lookup_data.get("claim_number")

            # Patient name: Override only if missing
            doc_patient_name = document_analysis.patient_name
            doc_name_missing = not doc_patient_name or str(doc_patient_name).lower() == "not specified"
            if doc_name_missing:
                if fetched_patient_name:
                    document_analysis.patient_name = fetched_patient_name
                    logger.info(f"🔄 Overrode patient_name from previous data: {fetched_patient_name}")
            else:
                # If present, check for match and log if mismatch (no override)
                if fetched_patient_name:
                    doc_name_norm = str(doc_patient_name).strip().lower()
                    fetched_name_norm = str(fetched_patient_name).strip().lower()
                    if doc_name_norm != fetched_name_norm:
                        logger.warning(f"⚠️ Patient name mismatch - document: '{doc_patient_name}', previous: '{fetched_patient_name}' (keeping document value)")

            # DOB: Override only if missing
            doc_dob = document_analysis.dob
            doc_dob_missing = not doc_dob or str(doc_dob).lower() == "not specified"
            if doc_dob_missing:
                if fetched_dob:
                    document_analysis.dob = fetched_dob
                    logger.info(f"🔄 Overrode DOB from previous data: {fetched_dob}")
            else:
                # If present, check for match and log if mismatch (no override)
                if fetched_dob:
                    doc_dob_norm = str(doc_dob).strip().lower()
                    fetched_dob_norm = str(fetched_dob).strip().lower()
                    if doc_dob_norm != fetched_dob_norm:
                        logger.warning(f"⚠️ DOB mismatch - document: '{doc_dob}', previous: '{fetched_dob}' (keeping document value)")

            # DOI: Override only if missing
            doc_doi = document_analysis.doi
            doc_doi_missing = not doc_doi or str(doc_doi).lower() == "not specified"
            if doc_doi_missing:
                if fetched_doi:
                    document_analysis.doi = fetched_doi
                    logger.info(f"🔄 Overrode DOI from previous data: {fetched_doi}")
            else:
                # If present, check for match and log if mismatch (no override)
                if fetched_doi:
                    doc_doi_str = str(doc_doi).strip()
                    fetched_doi_str = str(fetched_doi).strip()
                    if doc_doi_str != fetched_doi_str:
                        logger.warning(f"⚠️ DOI mismatch - document: '{doc_doi}', previous: '{fetched_doi}' (keeping document value)")

            # Claim number: Override only if missing
            doc_claim_number = document_analysis.claim_number
            doc_claim_missing = not doc_claim_number or str(doc_claim_number).lower() == "not specified"
            if doc_claim_missing:
                if fetched_claim_number:
                    document_analysis.claim_number = fetched_claim_number
                    logger.info(f"🔄 Overrode claim_number from previous data: {fetched_claim_number}")
            else:
                # If present, check for match and log if mismatch (no override)
                if fetched_claim_number:
                    doc_claim_norm = str(doc_claim_number).strip().lower()
                    fetched_claim_norm = str(fetched_claim_number).strip().lower()
                    if doc_claim_norm != fetched_claim_norm:
                        logger.warning(f"⚠️ Claim number mismatch - document: '{doc_claim_number}', previous: '{fetched_claim_number}' (keeping document value)")

        updated_claim_number_for_query = (
            document_analysis.claim_number
            if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
            else None
        )
        updated_patient_name_for_query = (
            document_analysis.patient_name
            if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"
            else "Unknown Patient"
        )

        # Re-check missing fields
        updated_required_fields = {
            "patient_name": document_analysis.patient_name,
            "dob": document_analysis.dob,
        }
        updated_missing_fields = [k for k, v in updated_required_fields.items() if not v or str(v).lower() == "not specified"]
        has_missing_required_fields = len(updated_missing_fields) > 0

        # ✅ NEW RULE: If document already has a valid claim number → don't treat missing fields or conflicts as failure
        if processed_data.get("has_claim_number"):
            logger.info("✅ Document has claim number — skipping failure conditions for missing fields or conflicts.")
            has_conflicting_claims = False
            has_missing_required_fields = False
            conflicting_claims_reason = None
            updated_missing_fields = []

        return {
            "lookup_data": lookup_data,
            "has_conflicting_claims": has_conflicting_claims,
            "conflicting_claims_reason": conflicting_claims_reason,
            "updated_missing_fields": updated_missing_fields,
            "has_missing_required_fields": has_missing_required_fields,
            "updated_claim_number_for_query": updated_claim_number_for_query,
            "updated_patient_name_for_query": updated_patient_name_for_query,
            "document_analysis": document_analysis,
            "is_first_time_claim_only": is_first_time_claim_only,
            "mode": processed_data.get("mode")  # Propagate mode
        }
    
    async def compare_and_determine_status(self, processed_data: dict, lookup_result: dict, db_service, physician_id: str) -> dict:
        """
        Step 3: Fetch previous documents, compare, and determine final status/reason.
        Now supports multiple body parts with separate summary snapshots.
        """
        document_analysis = lookup_result["document_analysis"]
        lookup_data = lookup_result["lookup_data"]
        is_first_time_claim_only = lookup_result.get("is_first_time_claim_only", False)
        mode = lookup_result.get("mode")

        # Always fetch all previous unverified documents for the patient
        db_response = await db_service.get_all_unverified_documents(
            patient_name=lookup_result["updated_patient_name_for_query"],
            physicianId=physician_id,
            claimNumber=lookup_result["updated_claim_number_for_query"],
            dob=processed_data["dob_for_query"]
        )

        previous_documents = db_response.get('documents', []) if db_response else []

        # Compare with previous documents using LLM - UPDATED: Pass raw deidentified_text instead of document_analysis
        analyzer = EnhancedReportAnalyzer()
        whats_new_data = analyzer.compare_with_previous_documents(
            processed_data["deidentified_text"],
            previous_documents
        )
        if whats_new_data is None or not isinstance(whats_new_data, dict):
            logger.warning(f"⚠️ Invalid whats_new data; using empty dict")
            whats_new_data = {}

        # Simplified claim logic
        claim_to_use = document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else "Not specified"
        pending_reason = None

        # Check for ambiguity only after lookup
        if lookup_data and lookup_data.get("total_documents", 0) > 1:
            logger.warning(f"⚠️ Multiple documents found via lookup ({lookup_data['total_documents']}); using primary values")

        # Determine status
        base_status = document_analysis.status
        
        if is_first_time_claim_only:
            logger.info("✅ First-time claim-only document - allowing despite missing patient name")
            document_status = base_status
            pending_reason = "First-time document with claim number only"
        elif lookup_result["has_missing_required_fields"]:
            document_status = "failed"
            pending_reason = f"Missing required fields after lookup: {', '.join(lookup_result['updated_missing_fields'])}"
        elif lookup_result["has_conflicting_claims"]:
            document_status = "failed"
            pending_reason = lookup_result["conflicting_claims_reason"]
        else:
            document_status = base_status

        # Prepare values for saving
        patient_name_to_use = document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else "Not specified"
        claim_to_save = claim_to_use if claim_to_use != "Not specified" else "Not specified"

        # For first-time claim-only documents, use a placeholder patient name
        if is_first_time_claim_only and patient_name_to_use == "Not specified":
            patient_name_to_use = f"Claim_{claim_to_save}_Patient"
            logger.info(f"🔄 Using placeholder patient name for first-time claim: {patient_name_to_use}")

        # CREATE MULTIPLE SUMMARY SNAPSHOTS FOR EACH BODY PART
        summary_snapshots = []
        
        # If we have detailed body parts analysis, create snapshot for each
        if hasattr(document_analysis, 'body_parts_analysis') and document_analysis.body_parts_analysis:
            logger.info(f"📊 Creating {len(document_analysis.body_parts_analysis)} summary snapshots for multiple body parts")
            
            for body_part_analysis in document_analysis.body_parts_analysis:
                snapshot = {
                    # Use snake_case for the application code (will be converted to camelCase for DB later)
                    "body_part": body_part_analysis.body_part,  # Keep snake_case for application code
                    "dx": body_part_analysis.diagnosis,
                    "key_concern": body_part_analysis.key_concern,
                    "next_step": body_part_analysis.extracted_recommendation or None,
                    "ur_decision": document_analysis.ur_decision or None,
                    "recommended": body_part_analysis.extracted_recommendation or None,
                    "ai_outcome": document_analysis.ai_outcome or None,
                    "consulting_doctor": document_analysis.consulting_doctor or None,
                    
                    # New fields
                    "key_findings": body_part_analysis.clinical_summary or None,
                    "treatment_approach": body_part_analysis.treatment_plan or None,
                    "clinical_summary": body_part_analysis.clinical_summary or None,
                    "referral_doctor": document_analysis.referral_doctor or None,
                }
                summary_snapshots.append(snapshot)
        else:
            # Fallback to single snapshot using primary body part
            logger.info("📊 Creating single summary snapshot (no multiple body parts analysis)")
            snapshot = {
                "body_part": document_analysis.body_part,
                "dx": document_analysis.diagnosis,
                "key_concern": document_analysis.key_concern,
                "next_step": document_analysis.extracted_recommendation or None,
                "ur_decision": document_analysis.ur_decision or None,
                "recommended": document_analysis.extracted_recommendation or None,
                "ai_outcome": document_analysis.ai_outcome or None,
                "consulting_doctor": document_analysis.consulting_doctor or None,
                
                # New fields
                "key_findings": document_analysis.diagnosis or None,
                "treatment_approach": document_analysis.extracted_recommendation or None,
                "clinical_summary": f"{document_analysis.diagnosis} - {document_analysis.key_concern}" or None,
                "referral_doctor": document_analysis.referral_doctor or None,
            }
            summary_snapshots.append(snapshot)

        # ADL data (can be shared across body parts or customized per body part)
        adl_data = {
            "adls_affected": document_analysis.adls_affected,
            "work_restrictions": document_analysis.work_restrictions
        }
        
        # For multiple body parts, you might want to customize ADL data per body part
        if len(summary_snapshots) > 1:
            logger.info("🔄 Multiple body parts detected - using shared ADL data")

        summary_text = " | ".join(document_analysis.summary_points) if document_analysis.summary_points else "No summary"
        document_summary = {
            "type": document_analysis.document_type,
            "created_at": datetime.now(),
            "summary": summary_text
        }

        return {
            "document_status": document_status,
            "pending_reason": pending_reason,
            "patient_name_to_use": patient_name_to_use,
            "claim_to_save": claim_to_save,
            "whats_new_data": whats_new_data,
            "summary_snapshots": summary_snapshots,
            "adl_data": adl_data,
            "document_summary": document_summary,
            "updated_missing_fields": lookup_result["updated_missing_fields"],
            "has_missing_required_fields": lookup_result["has_missing_required_fields"],
            "has_conflicting_claims": lookup_result["has_conflicting_claims"],
            "conflicting_claims_reason": lookup_result["conflicting_claims_reason"],
            "lookup_data": lookup_data,
            "previous_documents": previous_documents,
            "document_analysis": document_analysis,
            "is_first_time_claim_only": is_first_time_claim_only,
            "mode": mode,
            "has_multiple_body_parts": len(summary_snapshots) > 1
        }
    async def save_and_process_document(self, processed_data: dict, status_result: dict, data: dict, db_service) -> dict:
        document_analysis = status_result["document_analysis"]
        has_date_reasoning = processed_data["has_date_reasoning"]
        physician_id = processed_data["physician_id"]
        user_id = processed_data["user_id"]
        document_status = status_result["document_status"]
        pending_reason = status_result["pending_reason"]
        is_first_time_claim_only = status_result.get("is_first_time_claim_only", False)
        mode = status_result.get("mode")
        has_multiple_body_parts = status_result.get("has_multiple_body_parts", False)
        
        # Get summary snapshots (now a list)
        summary_snapshots = status_result["summary_snapshots"]

        # Check for existing document first
        # file_exists = await db_service.document_exists(
        #     processed_data["filename"],
        #     processed_data["file_size"]
        # )

        # if file_exists:
        #     logger.warning(f"⚠️ Document already exists: {processed_data['filename']}")
        #     user_friendly_msg = "Document already processed"

        #     # Emit skipped event
        #     emit_data = {
        #         'document_id': processed_data['document_id'],
        #         'filename': processed_data["filename"],
        #         'status': 'skipped',
        #         'reason': user_friendly_msg,
        #         "user_id": user_id,
        #         'blob_path': processed_data["blob_path"],
        #         'physician_id': physician_id
        #     }
        #     if user_id:
        #         await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
        #     else:
        #         await sio.emit('task_complete', emit_data)
        #     return {"status": "skipped", "document_id": processed_data.get('document_id'), "reason": "Document already processed", "blob_path": processed_data["blob_path"]}

        # UPDATED FAILURE LOGIC: Allow first-time claim-only documents to pass
        if document_status == "failed" and not is_first_time_claim_only:
            # Fail: save to FailDocs, no further processing
            fail_reason = pending_reason if pending_reason else f"Missing required fields: {', '.join(status_result['updated_missing_fields'])}"
            logger.warning(f"⚠️ Failing document {processed_data['filename']}: {fail_reason}")

            await db_service.save_fail_doc(
                reason=fail_reason,
                db=processed_data["dob"],
                doi=processed_data["doi"],
                claim_number=status_result["claim_to_save"],
                patient_name=status_result["patient_name_to_use"],
                document_text=processed_data["result_data"].get("text", ""),
                physician_id=physician_id,
                gcs_file_link=processed_data["gcs_url"],
                file_name=processed_data["filename"],
                file_hash=processed_data["file_hash"],
                blob_path=processed_data["blob_path"],
                mode=mode
            )

            # Emit failed event
            emit_data = {
                'document_id': processed_data['document_id'],
                'filename': processed_data["filename"],
                'status': 'failed',
                'missing_fields': status_result['updated_missing_fields'] if status_result['has_missing_required_fields'] else None,
                'pending_reason': pending_reason
            }
            if user_id:
                await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
            else:
                await sio.emit('task_complete', emit_data)

            logger.info(f"📡 Emitted 'task_complete' failed event: {emit_data}")

            return {
                "status": "failed",
                "document_id": processed_data.get('document_id'),
                "reason": fail_reason,
                "missing_fields": status_result['updated_missing_fields'] if status_result['has_missing_required_fields'] else None,
                "pending_reason": pending_reason
            }

        # Success: Proceed with saving (including first-time claim-only documents)
        logger.info(f"💾 Proceeding to save document {processed_data['filename']} - status: {document_status}")
        if has_multiple_body_parts:
            logger.info(f"📊 Saving {len(summary_snapshots)} body part snapshots")

        # Prepare dob_str for update
        dob_str = None
        updated_dob_for_query = None
        if document_analysis.dob and str(document_analysis.dob).lower() != "not specified":
            try:
                updated_dob_for_query = datetime.strptime(document_analysis.dob, "%Y-%m-%d")
                dob_str = updated_dob_for_query.strftime("%Y-%m-%d")
            except ValueError:
                updated_dob_for_query = processed_data["dob_for_query"]
                if updated_dob_for_query:
                    dob_str = updated_dob_for_query.strftime("%Y-%m-%d")

        # Mock ExtractionResult
        extraction_result = ExtractionResult(
            text=processed_data["result_data"].get("text", ""),
            pages=processed_data["result_data"].get("pages", 0),
            entities=processed_data["result_data"].get("entities", []),
            tables=processed_data["result_data"].get("tables", []),
            formFields=processed_data["result_data"].get("formFields", []),
            confidence=processed_data["result_data"].get("confidence", 0.0),
            success=processed_data["result_data"].get("success", False),
            gcs_file_link=processed_data["result_data"].get("gcs_file_link", processed_data["gcs_url"]),
            fileInfo=processed_data["result_data"].get("fileInfo", {}),
            summary=status_result["document_summary"]["summary"],
            comprehensive_analysis=processed_data["result_data"].get("comprehensive_analysis"),
            document_id=processed_data["result_data"].get("document_id", f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            date_reasoning=processed_data["date_reasoning_data"]
        )

        # Save the document to get the document_id
        # MODIFIED: Pass summary_snapshots (list) instead of summary_snapshot (single)
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=processed_data["filename"],
            file_size=processed_data["file_size"],
            mime_type=processed_data["mime_type"],
            processing_time_ms=processed_data["processing_time_ms"],
            blob_path=processed_data["blob_path"],
            file_hash=processed_data["file_hash"],
            gcs_file_link=processed_data["gcs_url"],
            patient_name=status_result["patient_name_to_use"],
            claim_number=status_result["claim_to_save"],
            dob=processed_data["dob"],
            doi=processed_data["doi"],
            status=document_status,
            brief_summary=processed_data["brief_summary"],
            summary_snapshots=summary_snapshots,  # Changed to plural
            whats_new=status_result["whats_new_data"],
            adl_data=status_result["adl_data"],
            document_summary=status_result["document_summary"],
            rd=processed_data["rd_for_db"],
            physician_id=physician_id,
            mode=mode,
            ur_denial_reason=document_analysis.ur_denial_reason
        )

        # ✅ DECREMENT PARSE COUNT AFTER SUCCESSFUL DOCUMENT SAVE
        parse_decremented = await db_service.decrement_parse_count(physician_id)
        if not parse_decremented:
            logger.warning(f"⚠️ Could not decrement parse count for physician {physician_id}")

        # AI Task Creation
        task_creator = TaskCreator()
        tasks = await task_creator.generate_tasks(document_analysis.dict(), processed_data["filename"])

        # Save tasks to DB
        prisma = Prisma()
        await prisma.connect()
        created_tasks = 0
        for task in tasks:
            try:
                # Map task fields
                mapped_task = {
                    "description": task.get("description"),
                    "department": task.get("department"),
                    "status": task.get("status", "Pending"),
                    "dueDate": None,
                    "patient": task.get("patient", "Unknown"),
                    "actions": task.get("actions") if isinstance(task.get("actions"), list) else [],
                    "sourceDocument": task.get("source_document") or task.get("sourceDocument") or processed_data.get("filename"),
                    "documentId": document_id,
                    "physicianId": physician_id,
                }

                # Normalize due date
                due_raw = task.get("due_date") or task.get("dueDate")
                if due_raw:
                    if isinstance(due_raw, str):
                        try:
                            mapped_task["dueDate"] = datetime.strptime(due_raw, "%Y-%m-%d")
                        except Exception:
                            mapped_task["dueDate"] = datetime.now() + timedelta(days=3)
                    else:
                        mapped_task["dueDate"] = due_raw

                await prisma.task.create(data=mapped_task)
                created_tasks += 1
            except Exception as task_err:
                logger.error(f"❌ Failed to create task for document {processed_data['filename']}: {task_err}", exc_info=True)
                continue

        logger.info(f"✅ {created_tasks} / {len(tasks)} tasks created for document {processed_data['filename']}")

        # Update previous documents' fields (skip for first-time claim-only as there are no previous)
        # FIXED: Only update if lookup found previous documents for this specific patient/claim (total_documents > 0)
        # This prevents updating docs for other patients when has_claim_number=True but no match
        total_previous_docs = status_result["lookup_data"].get("total_documents", 0) if status_result["lookup_data"] else 0
        should_update_previous = (
            document_status not in ["failed"] and
            total_previous_docs > 0 and  # Require confirmed previous docs for this patient
            not status_result["has_conflicting_claims"] and
            status_result["patient_name_to_use"] != "Not specified" and
            updated_dob_for_query is not None and
            not is_first_time_claim_only
        )

        if should_update_previous:
            updated_count = await db_service.update_previous_fields(
                patient_name=status_result["patient_name_to_use"],
                dob=dob_str,
                physician_id=physician_id,
                claim_number=status_result["claim_to_save"],
                doi=document_analysis.doi if document_analysis.doi and str(document_analysis.doi).lower() != "not specified" else None
            )
            logger.info(f"🔄 Updated {updated_count} previous documents' fields for patient '{status_result['patient_name_to_use']}' using new data")
        else:
            logger.info(f"ℹ️ Skipping previous update: status={document_status}, total_previous={total_previous_docs}, has_conflicts={status_result['has_conflicting_claims']}, patient={status_result['patient_name_to_use']}, has_dob={updated_dob_for_query is not None}, is_first_time_claim_only={is_first_time_claim_only}")

        logger.info(f"💾 Document saved via webhook with ID: {document_id}, status: {document_status}")

        # Emit success event - include parse count info
        emit_data = {
            'document_id': document_id,
            'filename': processed_data["filename"],
            'status': document_status,
            'missing_fields': status_result['updated_missing_fields'] if status_result['has_missing_required_fields'] else None,
            'pending_reason': pending_reason,
            'date_reasoning_confidence': processed_data["date_reasoning_data"]["confidence_scores"],
            'extracted_dates_count': len(processed_data["date_reasoning_data"]["extracted_dates"]),
            'lookup_used': status_result["lookup_data"].get("total_documents", 0) > 0,
            'fields_overridden_from_lookup': len(status_result['updated_missing_fields']) < len(processed_data.get("initial_missing_fields", [])),
            'is_first_time_claim_only': is_first_time_claim_only,
            'mode': mode,
            'parse_count_decremented': parse_decremented,  # Add this field
            'ur_denial_reason': document_analysis.ur_denial_reason or None,
            'body_parts_count': len(summary_snapshots),  # Add body parts count
            'has_multiple_body_parts': has_multiple_body_parts  # Add multiple body parts flag
        }
        if user_id:
            await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
        else:
            await sio.emit('task_complete', emit_data)

        logger.info(f"📡 Emitted 'task_complete' event: {emit_data}")

        return {
            "status": document_status,
            "document_id": document_id,
            "missing_fields": status_result['updated_missing_fields'] if status_result['has_missing_required_fields'] else None,
            "pending_reason": pending_reason,
            "is_first_time_claim_only": is_first_time_claim_only,
            "parse_count_decremented": parse_decremented,  # Add this field
            "date_reasoning_summary": {
                "used_reasoning": has_date_reasoning,
                "confidence_scores": processed_data["date_reasoning_data"]["confidence_scores"],
                "dates_extracted": len(processed_data["date_reasoning_data"]["extracted_dates"])
            },
            "lookup_summary": {
                "documents_found": status_result["lookup_data"].get("total_documents", 0),
                "has_conflicting_claims": status_result["has_conflicting_claims"],
                "unique_valid_claims": status_result["lookup_data"].get("unique_valid_claims", []) if status_result["lookup_data"] else [],
                "fields_fetched": {
                    "patient_name": bool(status_result["lookup_data"].get("patient_name")),
                    "dob": bool(status_result["lookup_data"].get("dob")),
                    "doi": bool(status_result["lookup_data"].get("doi")),
                    "claim_number": bool(status_result["lookup_data"].get("claim_number"))
                }
            },
            "mode": mode,
            "ur_denial_reason": document_analysis.ur_denial_reason or None,
            "body_parts_analysis": {
                "total_body_parts": len(summary_snapshots),
                "has_multiple_body_parts": has_multiple_body_parts,
                "body_parts": [snapshot["body_part"] for snapshot in summary_snapshots]
            }
        }
    
    async def handle_webhook(self, data: dict, db_service) -> dict:
        """
        Orchestrates the full webhook processing pipeline using the 4 steps.
        """
        # Step 1: Process document data
        processed_data = await self.process_document_data(data)

        # Extract physician_id
        physician_id = processed_data["physician_id"]

        # Step 2: Perform patient lookup
        lookup_result = await self.perform_patient_lookup(db_service, processed_data, physician_id)

        # Step 3: Compare and determine status
        status_result = await self.compare_and_determine_status(processed_data, lookup_result, db_service, physician_id)

        # Step 4: Save and process
        result = await self.save_and_process_document(processed_data, status_result, data, db_service)

        return result

    async def update_fail_document(self, fail_doc: Any, updated_fields: dict, user_id: str = None, db_service: Any = None) -> dict:
        """
        Handles updating and processing a failed document using webhook-like logic.
        Overrides fields from updated_fields, processes via service steps, saves, creates tasks, updates previous, and deletes fail_doc.
        """
        # Use updated values if provided, otherwise fallback to fail_doc values
        document_text = updated_fields.get("document_text") or fail_doc.documentText
        dob_str = updated_fields.get("dob") or fail_doc.db
        doi = updated_fields.get("doi") or fail_doc.doi
        claim_number = updated_fields.get("claim_number") or fail_doc.claimNumber
        patient_name = updated_fields.get("patient_name") or fail_doc.patientName
        physician_id = fail_doc.physicianId
        filename = fail_doc.fileName
        gcs_url = fail_doc.gcsFileLink
        blob_path = fail_doc.blobPath
        file_hash = fail_doc.fileHash

        # Construct webhook-like data
        result_data = {
            "text": document_text,
            "pages": 0,
            "entities": [],
            "tables": [],
            "formFields": [],
            "confidence": 0.0,
            "success": False,
            "gcs_file_link": gcs_url,
            "fileInfo": {},
            "comprehensive_analysis": None,
            "document_id": f"update_fail_{fail_doc.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        webhook_data = {
            "result": result_data,
            "filename": filename,
            "gcs_url": gcs_url,
            "blob_path": blob_path,
            "physician_id": physician_id,
            "file_size": 0,
            "mime_type": "application/octet-stream",
            "processing_time_ms": 0,
            "file_hash": file_hash,
            "user_id": user_id,
            "document_id": str(fail_doc.id)
        }

        # Step 1: Process document data
        processed_data = await self.process_document_data(webhook_data)

        # Override with updated fields
        document_analysis = processed_data["document_analysis"]
        if updated_fields.get("patient_name") and str(updated_fields["patient_name"]).lower() != "not specified":
            document_analysis.patient_name = updated_fields["patient_name"]
        if updated_fields.get("dob") and str(updated_fields["dob"]).lower() != "not specified":
            document_analysis.dob = updated_fields["dob"]
        if updated_fields.get("doi") and str(updated_fields["doi"]).lower() != "not specified":
            document_analysis.doi = updated_fields["doi"]
        if updated_fields.get("claim_number") and str(updated_fields["claim_number"]).lower() != "not specified":
            document_analysis.claim_number = updated_fields["claim_number"]

        # Re-parse dates after overrides
        dob = document_analysis.dob if document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else "Not specified"
        rd = document_analysis.rd if document_analysis.rd and str(document_analysis.rd).lower() != "not specified" else "Not specified"
        doi = document_analysis.doi if document_analysis.doi and str(document_analysis.doi).lower() != "not specified" else "Not specified"

        # Parse DOB for query
        dob_for_query = None
        if dob and dob.lower() != "not specified":
            try:
                dob_for_query = datetime.strptime(dob, "%Y-%m-%d")
            except ValueError:
                has_date_reasoning = processed_data["has_date_reasoning"]
                if has_date_reasoning:
                    for date_str in document_analysis.date_reasoning.extracted_dates:
                        try:
                            dob_for_query = datetime.strptime(date_str, "%Y-%m-%d")
                            logger.info(f"🔄 Used alternative date from reasoning for DOB: {date_str}")
                            break
                        except ValueError:
                            continue

        # Parse RD for DB (using 2025 as current year per route spec)
        rd_for_db = None
        if rd.lower() != "not specified":
            try:
                if '/' in rd:
                    month, day = rd.split('/')
                    year = 2025
                    full_date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif '-' in rd:
                    full_date_str = rd
                else:
                    raise ValueError("Invalid date format")
                rd_for_db = datetime.strptime(full_date_str, "%Y-%m-%d")
            except (ValueError, AttributeError):
                has_date_reasoning = processed_data["has_date_reasoning"]
                if has_date_reasoning:
                    for date_str in document_analysis.date_reasoning.extracted_dates:
                        try:
                            rd_for_db = datetime.strptime(date_str, "%Y-%m-%d")
                            logger.info(f"🔄 Used reasoning date for RD fallback: {date_str}")
                            break
                        except ValueError:
                            continue

        # Update processed_data
        processed_data.update({
            "dob": dob,
            "doi": doi,
            "dob_for_query": dob_for_query,
            "rd_for_db": rd_for_db
        })

        # Ensure lookup-related processed fields reflect any overrides provided in updated_fields
        # so that perform_patient_lookup uses the new claim/patient values
        processed_data["claim_number_for_query"] = (
            document_analysis.claim_number
            if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
            else None
        )
        processed_data["has_claim_number"] = bool(processed_data["claim_number_for_query"])
        processed_data["patient_name_for_query"] = (
            document_analysis.patient_name
            if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"
            else None
        )
        processed_data["has_patient_name"] = bool(processed_data["patient_name_for_query"])

        # Step 2: Perform patient lookup
        lookup_result = await self.perform_patient_lookup(db_service, processed_data, physician_id)

        # Step 3: Compare and determine status
        status_result = await self.compare_and_determine_status(processed_data, lookup_result, db_service, physician_id)

        # Integrate route-specific claim logic
        claim_numbers_response = await db_service.get_patient_claim_numbers(
            patient_name=lookup_result["updated_patient_name_for_query"],
            physicianId=physician_id,
            dob=processed_data["dob_for_query"]
        )
        claim_numbers_list = claim_numbers_response.get('claim_numbers', []) if isinstance(claim_numbers_response, dict) else claim_numbers_response if isinstance(claim_numbers_response, list) else []
        claim_numbers_list = list(set(claim_numbers_list))
        valid_claims_list = [c for c in claim_numbers_list if c and str(c).lower() != 'not specified']
        has_new_claim = document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
        previous_docs_count = len(lookup_result["lookup_data"].get("documents", []))

        claim_to_use = None
        claim_pending_reason = None
        if has_new_claim:
            claim_to_use = document_analysis.claim_number
            logger.info(f"ℹ️ Using extracted claim '{claim_to_use}' for patient '{document_analysis.patient_name}'")
        else:
            if previous_docs_count == 0:
                claim_to_use = "Not specified"
                logger.info(f"ℹ️ First document for patient '{document_analysis.patient_name}': No claim specified, proceeding as OK")
            elif len(valid_claims_list) == 0 and previous_docs_count > 0:
                claim_to_use = None
                claim_pending_reason = "No claim number specified and previous documents exist without valid claim"
                logger.warning(f"⚠️ {claim_pending_reason} for file {filename}")
            elif len(valid_claims_list) == 1:
                claim_to_use = valid_claims_list[0]
                logger.info(f"ℹ️ Using single previous valid claim '{claim_to_use}' for patient '{document_analysis.patient_name}'")
            else:
                claim_to_use = None
                claim_pending_reason = "No claim number specified and multiple previous valid claims found"
                logger.warning(f"⚠️ {claim_pending_reason} for file {filename}")

        # Update status_result based on claim logic
        if claim_to_use is None:
            status_result["document_status"] = "failed"
            status_result["pending_reason"] = claim_pending_reason or status_result["pending_reason"]
            status_result["claim_to_save"] = "Not specified"
        else:
            status_result["claim_to_save"] = claim_to_use
            # Override to success if claim resolved, unless other issues
            pr = status_result.get("pending_reason") or ""
            if status_result.get("document_status") == "failed" and (
                "Missing required fields" in pr or
                "Multiple conflicting claim numbers" in pr
            ):
                # If the pending_reason is a textual match (even with appended lists), accept the provided claim
                status_result["document_status"] = "success"
                status_result["pending_reason"] = None

        document_status = status_result["document_status"]
        pending_reason = status_result["pending_reason"]

        # Early return on failure
        if document_status == "failed":
            # Emit failed event
            emit_data = {
                'document_id': str(fail_doc.id),
                'filename': filename,
                'status': 'failed',
                'pending_reason': pending_reason
            }
            if user_id:
                await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
            else:
                await sio.emit('task_complete', emit_data)

            logger.info(f"📡 Emitted 'task_complete' failed event from update: {emit_data}")
            return {
                "status": "failed",
                "document_id": str(fail_doc.id),
                "reason": pending_reason,
                "pending_reason": pending_reason
            }

        # Step 4: Save and process (service handles saving, tasks, previous updates, emit)
        save_result = await self.save_and_process_document(processed_data, status_result, webhook_data, db_service)

        # Additional: Update previous claims if new claim provided
        dob_str_for_query = None
        if processed_data["dob_for_query"]:
            dob_str_for_query = processed_data["dob_for_query"].strftime("%Y-%m-%d")

        should_update_previous_claims = (
            has_new_claim and
            status_result["patient_name_to_use"] != "Not specified" and
            dob_str_for_query is not None
        )

        if should_update_previous_claims:
            db_response = await db_service.get_all_unverified_documents(
                patient_name=status_result["patient_name_to_use"],
                physicianId=physician_id,
                claimNumber=None,
                dob=processed_data["dob_for_query"]
            )
            previous_documents = db_response.get('documents', []) if db_response else []

            has_null_claim_docs = any(
                doc.get('claim_number') is None or str(doc.get('claim_number', '')).lower() == 'not specified'
                for doc in previous_documents
            )

            if has_null_claim_docs:
                await db_service.update_previous_fields(
                    patient_name=status_result["patient_name_to_use"],
                    dob=dob_str_for_query,
                    physician_id=physician_id,
                    claim_number=claim_to_use
                )
                logger.info(f"🔄 Updated previous documents' claim numbers for patient '{status_result['patient_name_to_use']}' using claim '{claim_to_use}'")
            else:
                logger.info(f"ℹ️ No previous documents with null claims for patient '{status_result['patient_name_to_use']}; skipping update")
        else:
            logger.info(f"ℹ️ Skipping previous claim update: has_new_claim={has_new_claim}, patient={status_result['patient_name_to_use']}, has_dob={dob_str_for_query is not None}")

        # Delete the FailDoc
        await db_service.delete_fail_doc(fail_doc.id)
        logger.info(f"🗑️ Deleted fail doc {fail_doc.id} after successful update")

        # Emit success event (service already emits, but ensure)
        emit_data = {
            'document_id': save_result['document_id'],
            'filename': filename,
            'status': document_status,
            'pending_reason': pending_reason
        }
        if user_id:
            await sio.emit('task_complete', emit_data, room=f"user_{user_id}")
        else:
            await sio.emit('task_complete', emit_data)

        logger.info(f"📡 Emitted 'task_complete' event from update: {emit_data}")

        return save_result