"""
OPTIMIZED Webhook Service - Clean Version with Redis Caching (No Duplication)
"""
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any,List
from models.schemas import ExtractionResult
from models.data_models import DocumentAnalysis
from helpers.helpers import clean_name_string
from services.database_service import get_database_service
from services.report_analyzer import ReportAnalyzer
from services.task_creation import TaskCreator
from services.patient_lookup_service import EnhancedPatientLookup
from utils.multi_report_detector import get_multi_report_detector
from utils.document_detector import detect_document_type
from utils.logger import logger
from prisma import Prisma
from concurrent.futures import ThreadPoolExecutor
import asyncio
import re
import json
# Import refactored service functions
from services.document_save_service import save_document as save_document_external
from services.fail_document_service import update_fail_document as update_fail_doc_service, update_multiple_fail_documents, generate_concise_brief_summary
# Dedicated thread pool for LLM operations (shared across all WebhookService instances)
LLM_EXECUTOR = ThreadPoolExecutor(max_workers=10)

class WebhookService:
    """
    Clean Webhook Service with essential features:
    - ReportAnalyzer for summaries
    - Patient lookup (no duplicate checking)
    - Task generation with conditions
    - Mode-aware processing (WC/GM)
    """
    
    def __init__(self, redis_client=None):
        logger.info("✅ WebhookService initialized")
        self.redis_client = redis_client
        self.patient_lookup = EnhancedPatientLookup(redis_client=redis_client)
    
    def _generate_document_filename(self, patient_name: str, document_type: str, report_date, original_filename: str) -> str:
        """
        Generate a structured filename in format: patientName_typeOfReport_dateOfReport.ext
        
        Args:
            patient_name: Patient's name
            document_type: Type of document/report
            report_date: Report date (datetime object or string)
            original_filename: Original filename to extract extension from
            
        Returns:
            Formatted filename string
        """
        import os
        
        # Get file extension from original filename
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = ".pdf"  # Default extension
        
        # Sanitize patient name (remove special characters, replace spaces with underscores)
        sanitized_patient = "Unknown"
        if patient_name and str(patient_name).lower() not in ["not specified", "unknown", "none", ""]:
            # Remove special characters and normalize
            sanitized_patient = re.sub(r'[^\w\s-]', '', str(patient_name))
            sanitized_patient = re.sub(r'\s+', '_', sanitized_patient.strip())
            sanitized_patient = sanitized_patient[:50]  # Limit length
        
        # Sanitize document type
        sanitized_type = "Document"
        if document_type and str(document_type).lower() not in ["not specified", "unknown", "none", ""]:
            sanitized_type = re.sub(r'[^\w\s-]', '', str(document_type))
            sanitized_type = re.sub(r'\s+', '_', sanitized_type.strip())
            sanitized_type = sanitized_type[:30]  # Limit length
        
        # Format report date
        date_str = "NoDate"
        if report_date:
            try:
                if isinstance(report_date, datetime):
                    date_str = report_date.strftime("%Y-%m-%d")
                elif isinstance(report_date, str) and report_date.lower() not in ["not specified", "unknown", "none", ""]:
                    # Try to parse and reformat the date string
                    for fmt in ["%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%m/%d/%y", "%d-%m-%Y", "%d/%m/%Y"]:
                        try:
                            parsed_date = datetime.strptime(report_date.strip(), fmt)
                            date_str = parsed_date.strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
                    else:
                        # If parsing fails, use sanitized original string
                        date_str = re.sub(r'[^\w-]', '-', str(report_date).strip())[:10]
            except Exception as e:
                logger.warning(f"⚠️ Could not format report date: {e}")
                date_str = "NoDate"
        
        # Construct the new filename
        new_filename = f"{sanitized_patient}_{sanitized_type}_{date_str}{ext}"
        logger.info(f"📝 Generated filename: {new_filename} (original: {original_filename})")
        
        return new_filename
    
    async def _generate_concise_brief_summary(self, structured_short_summary: Dict[str, Any], document_type: str = "Medical Document") -> Dict[str, List[str]]:
        """
        Uses the reducer to filter and prioritize the structured short summary.
        Wrapper for external service function.
        """
        return await generate_concise_brief_summary(structured_short_summary, document_type, LLM_EXECUTOR)
    
    async def process_document_data(self, data: dict) -> dict:
        logger.info(f"📥 Processing document: {data.get('document_id', 'unknown')}")

        if not data.get("result") or not data.get("filename") or not data.get("gcs_url"):
            raise HTTPException(status_code=400, detail="Missing required fields")

        result_data = data["result"]
        text = result_data.get("text", "")
        raw_text = result_data.get("raw_text") or ""  # Handle None case
        mode = data.get("mode", "wc")
        
        # Check for multi-report detection from Document AI processing
        is_multiple_reports = result_data.get("is_multiple_reports", False)
        multi_report_info = result_data.get("multi_report_info", {})

        logger.info(f"📋 Document mode: {mode}")
        logger.info(f"📊 Text lengths - Full OCR: {len(text)} chars, Document AI summary: {len(raw_text)} chars")
        
        # Log if raw_text is missing to help debug
        if not raw_text:
            logger.warning("⚠️ raw_text is empty - Document AI summarizer output not available, will use full OCR text as fallback")
        
        # OPTIMIZED: Detect document type ONCE and reuse result
        # Uses summarizer output first, falls back to raw_text if confidence is low
        loop = asyncio.get_event_loop()
        logger.info("🔍 Detecting document type (single call, reused across pipeline)...")
        doc_type_result = await loop.run_in_executor(
            LLM_EXECUTOR, 
            lambda: detect_document_type(summarizer_output=raw_text, raw_text=text)
        )
        logger.info(f"📄 Document type detected: {doc_type_result.get('doc_type')} (Conf: {doc_type_result.get('confidence', 0.0)}, Source: {doc_type_result.get('source', 'unknown')})")
        
        # ✅ Extract pre-extracted patient details from Document AI metadata
        pre_extracted_patient_details = result_data.get("metadata", {}).get("patient_details", {})
        if pre_extracted_patient_details:
            logger.info("📋 Found pre-extracted patient details from Document AI:")
            for key, value in pre_extracted_patient_details.items():
                if value:
                    logger.info(f"   - {key}: {value}")
        else:
            logger.info("⚠️ No pre-extracted patient details found in metadata")

        # 🚨 FAST FAIL: Check if Document Author is internal BEFORE generating summaries
        # This saves significant time/cost by not generating long/short summaries for internal docs
        extracted_author = pre_extracted_patient_details.get("author")
        if extracted_author and str(extracted_author).lower() not in ["not specified", "unknown", "none", "", "null"]:
            physician_id_chk = data.get("physician_id")
            if physician_id_chk:
                logger.info(f"🔍 Pre-check: Verifying if author '{extracted_author}' is internal...")
                try:
                    prisma = Prisma()
                    await prisma.connect()
                    
                    # Find all physicians in this clinic
                    users = await prisma.user.find_many(where={
                        "physicianId": physician_id_chk,
                        "role": "Physician"
                    })
                    
                    # Also check actual physician user
                    current_user = await prisma.user.find_first(where={
                        "OR": [
                            {"id": physician_id_chk},
                            {"physicianId": physician_id_chk}
                        ]
                    })
                    if current_user and not any(u.id == current_user.id for u in users):
                        users.append(current_user)
                        
                    await prisma.disconnect()
                    
                    # Check match
                    match_result = self._find_best_match(extracted_author, users)
                    if match_result and match_result[0]: # matched user found
                        logger.warning(f"⚠️ INTERNAL DOCUMENT DETECTED: Author '{extracted_author}' is from our clinic (matched: {match_result[0].firstName} {match_result[0].lastName})")
                        # Return validation failure immediately
                        return {
                            "document_analysis": None,
                            "brief_summary": f"Document authored by internal clinic member ({extracted_author}). processing skipped.",
                            "text_for_analysis": text,
                            "raw_text": raw_text,
                            "report_analyzer_result": {},
                            "patient_name": pre_extracted_patient_details.get("patient_name"),
                            "claim_number": pre_extracted_patient_details.get("claim_number"),
                            "dob": pre_extracted_patient_details.get("dob"),
                            "has_patient_name": bool(pre_extracted_patient_details.get("patient_name")),
                            "has_claim_number": bool(pre_extracted_patient_details.get("claim_number")),
                            "physician_id": physician_id_chk,
                            "user_id": data.get("user_id"),
                            "filename": data["filename"],
                            "gcs_url": data["gcs_url"],
                            "blob_path": data.get("blob_path"),
                            "file_size": data.get("file_size", 0),
                            "mime_type": data.get("mime_type", "application/octet-stream"),
                            "processing_time_ms": data.get("processing_time_ms", 0),
                            "file_hash": data.get("file_hash"),
                            "result_data": result_data,
                            "document_id": data.get("document_id", "unknown"),
                            "mode": mode,
                            "is_multiple_reports": False,
                            "internal_doc_skipped": True, # Flag for caller
                            "error_msg": f"Internal document from {extracted_author}"
                        }
                except Exception as e:
                    logger.error(f"❌ Error in internal author pre-check: {e}")
                    # Continue if check fails, safe fallback
        # 🎯 CHECK: Is this document valid for Summary Card (physician review)?
        is_valid_for_summary_card = doc_type_result.get('is_valid_for_summary_card', True)  # Default True for safety
        summary_card_reasoning = doc_type_result.get('summary_card_reasoning', '')
        
        logger.info(f"🎯 Summary Card Eligibility: {is_valid_for_summary_card}")
        logger.info(f"   Reasoning: {summary_card_reasoning[:100]}..." if len(summary_card_reasoning) > 100 else f"   Reasoning: {summary_card_reasoning}")
        
        # Initialize variables
        long_summary = ""
        short_summary = ""
        report_result = {}
        
        if is_valid_for_summary_card:
            # ✅ FULL EXTRACTION: Document requires physician review - generate summaries
            logger.info("📋 Document requires Summary Card - running full LLM extraction...")
            
            # Run ReportAnalyzer in dedicated LLM executor, passing pre-detected doc_type
            report_analyzer = ReportAnalyzer(mode)
            report_result = await loop.run_in_executor(
                LLM_EXECUTOR, 
                lambda: report_analyzer.extract_document(text, raw_text, doc_type_result=doc_type_result)
            )
            long_summary = report_result.get("long_summary", "")
            short_summary = report_result.get("short_summary", "")

            logger.info(f"✅ Generated long summary: {len(long_summary)} chars")
            logger.info(f"✅ Generated short summary: {type(short_summary)}")
           
        else:
            # ⏭️ TASK-ONLY MODE: Document is administrative - skip expensive LLM extraction
            logger.info("📌 Document is TASK-ONLY (administrative) - skipping LLM extraction for summaries")
            logger.info(f"   Document type: {doc_type_result.get('doc_type', 'Unknown')}")
            logger.info(f"   Reason: {summary_card_reasoning}")
            
            # Create minimal summary for task generation (just use raw_text as reference)
            detected_doc_type = doc_type_result.get('doc_type', 'Unknown')
            long_summary = f"[TASK-ONLY DOCUMENT]\nType: {detected_doc_type}\nReason: {summary_card_reasoning}\n\nThis document is administrative and does not require physician clinical review. Tasks have been generated for staff action."
            short_summary = ""  # No short summary for task-only docs
            report_result = {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "is_task_only": True,
                "task_only_reason": summary_card_reasoning
            }
            logger.info("⏭️ Skipped ReportAnalyzer - will proceed to task generation only")

        
        def safe_get(d, key, default="Not specified"):
            val = d.get(key)
            if not val or str(val).lower() in ["none", "null", ""]:
                return default
            return val

        # Map pre-extracted patient details
        patient_name_val = safe_get(pre_extracted_patient_details, "patient_name")
        claim_number_val = safe_get(pre_extracted_patient_details, "claim_number")
        dob_val = safe_get(pre_extracted_patient_details, "dob", "0000-00-00")
        doi_val = safe_get(pre_extracted_patient_details, "doi", "0000-00-00")
        report_date_val = safe_get(pre_extracted_patient_details, "date_of_report", "0000-00-00")
        author_val = safe_get(pre_extracted_patient_details, "author", "Not specified")
        
        detected_doc_type = doc_type_result.get('doc_type', 'Unknown')

        # Helper to convert structured short_summary dict to string
        raw_brief_summary_text = "Summary not available"
        if short_summary:
            if isinstance(short_summary, dict):
                # Try to extract meaningful text from structured summary
                try:
                    # 1. Try to get items texts
                    items = short_summary.get('summary', {}).get('items', [])
                    text_parts = []
                    for item in items:
                        if isinstance(item, dict):
                            # Prefer expanded text, fall back to collapsed
                            part = item.get('expanded') or item.get('collapsed')
                            if part:
                                text_parts.append(part)
                    
                    if text_parts:
                        raw_brief_summary_text = " ".join(text_parts)
                    elif short_summary.get('header', {}).get('title'):
                         # Fallback to Title if no items
                         raw_brief_summary_text = f"Report: {short_summary['header']['title']}"
                    else:
                        # Fallback to JSON string as last resort
                        raw_brief_summary_text = json.dumps(short_summary)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse structured short_summary: {e}")
                    raw_brief_summary_text = str(short_summary)
            else:
                raw_brief_summary_text = str(short_summary)
        elif not is_valid_for_summary_card:
            # For task-only documents, use a simple description
            raw_brief_summary_text = f"{detected_doc_type} - Administrative document for staff action"
        
        # ✅ Process the structured summary through the reducer (only for summary card eligible docs)
        if is_valid_for_summary_card and isinstance(short_summary, dict) and short_summary.get('summary', {}).get('items'):
            brief_summary_text = await self._generate_concise_brief_summary(
                short_summary,  # Pass the structured summary directly
                detected_doc_type
            )
        elif is_valid_for_summary_card and raw_brief_summary_text != "Summary not available":
            # Fallback: create minimal structure from raw text
            brief_summary_text = {"Summary": [raw_brief_summary_text]} if raw_brief_summary_text else {}
        else:
            # Skip reducer for task-only docs
            brief_summary_text = {"Summary": [raw_brief_summary_text]} if raw_brief_summary_text else {}

        document_analysis = DocumentAnalysis(
            patient_name=patient_name_val,
            claim_number=claim_number_val,
            dob=dob_val,
            doi=doi_val,
            status="Not specified",
            rd=report_date_val,
            body_part="Not specified",
            body_parts_analysis=[],
            diagnosis="See summary",
            key_concern="Medical evaluation",
            extracted_recommendation="See summary",
            extracted_decision="Not specified",
            ur_decision="",
            ur_denial_reason=None,
            adls_affected="Not specified",
            work_restrictions="Not specified",
            consulting_doctor=author_val,
            all_doctors=[],
            referral_doctor="Not specified",
            ai_outcome="Review required",
            document_type=detected_doc_type,
            summary_points=[],
            brief_summary=brief_summary_text,
            date_reasoning=None,
            is_task_needed=False,
            formatted_summary=brief_summary_text,
            extraction_confidence=1.0 if short_summary else 0.0,
            verified=True,
            verification_notes=["Simplified analysis from ReportAnalyzer"]
        )
        
        brief_summary = document_analysis.brief_summary

        patient_name = (
            document_analysis.patient_name
            if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"
            else None
        )

        claim_number = (
            document_analysis.claim_number
            if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"
            else None
        )

        dob = (
            document_analysis.dob
            if hasattr(document_analysis, 'dob') and document_analysis.dob and str(document_analysis.dob).lower() != "not specified"
            else None
        )

        return {
            "document_analysis": document_analysis,
            "brief_summary": brief_summary,
            "text_for_analysis": text,
            "raw_text": raw_text,
            "report_analyzer_result": report_result,
            "patient_name": patient_name,
            "claim_number": claim_number,
            "dob": dob,
            "has_patient_name": bool(patient_name),
            "has_claim_number": bool(claim_number),
            "physician_id": data.get("physician_id"),
            "user_id": data.get("user_id"),
            "filename": data["filename"],
            "gcs_url": data["gcs_url"],
            "blob_path": data.get("blob_path"),
            "file_size": data.get("file_size", 0),
            "mime_type": data.get("mime_type", "application/octet-stream"),
            "processing_time_ms": data.get("processing_time_ms", 0),
            "file_hash": data.get("file_hash"),
            "result_data": result_data,
            "document_id": data.get("document_id", "unknown"),
            "mode": mode,
            "is_multiple_reports": is_multiple_reports,
            "multi_report_info": multi_report_info,
            # 🎯 NEW: Summary Card eligibility flag
            "is_valid_for_summary_card": is_valid_for_summary_card,
            "summary_card_reasoning": summary_card_reasoning,
            "is_task_only": not is_valid_for_summary_card
        }

    async def save_to_redis_cache(self, document_id: str, document_data: dict):
        """Save document data to Redis cache"""
        if not self.redis_client:
            logger.warning("⚠️ Redis client not available - skipping cache")
            return False
        
        try:
            # Create cache key
            cache_key = f"document:{document_id}"
            
            # Prepare data for caching
            cache_data = {
                "document_id": document_id,
                "patient_name": document_data.get("patient_name"),
                "claim_number": document_data.get("claim_number"),
                "dob": document_data.get("dob"),
                "doi": document_data.get("doi"),
                "physician_id": document_data.get("physician_id"),
                "status": document_data.get("status"),
                "mode": document_data.get("mode"),
                "brief_summary": document_data.get("brief_summary"),
                "filename": document_data.get("filename"),
                "created_at": datetime.now().isoformat(),
                "cached_at": datetime.now().isoformat()
            }
            
            # Save to Redis with 24-hour expiration
            await self.redis_client.setex(
                cache_key, 
                86400,  # 24 hours in seconds
                json.dumps(cache_data)
            )
            
            logger.info(f"💾 Document {document_id} saved to Redis cache")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save document {document_id} to Redis: {str(e)}")
            return False

    async def _get_cached_patient_lookup(self, physician_id: str, patient_name: str, claim_number: str, dob: str, db_service) -> dict:
        """Get patient lookup data from cache or database with minimum 2-field matching"""
        
        # Create cache key matching your log format
        cache_key_parts = [f"patient_lookup:{physician_id}"]
        
        if patient_name and str(patient_name).lower() not in ["not specified", "unknown", "", "none"]:
            cache_key_parts.append(f"patient:{patient_name}")
        
        if dob and str(dob).lower() not in ["not specified", "unknown", "", "none"]:
            cache_key_parts.append(f"dob:{dob}")
        
        if claim_number and str(claim_number).lower() not in ["not specified", "unknown", "", "none"]:
            cache_key_parts.append(f"claim:{claim_number}")
        
        cache_key = ":".join(cache_key_parts)
        logger.info(f"🔑 Final cache key: {cache_key}")
        
        # Helper function to check if field is "bad"
        def is_bad_field(value):
            return not value or str(value).lower() in ["not specified", "unknown", "", "none", "null"]
        
        # Helper function to normalize field values
        def normalize_field(value):
            if not value:
                return ""
            return str(value).strip().lower()
        
        # Check if Redis client is available
        if not self.redis_client:
            logger.warning("❌ Redis client not available, skipping cache")
            lookup_data = await db_service.get_patient_claim_numbers(
                patient_name=patient_name,
                physicianId=physician_id,
                dob=dob,
                claim_number=claim_number
            )
            # Add original search criteria to lookup data for matching validation
            if lookup_data:
                lookup_data["_search_criteria"] = {
                    "patient_name": patient_name,
                    "dob": dob,
                    "claim_number": claim_number
                }
            return lookup_data
        
        # Try cache first
        try:
            logger.info(f"🔍 Checking Redis cache FIRST for key: {cache_key}")
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"💾 CACHE HIT: Found patient lookup data in cache for key: {cache_key}")
                lookup_data = json.loads(cached_data)
                
                # 🚨 CRITICAL: Validate that cached data meets minimum 2-field matching
                if lookup_data and lookup_data.get("total_documents", 0) > 0:
                    # Get the original search criteria from cache key or reconstruct
                    original_patient_name = patient_name
                    original_dob = dob
                    original_claim_number = claim_number
                    
                    # Count matching fields between original search and cached results
                    matching_fields = 0
                    
                    # Get the first document from cached results for field comparison
                    first_doc = lookup_data.get("documents", [{}])[0] if lookup_data.get("documents") else {}
                    cached_patient_name = first_doc.get("patientName") or lookup_data.get("patient_name")
                    cached_dob = first_doc.get("dob") or lookup_data.get("dob")
                    cached_claim_number = first_doc.get("claimNumber") or lookup_data.get("claim_number")
                    
                    # Check patient name match
                    original_patient_normalized = normalize_field(original_patient_name)
                    cached_patient_normalized = normalize_field(cached_patient_name)
                    patient_matches = (
                        not is_bad_field(original_patient_name) and 
                        not is_bad_field(cached_patient_name) and
                        original_patient_normalized == cached_patient_normalized
                    )
                    if patient_matches:
                        matching_fields += 1
                        logger.info(f"✅ Cached patient name matches: '{original_patient_name}' == '{cached_patient_name}'")
                    
                    # Check DOB match
                    original_dob_normalized = normalize_field(original_dob)
                    cached_dob_normalized = normalize_field(cached_dob)
                    dob_matches = (
                        not is_bad_field(original_dob) and 
                        not is_bad_field(cached_dob) and
                        original_dob_normalized == cached_dob_normalized
                    )
                    if dob_matches:
                        matching_fields += 1
                        logger.info(f"✅ Cached DOB matches: '{original_dob}' == '{cached_dob}'")
                    
                    # Check claim number match
                    original_claim_normalized = normalize_field(original_claim_number)
                    cached_claim_normalized = normalize_field(cached_claim_number)
                    claim_matches = (
                        not is_bad_field(original_claim_number) and 
                        not is_bad_field(cached_claim_number) and
                        original_claim_normalized == cached_claim_normalized
                    )
                    if claim_matches:
                        matching_fields += 1
                        logger.info(f"✅ Cached claim number matches: '{original_claim_number}' == '{cached_claim_number}'")
                    
                    logger.info(f"🔢 Cached data field matching: {matching_fields} fields match")
                    
                    # 🚨 Only return cached data if we have minimum 2-field match
                    if matching_fields >= 2:
                        logger.info("✅ Cached data meets minimum 2-field requirement - using cached results")
                        # Add search criteria to lookup data for later validation
                        lookup_data["_search_criteria"] = {
                            "patient_name": original_patient_name,
                            "dob": original_dob,
                            "claim_number": original_claim_number
                        }
                        return lookup_data
                    else:
                        logger.warning(f"🚨 Cached data FAILED 2-field requirement (only {matching_fields} matches) - fetching fresh from DB")
                        # Intentionally fall through to database fetch
                else:
                    logger.info("💾 Cached data has no documents - returning as-is")
                    return lookup_data
            else:
                logger.info(f"💾 CACHE MISS: No data found in cache for key: {cache_key}")
        except Exception as e:
            logger.error(f"⚠️ Cache read error for key {cache_key}: {e}")
        
        # Get from database (either cache miss or cache validation failed)
        logger.info("🗄️ Fetching patient lookup data from database...")
        lookup_data = await db_service.get_patient_claim_numbers(
            patient_name=patient_name,
            physicianId=physician_id,
            dob=dob,
            claim_number=claim_number
        )
        
        # 🚨 CRITICAL: Validate database results meet minimum 2-field matching BEFORE caching
        valid_for_cache = False
        if lookup_data and lookup_data.get("total_documents", 0) > 0:
            # Count matching fields between search criteria and database results
            matching_fields = 0
            
            # Get the first document from database results for field comparison
            first_doc = lookup_data.get("documents", [{}])[0] if lookup_data.get("documents") else {}
            db_patient_name = first_doc.get("patientName") or lookup_data.get("patient_name")
            db_dob = first_doc.get("dob") or lookup_data.get("dob")
            db_claim_number = first_doc.get("claimNumber") or lookup_data.get("claim_number")
            
            # Check patient name match
            original_patient_normalized = normalize_field(patient_name)
            db_patient_normalized = normalize_field(db_patient_name)
            patient_matches = (
                not is_bad_field(patient_name) and 
                not is_bad_field(db_patient_name) and
                original_patient_normalized == db_patient_normalized
            )
            if patient_matches:
                matching_fields += 1
            
            # Check DOB match
            original_dob_normalized = normalize_field(dob)
            db_dob_normalized = normalize_field(db_dob)
            dob_matches = (
                not is_bad_field(dob) and 
                not is_bad_field(db_dob) and
                original_dob_normalized == db_dob_normalized
            )
            if dob_matches:
                matching_fields += 1
            
            # Check claim number match
            original_claim_normalized = normalize_field(claim_number)
            db_claim_normalized = normalize_field(db_claim_number)
            claim_matches = (
                not is_bad_field(claim_number) and 
                not is_bad_field(db_claim_number) and
                original_claim_normalized == db_claim_normalized
            )
            if claim_matches:
                matching_fields += 1
            
            logger.info(f"🔢 Database results field matching: {matching_fields} fields match")
            
            # Only cache if we have minimum 2-field match
            if matching_fields >= 2:
                valid_for_cache = True
                logger.info("✅ Database results meet minimum 2-field requirement - will cache")
            else:
                logger.warning(f"🚨 Database results FAILED 2-field requirement (only {matching_fields} matches) - NOT caching")
        
        # Cache the result only if it meets 2-field requirement
        if valid_for_cache:
            try:
                # Convert any non-serializable objects to strings
                cacheable_data = {
                    "total_documents": lookup_data.get("total_documents", 0),
                    "patient_name": lookup_data.get("patient_name"),
                    "dob": lookup_data.get("dob"),
                    "claim_number": lookup_data.get("claim_number"),
                    "doi": lookup_data.get("doi"),
                    "has_conflicting_claims": lookup_data.get("has_conflicting_claims", False),
                    "documents": [
                        {
                            "patientName": doc.get("patientName"),
                            "dob": doc.get("dob"),
                            "claimNumber": doc.get("claimNumber"),
                            "physicianId": doc.get("physicianId")
                        }
                        for doc in lookup_data.get("documents", [])
                    ]
                }
                
                await self.redis_client.setex(cache_key, 3600, json.dumps(cacheable_data))
                logger.info(f"💾 CACHE STORE: Successfully cached patient lookup data for key: {cache_key}")
            except Exception as e:
                logger.error(f"⚠️ Cache write error for key {cache_key}: {e}")
        
        # Add search criteria to lookup data for later validation in perform_patient_lookup
        if lookup_data:
            lookup_data["_search_criteria"] = {
                "patient_name": patient_name,
                "dob": dob,
                "claim_number": claim_number
            }
        
        return lookup_data

    def _normalize_name(self, name):
        """
        Advanced name normalization:
        Delegates to centralized clean_name_string helper.
        """
        return clean_name_string(name)

    def _extract_name_parts(self, name):
        """
        Extract first name and last name, ignoring middle initials/names.
        Returns: (first_name, last_name, middle_parts)
        """
        if not name:
            return ("", "", [])
        
        normalized = self._normalize_name(name)
        parts = normalized.split()
        
        if len(parts) == 0:
            return ("", "", [])
        elif len(parts) == 1:
            return (parts[0], "", [])
        elif len(parts) == 2:
            return (parts[0], parts[1], [])
        else:
            return (parts[0], parts[-1], parts[1:-1])

    def _names_match(self, name1, name2, match_type="exact"):
        """Compare two names with different matching strategies."""
        first1, last1, middle1 = self._extract_name_parts(name1)
        first2, last2, middle2 = self._extract_name_parts(name2)
        
        if first1 and last1 and first1 == first2 and last1 == last2:
            return (True, 1.0, f"Exact match: {first1} {last1}")
        
        if match_type in ["partial", "fuzzy"]:
            if first1 and first1 == first2:
                return (True, 0.8, f"First name match: {first1}")
            if last1 and last1 == last2:
                return (True, 0.8, f"Last name match: {last1}")
        
        if first1 and last1 and first1 == last2 and last1 == first2:
            return (True, 0.9, f"Reversed match: {first1} {last1}")
        
        return (False, 0.0, "No match")

    def _find_best_match(self, doctor_name, users):
        """Find the best matching user for a given doctor name."""
        best_match = None
        best_confidence = 0.0
        best_details = ""
        
        for user in users:
            user_full_name = f"{user.firstName or ''} {user.lastName or ''}".strip()
            is_match, confidence, details = self._names_match(doctor_name, user_full_name, "exact")
            
            if is_match and confidence > best_confidence:
                best_match = user
                best_confidence = confidence
                best_details = details
                if confidence >= 1.0:
                    break
        
        if best_confidence < 1.0:
            for user in users:
                user_full_name = f"{user.firstName or ''} {user.lastName or ''}".strip()
                is_match, confidence, details = self._names_match(doctor_name, user_full_name, "partial")
                
                if is_match and confidence > best_confidence:
                    best_match = user
                    best_confidence = confidence
                    best_details = details
        
        return (best_match, best_confidence, best_details)


    def _extract_author_from_long_summary(self, long_summary: str) -> str:
        """
        Extract author from long summary using pattern matching.
        Looks for keys like: Author, Signature, Signed by, Electronic signature, etc.
        """
        if not long_summary:
            return None
        # logger.info(f"🔍 Extracting author from long summary... {long_summary[:800]}")  # Log first 100 chars for context
        try:
            # Patterns to look for author information
            # we only need the author who signed the report, not assistants or transcribers, or prepared by, directed by, etc.
            
            # Pattern for name - handles both "FirstName LastName" and "LastName, FirstName" formats
            # Captures the full name including credentials
            # Examples: "John Smith, MD", "Smith, John, MD", "Dr. John Smith", "John A. Smith, M.D."
            name_pattern_forward = r'(?:Dr\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:,?\s*(?:MD|M\.D\.|DO|D\.O\.|DC|D\.C\.|DPM|NP|PA|PhD|RN|R\.N\.|LVN|L\.V\.N\.|PA-C))?)'
            name_pattern_reverse = r'([A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,?\s*(?:MD|M\.D\.|DO|D\.O\.|DC|D\.C\.|DPM|NP|PA|PhD|RN|R\.N\.|LVN|L\.V\.N\.|PA-C))?)'
            
            # Combined pattern that tries both formats
            name_pattern = f'(?:{name_pattern_reverse}|{name_pattern_forward})'
            
            author_patterns = [
                # Pattern for "• Signature:" format (used by Pydantic formatted summaries)
                r'[•\-]\s*Signature[:\s]*' + name_pattern,
                # Pattern for various signature labels
                r'(?:Electronically\s+Signed\s+By|Electronic\s+Signature|Signed\s+By|Signature)[:\s]*' + name_pattern,
                # Pattern for "- Signature:" format  
                r'-\s*Signature[:\s]*(?:Electronically\s+Signed\s+By[:\s]*)?' + name_pattern,
                # Pattern for approval labels
                r'(?:Approved\s+By|Authenticated\s+By|Verified\s+By)[:\s]*' + name_pattern,
                # Pattern for "Evaluating Physician:" (used by QME)
                r'(?:Evaluating\s+Physician|Examining\s+Physician|QME\s+Physician)[:\s]*' + name_pattern,
                # Pattern for "Reviewer:" (used by UR)
                r'(?:Reviewing\s+Physician|Reviewer)[:\s]*' + name_pattern,
                # Pattern for "Radiologist:" field (common in imaging reports)
                r'(?:Radiologist)[:\s]*' + name_pattern,
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, long_summary, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Get the first non-None group (either group 1 or 2 depending on which pattern matched)
                    author = None
                    for group in match.groups():
                        if group:
                            author = group.strip()
                            break
                    
                    if author:
                        # Clean up the author name
                        author = re.sub(r'\s+', ' ', author)
                        if len(author) > 3:
                            logger.info(f"✅ Found author in long summary: {author}")
                            return author
            
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting author from long summary: {e}")
            return None

    async def _check_physician_author(self, physician_id: str, short_summary: str = None, long_summary: str = None) -> dict:
        """
        Check if the document author matches a clinic member.
        
        Logic:
        2. check long summary for author keys (Signature, Signed by, etc.)
        3. If no author found anywhere, return error for manual verification
        4. If author found, check if they are from our clinic:
           - If author IS from our clinic = INTERNAL document (cannot parse, fail)
           - If author is NOT from our clinic = EXTERNAL document (can create tasks)
        
        Returns a dict with:
        - author_found: bool - whether an author was found in the document
        - author_name: str - the extracted author name
        - is_clinic_member: bool - whether author is from our clinic
        - is_external_document: bool - True if document is external (can process)
        - is_internal_document: bool - True if document is internal (cannot process)
        - matching_user: User object if matched
        - error_message: str - error if no author found
        """
        try:
            logger.info(f"🔍 Checking for document author...")
            author_name = self._extract_author_from_long_summary(long_summary)
            author_source = "long_summary" if author_name else None
            
            # Step 3: If no author found anywhere, return error
            if not author_name:
                logger.warning("⚠️ No author found in document summaries")
                return {
                    "author_found": False,
                    "author_name": None,
                    "is_clinic_member": False,
                    "is_external_document": False,
                    "is_internal_document": False,
                    "matching_user": None,
                    "error_message": "There is no author in the report, manual verification required",
                    "author_source": None
                }
            
            logger.info(f"✅ Author found: {author_name} (from {author_source})")
            
            # Step 4: Check if author is from our clinic
            prisma = Prisma()
            await prisma.connect()
            
            # Find all physicians in this clinic
            users = await prisma.user.find_many(where={
                "physicianId": physician_id,
                "role": "Physician"
            })
            
            # Also check if physician_id itself is a user
            current_user = await prisma.user.find_first(where={
                "OR": [
                    {"id": physician_id},
                    {"physicianId": physician_id}
                ]
            })
            
            if current_user and not any(u.id == current_user.id for u in users):
                users.append(current_user)
            
            await prisma.disconnect()
            
            # Try to match author with clinic members
            matching_user, confidence, match_details = self._find_best_match(author_name, users)
            
            result = {
                "author_found": True,
                "author_name": author_name,
                "author_source": author_source,
                "is_clinic_member": False,
                "is_external_document": True,  # Default to external (can process)
                "is_internal_document": False,
                "matching_user": matching_user,
                "confidence": confidence,
                "match_details": match_details,
                "error_message": None
            }
            
            if matching_user:
                # Author is from our clinic - this is an INTERNAL document (cannot process)
                result["is_clinic_member"] = True
                result["is_external_document"] = False
                result["is_internal_document"] = True
                logger.warning(f"⚠️ Author '{author_name}' is a clinic member - INTERNAL document (cannot process)")
            else:
                # Author is NOT from our clinic - this is an EXTERNAL document (can create tasks)
                logger.info(f"✅ Author '{author_name}' is NOT a clinic member - EXTERNAL document (can process)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Physician author check failed: {str(e)}")
            return {
                "author_found": False,
                "author_name": None,
                "is_clinic_member": False,
                "is_external_document": False,
                "is_internal_document": False,
                "matching_user": None,
                "error_message": f"Error checking author: {str(e)}",
                "author_source": None
            }

    async def create_tasks_if_needed(self, document_analysis, document_id: str, physician_id: str, filename: str, processed_data: dict = None) -> int:
        """Step 3: Create tasks for documents"""
        # logger.info(f"🔧 Creating tasks for document {document_analysis}...")
        created_tasks = 0
        
        try:
            # Generate and create tasks
            task_creator = TaskCreator()
            
            tasks_result = await task_creator.generate_tasks(document_analysis=document_analysis, processed_data=processed_data)
            
            # Extract tasks
            generated_tasks = tasks_result.get("internal_tasks", [])
            logger.info(f"🔢 Generated {len(generated_tasks)} tasks for document")
            # Save tasks to database
            prisma = Prisma()
            await prisma.connect()
            
            # Process tasks
            for task in generated_tasks:
                try:
                    # Ensure task is a dict
                    if not isinstance(task, dict):
                        if hasattr(task, 'dict'):
                            task = task.dict()
                        elif hasattr(task, '__dict__'):
                            task = task.__dict__
                        elif isinstance(task, str):
                            try:
                                task = json.loads(task)
                            except:
                                logger.warning(f"⚠️ Skipping invalid task (not a dict): {task}")
                                continue
                        else:
                            logger.warning(f"⚠️ Skipping invalid task type: {type(task)}")
                            continue
                    
                    # Build quick notes JSON
                    quick_notes = task.get("quick_notes", {})
                    if not isinstance(quick_notes, dict):
                        quick_notes = {"status_update": "", "details": "", "one_line_note": ""}
                    quick_notes_json = json.dumps({
                        "status_update": quick_notes.get("status_update", ""),
                        "details": quick_notes.get("details", ""),
                        "one_line_note": quick_notes.get("one_line_note", "")
                    })
                    
                    mapped_task = {
                        "description": task.get("description"),
                        "department": task.get("department"),
                        "status": "Open",
                        "dueDate": datetime.now(),
                        "patient": task.get("patient", "Unknown"),
                        "actions": task.get("actions") if isinstance(task.get("actions"), list) else [],
                        "sourceDocument": task.get("source_document") or filename,
                        "physicianId": physician_id,
                        "type": "external",  # Set type as external
                        "quickNotes": quick_notes_json,
                    }
                    
                    # Connect to document if document_id exists and document was saved successfully
                    # Don't connect if this is a FailDoc ID (document relation only works with Document table)
                    if document_id:
                        # Verify the document actually exists in the Document table before connecting
                        try:
                            existing_doc = await prisma.document.find_unique(where={"id": document_id})
                            if existing_doc:
                                mapped_task["document"] = {"connect": {"id": document_id}}
                            else:
                                logger.warning(f"⚠️ Document {document_id} not found in Document table - creating task without document link")
                        except Exception as doc_check_err:
                            logger.warning(f"⚠️ Could not verify document {document_id}: {doc_check_err} - creating task without document link")
                    
                    await prisma.task.create(data=mapped_task)
                    created_tasks += 1
                    logger.info(f"✅ Created task: {task.get('description', 'Unknown task')}")
                    
                except Exception as task_err:
                    logger.error(f"❌ Failed to create task: {task_err}")
                    continue
            

            await prisma.disconnect()
            logger.info(f"✅ {created_tasks} tasks created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in task creation: {str(e)}")
        
        return created_tasks
    
    async def save_document(self, db_service, processed_data: dict, lookup_result: dict) -> dict:
        """Step 4: Save document to database and Redis cache - Wrapper for external service"""
        return await save_document_external(
            db_service=db_service,
            processed_data=processed_data,
            lookup_result=lookup_result,
            generate_filename_func=self._generate_document_filename,
            save_to_redis_cache_func=self.save_to_redis_cache
        )

    async def handle_webhook(self, data: dict, db_service) -> dict:
        """
        Clean webhook processing pipeline WITHOUT duplicate prevention
        Includes treatment history creation
        """
        try:
            # Step 1: Process document data
            processed_data = await self.process_document_data(data)

            # logger.info(f"📄 Document data processed successfully : {processed_data.keys()}, document_analysis: {processed_data.get('document_analysis', {})}")
            
            # 🚨 CHECK: Early internal document detection (from process_document_data pre-check)
            if processed_data.get("internal_doc_skipped"):
                # Internal document was already detected in process_document_data
                # Extract the author from the error message or use the extracted author
                error_msg = processed_data.get("error_msg", "Internal document detected")
                
                # Try to extract author name from error_msg (format: "Internal document from AuthorName")
                extracted_author = None
                if "Internal document from " in error_msg:
                    extracted_author = error_msg.replace("Internal document from ", "").strip()
                
                logger.warning(f"⚠️ INTERNAL DOCUMENT (early detection): Author '{extracted_author}' - saving to FailDocs")
                
                raw_text = processed_data.get("raw_text", "")
                text_for_analysis = processed_data.get("text_for_analysis", "")
                
                fail_doc_id = await db_service.save_fail_doc(
                    reason=f"Internal document detected - Author: {extracted_author}. Cannot process documents authored by clinic members.",
                    db=processed_data.get("dob"),
                    claim_number=processed_data.get("claim_number"),
                    patient_name=processed_data.get("patient_name"),
                    physician_id=processed_data.get("physician_id"),
                    gcs_file_link=processed_data.get("gcs_url"),
                    file_name=processed_data.get("filename"),
                    file_hash=processed_data.get("file_hash"),
                    blob_path=processed_data.get("blob_path"),
                    mode=processed_data.get("mode", "wc"),
                    document_text=text_for_analysis if text_for_analysis else raw_text,
                    doi=None,
                    ai_summarizer_text=raw_text,  # Store actual Document AI Summarizer output
                    author=extracted_author
                )
                
                parse_decremented = await db_service.decrement_parse_count(processed_data.get("physician_id"))
                
                return {
                    "status": "internal_document_fail",
                    "document_id": fail_doc_id,
                    "filename": processed_data.get("filename"),
                    "parse_count_decremented": parse_decremented,
                    "failure_reason": f"Internal document detected - Author: {extracted_author}",
                    "author_info": {
                        "author_name": extracted_author,
                        "author_source": "early_detection"
                    }
                }
            
            # SKIP author check for task-only documents (administrative docs don't need author verification)
            is_task_only = processed_data.get("is_task_only", False)
            
            if is_task_only:
                logger.info("📌 Task-only document - skipping author verification (administrative documents don't require author check)")
            else:
                # Get summaries for author check
                report_analyzer_result = processed_data.get("report_analyzer_result", {})
                short_summary = report_analyzer_result.get("short_summary", "") if isinstance(report_analyzer_result, dict) else ""
                long_summary = report_analyzer_result.get("long_summary", "") if isinstance(report_analyzer_result, dict) else ""
                
                author_info = await self._check_physician_author(
                    physician_id=processed_data["physician_id"],
                    short_summary=short_summary,
                    long_summary=long_summary
                )
                
                # If no author found in document, save to FailDocs
                if not author_info.get("author_found"):
                    logger.warning(f"⚠️ NO AUTHOR FOUND: {author_info.get('error_message')}")
                    logger.info("💾 Saving to FailDocs as per requirements...")
                    
                    raw_text = processed_data.get("raw_text", "")
                    text_for_analysis = processed_data.get("text_for_analysis", "")
                    
                    # Handle short_summary being either a string or dict
                    short_summary_text = short_summary.get('raw_summary', str(short_summary)) if isinstance(short_summary, dict) else (short_summary or 'N/A')
                    
                    fail_doc_id = await db_service.save_fail_doc(
                        reason=author_info.get('error_message', "No author found in document - manual verification required"),
                        db=processed_data.get("dob"),
                        claim_number=processed_data.get("claim_number"),
                        patient_name=processed_data.get("patient_name"),
                        physician_id=processed_data.get("physician_id"),
                        gcs_file_link=processed_data.get("gcs_url"),
                        file_name=processed_data.get("filename"),
                        file_hash=processed_data.get("file_hash"),
                        blob_path=processed_data.get("blob_path"),
                        mode=processed_data.get("mode", "wc"),
                        document_text=text_for_analysis if text_for_analysis else raw_text,
                        doi=None,
                        ai_summarizer_text=raw_text  # Store actual Document AI Summarizer output
                    )
                    
                    parse_decremented = await db_service.decrement_parse_count(processed_data.get("physician_id"))
                    
                    return {
                        "status": "no_author_found_fail",
                        "document_id": fail_doc_id,
                        "filename": processed_data.get("filename"),
                        "parse_count_decremented": parse_decremented,
                        "failure_reason": author_info.get('error_message', "No author found in document")
                    }
                
                # If author IS from our clinic (INTERNAL document), save to FailDocs
                if author_info.get("is_internal_document") or author_info.get("is_clinic_member"):
                    author_name = author_info.get("author_name", "Unknown")
                    logger.warning(f"⚠️ INTERNAL DOCUMENT DETECTED: Author '{author_name}' is from our clinic")
                    logger.info("💾 Saving to FailDocs as per requirements - cannot process internal documents...")
                    
                    raw_text = processed_data.get("raw_text", "")
                    text_for_analysis = processed_data.get("text_for_analysis", "")
                    
                    # Handle short_summary being either a string or dict
                    short_summary_text_internal = short_summary.get('raw_summary', str(short_summary)) if isinstance(short_summary, dict) else (short_summary or 'N/A')
                    
                    fail_doc_id = await db_service.save_fail_doc(
                        reason=f"Internal document detected - Author: {author_name}. Cannot process documents authored by clinic members.",
                        db=processed_data.get("dob"),
                        claim_number=processed_data.get("claim_number"),
                        patient_name=processed_data.get("patient_name"),
                        physician_id=processed_data.get("physician_id"),
                        gcs_file_link=processed_data.get("gcs_url"),
                        file_name=processed_data.get("filename"),
                        file_hash=processed_data.get("file_hash"),
                        blob_path=processed_data.get("blob_path"),
                        mode=processed_data.get("mode", "wc"),
                        document_text=text_for_analysis if text_for_analysis else raw_text,
                        doi=None,
                        ai_summarizer_text=raw_text,  # Store actual Document AI Summarizer output
                        author=author_name
                    )
                    
                    parse_decremented = await db_service.decrement_parse_count(processed_data.get("physician_id"))
                    
                    return {
                        "status": "internal_document_fail",
                        "document_id": fail_doc_id,
                        "filename": processed_data.get("filename"),
                        "parse_count_decremented": parse_decremented,
                        "failure_reason": f"Internal document detected - Author: {author_name}",
                        "author_info": {
                            "author_name": author_name,
                            "author_source": author_info.get("author_source")
                        }
                    }
                
                # EXTERNAL document - author is NOT from our clinic, can proceed with processing
                logger.info(f"✅ EXTERNAL document confirmed - Author: {author_info.get('author_name')} (not a clinic member)")
            
            
            # Step 2: Perform patient lookup with enhanced fuzzy matching (NO DUPLICATE CHECK)
            lookup_result = await self.patient_lookup.perform_patient_lookup(db_service, processed_data)
            
            
            # Step 3: Save document (ALL documents are saved - no duplicate blocking)
            save_result = await self.save_document(db_service, processed_data, lookup_result)
            
            # Step 3.5: Create treatment history (ONLY for successfully saved documents)
            treatment_history = {}

               # Step 4: Create tasks if document was saved successfully (NOT for FailDocs)
            tasks_created = 0
            tasks_created = await self.create_tasks_if_needed(
                processed_data["result_data"]["raw_text"],
                save_result["document_id"],
                processed_data["physician_id"],
                processed_data["filename"],
                processed_data['document_analysis']
            )
                
            # if tasks_created == -1:
            #     # This should ideally be caught by Step 1.2, but as a safety measure:
            #     logger.warning(f"⚠️ Task creation returned -1 (self-authored).")
            #     # We already saved the document, so we might need to move it to FailDocs here too
            #     # but Step 1.2 should have caught it.
            #     pass
            # else:
            #     logger.info(f"⚠️ Skipping task creation - document was saved as FailDoc (status: {save_result.get('status')})")
                            

            # Prepare final response - INCLUDE TREATMENT HISTORY INFORMATION
            result = {
                "status": save_result["status"],
                "document_id": save_result["document_id"],
                "filename": processed_data["filename"],
                "tasks_created": tasks_created,
                "treatment_history_created": len(treatment_history) > 0,
                "treatment_history_event_count": sum(len(v) for v in treatment_history.values()) if treatment_history else 0,
                "treatment_history_categories": list(treatment_history.keys()) if treatment_history else [],
                "mode": processed_data["mode"],
                "parse_count_decremented": save_result["parse_count_decremented"],
                "cache_success": save_result.get("cache_success", False)
            }
            
            if lookup_result["pending_reason"]:
                result["pending_reason"] = lookup_result["pending_reason"]
            
            # Add sample treatment history preview (first event from each category)
            if treatment_history:
                result["treatment_history_preview"] = {}
                for category, events in treatment_history.items():
                    if events and len(events) > 0:
                        result["treatment_history_preview"][category] = {
                            "count": len(events),
                            "latest_event": events[0] if len(events) > 0 else None,
                            "oldest_event": events[-1] if len(events) > 0 else None
                        }
            
            logger.info(f"✅ Webhook processing completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Webhook processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    async def handle_webhook_with_retry(self, data: dict, db_service, max_retries: int = 1) -> dict:
        """
        Wrapper that retries handle_webhook once on failure.
        If all retries fail, saves the document to FailDocs.
        
        Args:
            data: The webhook data
            db_service: Database service instance
            max_retries: Number of retry attempts (default: 1)
        
        Returns:
            Result dict with status and document_id
        """
        last_error = None
        filename = data.get("filename", "unknown")
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔄 Processing attempt {attempt + 1}/{max_retries + 1} for document: {filename}")
                result = await self.handle_webhook(data, db_service)
                
                # If we reach here, processing succeeded
                if attempt > 0:
                    logger.info(f"✅ Retry successful for document: {filename} (attempt {attempt + 1})")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for {filename}: {str(e)}")
                
                if attempt < max_retries:
                    # Wait briefly before retry
                    logger.info(f"⏳ Waiting 2 seconds before retry for {filename}...")
                    await asyncio.sleep(2)
                else:
                    # All retries exhausted - save to FailDocs
                    logger.error(f"❌ All {max_retries + 1} attempts failed for {filename}: {str(last_error)}")
        
        # All retries failed - save to FailDocs
        try:
            logger.info(f"💾 Saving failed document to FailDocs: {filename}")
            
            result_data = data.get("result", {})
            document_text = result_data.get("text", "")
            mode = data.get("mode", "wc")
            
            fail_doc_id = await db_service.save_fail_doc(
                reason=f"Processing failed after {max_retries + 1} attempts: {str(last_error)}",
                db=None,  # DOB unknown since processing failed
                claim_number=None,
                patient_name=None,
                physician_id=data.get("physician_id"),
                gcs_file_link=data.get("gcs_url"),
                file_name=filename,
                file_hash=data.get("file_hash"),
                blob_path=data.get("blob_path"),
                mode=mode,
                document_text=document_text,
                doi=None,
                ai_summarizer_text=result_data.get("raw_text", "")  # Try to get raw_text (summary) from result
            )
            
            # Decrement parse count for failed documents too
            parse_decremented = await db_service.decrement_parse_count(data.get("physician_id"))
            
            logger.info(f"✅ Failed document saved to FailDocs with ID: {fail_doc_id}")
            
            return {
                "status": "failed",
                "document_id": fail_doc_id,
                "filename": filename,
                "parse_count_decremented": parse_decremented,
                "failure_reason": f"Processing failed after {max_retries + 1} attempts: {str(last_error)}",
                "retries_attempted": max_retries + 1
            }
            
        except Exception as save_error:
            logger.error(f"❌ Failed to save to FailDocs: {str(save_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Processing failed and could not save to FailDocs: {str(last_error)}"
            )

    async def update_fail_document(self, fail_doc: Any, updated_fields: dict, user_id: str = None, db_service: Any = None) -> dict:
        """
        Updates and processes a failed document using the complete webhook-like logic.
        Wrapper for external service function.
        """
        return await update_fail_doc_service(
            fail_doc=fail_doc,
            updated_fields=updated_fields,
            user_id=user_id,
            db_service=db_service,
            patient_lookup=self.patient_lookup,
            save_document_func=self.save_document,
            create_tasks_func=self.create_tasks_if_needed,
            llm_executor=LLM_EXECUTOR
        )

    async def update_fail_documents_batch(
        self,
        fail_docs_data: List[dict],
        user_id: str,
        db_service: Any,
        max_concurrent: int = 3
    ) -> dict:
        """
        Updates and processes multiple failed documents concurrently.
        
        Args:
            max_concurrent: Maximum number of documents to process concurrently
        """
        results = {
            "total_documents": len(fail_docs_data),
            "successful": 0,
            "failed": 0,
            "documents": []
        }
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_document(doc_data):
            async with semaphore:
                fail_doc = doc_data.get("fail_doc")
                updated_fields = doc_data.get("updated_fields", {})
                
                if not fail_doc:
                    return {
                        "fail_doc_id": "unknown",
                        "status": "failed",
                        "error": "Missing fail_doc object"
                    }
                
                try:
                    document_result = await update_fail_doc_service(
                        fail_doc=fail_doc,
                        updated_fields=updated_fields,
                        user_id=user_id,
                        db_service=db_service,
                        patient_lookup=self.patient_lookup,
                        save_document_func=self.save_document,
                        create_tasks_func=self.create_tasks_if_needed,
                        llm_executor=LLM_EXECUTOR
                    )
                    
                    return {
                        "fail_doc_id": fail_doc.id,
                        "status": "success",
                        "document_id": document_result.get("document_id"),
                        "tasks_created": document_result.get("tasks_created", 0)
                    }
                except Exception as e:
                    logger.error(f"❌ Failed to process fail document {fail_doc.id}: {str(e)}")
                    return {
                        "fail_doc_id": fail_doc.id,
                        "status": "failed",
                        "error": str(e)
                    }
        
        # Process all documents concurrently
        tasks = [process_single_document(doc_data) for doc_data in fail_docs_data]
        individual_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in individual_results:
            if isinstance(result, Exception):
                results["failed"] += 1
                results["documents"].append({
                    "fail_doc_id": "unknown",
                    "status": "failed",
                    "error": str(result)
                })
            else:
                if result["status"] == "success":
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                results["documents"].append(result)
        
        return results