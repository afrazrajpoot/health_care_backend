"""
OPTIMIZED Webhook Service - Clean Version with Redis Caching (No Duplication)
"""
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any
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
from .treatment_history_generator import TreatmentHistoryGenerator
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
    
    async def verify_redis_connection(self):
        """Verify Redis connection is working"""
        if not self.redis_client:
            logger.error("❌ Redis client is None - not initialized")
            return False
        
        try:
            # Test the connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection verified")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def test_redis_basic(self):
        """Test basic Redis operations"""
        if not self.redis_client:
            print("❌ Redis client is None")
            return False
        
        try:
            test_key = "test_key_123"
            test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
            
            # Set value
            await self.redis_client.setex(test_key, 60, json.dumps(test_value))
            logger.info("✅ Test value set in Redis")
            
            # Get value
            retrieved = await self.redis_client.get(test_key)
            if retrieved:
                parsed_retrieved = json.loads(retrieved)
                logger.info(f"✅ Test value retrieved: {parsed_retrieved}")
                return True
            else:
                logger.error("❌ Test value not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Redis test failed: {e}")
            return False
    

    async def _generate_concise_brief_summary(self, raw_summary_text: str, document_type: str = "Medical Document") -> str:
        """
        Uses LLM to transform the raw summarizer output into a concise, accurate professional summary.
        Focuses on factual extraction without adding interpretations or missing critical details.
        """
        if not raw_summary_text or len(raw_summary_text) < 10:
            return "Summary not available"

        try:
            logger.info("🤖 Generating concise brief summary using LLM...")
            
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import AzureChatOpenAI
            from config.settings import CONFIG

            llm = AzureChatOpenAI(
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                deployment_name=CONFIG.get("azure_openai_deployment"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.0,  # Zero temperature for maximum factual accuracy
                timeout=30
            )

            system_template = """You are a medical documentation assistant specialized in accurate information extraction.

    CRITICAL RULES:
    1. Extract ONLY information explicitly stated in the raw summary - DO NOT infer, assume, or add any details
    2. If critical information is present, include it even if it makes the summary longer
    3. Preserve ALL specific medical details: diagnoses, medications (with dosages), test results, dates, measurements
    4. Use the EXACT medical terminology from the source - do not paraphrase medical terms
    5. If information is uncertain or not clearly stated, omit it rather than guessing
    6. Do not include generic statements like "patient was treated" without specifying what treatment

    STRUCTURE (only include sections with available information):
    - Primary diagnosis/condition with any relevant clinical findings
    - Key interventions, procedures, or medications (include specific names and dosages if mentioned)
    - Critical test results or measurements if present
    - Current status, follow-up plan, or next steps

    OUTPUT FORMAT:
    - Write in clear, concise paragraphs (NOT bullet points)
    - No headers, no "Here is the summary" preamble
    - Aim for 3-5 sentences, but extend if necessary to capture all critical information
    - Prioritize completeness and accuracy over brevity

    If the raw summary lacks substantive medical information, state "Limited clinical information available in source document" rather than fabricating content."""

            user_template = f"""Document Type: {document_type}

    Raw Summary Input:
    {raw_summary_text}

    Extract and present the concise summary following the rules above:"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("user", user_template)
            ])
            
            chain = prompt | llm
            
            # Run in executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                LLM_EXECUTOR, 
                lambda: chain.invoke({})
            )
            
            clean_summary = response.content.strip()
            
            # Validation: Check if summary is suspiciously short given substantial input
            if len(raw_summary_text) > 200 and len(clean_summary) < 50:
                logger.warning("⚠️ Generated summary may be incomplete - falling back to raw summary")
                return (raw_summary_text[:500] + "...") if len(raw_summary_text) > 500 else raw_summary_text
            
            # Validation: Check for generic/hallucinated content patterns
            hallucination_indicators = [
                "patient was seen",
                "routine care provided",
                "standard treatment given",
                "typical findings noted"
            ]
            if any(indicator in clean_summary.lower() for indicator in hallucination_indicators):
                if not any(indicator in raw_summary_text.lower() for indicator in hallucination_indicators):
                    logger.warning("⚠️ Detected potentially fabricated generic content - using raw summary")
                    return (raw_summary_text[:500] + "...") if len(raw_summary_text) > 500 else raw_summary_text
            
            logger.info(f"✅ Generated concise summary ({len(clean_summary)} chars)")
            return clean_summary

        except Exception as e:
            logger.error(f"❌ Failed to generate concise brief summary: {e}")
            # Fallback: Return truncated original text with better length handling
            return (raw_summary_text[:500] + "...") if len(raw_summary_text) > 500 else raw_summary_text
    
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
        # logger.info(f"📊 Multiple reports detected (from Document AI): {is_multiple_reports}")
        
        # 🚨 EARLY EXIT: If Document AI already detected multiple reports, return immediately
        # Do NOT run any further processing (no ReportAnalyzer, no summaries)
        # if is_multiple_reports:
        #     logger.warning("⚠️ MULTIPLE REPORTS DETECTED by Document AI - returning early, skipping all analysis")
        #     return {
        #         "document_analysis": None,
        #         "brief_summary": "",
        #         "text_for_analysis": text,
        #         "raw_text": raw_text,
        #         "report_analyzer_result": {},
        #         "patient_name": None,
        #         "claim_number": None,
        #         "dob": None,
        #         "has_patient_name": False,
        #         "has_claim_number": False,
        #         "physician_id": data.get("physician_id"),
        #         "user_id": data.get("user_id"),
        #         "filename": data["filename"],
        #         "gcs_url": data["gcs_url"],
        #         "blob_path": data.get("blob_path"),
        #         "file_size": data.get("file_size", 0),
        #         "mime_type": data.get("mime_type", "application/octet-stream"),
        #         "processing_time_ms": data.get("processing_time_ms", 0),
        #         "file_hash": data.get("file_hash"),
        #         "result_data": result_data,
        #         "document_id": data.get("document_id", "unknown"),
        #         "mode": mode,
        #         "is_multiple_reports": True,
        #         "multi_report_info": multi_report_info
        #     }
        
        # Log if raw_text is missing to help debug
        if not raw_text:
            logger.warning("⚠️ raw_text is empty - Document AI summarizer output not available, will use full OCR text as fallback")
        
        # 🔍 Check for multiple reports using MultiReportDetector
        # Use raw_text (Document AI summary) if available, otherwise use full OCR text
        # text_to_check = raw_text if raw_text else text
        # if text_to_check and len(text_to_check.strip()) > 50:
        #     try:
        #         logger.info("🔍 Running MultiReportDetector to check for multiple reports in document...")
        #         detector = get_multi_report_detector()
        #         loop = asyncio.get_event_loop()
        #         detection_result = await loop.run_in_executor(
        #             LLM_EXECUTOR, 
        #             detector.detect_multiple_reports, 
        #             text_to_check
        #         )
                
        #         # If multiple reports detected, override the flags and return early
        #         if detection_result.get("is_multiple", False):
        #             is_multiple_reports = True
        #             multi_report_info = {
        #                 "is_multiple": True,
        #                 "confidence": detection_result.get("confidence", "unknown"),
        #                 "reason": detection_result.get("reasoning", "Multiple reports detected in document"),
        #                 "report_count_estimate": detection_result.get("report_count", 2),
        #                 "reports_identified": detection_result.get("report_types", [])
        #             }
        #             logger.warning(f"⚠️ MULTIPLE REPORTS DETECTED by MultiReportDetector!")
        #             logger.warning(f"   Confidence: {multi_report_info.get('confidence')}")
        #             logger.warning(f"   Reason: {multi_report_info.get('reason')}")
        #             logger.warning(f"   Estimated count: {multi_report_info.get('report_count_estimate')}")
        #             logger.info("⏭️ Skipping further processing - will save to FailDocs")
                    
        #             # Return early with minimal data - save resources by not running expensive analyzers
        #             return {
        #                 "document_analysis": None,
        #                 "brief_summary": "",
        #                 "text_for_analysis": text,
        #                 "raw_text": raw_text,
        #                 "report_analyzer_result": {},
        #                 "patient_name": None,
        #                 "claim_number": None,
        #                 "dob": None,
        #                 "has_patient_name": False,
        #                 "has_claim_number": False,
        #                 "physician_id": data.get("physician_id"),
        #                 "user_id": data.get("user_id"),
        #                 "filename": data["filename"],
        #                 "gcs_url": data["gcs_url"],
        #                 "blob_path": data.get("blob_path"),
        #                 "file_size": data.get("file_size", 0),
        #                 "mime_type": data.get("mime_type", "application/octet-stream"),
        #                 "processing_time_ms": data.get("processing_time_ms", 0),
        #                 "file_hash": data.get("file_hash"),
        #                 "result_data": result_data,
        #                 "document_id": data.get("document_id", "unknown"),
        #                 "mode": mode,
        #                 "is_multiple_reports": is_multiple_reports,
        #                 "multi_report_info": multi_report_info
        #             }
        #         else:
        #             logger.info("✅ MultiReportDetector: Single report confirmed")
                    
        #     except Exception as e:
        #         logger.error(f"❌ Multi-report detection failed: {str(e)}")
        #         # Continue processing even if detection fails
        #         logger.warning("⚠️ Continuing with document processing despite detection error")
        
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
                    # Reuse the logic from _check_physician_author but just check the "is_clinic_member"
                    # We pass dummy summaries because we just want to match the author name
                    # But _check_physician_author extracts name from summary, so we need to construct a fake summary with signature
                    # Or better: refactor _check_physician_author to accept direct name.
                    # For now, let's duplicate the DB check logic slightly here for efficiency/speed
                    
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
        
        # Run ReportAnalyzer in dedicated LLM executor, passing pre-detected doc_type
        report_analyzer = ReportAnalyzer(mode)
        report_result = await loop.run_in_executor(
            LLM_EXECUTOR, 
            lambda: report_analyzer.extract_document(text, raw_text, doc_type_result=doc_type_result)
        )
        long_summary = report_result.get("long_summary", "")
        short_summary = report_result.get("short_summary", "")

        logger.info(f"✅ Generated long summary: {len(long_summary)} chars")
        logger.info(f"✅ Generated short summary: {short_summary}")

        # Construct DocumentAnalysis directly from ReportAnalyzer output + Pre-extracted details
        
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
        
        # ✅ Process the raw summary through the AI Condenser
        brief_summary_text = await self._generate_concise_brief_summary(
            raw_brief_summary_text, 
            detected_doc_type
        )

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
            "multi_report_info": multi_report_info
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

    async def create_treatment_history(self, 
                                    processed_data: dict, 
                                    lookup_result: dict,
                                    document_id: str = None,
                                    ai_summarizer_text: str = ""
                                    ) -> dict:
        """Step 3.5: Create or update treatment history"""
        logger.info(f"🔄 Treatment history creation is currently DISABLED in WebhookService")
        return {}
        # logger.info(f"🔄 Creating treatment history for {lookup_result.get('patient_name_to_use')}")
        # logger.info(f"   processed data: {processed_data.get('report_analyzer_result').get('short_summary', '')}")
        # 
        # try:
        #     # Initialize treatment history generator
        #     history_generator = TreatmentHistoryGenerator()
        #     
        #     # Get document analysis
        #     document_analysis = processed_data.get("document_analysis")
        #     
        #     # Extract report date from document analysis - NEVER use current date as fallback
        #     report_date = None
        #     if document_analysis:
        #         # Try to get rd (report date) from document analysis
        #         if hasattr(document_analysis, 'rd') and document_analysis.rd:
        #             rd_value = str(document_analysis.rd).lower()
        #             if rd_value not in ["not specified", "unknown", "none", ""]:
        #                 report_date = document_analysis.rd
        #         # Fallback to doi (date of injury) if rd not available
        #         if not report_date and hasattr(document_analysis, 'doi') and document_analysis.doi:
        #             doi_value = str(document_analysis.doi).lower()
        #             if doi_value not in ["not specified", "unknown", "none", ""]:
        #                 report_date = document_analysis.doi
        #     
        #     # Prepare current document data for history generator
        #     # CRITICAL: Only include date if actually extracted from document, NEVER use current date
        #     current_doc_data = {
        #         "short_summary": processed_data.get("brief_summary", ""),
        #         "long_summary": processed_data.get("text_for_analysis", ""),
        #         "briefSummary": processed_data.get("brief_summary", ""),
        #         "whatsNew": json.dumps({
        #             "long_summary": processed_data.get("text_for_analysis", ""),
        #             "short_summary": processed_data.get("brief_summary", "")
        #         }) if processed_data.get("text_for_analysis") else None
        #     }
        #     
        #     # Only add date fields if they were actually extracted from the document
        #     if report_date:
        #         current_doc_data["createdAt"] = report_date
        #         current_doc_data["documentDate"] = report_date
        #         current_doc_data["date"] = report_date
        #         logger.info(f"📅 Using extracted report date for treatment history: {report_date}")
        #     else:
        #         logger.warning(f"⚠️ No report date found in document - treatment history will use 'Date not specified'")
        #     
        #     # Add body part snapshots if available
        #     if hasattr(document_analysis, 'body_parts_analysis') and document_analysis.body_parts_analysis:
        #         current_doc_data["bodyPartSnapshots"] = []
        #         for bp in document_analysis.body_parts_analysis:
        #             if hasattr(bp, 'dict'):
        #                 current_doc_data["bodyPartSnapshots"].append(bp.dict())
        #             else:
        #                 current_doc_data["bodyPartSnapshots"].append({
        #                     "bodyPart": getattr(bp, 'body_part', None),
        #                     "condition": getattr(bp, 'condition', None),
        #                     "dx": getattr(bp, 'diagnosis', None),
        #                     "keyConcern": getattr(bp, 'key_concern', None),
        #                     "nextStep": getattr(bp, 'extracted_recommendation', None),
        #                     "clinicalSummary": getattr(bp, 'clinical_summary', None),
        #                     "treatmentApproach": getattr(bp, 'treatment_plan', None),
        #                     "adlsAffected": getattr(bp, 'adls_affected', None)
        #                 })
        #     
        #     # ✅ Step 3.4: Check if treatment history already exists
        #     existing_history = await history_generator.get_treatment_history(
        #         patient_name=lookup_result.get("patient_name_to_use"),
        #         dob=getattr(document_analysis, 'dob', None) if document_analysis else None,
        #         claim_number=lookup_result.get("claim_to_save"),
        #         physician_id=processed_data.get("physician_id")
        #     )
        #     
        #     # If history exists, we only generate history for the CURRENT document and then merge/archive
        #     # If history doesn't exist, we generate from ALL documents
        #     only_current = existing_history is not None
        #     if only_current:
        #         logger.info(f"🔄 Treatment history exists for {lookup_result.get('patient_name_to_use')}, generating new entries for archive/merge")
        #     # logger.info(f"📝 Generating treatment history for patient: {ai_summarizer_text}")
        #     # Create treatment history
        #     treatment_history = await history_generator.generate_treatment_history(
        #         patient_name=lookup_result.get("patient_name_to_use"),
        #         dob=getattr(document_analysis, 'dob', None) if document_analysis else None,
        #         claim_number=lookup_result.get("claim_to_save"),
        #         physician_id=processed_data.get("physician_id"),
        #         current_document_id=document_id,
        #         current_document_data=processed_data.get('report_analyzer_result', {}).get('short_summary', ''),
        #         only_current=only_current
        #         
        #     )
        #     
        #     # ✅ Step 3.6: Save treatment history to database (Moved from TreatmentHistoryGenerator)
        #     if treatment_history:
        #         logger.info(f"💾 Saving treatment history to database for {lookup_result.get('patient_name_to_use')}")
        #         await history_generator.save_treatment_history(
        #             patient_name=lookup_result.get("patient_name_to_use"),
        #             dob=getattr(document_analysis, 'dob', None) if document_analysis else None,
        #             claim_number=lookup_result.get("claim_to_save"),
        #             physician_id=processed_data.get("physician_id"),
        #             history_data=treatment_history,
        #             document_id=document_id
        #         )
        #     
        #     logger.info(f"✅ Treatment history created and saved for {lookup_result.get('patient_name_to_use')}")
        #     return treatment_history
        #     
        # except ImportError as e:
        #     logger.error(f"❌ TreatmentHistoryGenerator not found: {str(e)}")
        #     return {}
        # except Exception as e:
        #     logger.error(f"❌ Error creating/saving treatment history: {str(e)}")
        #     # Return empty template on error
        #     return {
        #         "musculoskeletal_system": [],
        #         "cardiovascular_system": [],
        #         "pulmonary_respiratory": [],
        #         "neurological": [],
        #         "gastrointestinal": [],
        #         "metabolic_endocrine": [],
        #         "other_systems": [],
        #         "general_treatments": []
        #     }
        # finally:
        #     # ✅ Ensure database connection is closed
        #     try:
        #         if 'history_generator' in locals():
        #             await history_generator.disconnect()
        #     except Exception as disconnect_error:
        #         logger.warning(f"⚠️ Error disconnecting history generator: {disconnect_error}")
    
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
        """Step 3: Create tasks if conditions are met - ONLY for EXTERNAL documents"""
        logger.info(f"🔧 Checking document {filename} for task creation...")
        created_tasks = 0
        
        try:
            # Get summaries for author check
            report_analyzer_result = processed_data.get("report_analyzer_result", {}) if processed_data else {}
            short_summary = report_analyzer_result.get("short_summary", "") if isinstance(report_analyzer_result, dict) else ""
            long_summary = report_analyzer_result.get("long_summary", "") if isinstance(report_analyzer_result, dict) else ""
            
            # Check for document author using summaries
            author_info = await self._check_physician_author(
                physician_id=physician_id,
                short_summary=short_summary,
                long_summary=long_summary
            )
            
            # If no author found, cannot process
            if not author_info.get("author_found"):
                logger.warning(f"⚠️ {author_info.get('error_message', 'No author found in document')}")
                return -2  # Return -2 to indicate no author found
            
            author_name = author_info.get("author_name", "Unknown")
            
            # If author is from our clinic (INTERNAL document), cannot process
            if author_info.get("is_internal_document") or author_info.get("is_clinic_member"):
                logger.warning(f"⚠️ Document author '{author_name}' is a clinic member - INTERNAL document cannot be processed")
                return -1  # Return -1 to indicate internal document
            
            # EXTERNAL document - can create tasks
            logger.info(f"✅ Document is EXTERNAL (author: {author_name}) - Creating tasks...")
            
            # Generate and create tasks for EXTERNAL documents
            task_creator = TaskCreator()
            document_data = document_analysis.dict()
            document_data["filename"] = filename
            document_data["document_id"] = document_id
            document_data["physician_id"] = physician_id
            
            # Get document text for task generation - prioritize raw_text (Document AI summarizer output)
            full_text = ""
            if processed_data:
                # PRIMARY: Use raw_text (Document AI summarizer output) for accurate context
                full_text = processed_data.get("raw_text", "")
                # FALLBACK: Use full OCR text only if raw_text is not available
                if not full_text:
                    full_text = processed_data.get("text_for_analysis", "")
                    if not full_text:
                        result_data = processed_data.get("result_data", {})
                        full_text = result_data.get("text", "")
            
            logger.info(f"📝 Passing {len(full_text)} characters to task generator (from {'Document AI summarizer' if processed_data.get('raw_text') else 'OCR text'})")
            
            tasks_result = await task_creator.generate_tasks(document_data, filename, full_text, author_name)
            
            # Extract tasks - for external documents we create tasks
            external_tasks = tasks_result.get("internal_tasks", [])  # Using same key for now
            
            logger.info(f"📋 Document is EXTERNAL (author: {author_name}) - Processing {len(external_tasks)} tasks")

            # Save tasks to database
            prisma = Prisma()
            await prisma.connect()
            
            # Process tasks for external documents
            for task in external_tasks:
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
                        "details": quick_notes.get("details", f"External document from {author_name}"),
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
                    logger.info(f"✅ Created EXTERNAL task: {task.get('description', 'Unknown task')}")
                    
                except Exception as task_err:
                    logger.error(f"❌ Failed to create task: {task_err}")
                    continue
            

            await prisma.disconnect()
            logger.info(f"✅ {created_tasks} tasks created successfully for external document")
            
        except Exception as e:
            logger.error(f"❌ Error in task creation: {str(e)}")
        
        return created_tasks
    async def save_document(self, db_service, processed_data: dict, lookup_result: dict) -> dict:
        """Step 4: Save document to database and Redis cache"""
        
        # ✅ Check if both DOB and claim number are not specified
        document_analysis = processed_data["document_analysis"]
        dob_not_specified = (
            not hasattr(document_analysis, 'dob') or 
            not document_analysis.dob or 
            str(document_analysis.dob).lower() in ["not specified", "none", ""]
        )
        
        claim_not_specified = (
            not lookup_result.get("claim_to_save") or 
            str(lookup_result["claim_to_save"]).lower() in ["not specified", "none", ""]
        )
        
        # If both DOB and claim number are not specified, save as fail document
        if dob_not_specified and claim_not_specified:
            # ✅ Get the actual parsed text from the result data
            parsed_text = processed_data["result_data"].get("text", "")
            
            # ✅ Also get the brief summary and other analysis data
            brief_summary = processed_data.get("brief_summary", "")
            
            # ✅ Combine text and summary for comprehensive document text
            full_document_text = f"ORIGINAL TEXT:\n{parsed_text}\n\nSUMMARY:\n{brief_summary}"
            
            fail_doc_id = await db_service.save_fail_doc(
                reason="Both DOB and claim number are not specified",
                db=document_analysis.dob if hasattr(document_analysis, 'dob') else None,
                claim_number=lookup_result.get("claim_to_save"),
                patient_name=lookup_result.get("patient_name_to_use"),
                physician_id=processed_data.get("physician_id"),
                gcs_file_link=processed_data.get("gcs_url"),
                file_name=processed_data.get("filename"),
                file_hash=processed_data.get("file_hash"),
                blob_path=processed_data.get("blob_path"),
                mode=processed_data.get("mode"),
                # ✅ SAVE THE PARSED TEXT AND SUMMARY
                document_text=full_document_text,
                doi=document_analysis.doi if hasattr(document_analysis, 'doi') else None,
                ai_summarizer_text=brief_summary
            )

            
            # ✅ Decrement parse count even for failed documents since they consumed resources
            parse_decremented = await db_service.decrement_parse_count(processed_data["physician_id"])
            
            return {
                "status": "failed",
                "document_id": fail_doc_id,
                "parse_count_decremented": parse_decremented,  # Now True for failed docs too
                "filename": processed_data["filename"],
                "cache_success": False,
                "failure_reason": "Both DOB and claim number are not specified"
            }
        
        # Continue with normal document saving process...
        # Create ExtractionResult
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
            summary=processed_data["brief_summary"],
            document_id=processed_data["result_data"].get("document_id", f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        )
        
        # Prepare the additional required parameters with default values
        document_analysis = processed_data["document_analysis"]
        
        # Get RD (Report Date) - use from analysis
        rd = None
        if hasattr(document_analysis, 'rd') and document_analysis.rd and str(document_analysis.rd).lower() != "not specified":
            try:
                date_str = str(document_analysis.rd).strip()
                logger.info(f"📅 Parsing report date: {date_str}")
                
                # Try multiple date formats
                date_formats = [
                    "%Y-%m-%d",      # 2025-11-25
                    "%m-%d-%Y",      # 11-25-2025
                    "%m/%d/%Y",      # 11/25/2025
                    "%m/%d/%y",      # 11/25/25
                    "%d-%m-%Y",      # 25-11-2025
                    "%d/%m/%Y",      # 25/11/2025
                    "%Y/%m/%d",      # 2025/11/25
                ]
                
                parsed = False
                for fmt in date_formats:
                    try:
                        rd = datetime.strptime(date_str, fmt)
                        # Set time to noon to avoid timezone shifting issues
                        rd = rd.replace(hour=12, minute=0, second=0, microsecond=0)
                        logger.info(f"✅ Parsed report date with format {fmt}: {rd}")
                        parsed = True
                        break
                    except ValueError:
                        continue
                
                if not parsed:
                    logger.warning(f"⚠️ Could not parse report date: {date_str}, keeping as None")
                    rd = None
                    
            except Exception as e:
                logger.warning(f"⚠️ Error parsing report date: {e}")
                rd = None  # Don't fallback to current date - leave as None
        
        # Create summary snapshots
        summary_snapshots = []
        if hasattr(document_analysis, 'body_parts_analysis') and document_analysis.body_parts_analysis:
            for body_part_analysis in document_analysis.body_parts_analysis:
                snapshot = {
                    "mode": processed_data["mode"],
                    "body_part": body_part_analysis.body_part if processed_data["mode"] == "wc" else None,
                    "condition": None if processed_data["mode"] == "wc" else body_part_analysis.body_part,
                    "dx": body_part_analysis.diagnosis,
                    "key_concern": body_part_analysis.key_concern,
                    "next_step": body_part_analysis.extracted_recommendation or None,
                    "ur_decision": document_analysis.ur_decision or None,
                    "recommended": body_part_analysis.extracted_recommendation or None,
                    "ai_outcome": document_analysis.ai_outcome or None,
                    "consulting_doctor": document_analysis.consulting_doctor or None,
                    "key_findings": body_part_analysis.clinical_summary or None,
                    "treatment_approach": body_part_analysis.treatment_plan or None,
                    "clinical_summary": body_part_analysis.clinical_summary or None,
                    "referral_doctor": document_analysis.referral_doctor or None,
                    "adls_affected": body_part_analysis.adls_affected or None,
                }
                summary_snapshots.append(snapshot)
        else:
            # Create a single snapshot if no body parts analysis
            snapshot = {
                "mode": processed_data["mode"],
                "body_part": document_analysis.body_part if processed_data["mode"] == "wc" else None,
                "condition": None if processed_data["mode"] == "wc" else document_analysis.body_part,
                "dx": document_analysis.diagnosis,
                "key_concern": document_analysis.key_concern,
                "next_step": document_analysis.extracted_recommendation or None,
                "ur_decision": document_analysis.ur_decision or None,
                "recommended": document_analysis.extracted_recommendation or None,
                "ai_outcome": document_analysis.ai_outcome or None,
                "consulting_doctor": document_analysis.consulting_doctor or None,
                "key_findings": document_analysis.diagnosis or None,
                "treatment_approach": document_analysis.extracted_recommendation or None,
                "clinical_summary": f"{document_analysis.diagnosis} - {document_analysis.key_concern}" or None,
                "referral_doctor": document_analysis.referral_doctor or None,
                "adls_affected": document_analysis.adls_affected or None,
            }
            summary_snapshots.append(snapshot)
        
        # ✅ FIXED: Get the ACTUAL long and short summaries from ReportAnalyzer
        report_analyzer_result = processed_data.get("report_analyzer_result", {})
        
        # Use the actual summaries from ReportAnalyzer if available
        if report_analyzer_result and isinstance(report_analyzer_result, dict):
            long_summary = report_analyzer_result.get("long_summary", "")
            short_summary = report_analyzer_result.get("short_summary", "")
        else:
            # Fallback to the existing data
            long_summary = processed_data.get("text_for_analysis", "")
            short_summary = processed_data["brief_summary"]
        
        whats_new = {
            "long_summary": long_summary,
            "short_summary": short_summary
        }
        
        # Create ADL data
        adl_data = {
            "mode": processed_data["mode"],
            "adls_affected": document_analysis.adls_affected if hasattr(document_analysis, 'adls_affected') else None,
            "work_restrictions": document_analysis.work_restrictions if hasattr(document_analysis, 'work_restrictions') else None
        }
        
        # Create document summary
        document_summary = {
            "type": document_analysis.document_type if hasattr(document_analysis, 'document_type') else "Unknown",
            "created_at": datetime.now(),
            "summary": " | ".join(document_analysis.summary_points) if hasattr(document_analysis, 'summary_points') and document_analysis.summary_points else processed_data["brief_summary"]
        }
        
        # Generate structured filename: patientName_typeOfReport_dateOfReport
        original_filename = processed_data["filename"]
        document_type = document_analysis.document_type if hasattr(document_analysis, 'document_type') else "Document"
        structured_filename = self._generate_document_filename(
            patient_name=lookup_result["patient_name_to_use"],
            document_type=document_type,
            report_date=rd,
            original_filename=original_filename
        )
        
        # Save document to database with all required parameters
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=structured_filename,  # New structured filename
            file_size=processed_data["file_size"],
            mime_type=processed_data["mime_type"],
            processing_time_ms=processed_data["processing_time_ms"],
            blob_path=processed_data["blob_path"],
            file_hash=processed_data["file_hash"],
            gcs_file_link=processed_data["gcs_url"],
            patient_name=lookup_result["patient_name_to_use"],
            claim_number=lookup_result["claim_to_save"],
            dob=document_analysis.dob if hasattr(document_analysis, 'dob') else None,
            doi=document_analysis.doi if hasattr(document_analysis, 'doi') else None,
            status=lookup_result["document_status"],
            brief_summary=processed_data["brief_summary"],
            physician_id=processed_data["physician_id"],
            mode=processed_data["mode"],
            # Add the missing required parameters
            rd=rd,
            summary_snapshots=summary_snapshots,
            whats_new=whats_new,
            adl_data=adl_data,
            document_summary=document_summary,
            original_name=original_filename,  # Store original filename
            ai_summarizer_text=processed_data.get("raw_text")
        )
        
        # SAVE TO REDIS CACHE (only for successful documents, not for fail documents)
        cache_success = False
        if document_id:
            # Prepare data for caching
            cache_data = {
                "patient_name": lookup_result["patient_name_to_use"],
                "claim_number": lookup_result["claim_to_save"],
                "dob": document_analysis.dob if hasattr(document_analysis, 'dob') else None,
                "doi": document_analysis.doi if hasattr(document_analysis, 'doi') else None,
                "physician_id": processed_data["physician_id"],
                "status": lookup_result["document_status"],
                "mode": processed_data["mode"],
                "brief_summary": processed_data["brief_summary"],
                "filename": processed_data["filename"],
                "document_analysis": document_analysis.dict() if hasattr(document_analysis, 'dict') else str(document_analysis),
                "summary_snapshots": summary_snapshots,
                "created_at": datetime.now().isoformat()
            }
            
            cache_success = await self.save_to_redis_cache(document_id, cache_data)
        
        # Decrement parse count
        parse_decremented = await db_service.decrement_parse_count(processed_data["physician_id"])
        
        return {
            "status": lookup_result["document_status"],
            "document_id": document_id,
            "parse_count_decremented": parse_decremented,
            "filename": processed_data["filename"],
            "cache_success": cache_success
        }

    async def handle_webhook(self, data: dict, db_service) -> dict:
        """
        Clean webhook processing pipeline WITHOUT duplicate prevention
        Includes treatment history creation
        """
        try:
            # Test Redis connection first
            redis_test = await self.test_redis_basic()
            if not redis_test:
                logger.warning("⚠️ Redis basic test failed - caching will be disabled")
            
            # Step 1: Process document data
            processed_data = await self.process_document_data(data)
            
            # Step 1.1: Check for multiple reports FIRST - if detected, save to FailDocs immediately
            # This must happen before author check since multi-report docs return early with no analysis
            if processed_data.get("is_multiple_reports", False):
                multi_report_info = processed_data.get("multi_report_info", {})
                confidence = multi_report_info.get("confidence", "unknown")
                reason = multi_report_info.get("reason", "Multiple reports detected")
                report_count = multi_report_info.get("report_count_estimate", 2)
                reports_identified = multi_report_info.get("reports_identified", [])
                
                # Get the original text/summary
                raw_text = processed_data.get("raw_text", "")
                text_for_analysis = processed_data.get("text_for_analysis", "")
                
                # Create enhanced summary with detection details
                summary_parts = []
                if raw_text:
                    summary_parts.append(f"DOCUMENT AI SUMMARY:\n{raw_text}")
                summary_parts.append(f"\n\n=== MULTI-REPORT DETECTION RESULTS ===")
                summary_parts.append(f"Confidence: {confidence}")
                summary_parts.append(f"Reason: {reason}")
                summary_parts.append(f"Estimated Report Count: {report_count}")
                if reports_identified:
                    summary_parts.append(f"Reports Identified: {', '.join(reports_identified)}")
                summary_text = "\n".join(summary_parts)
                
                logger.warning(f"⚠️ MULTIPLE REPORTS DETECTED (confidence: {confidence})")
                logger.warning(f"   Reason: {reason}")
                logger.warning(f"   Estimated count: {report_count}")
                logger.info("💾 Saving to FailDocs for manual review...")
                
                # Save to FailDocs with clear reason for multiple reports
                fail_doc_id = await db_service.save_fail_doc(
                    reason=f"Multiple reports detected. Found {report_count} reports with {confidence} confidence. Report types: {', '.join(reports_identified) if reports_identified else 'Unknown'}",
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
                    ai_summarizer_text=raw_text  # Store raw AI summarizer output directly
                )
                
                # Decrement parse count
                parse_decremented = await db_service.decrement_parse_count(processed_data.get("physician_id"))
                
                logger.info(f"✅ Multiple reports document saved to FailDocs with ID: {fail_doc_id}")
                
                return {
                    "status": "multiple_reports_detected",
                    "document_id": fail_doc_id,
                    "filename": processed_data.get("filename"),
                    "parse_count_decremented": parse_decremented,
                    "failure_reason": f"Multiple reports detected. {reason}",
                    "multi_report_info": multi_report_info
                }
            
            # Step 1.2: Check if author is from our clinic (INTERNAL document check)
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
                    ai_summarizer_text=f"No author detected in document.\nShort Summary: {short_summary_text[:200] if short_summary_text else 'N/A'}..."
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
                    reason=f"Internal document detected - author '{author_name}' is a clinic member. Cannot process internal documents.",
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
                    ai_summarizer_text=f"Internal document detected.\nShort Summary: {short_summary_text_internal[:200] if short_summary_text_internal else 'N/A'}..."
                )
                
                parse_decremented = await db_service.decrement_parse_count(processed_data.get("physician_id"))
                
                return {
                    "status": "internal_document_fail",
                    "document_id": fail_doc_id,
                    "filename": processed_data.get("filename"),
                    "parse_count_decremented": parse_decremented,
                    "failure_reason": f"Internal document detected - author '{author_name}' is a clinic member",
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
            # treatment_history = {}
            # if save_result["document_id"] and save_result["status"] != "failed":
            #     treatment_history = await self.create_treatment_history(
            #         processed_data=processed_data,
            #         lookup_result=lookup_result,
            #         document_id=save_result["document_id"],
            #         ai_summarizer_text=processed_data.get("raw_text", "")
            #     )
            #     logger.info(f"✅ Treatment history created with {sum(len(v) for v in treatment_history.values())} total events across {len(treatment_history)} categories")
            # else:
            #     logger.info("⚠️ Skipping treatment history creation - document not saved or failed")
            
            # Step 4: Create tasks if document was saved successfully (NOT for FailDocs)
            tasks_created = 0
            if save_result["document_id"] and save_result["status"] != "failed":
                tasks_created = await self.create_tasks_if_needed(
                    processed_data["document_analysis"],
                    save_result["document_id"],
                    processed_data["physician_id"],
                    processed_data["filename"],
                    processed_data  # Pass full processed_data for document text access
                )
                
                if tasks_created == -1:
                    # This should ideally be caught by Step 1.2, but as a safety measure:
                    logger.warning(f"⚠️ Task creation returned -1 (self-authored).")
                    # We already saved the document, so we might need to move it to FailDocs here too
                    # but Step 1.2 should have caught it.
                    pass
            else:
                logger.info(f"⚠️ Skipping task creation - document was saved as FailDoc (status: {save_result.get('status')})")
            
            # Step 5: Post-save check - Verify document doesn't contain multiple different reports
            # This check happens AFTER document is saved to catch cases where multiple reports
            # of the same patient (e.g., QME + PR2) are combined in one document
            # if save_result["document_id"] and save_result["status"] != "failed":
            #     try:
            #         logger.info("🔍 Running post-save multi-report detection check...")
                    
            #         # Get document text for analysis
            #         raw_text = processed_data.get("raw_text", "")
            #         text_for_analysis = processed_data.get("text_for_analysis", "")
            #         text_to_check = raw_text if raw_text else text_for_analysis
                    
            #         if text_to_check and len(text_to_check.strip()) > 50:
            #             detector = get_multi_report_detector()
            #             loop = asyncio.get_event_loop()
            #             detection_result = await loop.run_in_executor(
            #                 LLM_EXECUTOR,
            #                 detector.detect_multiple_reports,
            #                 text_to_check
            #             )
                        
            #             # If multiple reports detected, move document to FailDocs
            #             if detection_result.get("is_multiple", False):
            #                 confidence = detection_result.get("confidence", "unknown")
            #                 reason = detection_result.get("reason", "Multiple reports detected in document")
            #                 report_count = detection_result.get("report_count_estimate", 2)
            #                 reports_identified = detection_result.get("reports_identified", [])
                            
            #                 logger.warning(f"⚠️ POST-SAVE CHECK: Multiple reports detected in saved document!")
            #                 logger.warning(f"   Document ID: {save_result['document_id']}")
            #                 logger.warning(f"   Confidence: {confidence}")
            #                 logger.warning(f"   Reason: {reason}")
            #                 logger.warning(f"   Estimated count: {report_count}")
            #                 logger.info("🔄 Moving document from Documents to FailDocs...")
                            
            #                 # Get document details before deletion
            #                 document_details = await db_service.get_document(save_result["document_id"])
                            
            #                 # Create enhanced summary with detection details
            #                 summary_parts = []
            #                 if raw_text:
            #                     summary_parts.append(f"DOCUMENT AI SUMMARY:\n{raw_text}")
            #                 summary_parts.append(f"\n\n=== POST-SAVE MULTI-REPORT DETECTION RESULTS ===")
            #                 summary_parts.append(f"Confidence: {confidence}")
            #                 summary_parts.append(f"Reason: {reason}")
            #                 summary_parts.append(f"Estimated Report Count: {report_count}")
            #                 if reports_identified:
            #                     summary_parts.append(f"Reports Identified: {', '.join(reports_identified)}")
            #                 summary_parts.append(f"\nNote: Document was initially saved successfully but failed post-save validation.")
            #                 summary_text = "\n".join(summary_parts)
                            
            #                 # Save to FailDocs with all document information
            #                 fail_doc_id = await db_service.save_fail_doc(
            #                     reason=f"Multiple reports detected after save ({report_count} reports) - manual review required. {reason}",
            #                     db=document_details.get("dob") if document_details else processed_data.get("dob"),
            #                     claim_number=document_details.get("claimNumber") if document_details else processed_data.get("claim_number"),
            #                     patient_name=document_details.get("patientName") if document_details else processed_data.get("patient_name"),
            #                     physician_id=processed_data.get("physician_id"),
            #                     gcs_file_link=processed_data.get("gcs_url"),
            #                     file_name=processed_data.get("filename"),
            #                     file_hash=processed_data.get("file_hash"),
            #                     blob_path=processed_data.get("blob_path"),
            #                     mode=processed_data.get("mode", "wc"),
            #                     document_text=text_for_analysis if text_for_analysis else raw_text,
            #                     doi=document_details.get("doi") if document_details else None,
            #                     ai_summarizer_text=summary_text
            #                 )
                            
            #                 # Delete the document and all related records including treatment history
            #                 try:
            #                     # First, delete the treatment history if it exists
            #                     if treatment_history:
            #                         history_generator = TreatmentHistoryGenerator()
            #                         await history_generator.connect()
            #                         await history_generator.prisma.treatmenthistory.delete_many(
            #                             where={"documentId": save_result["document_id"]}
            #                         )
            #                         await history_generator.disconnect()
                                
            #                     # Delete the document
            #                     await db_service._delete_existing_document(save_result["document_id"])
            #                     logger.info(f"🗑️ Deleted document {save_result['document_id']} and related treatment history after multi-report detection")
            #                 except Exception as delete_error:
            #                     logger.error(f"❌ Failed to delete document {save_result['document_id']}: {str(delete_error)}")
            #                     # Continue even if deletion fails - document is already in FailDocs
                            
            #                 logger.info(f"✅ Document moved to FailDocs with ID: {fail_doc_id}")
                            
            #                 # Return failure status
            #                 return {
            #                     "status": "multiple_reports_detected_after_save",
            #                     "document_id": fail_doc_id,
            #                     "filename": processed_data["filename"],
            #                     "parse_count_decremented": save_result.get("parse_count_decremented", False),
            #                     "failure_reason": f"Multiple reports detected after save - document moved to FailDocs. {reason}",
            #                     "original_document_id": save_result["document_id"],
            #                     "multi_report_info": {
            #                         "confidence": confidence,
            #                         "reason": reason,
            #                         "report_count_estimate": report_count,
            #                         "reports_identified": reports_identified
            #                     }
            #                 }
            #             else:
            #                 logger.info("✅ Post-save check: Single report confirmed - document is valid")
            #         else:
            #             logger.info("⚠️ Post-save check skipped: Document text too short for analysis")
                        
            #     except Exception as check_error:
            #         logger.error(f"❌ Post-save multi-report check failed: {str(check_error)}")
            #         # Don't fail the document if the check itself fails - log and continue
            #         logger.warning("⚠️ Continuing with saved document despite check error")
            
            
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
        """
        # Use updated values if provided, otherwise fallback to fail_doc values
        document_text = updated_fields.get("document_text") or fail_doc.documentText
        dob_str = updated_fields.get("dob") or fail_doc.dob  # ✅ FIXED: Changed from fail_doc.db to fail_doc.dob
        doi = updated_fields.get("doi") or fail_doc.doi
        claim_number = updated_fields.get("claim_number") or fail_doc.claimNumber
        patient_name = updated_fields.get("patient_name") or fail_doc.patientName
        author = updated_fields.get("author") or fail_doc.author  # ✅ Get author from client or fail_doc
        physician_id = fail_doc.physicianId
        filename = fail_doc.fileName
        gcs_url = fail_doc.gcsFileLink
        blob_path = fail_doc.blobPath
        file_hash = fail_doc.fileHash
        mode = "wc"  # Default mode, you can extract from fail_doc if available

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

       

        try:
            # Step 1: Process document data through BOTH EnhancedReportAnalyzer and ReportAnalyzer
            logger.info(f"🔄 Processing failed document through LLM analysis: {fail_doc.id}")
            
            # Generate long summary using ReportAnalyzer (same as process_document_data)
            report_analyzer = ReportAnalyzer(mode)
            report_result = await asyncio.to_thread(
                report_analyzer.extract_document,
                document_text,
                document_text  # Use same text for raw_text parameter
            )
            
            # ✅ STORE THE ACTUAL REPORT ANALYZER RESULT
            long_summary = report_result.get("long_summary", "")
            short_summary = report_result.get("short_summary", "")
            logger.info(f"✅ ReportAnalyzer long summary in author: {author}")
            
            # ✅ If author provided by user, replace or inject it into long_summary as "Signature:" field
            if author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
                import re
                # Replace existing signature line or add new one
                signature_pattern = r'•\s*Signature:.*?(?=\n•|\n\n|$)'
                signature_line = f"• Signature: {author.strip()}"
                
                if re.search(signature_pattern, long_summary, re.IGNORECASE | re.DOTALL):
                    # Replace existing signature
                    long_summary = re.sub(signature_pattern, signature_line, long_summary, flags=re.IGNORECASE | re.DOTALL)
                    logger.info(f"✅ Replaced existing signature with: {author}")
                else:
                    # Add new signature line
                    long_summary = long_summary + f"\n\n{signature_line}"
                    logger.info(f"✅ Injected author into long_summary: {author}")
                
                # Update the report_result dictionary to reflect the modified long_summary
                report_result["long_summary"] = long_summary
                
                # ✅ Also update the author field in short_summary header
                if isinstance(short_summary, dict) and 'header' in short_summary:
                    short_summary['header']['author'] = author.strip()
                    report_result["short_summary"] = short_summary
                    logger.info(f"✅ Updated short_summary header author: {author}")
            
            logger.info(f"✅ Generated long summary: {long_summary}")
            logger.info(f"✅ Generated short summary: {short_summary}")
            
            # Use ReportAnalyzer output directly (avoiding EnhancedReportAnalyzer)
            detected_doc_type = report_result.get('doc_type', 'Unknown')
            
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
            
            # ✅ Process the raw summary through the AI Condenser
            brief_summary_text = await self._generate_concise_brief_summary(
                raw_brief_summary_text, 
                detected_doc_type
            )
            
            # Prepare fields for DocumentAnalysis
            da_patient_name = patient_name or "Not specified"
            da_claim_number = claim_number or "Not specified"
            da_dob = dob_str or "0000-00-00" 
            da_doi = doi or "0000-00-00"
            da_author = author or "Not specified"
            
            # Manually construct DocumentAnalysis (replicating normal flow without EnhancedReportAnalyzer)
            document_analysis = DocumentAnalysis(
                patient_name=da_patient_name,
                claim_number=da_claim_number,
                dob=da_dob,
                doi=da_doi,
                status="Not specified",
                rd="0000-00-00", 
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
                consulting_doctor=da_author,
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
                verification_notes=["Analysis from basic ReportAnalyzer (Update Fail Doc)"]
            )
            
            brief_summary = document_analysis.brief_summary
            
            # Override with updated fields from the user
            if updated_fields.get("patient_name") and str(updated_fields["patient_name"]).lower() != "not specified":
                document_analysis.patient_name = updated_fields["patient_name"]
                logger.info(f"✅ Overridden patient_name: {updated_fields['patient_name']}")
            
            if updated_fields.get("dob") and str(updated_fields["dob"]).lower() != "not specified":
                document_analysis.dob = updated_fields["dob"]
                logger.info(f"✅ Overridden DOB: {updated_fields['dob']}")
            
            if updated_fields.get("doi") and str(updated_fields["doi"]).lower() != "not specified":
                document_analysis.doi = updated_fields["doi"]
                logger.info(f"✅ Overridden DOI: {updated_fields['doi']}")
            
            if updated_fields.get("claim_number") and str(updated_fields["claim_number"]).lower() != "not specified":
                document_analysis.claim_number = updated_fields["claim_number"]
                logger.info(f"✅ Overridden claim_number: {updated_fields['claim_number']}")
            
            # ✅ Override consulting_doctor (author) if provided by user
            if author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
                document_analysis.consulting_doctor = author.strip()
                logger.info(f"✅ Overridden consulting_doctor (author): {author}")

            logger.info(f"author detected : {author}")

            # Prepare processed_data similar to process_document_data
            processed_data = {
                "document_analysis": document_analysis,
                "brief_summary": brief_summary,
                "text_for_analysis": document_text,
                "report_analyzer_result": report_result,
                "patient_name": document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else None,
                "claim_number": document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else None,
                "dob": document_analysis.dob if hasattr(document_analysis, 'dob') and document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else None,
                "has_patient_name": bool(document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"),
                "has_claim_number": bool(document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"),
                "physician_id": physician_id,
                "user_id": user_id,
                "filename": filename,
                "gcs_url": gcs_url,
                "blob_path": blob_path,
                "file_size": 0,
                "mime_type": "application/octet-stream",
                "processing_time_ms": 0,
                "file_hash": file_hash,
                "result_data": result_data,
                "document_id": str(fail_doc.id),
                "mode": mode
            }

            # Step 2: Perform patient lookup with enhanced fuzzy matching
            logger.info("🔍 Performing patient lookup for updated failed document...")
            lookup_result = await self.patient_lookup.perform_patient_lookup(db_service, processed_data)
            
            # Step 3: Save document to database
            logger.info("💾 Saving updated document to database...")
            save_result = await self.save_document(db_service, processed_data, lookup_result)
            
            # Step 4: Create tasks if needed
            tasks_created = 0
            if save_result["document_id"] and save_result["status"] != "failed":
                tasks_created = await self.create_tasks_if_needed(
                    processed_data["document_analysis"],
                    save_result["document_id"],
                    processed_data["physician_id"],
                    processed_data["filename"],
                    processed_data  # Pass full processed_data for document text access
                )
                save_result["tasks_created"] = tasks_created

            # Step 5: Delete the FailDoc only if successful
            if save_result["status"] != "failed" and save_result["document_id"]:
                await db_service.delete_fail_doc(fail_doc.id)
                logger.info(f"🗑️ Deleted fail doc {fail_doc.id} after successful update")
                logger.info(f"📡 Success event processed for document: {save_result['document_id']}")
            else:
                logger.warning(f"⚠️ Failed document update unsuccessful, keeping fail doc {fail_doc.id}")
                # Optionally update the fail doc with the new failure reason
                if save_result.get("failure_reason"):
                    logger.info(f"📝 Updating fail doc reason: {save_result['failure_reason']}")

            return save_result

        except Exception as e:
            logger.error(f"❌ Failed to update fail document {fail_doc.id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Update fail document processing failed: {str(e)}")