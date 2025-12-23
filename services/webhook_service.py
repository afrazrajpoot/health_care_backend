"""
OPTIMIZED Webhook Service - Clean Version with Redis Caching (No Duplication)
"""
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any
from models.schemas import ExtractionResult
from services.database_service import get_database_service
from services.report_analyzer import ReportAnalyzer
from services.task_creation import TaskCreator
from services.resoning_agent import EnhancedReportAnalyzer
from services.patient_lookup_service import EnhancedPatientLookup
from utils.logger import logger
from prisma import Prisma
from concurrent.futures import ThreadPoolExecutor
import asyncio
import re
import json

# Dedicated thread pool for LLM operations (shared across all WebhookService instances)
LLM_EXECUTOR = ThreadPoolExecutor(max_workers=10)

class WebhookService:
    """
    Clean Webhook Service with essential features:
    - EnhancedReportAnalyzer for data extraction
    - ReportAnalyzer for summaries
    - Patient lookup (no duplicate checking)
    - Task generation with conditions
    - Mode-aware processing (WC/GM)
    """
    
    def __init__(self, redis_client=None):
        logger.info("✅ WebhookService initialized")
        self.redis_client = redis_client
        self.patient_lookup = EnhancedPatientLookup(redis_client=redis_client)
    
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
    
    async def debug_redis_contents(self, pattern: str = "patient_lookup:*"):
        """Debug method to see what's actually in Redis"""
        if not self.redis_client:
            logger.error("❌ Redis client not available for debug")
            return
        
        try:
            keys = await self.redis_client.keys(pattern)
            logger.info(f"🔍 Found {len(keys)} Redis keys matching pattern: {pattern}")
            
            for key in keys:
                value = await self.redis_client.get(key)
                if value:
                    try:
                        parsed_value = json.loads(value)
                        logger.info(f"🔍 Key: {key}, Value type: {type(parsed_value)}, Data: {str(parsed_value)[:200]}...")
                    except:
                        logger.info(f"🔍 Key: {key}, Raw Value: {str(value)[:200]}...")
                else:
                    logger.info(f"🔍 Key: {key}, Value: None")
        except Exception as e:
            logger.error(f"❌ Debug Redis failed: {e}")
    
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
        logger.info(f"📊 Multiple reports detected: {is_multiple_reports}")
        
        # Log if raw_text is missing to help debug
        if not raw_text:
            logger.warning("⚠️ raw_text is empty - Document AI summarizer output not available, will use full OCR text as fallback")
        
        # Run ReportAnalyzer in dedicated LLM executor for better batch performance
        report_analyzer = ReportAnalyzer(mode)
        loop = asyncio.get_event_loop()
        report_result = await loop.run_in_executor(
            LLM_EXECUTOR, report_analyzer.extract_document, text, raw_text
        )
        long_summary = report_result.get("long_summary", "")
        short_summary = report_result.get("short_summary", "")

        logger.info(f"✅ Generated long summary: {len(long_summary)} chars")
        logger.info(f"✅ Generated short summary: {short_summary}")

        analyzer = EnhancedReportAnalyzer()

        # Run both analyzer functions in parallel using dedicated LLM executor
        analysis_task = loop.run_in_executor(
            LLM_EXECUTOR,
            lambda: analyzer.extract_document_data_with_reasoning(
                long_summary, None, None, mode
            )
        )

        summary_task = loop.run_in_executor(
            LLM_EXECUTOR,
            lambda: analyzer.generate_brief_summary(raw_text, mode)
        )

        document_analysis, brief_summary = await asyncio.gather(
            analysis_task, summary_task
        )

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

     
    async def create_tasks_if_needed(self, document_analysis, document_id: str, physician_id: str, filename: str, processed_data: dict = None) -> int:
        """Step 3: Create tasks if conditions are met"""
        logger.info(f"🔧 Creating tasks for document {filename}")
        created_tasks = 0
        
        try:
            # Find physician user
            prisma = Prisma()
            await prisma.connect()
            
            users = await prisma.user.find_many(where={
                "OR": [
                    {"physicianId": physician_id, "role": "Physician"},
                    {"id": physician_id, "role": "Physician"}
                ]
            })
            
            await prisma.disconnect()

            if not users:
                logger.warning(f"⚠️ No physician user found for ID: {physician_id}")
                return 0
            
            # Check if consulting doctor matches physician
            consulting_doctor = document_analysis.consulting_doctor or ""
            
            def normalize_name(name):
                """
                Advanced name normalization:
                - Removes titles (Dr, Dr., MD, M.D, DO, D.O, Prof, etc.)
                - Removes commas
                - Handles "LastName, FirstName" and "FirstName LastName" formats
                - Converts to lowercase for comparison
                """
                if not name:
                    return ""
                
                # Remove all titles and credentials (case-insensitive, with or without dots)
                name = re.sub(r'\b(Dr\.?|M\.?D\.?|D\.?O\.?|D\.?P\.?M\.?|D\.?C\.?|N\.?P\.?|P\.?A\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?|Doctor|QME)\b', '', name, flags=re.IGNORECASE)
                
                # Remove all commas
                name = name.replace(',', ' ')
                
                # Remove all standalone periods and extra whitespace
                name = name.replace('.', ' ')
                
                # Remove extra whitespace and convert to lowercase
                name = ' '.join(name.split()).lower()
                
                return name.strip()
            
            def extract_name_parts(name):
                """
                Extract first name and last name, ignoring middle initials/names.
                Returns: (first_name, last_name, middle_parts)
                
                Examples:
                - "Kevin B. Calhoun" → ("kevin", "calhoun", ["b"])
                - "Kevin Calhoun" → ("kevin", "calhoun", [])
                - "Calhoun, Kevin B." → ("kevin", "calhoun", ["b"])
                """
                if not name:
                    return ("", "", [])
                
                normalized = normalize_name(name)
                parts = normalized.split()
                
                if len(parts) == 0:
                    return ("", "", [])
                elif len(parts) == 1:
                    return (parts[0], "", [])
                elif len(parts) == 2:
                    # Could be "FirstName LastName" or "LastName FirstName"
                    return (parts[0], parts[1], [])
                else:
                    # 3+ parts: assume first is first name, last is last name, middle are middle names/initials
                    return (parts[0], parts[-1], parts[1:-1])
            
            def names_match(name1, name2, match_type="exact"):
                """
                Compare two names with different matching strategies.
                
                match_type options:
                - "exact": First and last name must match exactly
                - "partial": Either first OR last name matches
                - "fuzzy": First AND last name match (ignoring middle names)
                
                Returns: (is_match, confidence_score, match_details)
                """
                first1, last1, middle1 = extract_name_parts(name1)
                first2, last2, middle2 = extract_name_parts(name2)
                
                # Exact match on first + last (ignoring middle)
                if first1 and last1 and first1 == first2 and last1 == last2:
                    return (True, 1.0, f"Exact match: {first1} {last1}")
                
                # Partial match: first name OR last name
                if match_type in ["partial", "fuzzy"]:
                    if first1 and first1 == first2:
                        return (True, 0.8, f"First name match: {first1}")
                    if last1 and last1 == last2:
                        return (True, 0.8, f"Last name match: {last1}")
                
                # Check reversed order (LastName, FirstName vs FirstName LastName)
                if first1 and last1 and first1 == last2 and last1 == first2:
                    return (True, 0.9, f"Reversed match: {first1} {last1}")
                
                return (False, 0.0, "No match")
            
            def find_best_match(doctor_name, users):
                """
                Find the best matching user for a given doctor name.
                Returns: (matching_user, confidence, match_details) or (None, 0, "")
                """
                best_match = None
                best_confidence = 0.0
                best_details = ""
                
                for user in users:
                    user_full_name = f"{user.firstName or ''} {user.lastName or ''}".strip()
                    
                    # Try exact match first
                    is_match, confidence, details = names_match(doctor_name, user_full_name, "exact")
                    
                    if is_match and confidence > best_confidence:
                        best_match = user
                        best_confidence = confidence
                        best_details = details
                        
                        # If we found a perfect match, we can stop
                        if confidence >= 1.0:
                            break
                
                # If no exact match, try partial matching
                if best_confidence < 1.0:
                    for user in users:
                        user_full_name = f"{user.firstName or ''} {user.lastName or ''}".strip()
                        is_match, confidence, details = names_match(doctor_name, user_full_name, "partial")
                        
                        if is_match and confidence > best_confidence:
                            best_match = user
                            best_confidence = confidence
                            best_details = details
                
                return (best_match, best_confidence, best_details)
            
            # STEP 1: Try to match consulting_doctor first (highest priority)
            logger.info(f"🔍 Looking for match with consulting doctor: '{consulting_doctor}'")
            matching_user, confidence, match_details = find_best_match(consulting_doctor, users)
            match_source = "consulting_doctor"
            
            if matching_user:
                user_full_name = f"{matching_user.firstName or ''} {matching_user.lastName or ''}".strip()
                logger.info(f"✅ Physician name MATCH found in CONSULTING DOCTOR!")
                logger.info(f"   User: '{user_full_name}'")
                logger.info(f"   Consulting Doctor: '{consulting_doctor}'")
                logger.info(f"   Match Details: {match_details}")
                logger.info(f"   Confidence: {confidence}")
                logger.info(f"   User ID: {matching_user.id}")
            
            # STEP 2: If no match in consulting_doctor, try all_doctors list
            if not matching_user:
                logger.info(f"⚠️ No match found in consulting_doctor, checking all_doctors list...")
                
                # Get all_doctors list from document_analysis
                all_doctors = []
                if hasattr(document_analysis, 'all_doctors') and document_analysis.all_doctors:
                    all_doctors = document_analysis.all_doctors
                    logger.info(f"📋 Found {len(all_doctors)} doctors in all_doctors: {all_doctors}")
                else:
                    logger.info(f"📋 No all_doctors list found in document_analysis")
                
                # Try to match each doctor in all_doctors list
                best_overall_match = None
                best_overall_confidence = 0.0
                best_overall_details = ""
                matched_doctor_name = None
                
                for doctor_name in all_doctors:
                    if not doctor_name or doctor_name == consulting_doctor:
                        continue
                    
                    logger.info(f"🔍 Trying doctor from all_doctors: '{doctor_name}'")
                    user_match, conf, details = find_best_match(doctor_name, users)
                    
                    if user_match and conf > best_overall_confidence:
                        best_overall_match = user_match
                        best_overall_confidence = conf
                        best_overall_details = details
                        matched_doctor_name = doctor_name
                
                if best_overall_match:
                    matching_user = best_overall_match
                    confidence = best_overall_confidence
                    match_details = best_overall_details
                    match_source = "all_doctors"
                    
                    user_full_name = f"{matching_user.firstName or ''} {matching_user.lastName or ''}".strip()
                    logger.info(f"✅ Physician name MATCH found in ALL_DOCTORS!")
                    logger.info(f"   User: '{user_full_name}'")
                    logger.info(f"   Doctor from all_doctors: '{matched_doctor_name}'")
                    logger.info(f"   Match Details: {match_details}")
                    logger.info(f"   Confidence: {confidence}")
                    logger.info(f"   User ID: {matching_user.id}")
            
            # STEP 3: If still no match found, return 0
            if not matching_user:
                logger.warning(f"⚠️ No physician name matches found in consulting_doctor or all_doctors")
                logger.warning(f"   Consulting doctor: '{consulting_doctor}'")
                logger.warning(f"   All doctors: {all_doctors if 'all_doctors' in locals() else '[]'}")
                logger.warning(f"   Available users: {[f'{u.firstName} {u.lastName}' for u in users]}")
                return 0
            
            logger.info(f"🎯 Final match: User='{matching_user.firstName} {matching_user.lastName}', Confidence={confidence}, Source='{match_source}'")
            
            # Generate and create tasks
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
            
            tasks = await task_creator.generate_tasks(document_data, filename, full_text, matched_doctor_name)
            logger.info(f"📋 Generated {len(tasks)} tasks")

            # Save tasks to database
            prisma = Prisma()
            await prisma.connect()
            
            for task in tasks:
                try:
                    mapped_task = {
                        "description": task.get("description"),
                        "department": task.get("department"),
                        "status": "Open",
                        "dueDate": datetime.now(),
                        "patient": task.get("patient", "Unknown"),
                        "actions": task.get("actions") if isinstance(task.get("actions"), list) else [],
                        "sourceDocument": task.get("source_document") or filename,
                        "documentId": document_id,
                        "physicianId": physician_id,
                    }
                    
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
                doi=document_analysis.doi if hasattr(document_analysis, 'doi') else None
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
        
        # Save document to database with all required parameters
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name=processed_data["filename"],
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
            document_summary=document_summary
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
        """
        try:
            # Test Redis connection first
            redis_test = await self.test_redis_basic()
            if not redis_test:
                logger.warning("⚠️ Redis basic test failed - caching will be disabled")
            
            # Step 1: Process document data
            processed_data = await self.process_document_data(data)
            
            # Step 1.5: Check for multiple reports - if detected, save to FailDocs
            if processed_data.get("is_multiple_reports", False):
                multi_report_info = processed_data.get("multi_report_info", {})
                confidence = multi_report_info.get("confidence", "unknown")
                reason = multi_report_info.get("reason", "Multiple reports detected")
                summary_text = processed_data.get("raw_text", "")
                
                logger.warning(f"⚠️ MULTIPLE REPORTS DETECTED (confidence: {confidence})")
                logger.warning(f"   Reason: {reason}")
                logger.info("💾 Saving to FailDocs for manual review...")
                
                # Save to FailDocs with summary
                fail_doc_id = await db_service.save_fail_doc(
                    reason=f"Multiple reports detected - manual review required. {reason}",
                    db=processed_data.get("dob"),
                    claim_number=processed_data.get("claim_number"),
                    patient_name=processed_data.get("patient_name"),
                    physician_id=processed_data.get("physician_id"),
                    gcs_file_link=processed_data.get("gcs_url"),
                    file_name=processed_data.get("filename"),
                    file_hash=processed_data.get("file_hash"),
                    blob_path=processed_data.get("blob_path"),
                    mode=processed_data.get("mode", "wc"),
                    document_text=processed_data.get("text_for_analysis", ""),
                    doi=None,
                    summary=summary_text  # Store the Document AI summarizer output
                )
                
                # Decrement parse count
                parse_decremented = await db_service.decrement_parse_count(processed_data.get("physician_id"))
                
                logger.info(f"✅ Multiple reports document saved to FailDocs with ID: {fail_doc_id}")
                
                return {
                    "status": "multiple_reports_detected",
                    "document_id": fail_doc_id,
                    "filename": processed_data.get("filename"),
                    "parse_count_decremented": parse_decremented,
                    "failure_reason": f"Multiple reports detected - manual review required. {reason}",
                    "multi_report_info": multi_report_info
                }
            
            # DEBUG: Check Redis before patient lookup
            await self.debug_redis_contents("patient_lookup:*")
            
            # Step 2: Perform patient lookup with enhanced fuzzy matching (NO DUPLICATE CHECK)
            lookup_result = await self.patient_lookup.perform_patient_lookup(db_service, processed_data)
            
            # DEBUG: Check Redis after patient lookup
            await self.debug_redis_contents("patient_lookup:*")
            
            # Step 3: Save document (ALL documents are saved - no duplicate blocking)
            save_result = await self.save_document(db_service, processed_data, lookup_result)
            
            # Step 4: Create tasks if document was saved successfully
            tasks_created = 0
            if save_result["document_id"]:
                tasks_created = await self.create_tasks_if_needed(
                    processed_data["document_analysis"],
                    save_result["document_id"],
                    processed_data["physician_id"],
                    processed_data["filename"],
                    processed_data  # Pass full processed_data for document text access
                )
            
            # DEBUG: Final Redis check
            await self.debug_redis_contents("*")
            
            # Prepare final response
            result = {
                "status": save_result["status"],
                "document_id": save_result["document_id"],
                "filename": processed_data["filename"],
                "tasks_created": tasks_created,
                "mode": processed_data["mode"],
                "parse_count_decremented": save_result["parse_count_decremented"],
                "cache_success": save_result.get("cache_success", False)
            }
            
            if lookup_result["pending_reason"]:
                result["pending_reason"] = lookup_result["pending_reason"]
            
            logger.info(f"✅ Webhook processing completed: {result}")
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
                doi=None
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
        Calls both EnhancedReportAnalyzer and ReportAnalyzer for comprehensive processing.
        """
        # Use updated values if provided, otherwise fallback to fail_doc values
        document_text = updated_fields.get("document_text") or fail_doc.documentText
        dob_str = updated_fields.get("dob") or fail_doc.dob  # ✅ FIXED: Changed from fail_doc.db to fail_doc.dob
        doi = updated_fields.get("doi") or fail_doc.doi
        claim_number = updated_fields.get("claim_number") or fail_doc.claimNumber
        patient_name = updated_fields.get("patient_name") or fail_doc.patientName
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
            
            logger.info(f"✅ Generated long summary: {len(long_summary)} chars")
            logger.info(f"✅ Generated short summary: {short_summary}")
            
            # Use EnhancedReportAnalyzer for detailed analysis
            analyzer = EnhancedReportAnalyzer()
            
            # Run analysis and summary generation in parallel
            analysis_task = asyncio.create_task(
                asyncio.to_thread(
                    analyzer.extract_document_data_with_reasoning, 
                    long_summary,    # Use summary for analysis
                    None,            # page_zones
                    None,            # raw_text  
                    mode             # mode
                )
            )
            
            summary_task = asyncio.create_task(
                asyncio.to_thread(analyzer.generate_brief_summary, long_summary, mode)
            )
            
            # Wait for both to complete
            document_analysis, brief_summary = await asyncio.gather(analysis_task, summary_task)
            
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