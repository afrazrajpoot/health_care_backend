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
from utils.logger import logger
from prisma import Prisma
import asyncio
import re
import json

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
        """Step 1: Process document data using EnhancedReportAnalyzer"""
        logger.info(f"📥 Processing document: {data.get('document_id', 'unknown')}")
        
        # Validate required fields
        if not data.get("result") or not data.get("filename") or not data.get("gcs_url"):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        result_data = data["result"]
        text = result_data.get("text", "")
        mode = data.get("mode", "wc")
        
        logger.info(f"📋 Document mode: {mode}")
        
        # Generate long summary using ReportAnalyzer
        report_analyzer = ReportAnalyzer(mode)
        report_result = await asyncio.to_thread(
            report_analyzer.extract_document,
            text
        )
        
        # ✅ STORE THE ACTUAL REPORT ANALYZER RESULT
        # print(report_analyzer,'report analyer')
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
        
        # Prepare data for patient lookup
        patient_name = document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else None
        claim_number = document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else None
        
        # Extract DOB from document_analysis
        dob = document_analysis.dob if hasattr(document_analysis, 'dob') and document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else None
        
        return {
            "document_analysis": document_analysis,
            "brief_summary": brief_summary,
            "text_for_analysis": text,
            # ✅ ADD THE ACTUAL REPORT ANALYZER RESULT
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
            "mode": mode
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
    async def perform_patient_lookup(self, db_service, processed_data: dict) -> dict:
        """Step 2: Perform patient lookup and update fields bidirectionally (NO DUPLICATE CHECK)"""
        physician_id = processed_data["physician_id"]
        patient_name = processed_data["patient_name"]
        claim_number = processed_data["claim_number"]
        document_analysis = processed_data["document_analysis"]
        
        logger.info(f"🔍 Performing patient lookup for physician: {physician_id}")
        
        # Helper function to check if field is "bad" (not specified/empty)
        def is_bad_field(value):
            return not value or str(value).lower() in ["not specified", "unknown", "", "none", "null"]
        
        # 🚨 CRITICAL: Check if both DOB and claim number are not specified
        dob_not_specified = is_bad_field(processed_data["dob"])
        claim_not_specified = is_bad_field(claim_number)
        
        # If both DOB AND claim number are not specified, skip lookup and field updates
        if dob_not_specified and claim_not_specified:
            logger.warning("🚨 SKIPPING PATIENT LOOKUP: Both DOB and claim number are not specified - no updates will be performed")
            
            # Determine document status for this case
            document_status = "failed"
            pending_reason = "Missing both DOB and claim number - cannot identify patient"
            
            return {
                "lookup_data": None,
                "document_status": document_status,
                "pending_reason": pending_reason,
                "patient_name_to_use": patient_name or "Not specified",
                "claim_to_save": claim_number or "Not specified", 
                "document_analysis": document_analysis,
                "field_updates": [],
                "previous_docs_updated": 0,
                "lookup_skipped": True  # New flag to indicate lookup was skipped
            }
        
        # ✅ Continue with normal patient lookup for documents with at least DOB OR claim number
        # Verify Redis connection first
        redis_ok = await self.verify_redis_connection()
        if not redis_ok:
            logger.warning("⚠️ Redis not available - proceeding without cache")
        
        # Get patient lookup data (with Redis caching)
        lookup_data = await self._get_cached_patient_lookup(physician_id, patient_name, claim_number, processed_data["dob"], db_service)
        
        # Bidirectional field updating logic (only if we have lookup data)
        field_updates = []
        updated_previous_docs = 0
        
        if lookup_data and lookup_data.get("total_documents", 0) > 0:
            logger.info("🔄 Checking for bidirectional field updates...")
            
            # Get fields from lookup data
            fetched_patient_name = lookup_data.get("patient_name")
            fetched_dob = lookup_data.get("dob")
            fetched_claim_number = lookup_data.get("claim_number")
            fetched_doi = lookup_data.get("doi")
            
            # Update document analysis with good values from DB
            # Only update if current document has bad values AND DB has good values
            if is_bad_field(document_analysis.patient_name) and not is_bad_field(fetched_patient_name):
                old_name = document_analysis.patient_name
                document_analysis.patient_name = fetched_patient_name
                field_updates.append(f"patient_name: '{old_name}' → '{fetched_patient_name}'")
                logger.info(f"✅ Updated patient_name from DB: '{old_name}' → '{fetched_patient_name}'")
            
            if hasattr(document_analysis, 'dob') and is_bad_field(document_analysis.dob) and not is_bad_field(fetched_dob):
                old_dob = document_analysis.dob
                document_analysis.dob = fetched_dob
                field_updates.append(f"dob: '{old_dob}' → '{fetched_dob}'")
                logger.info(f"✅ Updated DOB from DB: '{old_dob}' → '{fetched_dob}'")
            
            if is_bad_field(document_analysis.claim_number) and not is_bad_field(fetched_claim_number):
                old_claim = document_analysis.claim_number
                document_analysis.claim_number = fetched_claim_number
                field_updates.append(f"claim_number: '{old_claim}' → '{fetched_claim_number}'")
                logger.info(f"✅ Updated claim_number from DB: '{old_claim}' → '{fetched_claim_number}'")
            
            if (hasattr(document_analysis, 'doi') and 
                is_bad_field(document_analysis.doi) and 
                not is_bad_field(fetched_doi)):
                old_doi = document_analysis.doi
                document_analysis.doi = fetched_doi
                field_updates.append(f"doi: '{old_doi}' → '{fetched_doi}'")
                logger.info(f"✅ Updated DOI from DB: '{old_doi}' → '{fetched_doi}'")
            
            # 🚨 IMPORTANT: Update previous documents ONLY if current document has good identification
            # Check if current document has at least one good identification field
            current_has_good_patient = not is_bad_field(document_analysis.patient_name)
            current_has_good_dob = hasattr(document_analysis, 'dob') and not is_bad_field(document_analysis.dob)
            current_has_good_claim = not is_bad_field(document_analysis.claim_number)
            current_has_good_doi = hasattr(document_analysis, 'doi') and not is_bad_field(document_analysis.doi)
            
            # Only update previous documents if we have at least DOB OR claim number in current document
            current_has_identification = current_has_good_dob or current_has_good_claim
            
            if current_has_identification and (current_has_good_patient or current_has_good_dob or current_has_good_claim or current_has_good_doi):
                try:
                    update_patient = document_analysis.patient_name if current_has_good_patient else None
                    update_dob = document_analysis.dob if current_has_good_dob else None
                    update_claim = document_analysis.claim_number if current_has_good_claim else None
                    update_doi = document_analysis.doi if current_has_good_doi else None
                    
                    if update_patient or update_dob or update_claim:
                        updated_previous_docs = await db_service.update_document_fields(
                            patient_name=update_patient or "Not specified",
                            dob=update_dob or "Not specified",
                            physician_id=physician_id,
                            claim_number=update_claim or "Not specified",
                            doi=update_doi
                        )
                        logger.info(f"🔄 Updated {updated_previous_docs} previous documents with current good fields")
                        
                        # Invalidate cache after updates
                        if updated_previous_docs > 0 and self.redis_client:
                            pattern = f"patient_lookup:{physician_id}:*"
                            keys = await self.redis_client.keys(pattern)
                            if keys:
                                await self.redis_client.delete(*keys)
                                logger.info(f"🗑️ Invalidated {len(keys)} patient lookup cache entries")
                    
                except Exception as update_err:
                    logger.error(f"❌ Error updating previous documents: {update_err}")
            else:
                logger.info("ℹ️ Skipping previous document updates - current document lacks sufficient identification")
            
            logger.info(f"🎯 Bidirectional updates completed: {field_updates}")
        
        # Update processed_data with overridden values
        processed_data["patient_name"] = document_analysis.patient_name
        processed_data["claim_number"] = document_analysis.claim_number
        processed_data["has_patient_name"] = not is_bad_field(document_analysis.patient_name)
        processed_data["has_claim_number"] = not is_bad_field(document_analysis.claim_number)
        
        # Determine document status (NO DUPLICATE CHECK)
        base_status = document_analysis.status
        
        if not processed_data["has_patient_name"] and not processed_data["has_claim_number"]:
            document_status = "failed"
            pending_reason = "Missing patient name and claim number"
        elif lookup_data and lookup_data.get("has_conflicting_claims", False):
            document_status = "failed"
            pending_reason = "Conflicting claim numbers found"
        else:
            document_status = base_status
            pending_reason = None
        
        return {
            "lookup_data": lookup_data,
            "document_status": document_status,
            "pending_reason": pending_reason,
            "patient_name_to_use": processed_data["patient_name"] or "Not specified",
            "claim_to_save": processed_data["claim_number"] or "Not specified",
            "document_analysis": document_analysis,
            "field_updates": field_updates,
            "previous_docs_updated": updated_previous_docs,
            "lookup_skipped": False  # Lookup was performed
        }
    async def _get_cached_patient_lookup(self, physician_id: str, patient_name: str, claim_number: str, dob: str, db_service) -> dict:
        """Get patient lookup data from cache or database"""
        
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
        
        # Check if Redis client is available
        if not self.redis_client:
            logger.warning("❌ Redis client not available, skipping cache")
            return await db_service.get_patient_claim_numbers(
                patient_name=patient_name,
                physicianId=physician_id,
                dob=dob,
                claim_number=claim_number
            )
        
        # Try cache first
        try:
            logger.info(f"🔍 Checking Redis cache FIRST for key: {cache_key}")
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"💾 CACHE HIT: Found patient lookup data in cache for key: {cache_key}")
                return json.loads(cached_data)
            else:
                logger.info(f"💾 CACHE MISS: No data found in cache for key: {cache_key}")
        except Exception as e:
            logger.error(f"⚠️ Cache read error for key {cache_key}: {e}")
        
        # Get from database
        logger.info("🗄️ Fetching patient lookup data from database...")
        lookup_data = await db_service.get_patient_claim_numbers(
            patient_name=patient_name,
            physicianId=physician_id,
            dob=dob,
            claim_number=claim_number
        )
        
        # Cache the result
        if lookup_data and lookup_data.get("total_documents", 0) > 0:
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
                logger.error(f"⚠️ Data that failed to cache: {lookup_data}")
        
        return lookup_data

    async def create_tasks_if_needed(self, document_analysis, document_id: str, physician_id: str, filename: str) -> int:
        """Step 3: Create tasks if conditions are met"""
        if not document_analysis.is_task_needed:
            logger.info(f"ℹ️ No tasks needed for document {filename}")
            return 0
        
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
                if not name:
                    return ""
                name = re.sub(r'^(Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)\s*', '', name, flags=re.IGNORECASE)
                return name.strip().lower()

            normalized_consulting_doctor = normalize_name(consulting_doctor)
            matching_user = None

            for user in users:
                user_full_name = f"{user.firstName or ''} {user.lastName or ''}".strip()
                normalized_user_name = normalize_name(user_full_name)
                
                if normalized_user_name and normalized_consulting_doctor and normalized_user_name == normalized_consulting_doctor:
                    matching_user = user
                    logger.info(f"✅ Physician matches consulting doctor - User ID: {user.id}")
                    break
            
            if not matching_user:
                logger.warning(f"⚠️ No physician name matches consulting doctor - skipping task creation")
                return 0
            
            # Generate and create tasks
            task_creator = TaskCreator()
            document_data = document_analysis.dict()
            document_data["filename"] = filename
            document_data["document_id"] = document_id
            document_data["physician_id"] = physician_id
            
            tasks = await task_creator.generate_tasks(document_data, filename)
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
                        "dueDate": datetime.now(),  # Set default due date
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
        
        # Get RD (Report Date) - use from analysis or current date
        rd = None
        if hasattr(document_analysis, 'rd') and document_analysis.rd and str(document_analysis.rd).lower() != "not specified":
            try:
                # Parse the RD date if available
                if '/' in document_analysis.rd:
                    month, day = document_analysis.rd.split('/')
                    year = datetime.now().year
                    rd = datetime.strptime(f"{year}-{month.zfill(2)}-{day.zfill(2)}", "%Y-%m-%d")
                elif '-' in document_analysis.rd:
                    rd = datetime.strptime(document_analysis.rd, "%Y-%m-%d")
            except Exception:
                rd = datetime.now()  # Fallback to current date
        
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
        # The ReportAnalyzer returns a dict with both summaries in process_document_data
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
            
            # DEBUG: Check Redis before patient lookup
            await self.debug_redis_contents("patient_lookup:*")
            
            # Step 2: Perform patient lookup (NO DUPLICATE CHECK)
            lookup_result = await self.perform_patient_lookup(db_service, processed_data)
            
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
                    processed_data["filename"]
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
                document_text
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

            # Step 2: Perform patient lookup with the updated data
            logger.info("🔍 Performing patient lookup for updated failed document...")
            lookup_result = await self.perform_patient_lookup(db_service, processed_data)
            
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
                    processed_data["filename"]
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