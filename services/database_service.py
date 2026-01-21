import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
try:
    from prisma import Prisma
except Exception:
    # Defer hard error until connect time so the module can be imported in dev/test environments
    Prisma = None
from models.schemas import ExtractionResult
from cryptography.fernet import Fernet
from cryptography.exceptions import InvalidKey
import base64
from datetime import datetime, timedelta
from helpers.helpers import normalize_name, normalize_claim, normalize_dob, is_same_patient

load_dotenv()
logger = logging.getLogger("document_ai")

class DatabaseService:
    """Service for database operations using your schema structure"""
   
    def __init__(self):
        # Instantiate Prisma client lazily; if Prisma is not installed, keep None
        self.prisma = Prisma() if Prisma is not None else None
        # Async lock to prevent concurrent connect/disconnect races
        self._connect_lock = asyncio.Lock()
        # Track connection state to avoid double-connect attempts
        self._connected = False
        self._init_encryption()
        if not os.getenv("DATABASE_URL"):
            raise ValueError("DATABASE_URL environment variable not set")
    
    def _init_encryption(self):
        """Initialize encryption suite with validation and optional key generation."""
        encryption_key_str = os.getenv('ENCRYPTION_KEY')
        if not encryption_key_str:
            logger.warning("⚠️ ENCRYPTION_KEY not set. Generating a new one for development (insecure for production).")
            new_key = Fernet.generate_key().decode()  # Use module-level Fernet
            logger.info(f"Generated key: {new_key}")
            logger.info("💡 Set this as ENCRYPTION_KEY in your .env file for production.")
            self.encryption_key = new_key.encode()
        else:
            try:
                self.encryption_key = encryption_key_str.encode('utf-8')
                self.cipher_suite = Fernet(self.encryption_key)
                # Quick validation: Try to create a dummy token
                dummy_token = self.cipher_suite.encrypt(b"test")
                self.cipher_suite.decrypt(dummy_token)  # Should succeed
                logger.info("🔐 Encryption key validated successfully.")
            except (ValueError, InvalidKey) as e:
                logger.error(f"❌ Invalid ENCRYPTION_KEY: {str(e)}")
                logger.info("💡 Generate a new one: from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
                raise ValueError(f"Invalid ENCRYPTION_KEY: {str(e)}. Please set a valid 32-byte base64 key.")
    
    
    async def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Run a SELECT query and return a single row as dict"""
        await self.connect()
        result = await self.prisma.query_raw(query, params or {})
        if result and len(result) > 0:
            return dict(result[0])
        return None

    async def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Run a SELECT query and return all rows"""
        await self.connect()
        results = await self.prisma.query_raw(query, params or {})
        return [dict(row) for row in results]

    async def execute(self, query: str, params: Optional[Dict] = None):
        """Run INSERT/UPDATE/DELETE queries"""
        await self.connect()
        return await self.prisma.execute_raw(query, params or {})

    async def document_exists_by_hash(self, file_hash: str, user_id: str, physician_id: Optional[str] = None) -> bool:
        """Check if a document with the same hash already exists for this user (and optionally physician)."""
        try:
            where_clause: Dict[str, Any] = {
                "fileHash": file_hash,
                "userId": user_id
            }
            if physician_id:
                where_clause["physicianId"] = physician_id

            # Use Prisma model query which is safer than raw SQL and avoids placeholder syntax
            doc = await self.prisma.document.find_first(where=where_clause)
            return doc is not None
        except Exception as e:
            logger.error(f"❌ Error checking document by hash: {str(e)}")
            # Fallback: return False so caller can proceed without blocking
            return doc is not None

    async def increment_workflow_stat(self, category: str, amount: int = 1):
        """Increment workflow counter by category (e.g. 'referralsProcessed')."""
        valid_fields = {
            "referralsProcessed": "referralsProcessed",
            "rfasMonitored": "rfasMonitored",
            "qmeUpcoming": "qmeUpcoming",
            "payerDisputes": "payerDisputes",
            "externalDocs": "externalDocs",
            "intakes_created": "intakes_created",
        }

        if category not in valid_fields:
            logger.warning(f"⚠️ Invalid workflow stat category: {category}")
            return
        field = valid_fields[category]

        # Use UTC day boundaries for the DateTime `date` field in the WorkflowStats model
        now_utc = datetime.utcnow()
        day_start = datetime(now_utc.year, now_utc.month, now_utc.day)
        next_day = day_start + timedelta(days=1)

        await self.connect()
        try:
            # Find today's row by checking the DateTime falls within [day_start, next_day)
            stats = await self.prisma.workflowstats.find_first(
                where={"date": {"gte": day_start, "lt": next_day}}
            )

            if stats:
                # Increment existing field
                current_value = getattr(stats, field, 0) or 0
                new_value = current_value + amount
                await self.prisma.workflowstats.update(
                    where={"id": stats.id},
                    data={field: new_value}
                )
            else:
                # Create new daily entry with the date set to the day's start
                await self.prisma.workflowstats.create(
                    data={"date": day_start, field: amount}
                )

            logger.info(f"📈 Incremented workflow stat: {field} (+{amount})")
        except Exception as e:
            logger.error(f"❌ Failed to increment workflow stat: {str(e)}")
            logger.debug("Exception details for increment_workflow_stat", exc_info=True)
        finally:
            await self.disconnect()


    async def connect(self):
        """Connect to the database"""
        # Use a lock to make connect/disconnect concurrency-safe
        async with self._connect_lock:
            if getattr(self, "_connected", False):
                logger.debug("🔁 Database already connected; skipping connect")
                return
            if self.prisma is None:
                raise RuntimeError("Prisma client is not available. Ensure the 'prisma' package is installed and configured.")
            try:
                await self.prisma.connect()
                self._connected = True
                logger.info("✅ Connected to database")
            except Exception as e:
                msg = str(e)
                # Handle benign 'Already connected' errors from Prisma/engine
                if "Already connected" in msg or "Already connected to the query engine" in msg:
                    logger.info("🔁 Prisma reports already connected; marking as connected")
                    self._connected = True
                    return
                logger.error(f"❌ Failed to connect to database: {msg}")
                raise
    
    async def disconnect(self):
        """Disconnect from the database"""
        async with self._connect_lock:
            if not getattr(self, "_connected", False):
                logger.debug("🔌 Database not connected; skipping disconnect")
                return
            try:
                await self.prisma.disconnect()
                self._connected = False
                logger.info("🔌 Disconnected from database")
            except Exception as e:
                msg = str(e)
                # If Prisma/engine indicates it's already disconnected, treat as success
                if "Already disconnected" in msg or "not connected" in msg:
                    logger.info("🔌 Prisma already disconnected; marking as disconnected")
                    self._connected = False
                    return
                logger.warning(f"⚠️ Error disconnecting from database: {msg}")
        
    async def save_fail_doc(
            self, 
            reason: str | Dict[str, Any], 
            db: Optional[str] = None,
            doi: Optional[str] = None,
            claim_number: Optional[str] = None,
            patient_name: Optional[str] = None,
            document_text: Optional[str] = None,
            physician_id: Optional[str] = None,
            gcs_file_link: Optional[str] = None,
            file_name: Optional[str] = None,
            file_hash: Optional[str] = None,
            blob_path: Optional[str] = None,
            mode : Optional[str] = None,
            ai_summarizer_text: Optional[str] = None,
            author: Optional[str] = None
        ) -> str:
            """Save a failed document record to the FailDocs table. Supports both arg-based and dict-based usage."""
            try:
                # Handle dict-based input (overloaded usage)
                if isinstance(reason, dict):
                    data = reason
                    # Ensure minimal required fields are present if passed via dict
                    if "reason" not in data:
                        data["reason"] = "Unknown failure"
                else:
                    # Handle arg-based logic (legacy usage)
                    data = {
                        "reason": reason,
                    }
                    if db is not None:
                        data["dob"] = db
                    if doi is not None:
                        data["doi"] = doi
                    if claim_number is not None:
                        data["claimNumber"] = claim_number
                    if patient_name is not None:
                        data["patientName"] = patient_name
                    if document_text is not None:
                        data["documentText"] = document_text
                    if physician_id is not None:
                        data["physicianId"] = physician_id
                    if gcs_file_link is not None:
                        data["gcsFileLink"] = gcs_file_link
                    if file_name is not None:
                        data["fileName"] = file_name
                    if file_hash is not None:
                        data["fileHash"] = file_hash
                    if blob_path is not None:
                        data["blobPath"] = blob_path
                    if ai_summarizer_text is not None:
                        data["aiSummarizerText"] = ai_summarizer_text
                    if author is not None:
                        data["author"] = author

                await self.connect()
                
                fail_doc = await self.prisma.faildocs.create(
                    data=data
                )
                logger.info(f"💾 Saved fail doc with ID: {fail_doc.id} (Physician ID: {data.get('physicianId') if data.get('physicianId') else 'None'})")
                return fail_doc.id
            except Exception as e:
                logger.error(f"❌ Error saving fail doc: {str(e)}")
                # raise  # Don't raise, just log
                return None
    
    async def get_fail_doc_by_id(self, fail_doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a failed document record by its ID from the FailDocs table."""
        try:
            fail_doc = await self.prisma.faildocs.find_unique(
                where={"id": fail_doc_id}
            )
            if fail_doc is None:
                logger.warning(f"⚠️ Fail doc not found with ID: {fail_doc_id}")
            else:
                logger.info(f"📄 Retrieved fail doc with ID: {fail_doc_id}")
            return fail_doc
        except Exception as e:
            logger.error(f"❌ Error retrieving fail doc {fail_doc_id}: {str(e)}")
            raise
    async def get_fail_docs_by_physician(self, physician_id: str) -> List[Dict[str, Any]]:
        """Fetch failed documents by physician ID."""
        try:
            fail_docs = await self.prisma.faildocs.find_many(
                where={
                    "physicianId": physician_id
                }
            )
            # Convert to dicts for response
            fail_docs_list = [
                {
                    "id": doc.id,
                    "reason": doc.reason,
                    "blobPath": doc.blobPath,
                    "physicianId": doc.physicianId
                }
                for doc in fail_docs
            ]
            logger.info(f"📋 Fetched {len(fail_docs_list)} fail docs for physician: {physician_id}")
            return fail_docs_list
        except Exception as e:
            logger.error(f"❌ Error fetching fail docs for physician {physician_id}: {str(e)}")
            raise
    
    async def document_exists(self, filename: str, file_size: int) -> bool:
        """Check if document already exists by filename and size (adjust where clause if needed)"""
        try:
            count = await self.prisma.document.count(where={
                "fileName": filename,
            })
            return count > 0
        except Exception as e:
            logger.error(f"❌ Error checking document existence: {str(e)}")
            return False
    async def delete_fail_doc_by_id(self, doc_id: str, physician_id: str) -> bool:
        """Delete a failed document record from the FailDocs table, scoped to physician ID."""
        try:
            # First, fetch to verify ownership
            fail_doc = await self.prisma.faildocs.find_unique(
                where={
                    "id": doc_id
                }
            )
            if not fail_doc or fail_doc.physicianId != physician_id:
                logger.warning(f"⚠️ Unauthorized delete attempt for doc {doc_id} by physician {physician_id}")
                return False
            
            # Delete from DB
            await self.prisma.faildocs.delete(
                where={
                    "id": doc_id
                }
            )
            logger.info(f"🗑️ Deleted fail doc {doc_id} from DB")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting fail doc {doc_id}: {str(e)}")
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            document = await self.prisma.document.find_unique(
                where={"id": document_id},
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                    # NO "whatsNew" - scalar Json, auto-included
                }
            )

            if not document:
                logger.info(f"📄 No document found with ID: {document_id}")
                return None

            logger.info(f"📄 Retrieved document: {document.gcsFileLink}")
            return document.dict()

        except Exception as e:
            logger.error(f"❌ Error retrieving document {document_id}: {str(e)}")
            raise

    async def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent documents with pagination"""
        try:
            documents = await self.prisma.document.find_many(
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                    # NO "whatsNew" - scalar Json
                },
                order={"createdAt": "desc"},
                take=limit
            )

            docs_list = [doc.dict() for doc in documents]

            logger.info(f"📋 Retrieved {len(docs_list)} recent documents")
            return docs_list

        except Exception as e:
            logger.error(f"❌ Error retrieving recent documents: {str(e)}")
            raise
    
    async def get_document_by_patient_details(
        self, 
        patient_name: str,
        physicianId: Optional[str] = None,
        dob: Optional[str] = None,
        doi: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve ALL matching documents for a patient using comprehensive matching rules.
        Uses DOI (Date of Injury) to differentiate between different cases for the same patient.
        
        🔥 MATCHING RULES IMPLEMENTED:
        - Rule 1: Different DOI → DIFFERENT CASE (different injury for same patient)
        - Rule 2: Same DOI → SAME CASE (highest priority)
        - Rule 3: Both DOI None → Use name + DOB logic
        
        Returns: Aggregated response with ALL matching documents for the specific case (DOI).
        """
        try:
            # Normalize search inputs for DOI and DOB only (name uses flexible matching)
            normalized_input_doi = normalize_dob(doi)  # Reuse normalize_dob for DOI
            normalized_input_dob = normalize_dob(dob)
            
            logger.info(f"🔍 Searching for patient case:")
            logger.info(f"  Input: name='{patient_name}', dob='{dob}', doi='{doi}'")
            logger.info(f"  Normalized: dob='{normalized_input_dob}', doi='{normalized_input_doi}'")
            
            # Fetch ALL documents for physician (broad query)
            where_clause = {}
            if physicianId:
                where_clause["physicianId"] = physicianId
            
            documents = await self.prisma.document.find_many(
                where=where_clause,
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True,
                    "bodyPartSnapshots": True
                },
                order={"createdAt": "desc"}
            )
            
            logger.info(f"📄 Retrieved {len(documents)} total documents from DB")
            
            # Apply matching logic to filter documents
            matched_docs = []
            
            for i, doc in enumerate(documents):
                # 🆕 CRITICAL: Pass ORIGINAL names (not normalized) to is_same_patient()
                # is_same_patient() will generate all name variations internally
                db_doi = normalize_dob(doc.doi)  # Normalize DOI for comparison
                db_dob = normalize_dob(doc.dob)
                
                logger.info(f"📋 Checking Doc {i+1}/{len(documents)}: Name='{doc.patientName}', DOB='{doc.dob}'→'{db_dob}', DOI='{doc.doi}'→'{db_doi}'")
                
                if is_same_patient(
                    patient_name, normalized_input_dob, normalized_input_doi,
                    doc.patientName, db_dob, db_doi
                ):
                    logger.info(f"✅ MATCH! Adding document {doc.id[:12]}...")
                    matched_docs.append(doc)
                else:
                    logger.debug(f"❌ NO MATCH for document {doc.id[:12]}...")
            
            logger.info(f"🎯 Found {len(matched_docs)} matching documents for this case")
            
            # Merge patient details from all matched documents
            # Priority: Use most complete/recent data
            merged_patient_name = patient_name
            merged_dob = dob
            merged_doi = doi
            
            for doc in matched_docs:
                # Use first non-empty DOI found
                if not merged_doi or str(merged_doi).lower() in ["not specified", "unknown"]:
                    if doc.doi and str(doc.doi).lower() not in ["not specified", "unknown"]:
                        merged_doi = doc.doi
                
                # Use first non-empty DOB found
                if not merged_dob or str(merged_dob).lower() in ["not specified", "unknown"]:
                    if doc.dob and str(doc.dob).lower() not in ["not specified", "unknown"]:
                        merged_dob = doc.dob
                
                # Use longest/most complete name
                if len(str(doc.patientName)) > len(str(merged_patient_name)):
                    merged_patient_name = doc.patientName
            
            logger.info(f"📝 Merged patient details: name='{merged_patient_name}', dob='{merged_dob}', doi='{merged_doi}'")
            
            response = {
                "patient_name": merged_patient_name,
                "dob": merged_dob,
                "doi": merged_doi,
                "claim_number": None,  # Keep for backward compatibility
                "total_documents": len(matched_docs),
                "documents": []
            }
            
            for i, doc in enumerate(matched_docs):
                doc_data = doc.dict()
                doc_data["document_index"] = i + 1
                doc_data["is_latest"] = i == 0
                response["documents"].append(doc_data)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents for {patient_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "patient_name": patient_name,
                "dob": dob,
                "doi": doi,
                "claim_number": None,
                "total_documents": 0,
                "documents": []
            }
    async def get_tasks_by_document_ids(self, document_ids: list[str], physician_id: Optional[str] = None) -> list[dict]:
        """Fetch tasks (with quickNotes and description) by document IDs, optionally filtered by physician_id"""
        where_clause = {
            "documentId": {"in": document_ids},
            "status": "Pending"  # ✅ Only fetch Pending status tasks
        }
        if physician_id:
            where_clause["physicianId"] = physician_id
        
        tasks = await self.prisma.task.find_many(
            where=where_clause,
            order={"createdAt": "asc"}
        )
        
        # Manually select the fields you care about
        tasks_data = [
            {
                "id": t.id,
                "documentId": t.documentId,
                "description": t.description,
                "quickNotes": getattr(t, "quickNotes", None)
            }
            for t in tasks
        ]
        return tasks_data

    async def get_tasks_by_patient_details(
        self, 
        patient_name: str, 
        dob: Optional[str] = None, 
        doi: Optional[str] = None, 
        physician_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch all tasks (with quickNotes and description) by patient details (name, dob, doi),
        optionally filtered by physician_id. Includes all statuses (not just Pending).
        Uses DOI to identify the specific case/injury.
        """
        # await self.initialize_db()  # This initializes DocumentAggregationService's db_service

        # ✅ Handle DOB (accept both date strings or keep as-is if parsing fails)
        dob_value = None
        if dob:
            # Use the same logic as _parse_date to handle datetime objects
            if isinstance(dob, datetime):
                dob_value = dob
            else:
                try:
                    dob_value = datetime.strptime(dob, "%Y-%m-%d")
                except ValueError:
                    try:
                        dob_value = datetime.strptime(dob, "%m/%d/%Y")
                    except ValueError:
                        dob_value = dob  # keep string if parsing fails

        # ✅ Fetch related documents first using DOI-based matching
        document_data = await self.get_document_by_patient_details(
            patient_name=patient_name,
            physicianId=physician_id,
            dob=dob_value,
            doi=doi
        )

        # ✅ If no documents found, return empty list
        if not document_data or document_data.get("total_documents", 0) == 0:
            return []

        # ✅ Collect all document IDs
        documents = document_data["documents"]
        document_ids = [doc["id"] for doc in documents]

        # ✅ Build query for Prisma - FIXED: access prisma through db_service
        where_clause = {
            "documentId": {"in": document_ids}
        }
        if physician_id:
            where_clause["physicianId"] = physician_id

        # ✅ Fetch all tasks linked to those documents - FIXED: use db_service.prisma
        tasks = await self.prisma.task.find_many(
            where=where_clause,
            order={"createdAt": "asc"}
        )

        # ✅ Format output
        tasks_data = [
            {
                "id": t.id,
                "documentId": t.documentId,
                "description": t.description,
                "quickNotes": getattr(t, "quickNotes", None)
            }
            for t in tasks
        ]

        return tasks_data

    async def get_all_unverified_documents(
        self, 
        patient_name: Optional[str] = None,
        physicianId: Optional[str] = None,
        claimNumber: Optional[str] = None,
        dob: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all unverified documents where status is NOT 'verified'.
        Filters by patient_name if provided. 
        If claimNumber is provided and valid, filter by it (in addition to patient_name and dob if available).
        Otherwise, filter by patient_name and dob if available.
        At least one of patient_name, claimNumber, or dob must be provided.
        """
        try:
            filters_provided = [f for f in [patient_name, claimNumber, dob] if f is not None]
            if not filters_provided:
                logger.warning("❌ No filters provided for retrieving unverified documents")
                return None

            # Normalize claimNumber: set to None if "not specified"
            if claimNumber is not None and str(claimNumber).lower() == "not specified":
                claimNumber = None

            logger.info(f"🔍 Getting unverified documents with filters: patient_name={patient_name}, physicianId={physicianId}, claimNumber={claimNumber}, dob={dob}")
            print(patient_name, physicianId, claimNumber, dob, 'patient_name, physicianId, claimNumber, dob')

            # Base where clause
            where_clause = {
                "status": {"not": "verified"}
            }

            # Add patient_name if provided
            if patient_name:
                where_clause["patientName"] = patient_name

            # Add physicianId if provided
            if physicianId:
                where_clause["physicianId"] = physicianId

            # 🆕 Updated: Add claimNumber filter if provided (valid); always add dob filter if provided (no elif)
            if claimNumber:
                where_clause["claimNumber"] = claimNumber
                logger.info(f"📌 Using claimNumber filter: {claimNumber}")
            if dob:
                dob_str = dob.strftime("%Y-%m-%d")
                where_clause["dob"] = dob_str
                logger.info(f"📌 Using dob string filter: {dob_str}")

            # Fetch documents
            documents = await self.prisma.document.find_many(
                where=where_clause,
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                },
                order={"createdAt": "desc"}
            )

            if not documents:
                logger.warning(f"❌ No non-verified documents found with provided filters")
                return None

            # Determine patient_name for logging/response (use first document's if not provided)
            response_patient_name = patient_name or documents[0].patientName

            logger.info(f"📋 Found {len(documents)} unverified documents for patient: {response_patient_name}")

            # Structured response
            response = {
                "patient_name": response_patient_name,
                "total_documents": len(documents),
                "documents": []
            }

            for i, doc in enumerate(documents):
                doc_data = doc.dict()
                doc_data["document_index"] = i + 1
                doc_data["is_latest"] = i == 0
                response["documents"].append(doc_data)
                logger.info(f"📄 Added document {i+1}: ID {doc.id}")

            return response

        except Exception as e:
            logger.error(f"❌ Error retrieving unverified documents: {str(e)}")
            raise
  
    async def update_document_fields(
        self,
        patient_name: str,
        dob: str,
        physician_id: str,
        doi: str,
        claim_number: Optional[str] = None,
    ) -> int:
        print(patient_name, physician_id, doi, dob, 'patient data for update')
        
        # 🚨 CRITICAL FIX: Use STRICT patient matching criteria with DOI as primary identifier
        # Only update documents that definitely belong to the SAME case (same patient + DOI)
        
        or_conditions = []
        
        # Condition 1: Exact DOI match (strongest identifier for same case/injury)
        if doi and doi.lower() != "not specified":
            or_conditions.append({"doi": doi})
        
        # Condition 2: Patient name + DOB match (strong identifier)
        if (patient_name and patient_name.lower() != "not specified" and 
            dob and dob.lower() != "not specified"):
            or_conditions.append({
                "AND": [
                    {"patientName": patient_name},
                    {"dob": dob}
                ]
            })
        
        # Condition 3: If we have claim_number + patient name match (backward compatibility)
        if (claim_number and claim_number.lower() != "not specified" and 
            patient_name and patient_name.lower() != "not specified"):
            or_conditions.append({
                "AND": [
                    {"patientName": patient_name},
                    {"claimNumber": claim_number}
                ]
            })
        
        # 🚨 If no strong matching criteria, DON'T update any documents
        if not or_conditions:
            logger.warning(f"🚨 Cannot update documents - insufficient patient identification criteria")
            return 0
        
        fetch_where = {
            "physicianId": physician_id,
            "OR": or_conditions
        }
        
        # 🆕 FIXED: No 'select' parameter
        documents_to_update = await self.prisma.document.find_many(
            where=fetch_where,
        )
        
        logger.info(f"🔍 Found {len(documents_to_update)} documents for SAME CASE (physician '{physician_id}')")
        logger.info(f"   Matching criteria: {or_conditions}")
        
        updated_count = 0
        for doc in documents_to_update:
            # Determine updates needed for this doc
            update_data = {}
            
            # Update patientName if missing or doesn't match new one
            if (not doc.patientName or str(doc.patientName).lower() == "not specified" or
                str(doc.patientName).lower() != patient_name.lower()):
                update_data["patientName"] = patient_name
                logger.debug(f"  - Will update patientName for doc {doc.id}: '{doc.patientName}' -> '{patient_name}'")
            
            # Update dob if missing or doesn't match new one
            if (not doc.dob or str(doc.dob).lower() == "not specified" or
                str(doc.dob) != dob):
                update_data["dob"] = dob
                logger.debug(f"  - Will update dob for doc {doc.id}: '{doc.dob}' -> '{dob}'")
            
            # Update doi if missing (primary identifier for case)
            if not doc.doi or str(doc.doi).lower() == "not specified":
                update_data["doi"] = doi
                logger.debug(f"  - Will update doi for doc {doc.id}: '{doc.doi}' -> '{doi}'")
            
            # Update claimNumber if provided and missing or doesn't match
            if claim_number and str(claim_number).lower() != "not specified":
                if (not doc.doi or str(doc.doi).lower() == "not specified" or
                    str(doc.doi) != doi):
                    update_data["doi"] = doi
                    logger.debug(f"  - Will update doi for doc {doc.id}: '{doc.doi}' -> '{doi}'")
            
            # Only update if there's something to change
            if update_data:
                await self.prisma.document.update(
                    where={"id": doc.id},
                    data=update_data
                )
                updated_count += 1
                logger.info(f"  ✅ Updated doc {doc.id} with: {update_data}")
            else:
                logger.debug(f"  ℹ️ No updates needed for doc {doc.id}")
        
        logger.info(f"🔄 Updated {updated_count} documents for SAME PATIENT: '{patient_name}' (DOB: {dob}, Claim: {claim_number})")
        
        return updated_count
    
    async def get_patient_claim_numbers(
        self,
        patient_name: Optional[str] = None,
        physicianId: Optional[str] = None,
        dob: Optional[any] = None,  # 🆕 Change to 'any' to accept both string and datetime
        doi: Optional[str] = None,  # 🆕 Use DOI instead of claim_number
    ) -> Dict[str, Any]:
        try:
            logger.info(f"🎯 DEBUG - get_patient_claim_numbers CALLED WITH:")
            logger.info(f"  - patient_name: '{patient_name}'")
            logger.info(f"  - physicianId: '{physicianId}'")
            logger.info(f"  - dob: '{dob}' (type: {type(dob)})")  # 🆕 Log the type
            logger.info(f"  - doi: '{doi}'")
            
            # 🆕 ENHANCED: Build multiple WHERE clauses for better matching
            where_conditions = []
            
            # Always include physicianId if provided
            if physicianId:
                where_conditions.append({"physicianId": physicianId})
            
            # 🆕 FIX: Handle DOB conversion safely
            def format_dob_for_query(dob_value):
                if not dob_value:
                    return None
                if isinstance(dob_value, str):
                    # Already a string, return as-is
                    return dob_value
                elif hasattr(dob_value, 'strftime'):
                    # It's a datetime object, format it
                    return dob_value.strftime("%Y-%m-%d")
                else:
                    # Fallback - convert to string
                    return str(dob_value)
            
            formatted_dob = format_dob_for_query(dob)
            
            # 🆕 MULTI-FIELD MATCHING: Try different combinations using DOI
            if doi and patient_name and formatted_dob:
                # Case 1: All three fields provided - strongest match
                where_conditions.append({
                    "OR": [
                        {"doi": doi},
                        {"AND": [
                            {"patientName": patient_name},
                            {"dob": formatted_dob}  # 🆕 Use formatted_dob
                        ]}
                    ]
                })
                logger.info(f"🔍 Using multi-field match: doi + patient_name + dob")
            
            elif doi and patient_name:
                # Case 2: DOI + patient name
                where_conditions.append({
                    "OR": [
                        {"doi": doi},
                        {"patientName": patient_name}
                    ]
                })
                logger.info(f"🔍 Using dual-field match: doi + patient_name")
            
            elif doi and formatted_dob:
                # Case 3: DOI + DOB
                where_conditions.append({
                    "OR": [
                        {"doi": doi},
                        {"dob": formatted_dob}  # 🆕 Use formatted_dob
                    ]
                })
                logger.info(f"🔍 Using dual-field match: doi + dob")
            
            elif patient_name and formatted_dob:
                # Case 4: Patient name + DOB
                where_conditions.append({
                    "AND": [
                        {"patientName": patient_name},
                        {"dob": formatted_dob}  # 🆕 Use formatted_dob
                    ]
                })
                logger.info(f"🔍 Using dual-field match: patient_name + dob")
            
            elif doi:
                # Case 5: Only DOI
                where_conditions.append({"doi": doi})
                logger.info(f"🔍 Using single-field match: doi")
            
            elif patient_name:
                # Case 6: Only patient name
                where_conditions.append({"patientName": patient_name})
                logger.info(f"🔍 Using single-field match: patient_name")
            
            elif formatted_dob:
                # Case 7: Only DOB (least specific)
                where_conditions.append({"dob": formatted_dob})  # 🆕 Use formatted_dob
                logger.info(f"🔍 Using single-field match: dob")
            
            # Combine all conditions with AND
            final_where = {}
            if where_conditions:
                if len(where_conditions) == 1:
                    final_where = where_conditions[0]
                else:
                    final_where = {"AND": where_conditions}
            
            logger.info(f"🎯 DEBUG - FINAL WHERE CLAUSE: {final_where}")
            
            # Execute query
            documents = await self.prisma.document.find_many(where=final_where)
            
            logger.info(f"🎯 DEBUG - RAW DOCUMENTS FOUND: {len(documents)}")
            for i, doc in enumerate(documents):
                logger.info(f"  Doc {i+1}: patientName='{getattr(doc, 'patientName', None)}', dob='{getattr(doc, 'dob', None)}', doi='{getattr(doc, 'doi', None)}', claimNumber='{getattr(doc, 'claimNumber', None)}', physicianId='{getattr(doc, 'physicianId', None)}'")
            
            # Extract fields from documents
            claim_numbers = [
                doc.claimNumber for doc in documents if getattr(doc, "claimNumber", None)
            ]
            patient_names = [
                doc.patientName for doc in documents if getattr(doc, "patientName", None)
            ]
            dobs = [
                doc.dob for doc in documents if getattr(doc, "dob", None)
            ]
            dois = [
                doc.doi for doc in documents if getattr(doc, "doi", None)
            ]
            
            # Get primary values
            def get_primary_value(lst):
                if not lst:
                    return None
                # First try to find a valid (non-"not specified") value
                for item in lst:
                    if item and str(item).lower() != "not specified":
                        return item
                # If no valid values found, return the first one (even if "not specified")
                return lst[0] if lst else None
            
            primary_patient_name = get_primary_value(patient_names)
            primary_dob = get_primary_value(dobs)
            primary_doi = get_primary_value(dois)
            primary_claim_number = get_primary_value(claim_numbers)

            # Detect conflicting DOIs (different cases for same patient)
            valid_dois_set = set([d for d in dois if d and str(d).lower() != 'not specified'])
            has_conflicting_cases = len(valid_dois_set) > 1

            logger.info(f"✅ Found {len(documents)} documents for lookup: patient_name={primary_patient_name}, dob={primary_dob}, doi={primary_doi}, claim={primary_claim_number}, conflicting_cases={has_conflicting_cases}")
            
            return {
                "patient_name": primary_patient_name,
                "dob": primary_dob,
                "doi": primary_doi,
                "claim_number": primary_claim_number,
                "total_documents": len(documents),
                "has_conflicting_cases": has_conflicting_cases,
                "unique_valid_dois": list(valid_dois_set),
                "documents": [
                    {
                        "patientName": doc.patientName,
                        "dob": doc.dob,
                        "doi": doc.doi,
                        "claimNumber": doc.claimNumber,
                        "id": doc.id
                    } for doc in documents
                ]
            }

        except Exception as e:
            logger.error(f"❌ Error retrieving data for lookup: {str(e)}")
            return {
                "patient_name": None,
                "dob": None,
                "doi": None,
                "claim_number": None,
                "total_documents": 0,
                "has_conflicting_claims": False,
                "unique_valid_claims": [],
                "documents": []
            }
    async def get_last_document_for_patient(self, patient_name: str, claim_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent document for a specific patient and claim number.
        """
        try:
            document = await self.prisma.document.find_first(
                where={
                    "patientName": patient_name,
                    "claimNumber": claim_number
                },
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True,
                    # NO "whatsNew" - scalar, auto-included
                },
                order={"createdAt": "desc"}
            )
            
            if document:
                logger.info(f"📄 Found previous document for {patient_name} (Claim: {claim_number})")
                return document.dict()
            else:
                logger.info(f"📄 No previous document found for {patient_name} (Claim: {claim_number})")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving document for {patient_name}: {str(e)}")
            raise
    async def save_document_analysis(
            self,
            extraction_result: ExtractionResult,
            file_name: str,
            file_size: int,
            mime_type: str,
            processing_time_ms: int,
            gcs_file_link: str,
            patient_name: str,
            claim_number: str,
            dob: str,
            doi: str,
            rd: datetime,
            status: str,
            brief_summary: Any, # Changed to Any to support JSON summary
            summary_snapshots: List[Dict[str, Any]],
            whats_new: Dict[str, Any],
            adl_data: Dict[str, Any],
            document_summary: Dict[str, Any],
            physician_id: Optional[str] = None,
            blob_path: Optional[str] = None,
            file_hash: Optional[str] = None,
            mode: Optional[str] = None,
            ur_denial_reason: Optional[str] = None,
            original_name: Optional[str] = None,
            ai_summarizer_text: Optional[str] = None
        ) -> str:
            """
            Save document analysis results to the database.
            FIXED: bodyPart is ALWAYS required in SummarySnapshot, even in GM mode
            NOTE: Each document is saved as a NEW separate document (no duplicate checking)
            """
            try:
                print(f"📊 Saving {len(summary_snapshots)} summary snapshots for document in {mode.upper()} mode")

                # ℹ️ NOTE: We do NOT check for duplicates or delete existing documents
                # Real-world scenario: A patient may have multiple reports with same DOB/claim
                # (e.g., X-ray, Left Foot MRI, Right Foot MRI - all separate documents for same patient)
                # Each document is UNIQUE and should be saved separately

                # ✅ Step 1: Ensure document_summary has 'date'
                if "createdAt" in document_summary and "date" not in document_summary:
                    document_summary["date"] = document_summary["createdAt"]

                # ✅ Step 2: Handle whatsNew and briefSummary as JSON strings if needed
                whats_new_json = json.dumps(whats_new) if whats_new else None

                # Serialize brief_summary if it's a dict/list (for Prisma String field compatibility)
                if isinstance(brief_summary, (dict, list)):
                    brief_summary_val = json.dumps(brief_summary)
                else:
                    brief_summary_val = brief_summary

                # ✅ Step 3: Use the FIRST snapshot as primary summarySnapshot (for backward compatibility)
                primary_snapshot = summary_snapshots[0] if summary_snapshots else {}
                
                # ✅ Step 4: Create Document with nested relations including bodyPartSnapshots
                document_data = {
                    "patientName": patient_name,
                    "claimNumber": claim_number,
                    "dob": dob,
                    "doi": doi,
                    "status": status,
                    "gcsFileLink": gcs_file_link,
                    "briefSummary": brief_summary_val,
                    "whatsNew": whats_new_json,
                    "physicianId": physician_id,
                    "reportDate": rd,  # Use actual report date only, None if not found (do NOT fallback to current date)
                    "blobPath": blob_path,
                    "fileName": file_name,
                    "originalName": original_name,
                    "mode": mode,
                    "mode": mode,
                    "ur_denial_reason": ur_denial_reason,
                    "aiSummarizerText": ai_summarizer_text,
                    **({"fileHash": file_hash} if file_hash else {}),
                }

                # ✅ Step 5: Primary summary snapshot (for backward compatibility)
                # 🚨 CRITICAL FIX: bodyPart is ALWAYS required in SummarySnapshot!
                if summary_snapshots:
                    # For GM mode, use the condition as bodyPart, or fallback to "General"
                    body_part_value = primary_snapshot.get("body_part", "")
                    if mode == "gm" and not body_part_value:
                        body_part_value = primary_snapshot.get("condition", "General Condition")
                    
                    summary_snapshot_data = {
                        # BASIC FIELDS ONLY - these exist in SummarySnapshot model
                        "dx": primary_snapshot.get("dx", ""),
                        "keyConcern": primary_snapshot.get("key_concern", ""),
                        "nextStep": primary_snapshot.get("next_step", ""),
                        "bodyPart": body_part_value,  # 🚨 ALWAYS provide a value - never None!
                        "urDecision": primary_snapshot.get("ur_decision"),
                        "recommended": primary_snapshot.get("recommended"),
                        "aiOutcome": primary_snapshot.get("ai_outcome"),
                        "consultingDoctor": primary_snapshot.get("consulting_doctor", ""),
                        # Basic detail fields
                        "keyFindings": primary_snapshot.get("key_findings"),
                        "treatmentApproach": primary_snapshot.get("treatment_approach"),
                        "clinicalSummary": primary_snapshot.get("clinical_summary"),
                        "referralDoctor": primary_snapshot.get("referral_doctor")
                    }
                    
                    # Remove None values for optional fields, but ensure required fields have values
                    required_fields = ["dx", "keyConcern", "nextStep", "bodyPart"]
                    for field in required_fields:
                        if not summary_snapshot_data[field]:
                            summary_snapshot_data[field] = "Not specified"  # Fallback value
                    
                    # Remove None values from optional fields
                    summary_snapshot_data = {k: v for k, v in summary_snapshot_data.items() if v is not None}
                    
                    document_data["summarySnapshot"] = {
                        "create": summary_snapshot_data
                    }

                # ✅ Step 8: Multiple body part snapshots using the bodyPartSnapshots relation
                # THIS IS WHERE ALL MODE-SPECIFIC FIELDS GO!
                if summary_snapshots:
                    body_part_snapshots_data = []
                    
                    for snapshot in summary_snapshots:
                        # For GM mode, use condition field; for WC mode, use body_part field
                        body_part_value = snapshot.get("body_part")
                        condition_value = snapshot.get("condition")
                        
                        if mode == "gm":
                            # In GM mode, condition is primary, bodyPart can be None
                            condition_value = condition_value or body_part_value or "General Condition"
                            body_part_value = None
                        else:
                            # In WC mode, bodyPart is primary, condition can be None
                            body_part_value = body_part_value or "Not specified"
                            condition_value = None
                        
                        # Start with basic fields that exist in BodyPartSnapshot
                        snapshot_data = {
                            "mode": mode,  # Critical: include mode in each snapshot
                            "bodyPart": body_part_value,
                            "condition": condition_value,
                            "dx": snapshot.get("dx", ""),
                            "keyConcern": snapshot.get("key_concern", ""),
                            "nextStep": snapshot.get("next_step"),
                            "urDecision": snapshot.get("ur_decision"),
                            "recommended": snapshot.get("recommended"),
                            "aiOutcome": snapshot.get("ai_outcome"),
                            "consultingDoctor": snapshot.get("consulting_doctor", ""),
                            # Shared detail fields
                            "keyFindings": snapshot.get("key_findings"),
                            "treatmentApproach": snapshot.get("treatment_approach"),
                            "clinicalSummary": snapshot.get("clinical_summary"),
                            "referralDoctor": snapshot.get("referral_doctor"),
                            # Quality of life impact fields
                            "adlsAffected": snapshot.get("adls_affected"),
                            "painLevel": snapshot.get("pain_level"),
                            "functionalLimitations": snapshot.get("functional_limitations"),
                        }
                        
                        # 🆕 WC-SPECIFIC FIELDS - ONLY IN BODY PART SNAPSHOTS!
                        if mode == "wc":
                            # Workers Comp specific fields
                            wc_fields = {
                                "injuryType": snapshot.get("injury_type"),
                                "workRelatedness": snapshot.get("work_relatedness"),
                                "permanentImpairment": snapshot.get("permanent_impairment"),
                                "mmiStatus": snapshot.get("mmi_status"),
                                "returnToWorkPlan": snapshot.get("return_to_work_plan"),
                            }
                            snapshot_data.update({k: v for k, v in wc_fields.items() if v is not None})
                        else:
                            # General Medicine specific fields
                            gm_fields = {
                                "conditionSeverity": snapshot.get("condition_severity"),
                                "symptoms": snapshot.get("symptoms"),
                                "medications": snapshot.get("medications"),
                                "chronicCondition": snapshot.get("chronic_condition", False),
                                "comorbidities": snapshot.get("comorbidities"),
                                "lifestyleRecommendations": snapshot.get("lifestyle_recommendations"),
                            }
                            snapshot_data.update({k: v for k, v in gm_fields.items() if v is not None})
                        
                        # Remove None values to avoid Prisma errors
                        snapshot_data = {k: v for k, v in snapshot_data.items() if v is not None}
                        body_part_snapshots_data.append(snapshot_data)
                    
                    document_data["bodyPartSnapshots"] = {
                        "create": body_part_snapshots_data
                    }

                # ✅ Step 9: ADL (Activities of Daily Living) with mode-specific fields
                adl_create_data = {
                    "mode": mode,  # Critical: include mode in ADL
                    "adlsAffected": adl_data.get("adls_affected", ""),
                    "workRestrictions": adl_data.get("work_restrictions", ""),
                }
                
                # 🆕 MODE-SPECIFIC ADL FIELDS
                if mode == "wc":
                    # Workers Comp specific ADL fields
                    wc_adl_fields = {
                        "workImpact": adl_data.get("work_impact"),
                        "physicalDemands": adl_data.get("physical_demands"),
                        "workCapacity": adl_data.get("work_capacity"),
                    }
                    adl_create_data.update({k: v for k, v in wc_adl_fields.items() if v is not None})
                else:
                    # General Medicine specific ADL fields
                    gm_adl_fields = {
                        "dailyLivingImpact": adl_data.get("daily_living_impact"),
                        "functionalLimitations": adl_data.get("functional_limitations"),
                        "symptomImpact": adl_data.get("symptom_impact"),
                        "qualityOfLife": adl_data.get("quality_of_life"),
                    }
                    adl_create_data.update({k: v for k, v in gm_adl_fields.items() if v is not None})
                
                document_data["adl"] = {
                    "create": adl_create_data
                }

                # ✅ Step 10: Document Summary
                doc_summary_val = document_summary.get("summary", "")
                if isinstance(doc_summary_val, (dict, list)):
                    doc_summary_val = json.dumps(doc_summary_val)

                document_data["documentSummary"] = {
                    "create": {
                        "type": document_summary.get("type", ""),
                        "date": rd if rd else datetime.now(),
                        "summary": doc_summary_val
                    }
                }

                # ✅ Step 11: Create the document with all nested relations
                document = await self.prisma.document.create(
                    data=document_data,
                    include={
                        "summarySnapshot": True,
                        "adl": True,
                        "documentSummary": True,
                        "bodyPartSnapshots": True
                    }
                )

                # ✅ Step 12: Logging and response
                logger.info(f"✅ Document saved with ID: {document.id} in {mode.upper()} mode")
                logger.info(f"📊 Created {len(summary_snapshots)} body part snapshots")
                
                # Log what was saved
                if summary_snapshots:
                    logger.info(f"🔍 SummarySnapshot saved with bodyPart: {summary_snapshot_data.get('bodyPart', 'Not specified')}")
                    logger.info(f"🔍 BodyPartSnapshots saved with {len(body_part_snapshots_data)} records")
                    
                return document.id

            except Exception as e:
                logger.error(f"❌ Error saving document analysis in {mode.upper()} mode: {str(e)}")
                raise

    async def _find_existing_document(
            self,
            patient_name: str,
            claim_number: str,
            dob: str,
            doi: str,
            report_date: datetime,
            document_type: Optional[str] = None,
            physician_id: Optional[str] = None
        ) -> Optional[Any]:
            """
            Find existing document with the same patient details, claim number, and document type.
            Strict duplicate checking to prevent multiple documents for the same case.
            """
            try:
                # Base query for matching core identifiers
                where_conditions = {
                    "patientName": patient_name,
                    "claimNumber": claim_number,
                    "dob": dob,
                    "doi": doi,
                }
                
                # Add physician ID if provided
                if physician_id:
                    where_conditions["physicianId"] = physician_id
                
                # Add report date if provided (with some tolerance for time differences)
                if report_date:
                    # Look for documents with the same report date (within same day)
                    start_of_day = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_of_day = report_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                    where_conditions["reportDate"] = {
                        "gte": start_of_day,
                        "lte": end_of_day
                    }
                
                # Find documents matching the core criteria
                existing_documents = await self.prisma.document.find_many(
                    where=where_conditions,
                    include={
                        "documentSummary": True
                    },
                    order={"createdAt": "desc"}  # Get the most recent one first
                )
                
                # If we have document type, filter further
                if document_type and existing_documents:
                    for doc in existing_documents:
                        if (doc.documentSummary and 
                            doc.documentSummary.type and 
                            doc.documentSummary.type.lower() == document_type.lower()):
                            return doc
                    # If no exact type match but we have documents, return the most recent one
                    if existing_documents:
                        return existing_documents[0]
                
                # Return the most recent document if any found
                return existing_documents[0] if existing_documents else None
                
            except Exception as e:
                logger.error(f"❌ Error finding existing document: {str(e)}")
                return None
   
    async def _delete_existing_document(self, document_id: str) -> None:
        """
        Delete an existing document and all its related records.
        Manually delete all relations due to missing cascade constraints.
        """
        try:
            # ✅ Step 1: Delete all related records manually first
            # Delete body part snapshots
            await self.prisma.bodypartsnapshot.delete_many(where={"documentId": document_id})
            
            # Delete tasks
            await self.prisma.task.delete_many(where={"documentId": document_id})
            
            # Delete ADL
            await self.prisma.adl.delete_many(where={"documentId": document_id})
            
            # Delete document summary
            await self.prisma.documentsummary.delete_many(where={"documentId": document_id})
            
            # Delete summary snapshot (this has unique constraint, so use delete_many with where)
            await self.prisma.summarysnapshot.delete_many(where={"documentId": document_id})
            
            # ✅ Step 2: Now delete the main document
            await self.prisma.document.delete(where={"id": document_id})
            
            logger.info(f"🗑️ Successfully deleted document with ID: {document_id} and all related records")
            
        except Exception as e:
            logger.error(f"❌ Error deleting existing document {document_id}: {str(e)}")
            raise
    async def check_duplicate_by_hash(self, file_hash: str, physician_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a file with the same hash already exists for this physician.
        Returns the existing document if found, None otherwise.
        This prevents processing renamed versions of the same file.
        """
        try:
            existing_doc = await self.prisma.document.find_first(
                where={
                    "fileHash": file_hash,
                    "physicianId": physician_id
                },
                include={
                    "summarySnapshot": True,
                    "documentSummary": True
                }
            )
            
            if existing_doc:
                logger.info(f"🔍 Duplicate file detected - Hash: {file_hash[:16]}... for physician: {physician_id}")
                logger.info(f"   Original file: {existing_doc.fileName or 'unknown'}")
                logger.info(f"   Upload date: {existing_doc.createdAt}")
                return {
                    "id": existing_doc.id,
                    "fileName": existing_doc.fileName,
                    "patientName": existing_doc.patientName,
                    "claimNumber": existing_doc.claimNumber,
                    "status": existing_doc.status,
                    "createdAt": existing_doc.createdAt,
                    "gcsFileLink": existing_doc.gcsFileLink
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking duplicate by hash: {str(e)}")
            return None
    
    async def check_duplicate_document(self, patient_name: str, doi: Optional[str], report_date: Optional[str], 
                                        document_type: Optional[str], physician_id: str, 
                                        patient_dob: Optional[str] = None, claim_number: Optional[str] = None) -> bool:
        """
        Check if a document with the same patient name, DOB, claim number, document type, and physician already exists.
        Returns True if duplicate found (ALL fields match), False otherwise.
        """
        try:
            prisma = Prisma()
            await prisma.connect()
            
            # Build where conditions - must match ALL fields
            where_conditions = {
                "patientName": patient_name,
                "physicianId": physician_id,
            }
            
            # Add DOB condition if available
            if patient_dob and patient_dob.lower() not in ["not specified", "unknown", ""]:
                where_conditions["dob"] = patient_dob
            else:
                # If DOB is not available, we can't do complete matching
                logger.warning("⚠️ DOB not available for complete duplicate check")
                await prisma.disconnect()
                return False
            
            # Add claim number condition if available  
            if claim_number and claim_number.lower() not in ["not specified", "unknown", ""]:
                where_conditions["claimNumber"] = claim_number
            else:
                # If claim number is not available, we can't do complete matching
                logger.warning("⚠️ Claim number not available for complete duplicate check")
                await prisma.disconnect()
                return False
            
            # Add DOI condition if available
            if doi and doi.lower() not in ["not specified", "unknown", ""]:
                where_conditions["doi"] = doi
            else:
                # If DOI is not available, we can't do complete matching
                logger.warning("⚠️ DOI not available for complete duplicate check")
                await prisma.disconnect()
                return False
            
            # Add report date condition if available
            if report_date and report_date.lower() not in ["not specified", "unknown", ""]:
                try:
                    report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
                    where_conditions["reportDate"] = report_date_obj
                except ValueError:
                    logger.warning(f"⚠️ Could not parse report date: {report_date}")
                    await prisma.disconnect()
                    return False
            else:
                # If report date is not available, we can't do complete matching
                logger.warning("⚠️ Report date not available for complete duplicate check")
                await prisma.disconnect()
                return False
            
            # Add document type condition
            if document_type and document_type.lower() not in ["not specified", "unknown", ""]:
                # Query for documents that match ALL conditions AND have the same document type
                existing_docs = await prisma.document.find_many(
                    where={
                        **where_conditions,
                        "documentSummary": {
                            "is": {
                                "type": document_type
                            }
                        }
                    },
                    include={
                        "documentSummary": True
                    },
                    take=1
                )
            else:
                # If document type is not available, we can't do complete matching
                logger.warning("⚠️ Document type not available for complete duplicate check")
                await prisma.disconnect()
                return False
            
            await prisma.disconnect()
            
            is_duplicate = len(existing_docs) > 0
            
            if is_duplicate:
                existing_doc = existing_docs[0]
                logger.info(f"🔍 EXACT DUPLICATE DOCUMENT FOUND:")
                logger.info(f"   Patient: {patient_name}")
                logger.info(f"   DOB: {patient_dob}")
                logger.info(f"   Claim#: {claim_number}")
                logger.info(f"   DOI: {doi}")
                logger.info(f"   Report Date: {report_date}")
                logger.info(f"   Document Type: {document_type}")
                logger.info(f"   Physician ID: {physician_id}")
                logger.info(f"   Existing Document ID: {existing_doc.id}")
                
                if existing_doc.documentSummary:
                    logger.info(f"   Summary ID: {existing_doc.documentSummary.id}")
            
            return is_duplicate
            
        except Exception as e:
            logger.error(f"❌ Error checking for duplicate document: {str(e)}", exc_info=True)
            # In case of error, allow the document to be saved (fail open)
            return False
    
    
    def decrypt_patient_token(self, token: str) -> Dict[str, Any]:
        """
        Decrypts the token and returns patient data.
        Use this in the FastAPI route.
        """
        print(token,'token')
        try:
            # Pad the token if needed for base64 (since we rstrip'd '=')
            padded_token = token + '=' * (4 - len(token) % 4)
            encrypted_bytes = base64.urlsafe_b64decode(padded_token)
            decrypted_json = self.cipher_suite.decrypt(encrypted_bytes).decode('utf-8')
            patient_data = json.loads(decrypted_json)
            # No need to convert to datetime objects since they're stored as strings
            # patient_data["dob"] and patient_data["doi"] remain as strings
            return patient_data
        except Exception as e:
            logger.error(f"❌ Decryption failed: {str(e)}")
            raise ValueError("Invalid or expired token")
    
    async def delete_fail_doc(
        self,
        fail_doc_id: str
    ) -> dict:
        """Delete a failed document record from the FailDocs table."""
        try:
            deleted_fail_doc = await self.prisma.faildocs.delete(
                where={"id": fail_doc_id}
            )
            logger.info(f"🗑️ Deleted fail doc with ID: {fail_doc_id}")
            return deleted_fail_doc
        except Exception as e:
            logger.error(f"❌ Error deleting fail doc {fail_doc_id}: {str(e)}")
        raise
    async def get_patient_quiz(self, patient_name: str, dob: str, doi: str) -> Optional[Dict[str, Any]]:
        """Retrieve a PatientQuiz by matching patientName and DATE (ignoring time)"""
        print(patient_name, dob, doi, 'patient_name,dob,doi')
        try:
            # Since dob and doi are now strings, we need to parse them for date comparison
            # Assuming the format is "YYYY-MM-DD" or "Not specified"
            if dob.lower() != "not specified":
                try:
                    dob_start = datetime.strptime(dob, "%Y-%m-%d")
                    dob_end = dob_start + timedelta(days=1)
                except ValueError:
                    # If date parsing fails, skip the date filter
                    dob_start = None
                    dob_end = None
            else:
                dob_start = None
                dob_end = None

            if doi.lower() != "not specified":
                try:
                    doi_start = datetime.strptime(doi, "%Y-%m-%d")
                    doi_end = doi_start + timedelta(days=1)
                except ValueError:
                    # If date parsing fails, skip the date filter
                    doi_start = None
                    doi_end = None
            else:
                doi_start = None
                doi_end = None

            # Build the where clause
            where_clause = {"patientName": patient_name}
            
            if dob_start and dob_end:
                where_clause["dob"] = {
                    "gte": dob_start.isoformat(),
                    "lt": dob_end.isoformat(),
                }
                
            if doi_start and doi_end:
                where_clause["doi"] = {
                    "gte": doi_start.isoformat(),
                    "lt": doi_end.isoformat(),
                }

            quiz = await self.prisma.patientquiz.find_first(
                where=where_clause
            )

            print(quiz, 'quiz')
            if quiz:
                logger.info(f"✅ Found PatientQuiz for patient: {patient_name}")
            else:
                logger.info(f"ℹ️ No PatientQuiz found for patient: {patient_name}")
            return quiz.dict() if quiz else None
        except Exception as e:
            logger.error(f"❌ Error retrieving PatientQuiz: {str(e)}")
            return None
        

# services/database_service.py

    async def decrement_parse_count(self, physician_id: str) -> bool:
        """
        Decrement the documentParse count for a physician's subscription
        Returns True if successful, False if no subscription found or count already 0
        """
        try:
            prisma = Prisma()
            await prisma.connect()
            
            # Find the active subscription for the physician
            subscription = await prisma.subscription.find_first(
                where={
                    "physicianId": physician_id,
                    "status": "active",
                    "documentParse": {"gt": 0}  # Only update if count > 0
                }
            )
            
            if not subscription:
                logger.warning(f"No active subscription with parse count found for physician: {physician_id}")
                return False
            
            # Decrement the parse count
            updated_subscription = await prisma.subscription.update(
                where={"id": subscription.id},
                data={"documentParse": {"decrement": 1}}
            )
            
            logger.info(f"✅ Decremented parse count for physician {physician_id}. New count: {updated_subscription.documentParse}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to decrement parse count for physician {physician_id}: {e}")
            return False
        finally:
            await prisma.disconnect()
_db_service = None

async def get_database_service() -> DatabaseService:
    """Get or create database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
        await _db_service.connect()
    return _db_service

async def cleanup_database_service():
    """Cleanup database connection"""
    global _db_service
    if _db_service:
        await _db_service.disconnect()
        _db_service = None
