import asyncio
import json
import logging
import os
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
            reason: str, 
            db: Optional[str] = None,
            doi: Optional[str] = None,
            claim_number: Optional[str] = None,
            patient_name: Optional[str] = None,
            document_text: Optional[str] = None,
            physician_id: Optional[str] = None,
            gcs_file_link: Optional[str] = None,
            file_name: Optional[str] = None,
            file_hash: Optional[str] = None,
            blob_path: Optional[str] = None
        ) -> str:
            """Save a failed document record to the FailDocs table."""
            try:
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
                
                fail_doc = await self.prisma.faildocs.create(
                    data=data
                )
                logger.info(f"💾 Saved fail doc with ID: {fail_doc.id} (Physician ID: {physician_id if physician_id else 'None'})")
                return fail_doc.id
            except Exception as e:
                logger.error(f"❌ Error saving fail doc: {str(e)}")
                raise
    
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
            count = await self.prisma.document.count(
                where={
                    # "gcsFileLink": {"contains": filename},
                    "fileName": filename,
                    # Add "fileSize": file_size if you add that field to schema
                }
            )
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
            # doi: Optional[str] = None,
            claim_number: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Retrieve last two documents for patient.
            Handles dob and doi as strings (or converts datetime to string if needed).
            """
            try:
                where_clause = {"patientName": patient_name}
                
                if physicianId:
                    where_clause["physicianId"] = physicianId

                # ✅ Handle dob (string or datetime)
                if dob:
                    if isinstance(dob, datetime):
                        dob_str = dob.strftime("%Y-%m-%d")
                    else:
                        # Try parsing string into datetime for normalization
                        try:
                            parsed_dob = datetime.fromisoformat(dob)
                            dob_str = parsed_dob.strftime("%Y-%m-%d")
                        except ValueError:
                            dob_str = dob  # Already correct format
                    where_clause["dob"] = dob_str

                # ✅ Handle doi (string or datetime)
                # if doi:
                #     if isinstance(doi, datetime):
                #         doi_str = doi.strftime("%Y-%m-%d")
                #     else:
                #         try:
                #             parsed_doi = datetime.fromisoformat(doi)
                #             doi_str = parsed_doi.strftime("%Y-%m-%d")
                #         except ValueError:
                #             doi_str = doi
                #     where_clause["doi"] = doi_str

                if claim_number:
                    where_clause["claimNumber"] = claim_number

                logger.info(f"🔍 Fetching last 2 documents with filters: {where_clause}")
                
                documents = await self.prisma.document.find_many(
                    where=where_clause,
                    include={
                        "summarySnapshot": True,
                        "adl": True,
                        "documentSummary": True
                    },
                    order={"createdAt": "desc"},
                    take=2
                )

                logger.info(f"📋 Found {len(documents)} documents for {patient_name}")

                response = {
                    "patient_name": patient_name,
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
                logger.error(f"❌ Error retrieving documents for {patient_name}: {str(e)}")
                return {
                    "patient_name": patient_name,
                    "total_documents": 0,
                    "documents": []
                }
            

    async def get_tasks_by_document_ids(self, document_ids: list[str], physician_id: Optional[str] = None) -> list[dict]:
        """Fetch tasks (with quickNotes) by document IDs, optionally filtered by physician_id"""
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
                "quickNotes": getattr(t, "quickNotes", None)
            }
            for t in tasks
        ]
        return tasks_data

    # async def get_all_unverified_documents(
    #     self, 
    #     patient_name: str, 
    #     physicianId: Optional[str] = None,
    #     dob: Optional[datetime] = None,
    # ) -> Optional[Dict[str, Any]]:
    #     """
    #     Retrieve all unverified documents for patient where status is NOT 'verified'
    #     Returns structured response with multiple documents
    #     """
    #     try:
    #         logger.info(f"🔍 Getting unverified documents for patient: {patient_name}")
            
    #         where_clause = {
    #             "patientName": patient_name,
    #             "dob":dob,
    #             "status": {"not": "verified"}
    #         }
    #         if physicianId:
    #             where_clause["physicianId"] = physicianId
            
    #         # Get all documents where status is NOT verified
    #         documents = await self.prisma.document.find_many(
    #             where=where_clause,
    #             include={
    #                 "summarySnapshot": True,
    #                 "adl": True,
    #                 "documentSummary": True
    #                 # NO "whatsNew" - scalar Json
    #             },
    #             order={"createdAt": "desc"}
    #         )
            
    #         if not documents:
    #             logger.warning(f"❌ No non-verified documents found for patient: {patient_name}")
    #             return None
            
    #         logger.info(f"📋 Found {len(documents)} documents for {patient_name}")
            
    #         # Always return the multi-document structure
    #         response = {
    #             "patient_name": patient_name,
    #             "total_documents": len(documents),
    #             "documents": []
    #         }
            
    #         for i, doc in enumerate(documents):
    #             doc_data = doc.dict()
    #             doc_data["document_index"] = i + 1
    #             doc_data["is_latest"] = i == 0
    #             response["documents"].append(doc_data)
    #             logger.info(f"📄 Added document {i+1}: ID {doc.id}")
            
    #         return response
                    
    #     except Exception as e:
    #         logger.error(f"❌ Error retrieving documents for {patient_name}: {str(e)}")
    #         raise


  

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
    


  
    async def update_previous_fields(
        self,
        patient_name: str,
        dob: str,
        physician_id: str,
        claim_number: str,
        doi: Optional[str] = None,
    ) -> int:
        print(patient_name, physician_id, claim_number, dob, 'patient data for update')
        
        # 🆕 IMPROVED: Fetch based on looser criteria to catch docs with incorrect/missing fields
        # Primary: physician_id + (patient_name OR claim_number match/missing)
        or_conditions = []
        
        # Condition 1: Matches patient_name (if provided)
        if patient_name and patient_name.lower() != "not specified":
            or_conditions.append({"patientName": patient_name})
        
        # Condition 2: Matches claim_number or missing
        if claim_number and claim_number.lower() != "not specified":
            or_conditions.append({"claimNumber": {"in": [claim_number, "Not specified"]}})
        
        # If no OR conditions, fallback to just physician_id
        fetch_where = {
            "physicianId": physician_id,
        }
        
        if or_conditions:
            fetch_where["OR"] = or_conditions
        
        # 🆕 FIXED: No 'select' parameter
        documents_to_update = await self.prisma.document.find_many(
            where=fetch_where,
        )
        
        logger.info(f"🔍 Found {len(documents_to_update)} documents to potentially update for physician '{physician_id}'")
        
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
            
            # Update claimNumber if missing
            if not doc.claimNumber or str(doc.claimNumber).lower() == "not specified":
                update_data["claimNumber"] = claim_number
                logger.debug(f"  - Will update claimNumber for doc {doc.id}: '{doc.claimNumber}' -> '{claim_number}'")
            
            # Update doi if provided and missing or doesn't match
            if doi and str(doi).lower() != "not specified":
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
        
        logger.info(f"🔄 Updated {updated_count} previous documents for patient '{patient_name}' (DOB: {dob}, Physician: {physician_id}) with fields: patientName={patient_name}, dob={dob}, claimNumber={claim_number}, doi={doi}")
        
        return updated_count
    
    async def get_patient_claim_numbers(
        self,
        patient_name: Optional[str] = None,
        physicianId: Optional[str] = None,
        dob: Optional[datetime] = None,
        claim_number: Optional[str] = None,  # 🆕 NEW: Optional claim_number for lookup
    ) -> Dict[str, Any]:
        try:
            where_clause = {}
            
            # 🆕 ENHANCED: Support multiple lookup strategies (prioritize claim_number if provided)
            if claim_number:
                where_clause["claimNumber"] = claim_number
                logger.info(f"🔍 Fetching data using claim number: {claim_number}")
            else:
                # Fallback to patient_name + optional dob/physician
                if patient_name:
                    where_clause["patientName"] = patient_name
                if physicianId:
                    where_clause["physicianId"] = physicianId
                if dob:
                    dob_str = dob.strftime("%Y-%m-%d")
                    where_clause["dob"] = dob_str
                logger.info(f"🔍 Fetching data for patient: {patient_name or 'unknown'}")

            documents = await self.prisma.document.find_many(where=where_clause)
            
            # 🆕 EXPANDED: Extract not just claim_numbers, but also patient_name, dob, doi
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
            
            # 🆕 DEDUPLICATE AND PRIORITIZE: Use first non-None/"Not specified" value for each field
            def get_first_valid(lst):
                for item in lst:
                    if item and str(item).lower() != "not specified":
                        return item
                return None
            
            primary_patient_name = get_first_valid(patient_names) or patient_names[0] if patient_names else None
            primary_dob = get_first_valid(dobs) or dobs[0] if dobs else None
            primary_doi = get_first_valid(dois) or dois[0] if dois else None
            primary_claim_number = get_first_valid(claim_numbers) or claim_numbers[0] if claim_numbers else None

            # 🆕 NEW: Detect conflicting claim numbers
            valid_claims_set = set([c for c in claim_numbers if c and str(c).lower() != 'not specified'])
            has_conflicting_claims = len(valid_claims_set) > 1

            logger.info(f"✅ Found data for lookup: patient_name={primary_patient_name}, dob={primary_dob}, doi={primary_doi}, claim={primary_claim_number}, conflicting_claims={has_conflicting_claims}")
            
            return {
                "patient_name": primary_patient_name,
                "dob": primary_dob,
                "doi": primary_doi,
                "claim_number": primary_claim_number,
                "total_documents": len(documents),
                "has_conflicting_claims": has_conflicting_claims,
                "unique_valid_claims": list(valid_claims_set),
                "documents": [  # 🆕 OPTIONAL: Include full docs if needed for further processing
                    {
                        "patientName": doc.patientName,
                        "dob": doc.dob,
                        "doi": doc.doi,
                        "claimNumber": doc.claimNumber,
                        "id": doc.id  # For reference
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
        dob: str,  # Now a string
        doi: str,  # Now a string
        rd: datetime,
        status: str,
        brief_summary: str,
        summary_snapshot: Dict[str, str],
        whats_new: Dict[str, str],
        adl_data: Dict[str, str],
        document_summary: Dict[str, Any],
        physician_id: Optional[str] = None,
        blob_path: Optional[str] = None
    ,
    file_hash: Optional[str] = None
        
    ) -> str:
        """
        Save document analysis results to database.
        whatsNew set as JSON string for compatibility.
        Encrypts patient details in URL token.
        """
        try:
            logger.info(f"💾 Saving document analysis for {patient_name} (Claim: {claim_number})")
            
            # ✅ NEW: Check if document already exists (using filename in gcsFileLink)
            if await self.document_exists(file_name, file_size):
                existing_doc = await self.prisma.document.find_first(
                    where={"gcsFileLink": {"contains": file_name}},
                    order={"createdAt": "desc"}
                )
                logger.warning(f"⚠️ Document already exists: {file_name} (ID: {existing_doc.id if existing_doc else 'N/A'}). Skipping save.")
                return existing_doc.id if existing_doc else "unknown"  # Return existing ID or handle as needed
            
            # Ensure document_summary has 'date' key
            if "createdAt" in document_summary and "date" not in document_summary:
                document_summary["date"] = document_summary["createdAt"]
            
            # Handle whatsNew as JSON string for Prisma Json field
            whats_new_json = json.dumps(whats_new) if whats_new else None
            
            # Encrypt patient details into a URL-safe token
            # patient_data = {
            #     "patientName": patient_name,
            #     "dob": dob,  # Already a string, no need for isoformat()
            #     "doi": doi   # Already a string, no need for isoformat()
            # }
            # patient_json = json.dumps(patient_data)
            # encrypted_token = self.cipher_suite.encrypt(patient_json.encode())
            # Base64 URL-safe encode for URL (Fernet is already base64, but ensure URL-safe)
            # url_safe_token = base64.urlsafe_b64encode(encrypted_token).decode('utf-8').rstrip('=')
            
            document = await self.prisma.document.create(
                data={
                    "patientName": patient_name,
                    "claimNumber": claim_number,
                    "dob": dob,  # String value
                    "doi": doi,  # String value
                    "status": status,
                    "gcsFileLink": gcs_file_link,
                    "briefSummary": brief_summary,
                    "whatsNew": whats_new_json,  # JSON string for scalar Json field
                    # Optional file hash for deduplication
                    **({"fileHash": file_hash} if file_hash else {}),
                    "physicianId": physician_id,
                    # "patientQuizPage": f"http://localhost:3000/intake-form?token={url_safe_token}",
                  
                    "reportDate": rd if rd else datetime.now(),
                    "blobPath": blob_path,
                    # Optional: Add these if you extend schema
                    "fileName": file_name,
                    # "fileSize": file_size,
                    # etc.
                    "summarySnapshot": {
                        "create": {
                            "dx": summary_snapshot.get("dx", ""),
                            "keyConcern": summary_snapshot.get("keyConcern", ""),
                            "nextStep": summary_snapshot.get("nextStep", "")
                        }
                    },
                    "adl": {
                        "create": {
                            "adlsAffected": adl_data.get("adlsAffected", ""),
                            "workRestrictions": adl_data.get("workRestrictions", "")
                        }
                    },
                    "documentSummary": {
                        "create": {
                            "type": document_summary.get("type", ""),
                            "date": rd if rd else datetime.now(),
                            "summary": document_summary.get("summary", "")
                        }
                    }
                },
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True,
                    # NO "whatsNew" - scalar
                }
            )
            
            logger.info(f"✅ Document saved with ID: {document.id}")
            logger.info(f"📊 WhatsNew JSON: {whats_new_json[:100]}..." if whats_new_json else "📊 WhatsNew: None")
            # logger.info(f"🔐 Encrypted token: {url_safe_token[:20]}...")
            return document.id
            
        except Exception as e:
            logger.error(f"❌ Error saving document analysis: {str(e)}")
            raise
    
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
