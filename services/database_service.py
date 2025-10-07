import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from prisma import Prisma
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
        self.prisma = Prisma()
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
    
    async def connect(self):
        """Connect to the database"""
        try:
            await self.prisma.connect()
            logger.info("✅ Connected to database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise
    
    async def disconnect(self):
        """Disconnect from the database"""
        try:
            await self.prisma.disconnect()
            logger.info("🔌 Disconnected from database")
        except Exception as e:
            logger.warning(f"⚠️ Error disconnecting from database: {str(e)}")
    
    async def document_exists(self, filename: str, file_size: int) -> bool:
        """Check if document already exists by filename and size (adjust where clause if needed)"""
        try:
            count = await self.prisma.document.count(
                where={
                    "gcsFileLink": {"contains": filename},
                    # Add "fileSize": file_size if you add that field to schema
                }
            )
            return count > 0
        except Exception as e:
            logger.error(f"❌ Error checking document existence: {str(e)}")
            return False

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
        dob: Optional[datetime] = None,
        doi: Optional[datetime] = None,
        claim_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve last two documents for patient
        Returns structured response with multiple documents
        """
        try:
            where_clause = {"patientName": patient_name}
            
            if physicianId:
                where_clause["physicianId"] = physicianId
            
            if dob:
                # Match date ignoring time: gte midnight, lt next midnight
                dob_start = dob.replace(hour=0, minute=0, second=0, microsecond=0)
                dob_end = dob_start + timedelta(days=1)
                where_clause["dob"] = {"gte": dob_start, "lt": dob_end}
            
            if doi:
                # Match date ignoring time: gte midnight, lt next midnight
                doi_start = doi.replace(hour=0, minute=0, second=0, microsecond=0)
                doi_end = doi_start + timedelta(days=1)
                where_clause["doi"] = {"gte": doi_start, "lt": doi_end}
            
            if claim_number:
                where_clause["claimNumber"] = claim_number
            
            logger.info(f"🔍 Getting last 2 documents for patient: {patient_name}")
            
            # Get the last two documents for this patient name
            documents = await self.prisma.document.find_many(
                where=where_clause,
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                    # NO "whatsNew" - scalar Json
                },
                order={"createdAt": "desc"},
                take=2  # Limit to last two
            )
            
            logger.info(f"📋 Found {len(documents)} documents for {patient_name}")
            
            # Always return the multi-document structure
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
   
    async def get_all_unverified_documents(
        self, 
        patient_name: str, 
        physicianId: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all unverified documents for patient where status is NOT 'verified'
        Returns structured response with multiple documents
        """
        try:
            logger.info(f"🔍 Getting unverified documents for patient: {patient_name}")
            
            where_clause = {
                "patientName": patient_name,
                "status": {"not": "verified"}
            }
            if physicianId:
                where_clause["physicianId"] = physicianId
            
            # Get all documents where status is NOT verified
            documents = await self.prisma.document.find_many(
                where=where_clause,
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                    # NO "whatsNew" - scalar Json
                },
                order={"createdAt": "desc"}
            )
            
            if not documents:
                logger.warning(f"❌ No non-verified documents found for patient: {patient_name}")
                return None
            
            logger.info(f"📋 Found {len(documents)} documents for {patient_name}")
            
            # Always return the multi-document structure
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
            raise

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
        dob: datetime,
        doi: datetime,
        rd: datetime,
        status: str,
        brief_summary: str,
        summary_snapshot: Dict[str, str],
        whats_new: Dict[str, str],
        adl_data: Dict[str, str],
        document_summary: Dict[str, Any],
        physician_id: Optional[str] = None,
        blob_path: Optional[str] = None
        
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
            patient_data = {
                "patientName": patient_name,
                "dob": dob.isoformat(),  # Serialize datetime to string
                "doi": doi.isoformat()
            }
            patient_json = json.dumps(patient_data)
            encrypted_token = self.cipher_suite.encrypt(patient_json.encode())
            # Base64 URL-safe encode for URL (Fernet is already base64, but ensure URL-safe)
            url_safe_token = base64.urlsafe_b64encode(encrypted_token).decode('utf-8').rstrip('=')
            
            document = await self.prisma.document.create(
                data={
                    "patientName": patient_name,
                    "claimNumber": claim_number,
                    "dob": dob,
                    "doi": doi,
                    "status": status,
                    "gcsFileLink": gcs_file_link,
                    "briefSummary": brief_summary,
                    "whatsNew": whats_new_json,  # JSON string for scalar Json field
                    "physicianId": physician_id,
                    "patientQuizPage": f"http://localhost:3000/intake-form?token={url_safe_token}",
                    "createdAt":rd if rd else datetime.now(),
                    "blobPath": blob_path,
                    # Optional: Add these if you extend schema
                    # "originalName": file_name,
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
                            "date": document_summary.get("date", datetime.now()),
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
            logger.info(f"🔐 Encrypted token: {url_safe_token[:20]}...")
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
            # Convert strings back to datetime objects
            patient_data["dob"] = datetime.fromisoformat(patient_data["dob"])
            patient_data["doi"] = datetime.fromisoformat(patient_data["doi"])
            return patient_data
        except Exception as e:
            logger.error(f"❌ Decryption failed: {str(e)}")
            raise ValueError("Invalid or expired token")
# Singleton instance

    async def get_patient_quiz(self, patient_name: str, dob: str, doi: str) -> Optional[Dict[str, Any]]:
        """Retrieve a PatientQuiz by matching patientName and DATE (ignoring time)"""
        print(patient_name, dob, doi, 'patient_name,dob,doi')
        try:
            # Parse the date strings into datetime objects (assuming dob and doi are 'YYYY-MM-DD')
            dob_start = datetime.strptime(dob, "%Y-%m-%d")
            dob_end = dob_start + timedelta(days=1)

            doi_start = datetime.strptime(doi, "%Y-%m-%d")
            doi_end = doi_start + timedelta(days=1)

            quiz = await self.prisma.patientquiz.find_first(
                where={
                    "patientName": patient_name,
                    "dob": {
                        "gte": dob_start.isoformat(),
                        "lt": dob_end.isoformat(),
                    },
                    "doi": {
                        "gte": doi_start.isoformat(),
                        "lt": doi_end.isoformat(),
                    },
                }
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