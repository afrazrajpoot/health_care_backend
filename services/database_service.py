import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from prisma import Prisma
from models.schemas import ExtractionResult

load_dotenv()
logger = logging.getLogger("document_ai")

class DatabaseService:
    """Service for database operations using your schema structure"""
    
    def __init__(self):
        self.prisma = Prisma()
        
        if not os.getenv("DATABASE_URL"):
            raise ValueError("DATABASE_URL environment variable not set")
    
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
            if claim_number:
                where_clause["claimNumber"] = claim_number
            if dob:
                where_clause["dob"] = dob
            if doi:
                where_clause["doi"] = doi
            
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
                # take=2
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
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all unverified documents for patient where status is NOT 'verified'
        Returns structured response with multiple documents
        """
        try:
            logger.info(f"🔍 Getting unverified documents for patient: {patient_name}")
            
            # Get all documents where status is NOT verified
            documents = await self.prisma.document.find_many(
                where={
                    "patientName": patient_name,
                    "status": {
                        "not": "verified"   # 👈 This is the key change
                    }
                },
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                    # NO "whatsNew" - scalar Json
                },
                order={"createdAt": "desc"}  # ✅ FIXED: Consistent with other methods
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
    async def get_all_unverified_documents(
        self, 
        patient_name: str, 
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve all unverified documents for patient where status is NOT 'verified'
        Returns structured response with multiple documents
        """
        try:
            logger.info(f"🔍 Getting unverified documents for patient: {patient_name}")
            
            # Get all documents where status is NOT verified
            documents = await self.prisma.document.find_many(
                where={
                    "patientName": patient_name,
                    "status": {
                        "not": "verified"   # 👈 This is the key change
                    }
                },
                include={
                    "summarySnapshot": True,
                    "adl": True,
                    "documentSummary": True
                    # NO "whatsNew" - scalar Json
                },
                order={
                    "createdAt": "desc"
                }
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
        status: str,
        brief_summary: str,
        summary_snapshot: Dict[str, str],
        whats_new: Dict[str, str],
        adl_data: Dict[str, str],
        document_summary: Dict[str, Any]
    ) -> str:
        """
        Save document analysis results to database.
        whatsNew set as JSON string for compatibility.
        """
        try:
            logger.info(f"💾 Saving document analysis for {patient_name} (Claim: {claim_number})")
            
            # Ensure document_summary has 'date' key
            if "createdAt" in document_summary and "date" not in document_summary:
                document_summary["date"] = document_summary["createdAt"]
            
            # Handle whatsNew as JSON string for Prisma Json field
            whats_new_json = json.dumps(whats_new) if whats_new else None
            
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
            return document.id
            
        except Exception as e:
            logger.error(f"❌ Error saving document analysis: {str(e)}")
            raise

# Singleton instance
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