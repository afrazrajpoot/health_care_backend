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
    
    async def get_last_document_for_patient(self, patient_name: str, dob: datetime, doi: datetime) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent document for a specific patient using name, DOB, and DOI
        """
        try:
            document = await self.prisma.document.find_first(
                where={
                    "patientName": patient_name,
                    "dob": dob,
                    "doi": doi
                },
                include={
                    "summarySnapshot": True,
                    "whatsNew": True,
                    "adl": True,
                    "documentSummary": True
                },
                order={"createdAt": "desc"}
            )
            
            if document:
                logger.info(f"📄 Found previous document for {patient_name} (DOB: {dob}, DOI: {doi})")
                return document.dict()
            else:
                logger.info(f"📄 No previous document found for {patient_name} (DOB: {dob}, DOI: {doi})")
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
        summary_snapshot: Dict[str, str],
        whats_new: Dict[str, str],
        adl_data: Dict[str, str],
        document_summary: Dict[str, Any]
    ) -> str:
        """
        Save document analysis results to database with proper relationships
        """
        try:
            logger.info(f"💾 Saving document analysis for {patient_name} (Claim: {claim_number})")
            
            # Create main document record
            document = await self.prisma.document.create(
                data={
                    "patientName": patient_name,
                    "claimNumber": claim_number,
                    "dob": dob,
                    "doi": doi,
                    "status": status,
                    "gcsFileLink": gcs_file_link,
                    "originalName": file_name,
                    "fileSize": file_size,
                    "mimeType": mime_type,
                    "processingTimeMs": processing_time_ms,
                    "extractedText": extraction_result.text or "",
                    "pages": extraction_result.pages,
                    "confidence": extraction_result.confidence,
                    # Create related records
                    "summarySnapshot": {
                        "create": {
                            "dx": summary_snapshot.get("dx", ""),
                            "keyConcern": summary_snapshot.get("keyConcern", ""),
                            "nextStep": summary_snapshot.get("nextStep", "")
                        }
                    },
                    "whatsNew": {
                        "create": {
                            "diagnostic": whats_new.get("diagnostic", ""),
                            "qme": whats_new.get("qme", ""),
                            "urDecision": whats_new.get("urDecision", ""),
                            "legal": whats_new.get("legal", "")
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
                    "whatsNew": True,
                    "adl": True,
                    "documentSummary": True
                }
            )
            
            logger.info(f"✅ Document saved with ID: {document.id}")
            logger.info(f"📊 Created related records:")
            logger.info(f"   - SummarySnapshot: {document.summarySnapshot.id if document.summarySnapshot else 'None'}")
            logger.info(f"   - WhatsNew: {document.whatsNew.id if document.whatsNew else 'None'}")
            logger.info(f"   - ADL: {document.adl.id if document.adl else 'None'}")
            logger.info(f"   - DocumentSummary: {document.documentSummary.id if document.documentSummary else 'None'}")
            
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