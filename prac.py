# services/database_service.py
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from prisma import Prisma
from prisma.models import Document
from models.schemas import ExtractionResult
from utils.logger import logger

class DatabaseService:
    def __init__(self, db: Prisma):
        self.db = db

 
    async def save_document_analysis(
        self,
        extraction_result: ExtractionResult,
        file_name: str,
        file_size: int,
        mime_type: str,
        processing_time_ms: int,
        blob_path: str,
        file_hash: str,
        gcs_file_link: str,
        patient_name: str,
        claim_number: str,
        dob: str,
        doi: str,
        status: str,
        brief_summary: str,
        summary_snapshots: List[Dict],
        whats_new: Dict,
        adl_data: Dict,
        document_summary: Dict,
        rd: datetime = None,
        physician_id: str = None,
        mode: str = "wc",
        ur_denial_reason: str = None
    ) -> str:
        """Save document analysis to database"""
        try:
            # Convert summary_snapshots to JSON
            summary_snapshots_json = json.dumps(summary_snapshots)
            
            # Create document in database
            document = await self.db.document.create(
                data={
                    "file_name": file_name,
                    "file_size": file_size,
                    "mime_type": mime_type,
                    "processing_time_ms": processing_time_ms,
                    "blob_path": blob_path,
                    "file_hash": file_hash,
                    "gcs_file_link": gcs_file_link,
                    "patient_name": patient_name,
                    "claim_number": claim_number,
                    "dob": dob,
                    "doi": doi,
                    "status": status,
                    "brief_summary": brief_summary,
                    "summary_snapshots": summary_snapshots_json,
                    "whats_new": json.dumps(whats_new),
                    "adl_data": json.dumps(adl_data),
                    "document_summary": json.dumps(document_summary),
                    "rd": rd,
                    "physician_id": physician_id,
                    "mode": mode,
                    "ur_denial_reason": ur_denial_reason,
                    "extraction_result": json.dumps(extraction_result.dict()),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            )
            return document.id
        except Exception as e:
            logger.error(f"Error saving document analysis: {e}")
            raise

    async def get_patient_claim_numbers(
        self,
        patient_name: str = None,
        physicianId: str = None,
        dob: datetime = None,
        claim_number: str = None
    ) -> Dict[str, Any]:
        """Get patient claim numbers with conflict detection"""
        try:
            where_conditions = {}
            
            if patient_name and patient_name.lower() != "not specified":
                where_conditions["patient_name"] = patient_name
            if physicianId:
                where_conditions["physician_id"] = physicianId
            if dob:
                where_conditions["dob"] = dob.strftime("%Y-%m-%d")
            if claim_number and claim_number.lower() != "not specified":
                where_conditions["claim_number"] = claim_number

            documents = await self.db.document.find_many(
                where=where_conditions,
                order={"created_at": "desc"}
            )

            if not documents:
                return {
                    "total_documents": 0,
                    "unique_valid_claims": [],
                    "has_conflicting_claims": False,
                    "patient_name": None,
                    "dob": None,
                    "doi": None,
                    "claim_number": None
                }

            # Extract unique valid claim numbers
            valid_claims = list(set(
                doc.claim_number for doc in documents 
                if doc.claim_number and doc.claim_number.lower() != "not specified"
            ))

            # Check for conflicts
            has_conflicting_claims = len(valid_claims) > 1

            # Get most recent document values
            latest_doc = documents[0]
            
            return {
                "total_documents": len(documents),
                "unique_valid_claims": valid_claims,
                "has_conflicting_claims": has_conflicting_claims,
                "patient_name": latest_doc.patient_name,
                "dob": latest_doc.dob,
                "doi": latest_doc.doi,
                "claim_number": latest_doc.claim_number
            }
        except Exception as e:
            logger.error(f"Error getting patient claim numbers: {e}")
            return {
                "total_documents": 0,
                "unique_valid_claims": [],
                "has_conflicting_claims": False,
                "patient_name": None,
                "dob": None,
                "doi": None,
                "claim_number": None
            }

    async def get_all_unverified_documents(
        self,
        patient_name: str,
        physicianId: str,
        claimNumber: str = None,
        dob: datetime = None
    ) -> Dict[str, Any]:
        """Get all unverified documents for a patient"""
        try:
            where_conditions = {
                "patient_name": patient_name,
                "physician_id": physicianId,
                "status": {"not": "failed"}
            }
            
            if claimNumber and claimNumber.lower() != "not specified":
                where_conditions["claim_number"] = claimNumber
            if dob:
                where_conditions["dob"] = dob.strftime("%Y-%m-%d")

            documents = await self.db.document.find_many(
                where=where_conditions,
                order={"created_at": "desc"}
            )

            return {
                "documents": [
                    {
                        "id": doc.id,
                        "file_name": doc.file_name,
                        "patient_name": doc.patient_name,
                        "claim_number": doc.claim_number,
                        "dob": doc.dob,
                        "doi": doc.doi,
                        "status": doc.status,
                        "brief_summary": doc.brief_summary,
                        "summary_snapshots": json.loads(doc.summary_snapshots) if doc.summary_snapshots else [],
                        "created_at":