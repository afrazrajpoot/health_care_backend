import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from fastapi import HTTPException

from services.database_service import get_database_service
from utils.logger import logger

class DocumentAggregationService:
    """
    Service for aggregating and formatting patient documents.
    Handles fetching documents, tasks, and formatting aggregated responses.
    """

    def __init__(self):
        self.db_service = None  # Will be initialized asynchronously

    async def initialize_db(self):
        """Initialize database service."""
        if self.db_service is None:
            self.db_service = await get_database_service()

    async def get_aggregated_document(
        self,
        patient_name: str,
        dob: str,
        physician_id: Optional[str] = None,
        claim_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch and aggregate documents for a patient.
        Returns a formatted aggregated response.
        """
        await self.initialize_db()
        logger.info(f"📄 Fetching aggregated document for patient: {patient_name}")

        # Parse date strings using helper
        dob_date = self._parse_date(dob, "Date of Birth")

        # Get all documents (includes bodyPartSnapshots via relation, matched by patient details)
        document_data = await self.db_service.get_document_by_patient_details(
            patient_name=patient_name,
            physicianId=physician_id,
            dob=dob_date,
            claim_number=claim_number
        )

        if not document_data or document_data.get("total_documents") == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for patient: {patient_name}"
            )

        # Fetch tasks for the document IDs, filtered by physicianId if provided
        documents = document_data["documents"]
        document_ids = [doc["id"] for doc in documents]
        tasks = await self.db_service.get_tasks_by_patient_details(
            patient_name=patient_name,
            dob=dob_date,
            claim_number=claim_number,
            physician_id=physician_id
        )
     
        # Create a mapping of document_id to list of tasks (to handle multiple tasks per document)
        tasks_dict = {}
        for task in tasks:
            doc_id = task["documentId"]
            if doc_id not in tasks_dict:
                tasks_dict[doc_id] = []
            tasks_dict[doc_id].append(task)

        response = await self._format_aggregated_document_response(
            all_documents_data=document_data,
            tasks_dict=tasks_dict
        )
        
        # 🆕 Use merged patient details from database service
        response["patient_name"] = document_data.get("patient_name", patient_name)
        response["dob"] = document_data.get("dob", dob)
        response["claim_number"] = document_data.get("claim_number", claim_number)

        logger.info(f"✅ Returned aggregated document for: {response['patient_name']}")
        return response

    def _parse_date(self, date_str: str, field_name: str) -> Optional[datetime]:
        """
        Parse a date string into a datetime object.
        Supports YYYY-MM-DD and other common formats.
        """
        if not date_str or str(date_str).lower() == "not specified":
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                logger.warning(f"⚠️ Failed to parse {field_name}: {date_str}")
                return None

    async def _format_aggregated_document_response(
        self,
        all_documents_data: Dict[str, Any],
        tasks_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format the aggregated document response from all documents."""
        documents = all_documents_data["documents"]

        if not documents:
            return {
                "patient_name": all_documents_data["patient_name"],
                "total_documents": 0,
                "documents": [],
                "patient_quiz": None,
                "is_multiple_documents": False
            }

        # Sort documents by reportDate descending (latest first)
        sorted_documents = sorted(documents, key=self._parse_report_date, reverse=True)

        # Grouped data by document_id
        whats_new_by_document = {}  # Stores original whatsNew as is, without grouping
        body_part_by_document = {}
        brief_summary_by_document = {}
        document_summary_by_document = {}
        adl_by_document = {}
        summary_snapshot_by_document = {}

        for doc in sorted_documents:
            doc_id = doc["id"]
            created_at = self._format_date_field(doc.get("createdAt"))
            report_date = self._format_date_field(doc.get("reportDate"))

            # Group body_part_snapshots by document
            body_part_snapshots = doc.get("bodyPartSnapshots", [])
            grouped_body_parts = []
            for snapshot in body_part_snapshots:
                snapshot_with_context = snapshot.copy()
                snapshot_with_context["document_created_at"] = created_at
                snapshot_with_context["document_report_date"] = report_date
                grouped_body_parts.append(snapshot_with_context)
            if doc_id not in body_part_by_document:
                body_part_by_document[doc_id] = []
            body_part_by_document[doc_id].extend(grouped_body_parts)

            # Send whats_new as is, without processing or grouping
            whats_new_data = doc.get("whatsNew")
            whats_new_by_document[doc_id] = whats_new_data if whats_new_data is not None else []

            # Group brief_summary by document
            brief_summary = doc.get("briefSummary")
            if brief_summary:
                brief_summary_by_document[doc_id] = brief_summary

            # Group document_summary by document
            doc_summary = doc.get("documentSummary")
            if doc_summary:
                summary_entry = {
                    "date": self._format_date_field(doc_summary.get("date")),
                    "summary": doc_summary.get("summary"),
                    "type": doc_summary.get("type", "unknown")
                }
                document_summary_by_document[doc_id] = summary_entry

            # Group ADL by document - INCLUDING ALL FIELDS
            adl_data = doc.get("adl")
            if adl_data:
                # Create complete ADL entry with all fields
                complete_adl_entry = {
                    # Shared fields
                    "adls_affected": adl_data.get("adlsAffected"),
                    "work_restrictions": adl_data.get("workRestrictions"),
                    "mode": adl_data.get("mode", "wc"),
                    
                    # GM-specific fields
                    "daily_living_impact": adl_data.get("dailyLivingImpact"),
                    "functional_limitations": adl_data.get("functionalLimitations"),
                    "symptom_impact": adl_data.get("symptomImpact"),
                    "quality_of_life": adl_data.get("qualityOfLife"),
                    
                    # WC-specific fields
                    "work_impact": adl_data.get("workImpact"),
                    "physical_demands": adl_data.get("physicalDemands"),
                    "work_capacity": adl_data.get("workCapacity"),
                    
                    # Metadata
                    "created_at": self._format_date_field(adl_data.get("createdAt")),
                    "updated_at": self._format_date_field(adl_data.get("updatedAt"))
                }
                
                if doc_id not in adl_by_document:
                    adl_by_document[doc_id] = {
                        "adls_affected": [],
                        "work_restrictions": [],
                        "complete_adl_data": []  # New field for complete ADL data
                    }
                
                # Add to complete ADL data
                adl_by_document[doc_id]["complete_adl_data"].append(complete_adl_entry)
                
                # Also maintain the legacy structure for backward compatibility
                if "adls_affected" not in adl_by_document[doc_id]:
                    adl_by_document[doc_id]["adls_affected"] = []
                if adls_affected := adl_data.get("adlsAffected"):
                    if isinstance(adl_by_document[doc_id]["adls_affected"], list):
                        adl_by_document[doc_id]["adls_affected"].append(adls_affected)
                    else:
                        adl_by_document[doc_id]["adls_affected"] = [adl_by_document[doc_id]["adls_affected"], adls_affected]
                
                if "work_restrictions" not in adl_by_document[doc_id]:
                    adl_by_document[doc_id]["work_restrictions"] = []
                if work_restrictions := adl_data.get("workRestrictions"):
                    if isinstance(adl_by_document[doc_id]["work_restrictions"], list):
                        adl_by_document[doc_id]["work_restrictions"].append(work_restrictions)
                    else:
                        adl_by_document[doc_id]["work_restrictions"] = [adl_by_document[doc_id]["work_restrictions"], work_restrictions]

            # Group summary_snapshot by document
            summary_snapshot = doc.get("summarySnapshot")
            if summary_snapshot:
                summary_snapshot_by_document[doc_id] = summary_snapshot

        # Get unique document IDs, sorted by latest report date
        unique_doc_ids = list(dict.fromkeys([doc["id"] for doc in sorted_documents]))
        id_to_latest_doc = {doc["id"]: doc for doc in sorted_documents}
        latest_docs = [id_to_latest_doc[doc_id] for doc_id in unique_doc_ids]

        # Create list of per-document responses
        per_document_responses = []
        
        # Create merged patient data for consistent info across all documents
        merged_patient_data = {
            "patient_name": all_documents_data.get("patient_name"),
            "dob": all_documents_data.get("dob"),
            "claim_number": all_documents_data.get("claim_number")
        }
        
        for latest_doc in latest_docs:
            doc_id = latest_doc["id"]
            base_doc = await self._format_single_document_base(latest_doc, merged_patient_data)
            
            # Get ADL data for this document
            document_adl_data = adl_by_document.get(doc_id, {
                "adls_affected": [], 
                "work_restrictions": [],
                "complete_adl_data": []
            })
            
            # Get task quick notes for this document (list of quickNotes JSON from tasks)
            document_tasks = tasks_dict.get(doc_id, []) if tasks_dict else []
            task_quick_notes = [task.get("quickNotes", {}) for task in document_tasks]
            
            base_doc.update({
                "body_part_snapshots": body_part_by_document.get(doc_id, []),
                "whats_new": whats_new_by_document.get(doc_id, []),  # Original structure as is
                "brief_summary": brief_summary_by_document.get(doc_id),
                "document_summary": document_summary_by_document.get(doc_id),
                "adl": document_adl_data,  # Now includes complete_adl_data with all fields
                "summary_snapshot": summary_snapshot_by_document.get(doc_id),
                "task_quick_notes": task_quick_notes,  # New field: list of quickNotes per task for this document
                "document_index": latest_docs.index(latest_doc) + 1,
                "is_latest": doc_id == latest_docs[0]["id"]
            })
            per_document_responses.append(base_doc)

        # Top-level aggregations
        total_body_parts = sum(len(snapshots) for snapshots in body_part_by_document.values())
        unique_total = len(unique_doc_ids)

        # Wrap in response structure
        return {
            "patient_name": all_documents_data["patient_name"],
            "total_documents": unique_total,
            "documents": per_document_responses,
            "patient_quiz": None,
            "is_multiple_documents": unique_total > 1,
            "total_body_parts": total_body_parts
        }
        
    def _parse_report_date(self, doc: Dict[str, Any]) -> datetime:
        """Parse reportDate for sorting, fallback to createdAt."""
        report_date = doc.get("reportDate")
        if report_date:
            if isinstance(report_date, str):
                try:
                    return datetime.fromisoformat(report_date.replace('Z', '+00:00'))
                except ValueError:
                    pass
            elif isinstance(report_date, datetime):
                return report_date
        # Fallback to createdAt
        return self._parse_created_at(doc)

    def _parse_created_at(self, doc: Dict[str, Any]) -> datetime:
        """Parse createdAt for sorting."""
        created_at = doc.get("createdAt")
        if created_at:
            if isinstance(created_at, str):
                try:
                    return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except ValueError:
                    pass
            elif isinstance(created_at, datetime):
                return created_at
        return datetime.min

    async def _format_single_document_base(self, document: Dict[str, Any], merged_patient_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format base info for a single document with all fields from the model.
        Uses merged_patient_data if provided for consolidated patient info across multiple documents.
        """
        # Use merged data if available, otherwise use document data
        if merged_patient_data:
            patient_name = merged_patient_data.get("patient_name") or document.get("patientName")
            dob = merged_patient_data.get("dob") or document.get("dob")
            claim_number = merged_patient_data.get("claim_number") or document.get("claimNumber")
        else:
            patient_name = document.get("patientName")
            dob = document.get("dob")
            claim_number = document.get("claimNumber")
        
        return {
            "document_id": document.get("id"),
            "patient_name": patient_name,
            "dob": dob,  # Merged from all documents
            "doi": document.get("doi"),
            "claim_number": claim_number,  # Merged from all documents
            "status": document.get("status"),
            "gcs_file_link": document.get("gcsFileLink"),
            "blob_path": document.get("blobPath"),
            "file_name": document.get("fileName"),
            "file_hash": document.get("fileHash"),
            "mode": document.get("mode"),
            "original_name": document.get("originalName"),
            "physician_id": document.get("physicianId"),
            "ur_denial_reason": document.get("ur_denial_reason"),
            "user_id": document.get("userId"),
            "created_at": self._format_date_field(document.get("createdAt")),
            "updated_at": self._format_date_field(document.get("updatedAt")),
            "report_date": self._format_date_field(document.get("reportDate")),
        }

    async def _format_adl_complete(self, adl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format complete ADL data with all fields."""
        if not adl_data:
            return None
            
        return {
            # Shared fields
            "adls_affected": adl_data.get("adlsAffected"),
            "work_restrictions": adl_data.get("workRestrictions"),
            "mode": adl_data.get("mode", "wc"),
            
            # GM-specific fields
            "daily_living_impact": adl_data.get("dailyLivingImpact"),
            "functional_limitations": adl_data.get("functionalLimitations"),
            "symptom_impact": adl_data.get("symptomImpact"),
            "quality_of_life": adl_data.get("qualityOfLife"),
            
            # WC-specific fields
            "work_impact": adl_data.get("workImpact"),
            "physical_demands": adl_data.get("physicalDemands"),
            "work_capacity": adl_data.get("workCapacity"),
            
            # Metadata
            "created_at": self._format_date_field(adl_data.get("createdAt")),
            "updated_at": self._format_date_field(adl_data.get("updatedAt"))
        }

    def _format_date_field(self, field_value: Any) -> Optional[str]:
        """Safely format datetime field to ISO string."""
        if field_value is None:
            return None
        if isinstance(field_value, datetime):
            return field_value.isoformat()
        if isinstance(field_value, str):
            return field_value
        return None