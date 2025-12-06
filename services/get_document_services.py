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

        logger.info(f"✅ Returned aggregated document for: {patient_name}")
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
        """
        Format the aggregated document response from all documents.
        Returns array of documents, but each document includes ALL related data from ALL matching patient records.
        """
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

        # 🆕 AGGREGATE ALL DATA ACROSS ALL DOCUMENTS (not grouped by document_id)
        all_body_part_snapshots = []
        all_whats_new = []
        all_document_summaries = []
        all_adl_data = []
        all_summary_snapshots = []
        all_task_quick_notes = []

        for doc in sorted_documents:
            doc_id = doc["id"]
            created_at = self._format_date_field(doc.get("createdAt"))
            report_date = self._format_date_field(doc.get("reportDate"))

            # Collect ALL body_part_snapshots from ALL documents
            body_part_snapshots = doc.get("bodyPartSnapshots", [])
            for snapshot in body_part_snapshots:
                snapshot_with_context = snapshot.copy()
                snapshot_with_context["document_id"] = doc_id
                snapshot_with_context["document_created_at"] = created_at
                snapshot_with_context["document_report_date"] = report_date
                snapshot_with_context["file_name"] = doc.get("fileName")
                snapshot_with_context["gcs_file_link"] = doc.get("gcsFileLink")
                all_body_part_snapshots.append(snapshot_with_context)

            # Collect ALL whats_new from ALL documents
            whats_new_data = doc.get("whatsNew")
            if whats_new_data:
                all_whats_new.append(whats_new_data)

            # Collect ALL document summaries from ALL documents
            doc_summary = doc.get("documentSummary")
            if doc_summary:
                summary_entry = {
                    "date": self._format_date_field(doc_summary.get("date")),
                    "summary": doc_summary.get("summary"),
                    "type": doc_summary.get("type", "unknown")
                }
                all_document_summaries.append(summary_entry)

            # Collect ALL ADL data from ALL documents
            adl_data = doc.get("adl")
            if adl_data:
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
                all_adl_data.append(complete_adl_entry)

            # Collect ALL summary snapshots from ALL documents
            summary_snapshot = doc.get("summarySnapshot")
            if summary_snapshot:
                all_summary_snapshots.append(summary_snapshot)

            # Collect ALL task quick notes from ALL documents
            document_tasks = tasks_dict.get(doc_id, []) if tasks_dict else []
            for task in document_tasks:
                task_note = task.get("quickNotes", {})
                if task_note:
                    all_task_quick_notes.append(task_note)

        # Get the latest document to use as the base
        latest_document = sorted_documents[0]
        base_doc = await self._format_single_document_base(latest_document)
        
        # Merge ADL fields (combine lists and text)
        aggregated_adl = {
            "adls_affected": [],
            "work_restrictions": [],
            "complete_adl_data": all_adl_data  # All ADL records with context
        }
        
        for adl_entry in all_adl_data:
            if adl_entry.get("adls_affected"):
                aggregated_adl["adls_affected"].append(adl_entry.get("adls_affected"))
            if adl_entry.get("work_restrictions"):
                aggregated_adl["work_restrictions"].append(adl_entry.get("work_restrictions"))

        base_doc.update({
            "body_part_snapshots": all_body_part_snapshots,  # ALL body parts from ALL documents
            "whats_new": all_whats_new,  # ALL whats_new from ALL documents
            "brief_summary": latest_document.get("briefSummary"),  # Keep as single field from latest
            "document_summary": all_document_summaries[0] if all_document_summaries else None,  # Keep as single from latest
            "adl": aggregated_adl,  # AGGREGATED ADL data from ALL documents
            "summary_snapshot": all_summary_snapshots[0] if all_summary_snapshots else None,  # Keep as single from latest
            "task_quick_notes": all_task_quick_notes,  # ALL task quick notes from ALL documents
            "document_index": 1,
            "is_latest": True
        })

        # Wrap in response structure - keep as array with single aggregated document
        return {
            "patient_name": all_documents_data["patient_name"],
            "total_documents": len(sorted_documents),
            "documents": [base_doc],  # Array with single aggregated document
            "patient_quiz": None,
            "is_multiple_documents": len(sorted_documents) > 1,
            "total_body_parts": len(all_body_part_snapshots)
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

    async def _format_single_document_base(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Format base info for a single document with all fields from the model."""
        return {
            "document_id": document.get("id"),
            "patient_name": document.get("patientName"),
            "dob": document.get("dob"),  # String as per schema
            "doi": document.get("doi"),  # String as per schema
            "claim_number": document.get("claimNumber"),
            "status": document.get("status"),
            "gcs_file_link": document.get("gcsFileLink"),
            "blob_path": document.get("blobPath"),
            "file_name": document.get("fileName"),
            "file_hash": document.get("fileHash"),  # ✅ Added fileHash field
            "mode": document.get("mode"),  # ✅ Added mode field (default: "wc")
            "original_name": document.get("originalName"),  # ✅ Added originalName field
            "physician_id": document.get("physicianId"),  # ✅ Added physicianId field
            "ur_denial_reason": document.get("ur_denial_reason"),  # ✅ Added ur_denial_reason field
            "user_id": document.get("userId"),  # ✅ Added userId field
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