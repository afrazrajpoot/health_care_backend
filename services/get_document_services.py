# services/document_aggregation_service.py
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
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

        # Get all documents
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
        tasks = await self.db_service.get_tasks_by_document_ids(
            document_ids=document_ids,
            physician_id=physician_id  # Optional filter
        )
        # Create a mapping of document_id to quickNotes
        tasks_dict = {task["documentId"]: task["quickNotes"] for task in tasks}

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
        """Format the aggregated document response from all documents."""
        documents = all_documents_data["documents"]

        if not documents:
            return {
                "patient_name": all_documents_data["patient_name"],
                "total_documents": 0,
                "documents": [],
                "patient_quiz": None,  # Commented out in original
                "is_multiple_documents": False
            }

        # Sort documents by reportDate ascending (oldest first)
        sorted_documents = sorted(documents, key=self._parse_report_date)

        # Find the latest document based on reportDate
        latest_doc = max(documents, key=self._parse_report_date)

        # Find the last saved document based on createdAt for full whats_new chain
        last_saved_doc = max(documents, key=self._parse_created_at)

        # Use the latest document for base info
        base_response = await self._format_single_document_base(latest_doc)

        # ✅ CHANGED: Collect all bodyPartSnapshots in chronological order, reverse to latest first
        all_body_part_snapshots = []
        for doc in sorted_documents:
            body_part_snapshots = doc.get("bodyPartSnapshots", [])
            if body_part_snapshots:
                # Add document context to each body part snapshot
                for snapshot in body_part_snapshots:
                    snapshot_with_context = snapshot.copy()
                    snapshot_with_context["document_id"] = doc["id"]
                    snapshot_with_context["document_created_at"] = self._format_date_field(doc.get("createdAt"))
                    snapshot_with_context["document_report_date"] = self._format_date_field(doc.get("reportDate"))
                    all_body_part_snapshots.append(snapshot_with_context)
        
        # Reverse to show latest first
        body_part_snapshots = all_body_part_snapshots[::-1]

        # Collect quick notes snapshots in chronological order, reverse to latest first, filter out None/null
        quick_notes_snapshots_list = [
            tasks_dict.get(doc["id"]) if tasks_dict else None
            for doc in sorted_documents
        ]
        # Filter out None/null values
        quick_notes_snapshots_filtered = [note for note in quick_notes_snapshots_list if note is not None]
        quick_notes_snapshots = quick_notes_snapshots_filtered[::-1]

        # ADL from latest document
        adl = await self._format_adl(latest_doc)

        # whats_new from the last saved document (most recent comparison, full chain)
        whats_new = last_saved_doc.get("whatsNew", {})

        # Status from the last saved document
        status = last_saved_doc.get("status")

        # Document_summary: group by type (chronological reportDate order)
        grouped_summaries = {}
        grouped_brief_summaries = {}
        for doc in sorted_documents:
            # Group brief_summary by type
            doc_summary = doc.get("documentSummary")
            doc_type = doc_summary.get("type", "unknown") if doc_summary else "unknown"

            brief_summary = doc.get("briefSummary")
            if brief_summary:
                if doc_type not in grouped_brief_summaries:
                    grouped_brief_summaries[doc_type] = []
                grouped_brief_summaries[doc_type].append(brief_summary)

            # Group document_summary
            if doc_summary:
                if doc_type not in grouped_summaries:
                    grouped_summaries[doc_type] = []
                summary_entry = {
                    "date": self._format_date_field(doc_summary.get("date")),
                    "summary": doc_summary.get("summary")
                }
                grouped_summaries[doc_type].append(summary_entry)

        base_response.update({
            "body_part_snapshots": body_part_snapshots,  # ✅ CHANGED: Now using body_part_snapshots
            "quick_notes_snapshots": quick_notes_snapshots,  # Filtered, no nulls
            "whats_new": whats_new,
            "adl": adl,
            "document_summary": grouped_summaries,
            "brief_summary": grouped_brief_summaries,
            "document_index": 1,
            "is_latest": True,
            "status": status  # Override from last saved doc
        })

        # Wrap in single document structure
        return {
            "patient_name": all_documents_data["patient_name"],
            "total_documents": all_documents_data["total_documents"],
            "documents": [base_response],
            "patient_quiz": None,  # Commented out in original
            "is_multiple_documents": len(documents) > 1,
            "total_body_parts": len(body_part_snapshots)  # ✅ ADDED: Total body parts count
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
        """Format base info for a single document."""
        return {
            "document_id": document.get("id"),
            "patient_name": document.get("patientName"),
            "dob": document.get("dob"),  # String as per schema
            "doi": document.get("doi"),  # String as per schema
            "claim_number": document.get("claimNumber"),
            "status": document.get("status"),
            "gcs_file_link": document.get("gcsFileLink"),
            "blob_path": document.get("blobPath"),  # Adjust key if needed
            "created_at": self._format_date_field(document.get("createdAt")),
            "updated_at": self._format_date_field(document.get("updatedAt")),
        }

    async def _format_adl(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Format ADL data."""
        adl_data = document.get("adl")
        if adl_data:
            return {
                "adls_affected": adl_data.get("adlsAffected"),
                "work_restrictions": adl_data.get("workRestrictions")
            }
        return None

    def _format_date_field(self, field_value: Any) -> Optional[str]:
        """Safely format datetime field to ISO string."""
        if field_value is None:
            return None
        if isinstance(field_value, datetime):
            return field_value.isoformat()
        if isinstance(field_value, str):
            return field_value
        return None