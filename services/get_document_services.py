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
        print(tasks,'taks')
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
        whats_new_by_document = {}  # Now stores simple arrays of bullet points
        body_part_by_document = {}
        brief_summary_by_document = {}
        document_summary_by_document = {}
        adl_by_document = {}

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

            # UPDATED: Handle whats_new as simple array of bullet point strings
            grouped_whats_new = []
            whats_new_data = doc.get("whatsNew", [])
            
            # Handle both array format and legacy dict format for backward compatibility
            if isinstance(whats_new_data, list):
                # New format: array of bullet point strings
                for bullet_point in whats_new_data:
                    if bullet_point and isinstance(bullet_point, str) and bullet_point.strip():
                        grouped_whats_new.append(bullet_point)
            elif isinstance(whats_new_data, dict):
                # Legacy format: convert dict values to array
                for category, value in whats_new_data.items():
                    if value and isinstance(value, str) and value.strip() and value.lower() != 'none':
                        grouped_whats_new.append(value)

            # Add quick_notes from tasks as separate bullet points
            if doc_id in tasks_dict:
                for task in tasks_dict[doc_id]:
                    quick_notes_data = task.get("quickNotes", {})
                    content = quick_notes_data.get("one_line_note") or quick_notes_data.get("status_update")
                    description = task.get("description", "")
                    if content:
                        grouped_whats_new.append(content)

            # Store simple array of bullet points by document_id
            whats_new_by_document[doc_id] = grouped_whats_new

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

            # Group ADL by document
            adl_data = doc.get("adl")
            if adl_data:
                if doc_id not in adl_by_document:
                    adl_by_document[doc_id] = {
                        "adls_affected": [],
                        "work_restrictions": []
                    }
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

        # Get unique document IDs, sorted by latest report date
        unique_doc_ids = list(dict.fromkeys([doc["id"] for doc in sorted_documents]))
        id_to_latest_doc = {doc["id"]: doc for doc in sorted_documents}
        latest_docs = [id_to_latest_doc[doc_id] for doc_id in unique_doc_ids]

        # Create list of per-document responses
        per_document_responses = []
        for latest_doc in latest_docs:
            doc_id = latest_doc["id"]
            base_doc = await self._format_single_document_base(latest_doc)
            base_doc.update({
                "body_part_snapshots": body_part_by_document.get(doc_id, []),
                "whats_new": whats_new_by_document.get(doc_id, []),  # Simple array of bullet point strings
                "brief_summary": brief_summary_by_document.get(doc_id),
                "document_summary": document_summary_by_document.get(doc_id),
                "adl": adl_by_document.get(doc_id, {"adls_affected": [], "work_restrictions": []}),
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