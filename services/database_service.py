# services/database_service.py
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from prisma import Prisma
from models.schemas import ComprehensiveAnalysis, ExtractionResult, PatientInfo, WorkStatusAlert

# Load environment variables
load_dotenv()

logger = logging.getLogger("document_ai")


class DatabaseService:
    """Service for database operations related to document analysis using Prisma ORM"""

    def __init__(self):
        self.prisma = Prisma()

        if not os.getenv("DATABASE_URL"):
            raise ValueError("DATABASE_URL environment variable not set")

    async def connect(self):
        try:
            await self.prisma.connect()
            logger.info("✅ Connected to database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise

    async def disconnect(self):
        try:
            await self.prisma.disconnect()
            logger.info("🔌 Disconnected from database")
        except Exception as e:
            logger.warning(f"⚠️ Error disconnecting from database: {str(e)}")

    async def save_document_analysis(
        self,
        extraction_result: ExtractionResult,
        report_title: str,
        file_name: str,
        file_size: int,
        mime_type: str,
        processing_time_ms: int,
        gcs_file_link: str,
        last_changes: Optional[Any] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        review_tickets: Optional[List[Dict[str, Any]]] = None,
        compliance_nudges: Optional[List[Dict[str, Any]]] = None,
        referrals: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Save document analysis results to database.
        Accepts deterministic alerts/actions produced by RuleEngine.
        """
        try:

            analysis = extraction_result.comprehensive_analysis
            logger.info(f"💾 Saving document analysis to database: {analysis}")
            if not analysis:
                logger.warning("⚠️ No comprehensive analysis data, using fallback")
                analysis = ComprehensiveAnalysis(
                    original_report=extraction_result.text or "",
                    report_json=PatientInfo(
                        patient_name=None,
                        patient_email=None,
                        claim_no=None,
                        report_title=report_title,
                        time_day=datetime.now().isoformat(),
                        status="normal"
                    ),
                    summary=["Document processed but no analysis available"],
                    work_status_alert=[]
                )

            report_date = None
            if analysis.report_json and analysis.report_json.time_day:
                try:
                    parsed_date = datetime.fromisoformat(str(analysis.report_json.time_day).replace('Z', '+00:00'))
                    report_date = parsed_date.replace(tzinfo=None)
                except Exception:
                    logger.warning(f"⚠️ Could not parse report date: {analysis.report_json.time_day}")

            document_data = {
                "originalName": file_name,
                "fileSize": file_size,
                "mimeType": mime_type,
                "extractedText": extraction_result.text or "",
                "pages": extraction_result.pages,
                "confidence": extraction_result.confidence,
                "entities": [json.dumps(entity) for entity in extraction_result.entities] if extraction_result.entities else [],
                "tables": [json.dumps(table) for table in extraction_result.tables] if extraction_result.tables else [],
                "formFields": [json.dumps(field) for field in extraction_result.formFields] if extraction_result.formFields else [],
                "patientName": analysis.report_json.patient_name,
                "patientEmail": analysis.report_json.patient_email,
                "claimNumber": analysis.report_json.claim_no,
                "reportTitle": analysis.report_json.report_title or report_title,
                "reportDate": report_date,
                "status": analysis.report_json.status,
                "summary": analysis.summary or [],
                "originalReport": analysis.original_report,
                "processingTimeMs": processing_time_ms,
                "analysisSuccess": extraction_result.success,
                "errorMessage": extraction_result.error,
                "gcsFileLink": gcs_file_link,
                "lastchanges": json.dumps(last_changes or []),  
                # "actions": json.dumps(actions or []),
                # "alerts": json.dumps(alerts or []),
                "complianceNudges": json.dumps(compliance_nudges or []),
                "referrals": json.dumps(referrals or []),
            }

            logger.info("🔍 DB payload inspection:")
            logger.info("   • lastchanges type=%s | preview=%s", type(document_data.get("lastchanges")), str(document_data.get("lastchanges"))[:500])
            logger.info("   • actions type=%s | raw preview=%s", type(actions), str(actions)[:500] if actions else "[]")
            logger.info("   • alerts type=%s | raw preview=%s", type(alerts), str(alerts)[:500] if alerts else "[]")
            logger.info("   • review_tickets type=%s | raw preview=%s", type(review_tickets), str(review_tickets)[:500] if review_tickets else "[]")

            # Create document record
            document = await self.prisma.document.create(data=document_data)

            # Try to create alert records in dedicated alerts model (if exists)
            created_alert_ids = []
            if alerts:
                try:
                    if hasattr(self.prisma, "alert"):
                        for alert_data in alerts:
                            alert = await self.prisma.alert.create(
                                data={
                                    "alertType": alert_data.get("alert_type"),  # map snake_case → camelCase
                                    "title": alert_data.get("title"),
                                    "date": alert_data.get("date"),
                                    "status": alert_data.get("status"),
                                    "description": alert_data.get("source") or "",  # map source/rule_id → description
                                    "documentId": document.id,
                                    # remove metadata → not in Prisma schema
                                }
                            )
                            created_alert_ids.append(alert.id)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to persist alerts to alert model: {str(e)}")
                    # alerts are still stored in document.actions JSON field

            # Persist actions: if 'task' model exists create tasks else leave actions in JSON
            created_task_ids = []
            if actions:
                try:
                    if hasattr(self.prisma, "task"):
                        for act in actions:
                            task = await self.prisma.task.create(
                                data={
                                    "title": act.get("type"),
                                    "description": act.get("reason"),
                                    "dueDate": act.get("due_date"),
                                    "assignee": act.get("assignee"),
                                    "documentId": document.id,
                                    "metadata": json.dumps(act)
                                }
                            )
                            created_task_ids.append(task.id)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to persist actions to task model: {str(e)}")
                    # actions remain serialized on document

            logger.info(f"✅ Document saved to database with ID: {document.id}")
            logger.info(f"🚨 Created {len(created_alert_ids)} alerts, {len(created_task_ids)} tasks (if models exist)")

            return document.id
        except Exception as e:
            logger.error(f"❌ Error saving document analysis: {str(e)}")
            raise

 
 
    # (other methods unchanged but kept to ensure compatibility)
    async def get_last_document_for_patient(self, patient_name: str) -> Optional[Dict[str, Any]]:
        try:
            document = await self.prisma.document.find_first(
                where={"patientName": patient_name},
                include={"alerts": True},
                order={"createdAt": "desc"}
            )

            if not document:
                logger.info(f"📄 No document found for patient: {patient_name}")
                return None

            logger.info(f"📄 Retrieved most recent document for patient {patient_name}: {document.originalName}")
            return document.dict()

        except Exception as e:
            logger.error(f"❌ Error retrieving document for patient {patient_name}: {str(e)}")
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            document = await self.prisma.document.find_unique(
                where={"id": document_id},
                include={"alerts": True}
            )

            if not document:
                return None

            logger.info(f"📄 Retrieved document: {document.originalName}")
            return document.dict()

        except Exception as e:
            logger.error(f"❌ Error retrieving document {document_id}: {str(e)}")
            raise

    async def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            documents = await self.prisma.document.find_many(
                include={"alerts": True},
                order={"createdAt": "desc"},
                take=limit
            )

            docs_list = []
            for doc in documents:
                doc_dict = doc.dict()
                doc_dict["alert_count"] = len(doc.alerts)
                doc_dict["urgent_alert_count"] = len([a for a in doc.alerts if a.status == "urgent" and not a.isResolved])
                docs_list.append(doc_dict)

            logger.info(f"📋 Retrieved {len(docs_list)} recent documents")
            return docs_list

        except Exception as e:
            logger.error(f"❌ Error retrieving recent documents: {str(e)}")
            raise

    # ... other methods unchanged (get_urgent_alerts, resolve_alert, statistics, etc.)

# Singleton helper
_db_service = None


async def get_database_service() -> DatabaseService:
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
        await _db_service.connect()
    return _db_service


async def cleanup_database_service():
    global _db_service
    if _db_service:
        await _db_service.disconnect()
        _db_service = None
