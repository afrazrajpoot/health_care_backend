
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
    
    async def save_document_analysis(
        self,
        extraction_result: ExtractionResult,
        file_name: str,
        file_size: int,
        mime_type: str,
        processing_time_ms: int,
        gcs_file_link: str,
        last_changes: Optional[str] = None
    ) -> str:
        """
        Save document analysis results to database
        """
        try:
            logger.info(f"💾 Saving document analysis to database: {file_name}")
            
            # Extract comprehensive analysis data or create fallback
            analysis = extraction_result.comprehensive_analysis
            if not analysis:
                logger.warning("⚠️ No comprehensive analysis data, using fallback")
                analysis = ComprehensiveAnalysis(
                    original_report=extraction_result.text or "",
                    report_json=PatientInfo(
                        patient_name=None,
                        patient_email=None,
                        claim_no=None,
                        report_title="Unknown Document",
                        time_day=datetime.now().isoformat(),
                        status="normal"
                    ),
                    summary=["Document processed but no analysis available"],
                    work_status_alert=[]
                )
            
            # Parse report date
            report_date = None
            if analysis.report_json.time_day:
                try:
                    parsed_date = datetime.fromisoformat(analysis.report_json.time_day.replace('Z', '+00:00'))
                    report_date = parsed_date.replace(tzinfo=None)
                except (ValueError, AttributeError):
                    logger.warning(f"⚠️ Could not parse report date: {analysis.report_json.time_day}")
            
            # Prepare document data
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
                "reportTitle": analysis.report_json.report_title,
                "reportDate": report_date,
                "status": analysis.report_json.status,
                "summary": analysis.summary or [],
                "originalReport": analysis.original_report,
                "processingTimeMs": processing_time_ms,
                "analysisSuccess": extraction_result.success,
                "errorMessage": extraction_result.error,
                "gcsFileLink": gcs_file_link
            }
            
            # Include lastchanges if provided
            if last_changes:
                document_data["lastchanges"] = last_changes
            
            # Create document record
            document = await self.prisma.document.create(data=document_data)
            
            # Save alerts
            alert_ids = []
            if analysis.work_status_alert:
                for alert_data in analysis.work_status_alert:
                    alert = await self.prisma.alert.create(
                        data={
                            "alertType": alert_data.alert_type,
                            "title": alert_data.title,
                            "date": alert_data.date,
                            "status": alert_data.status,
                            "documentId": document.id,
                        }
                    )
                    alert_ids.append(alert.id)
            
            logger.info(f"✅ Document saved to database with ID: {document.id}")
            logger.info(f"🚨 Created {len(alert_ids)} alerts")
            
            return document.id
            
        except Exception as e:
            logger.error(f"❌ Error saving document analysis: {str(e)}")
            raise
    
    async def get_last_document_for_patient(self, patient_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent document for a specific patient by name
        """
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
            print(document.dict().get('summary'),'previous doc')
            return document.dict()
            
        except Exception as e:
            logger.error(f"❌ Error retrieving document for patient {patient_name}: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID with related alerts
        """
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
        """
        Get recent documents with alerts
        """
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
    
    async def get_urgent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get urgent unresolved alerts
        """
        try:
            alerts = await self.prisma.alert.find_many(
                where={"status": "urgent", "isResolved": False},
                include={"document": True},
                order={"createdAt": "desc"},
                take=limit
            )
            
            alerts_list = [alert.dict() for alert in alerts]
            logger.info(f"🚨 Retrieved {len(alerts_list)} urgent alerts")
            return alerts_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving urgent alerts: {str(e)}")
            raise
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """
        Mark alert as resolved
        """
        try:
            alert = await self.prisma.alert.update(
                where={"id": alert_id},
                data={
                    "isResolved": True,
                    "resolvedAt": datetime.now(),
                    "resolvedBy": resolved_by,
                }
            )
            
            success = alert is not None
            if success:
                logger.info(f"✅ Alert {alert_id} resolved by {resolved_by}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error resolving alert {alert_id}: {str(e)}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for dashboard
        """
        try:
            total_documents = await self.prisma.document.count()
            urgent_alerts = await self.prisma.alert.count(
                where={"status": "urgent", "isResolved": False}
            )
            resolved_alerts = await self.prisma.alert.count(
                where={"isResolved": True}
            )
            
            seven_days_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            seven_days_ago = seven_days_ago.replace(day=seven_days_ago.day - 7)
            
            recent_documents = await self.prisma.document.count(
                where={"createdAt": {"gte": seven_days_ago}}
            )
            
            stats = {
                "total_documents": total_documents,
                "urgent_alerts": urgent_alerts,
                "resolved_alerts": resolved_alerts,
                "recent_documents": recent_documents,
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"📊 Database statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {str(e)}")
            return {
                "total_documents": 0,
                "urgent_alerts": 0,
                "resolved_alerts": 0,
                "recent_documents": 0,
                "error": str(e)
            }
    
    async def get_all_alerts(self, limit: int = 50, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """
        Get all alerts with optional filtering
        """
        try:
            where = {} if include_resolved else {"isResolved": False}
            alerts = await self.prisma.alert.find_many(
                where=where,
                include={"document": True},
                order={"createdAt": "desc"},
                take=limit
            )
            
            alerts_list = [alert.dict() for alert in alerts]
            logger.info(f"🚨 Retrieved {len(alerts_list)} alerts")
            return alerts_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving alerts: {str(e)}")
            raise
    
    async def get_document_alerts(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all alerts for a specific document
        """
        try:
            alerts = await self.prisma.alert.find_many(
                where={"documentId": document_id},
                order={"createdAt": "desc"}
            )
            
            alerts_list = [alert.dict() for alert in alerts]
            logger.info(f"📋 Retrieved {len(alerts_list)} alerts for document {document_id}")
            return alerts_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving document alerts: {str(e)}")
            raise
    
    async def search_documents(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search documents by patient name, claim number, or report title
        """
        try:
            documents = await self.prisma.document.find_many(
                where={
                    "OR": [
                        {"patientName": {"contains": query, "mode": "insensitive"}},
                        {"claimNumber": {"contains": query, "mode": "insensitive"}},
                        {"reportTitle": {"contains": query, "mode": "insensitive"}},
                        {"originalName": {"contains": query, "mode": "insensitive"}},
                    ]
                },
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
            
            logger.info(f"🔍 Found {len(docs_list)} documents matching '{query}'")
            return docs_list
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {str(e)}")
            raise

# Global database service instance
_db_service = None

async def get_database_service() -> DatabaseService:
    """Get singleton database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
        await _db_service.connect()
    return _db_service

async def cleanup_database_service():
    """Cleanup database service on shutdown"""
    global _db_service
    if _db_service:
        await _db_service.disconnect()
        _db_service = None
