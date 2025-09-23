import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncpg
from dotenv import load_dotenv
from models.schemas import ComprehensiveAnalysis, ExtractionResult

# Load environment variables
load_dotenv()

logger = logging.getLogger("document_ai")

class DatabaseService:
    """Service for database operations related to document analysis"""
    
    def __init__(self):
        load_dotenv()
        # Remove schema parameter from URL as asyncpg doesn't support it
        database_url = os.getenv("DATABASE_URL", "")
        self.database_url = database_url.replace("?schema=public", "")
        self.pool = None
        
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
    
    async def connect(self):
        """Connect to the database"""
        try:
            self.pool = await asyncpg.create_pool(self.database_url)
            logger.info("✅ Connected to database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise
    
    async def disconnect(self):
        """Disconnect from the database"""
        try:
            if self.pool:
                await self.pool.close()
            logger.info("🔌 Disconnected from database")
        except Exception as e:
            logger.warning(f"⚠️ Error disconnecting from database: {str(e)}")
    
    async def save_document_analysis(
        self, 
        extraction_result: ExtractionResult, 
        file_name: str, 
        file_size: int, 
        mime_type: str,
        processing_time_ms: int
    ) -> str:
        """
        Save document analysis results to database
        
        Args:
            extraction_result: Complete extraction results
            file_name: Original filename
            file_size: File size in bytes
            mime_type: MIME type of the file
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Document ID
        """
        async with self.pool.acquire() as conn:
            try:
                logger.info(f"💾 Saving document analysis to database: {file_name}")
                
                # Extract comprehensive analysis data
                analysis = extraction_result.comprehensive_analysis
                
                if not analysis:
                    raise ValueError("No comprehensive analysis data to save")
                
                # Parse report date
                report_date = None
                if analysis.report_json.time_day:
                    try:
                        # Parse and convert to timezone-naive datetime for PostgreSQL
                        parsed_date = datetime.fromisoformat(analysis.report_json.time_day.replace('Z', '+00:00'))
                        report_date = parsed_date.replace(tzinfo=None)  # Remove timezone info
                    except (ValueError, AttributeError):
                        logger.warning(f"⚠️ Could not parse report date: {analysis.report_json.time_day}")
                
                # Create document record with proper JSON serialization
                document_id = await conn.fetchval("""
                    INSERT INTO documents (
                        id, "originalName", "fileSize", "mimeType", "extractedText", 
                        pages, confidence, entities, tables, "formFields",
                        "patientName", "patientEmail", "claimNumber", "reportTitle", 
                        "reportDate", status, summary, "originalReport",
                        "processingTimeMs", "analysisSuccess", "errorMessage",
                        "createdAt", "updatedAt"
                    ) VALUES (
                        gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, NOW(), NOW()
                    ) RETURNING id
                """, 
                    file_name,
                    file_size,
                    mime_type,
                    extraction_result.text or "",
                    extraction_result.pages,
                    extraction_result.confidence,
                    [json.dumps(entity) for entity in extraction_result.entities] if extraction_result.entities else [],
                    [json.dumps(table) for table in extraction_result.tables] if extraction_result.tables else [], 
                    [json.dumps(field) for field in extraction_result.formFields] if extraction_result.formFields else [],
                    analysis.report_json.patient_name,
                    analysis.report_json.patient_email,
                    analysis.report_json.claim_no,
                    analysis.report_json.report_title,
                    report_date,
                    analysis.report_json.status,
                    analysis.summary or [],
                    analysis.original_report,
                    processing_time_ms,
                    extraction_result.success,
                    extraction_result.error
                )
                
                # Save alerts
                alert_ids = []
                if analysis.work_status_alert:
                    for alert_data in analysis.work_status_alert:
                        alert_id = await conn.fetchval("""
                            INSERT INTO alerts (id, "alertType", "title", "date", "status", "documentId", "createdAt", "updatedAt")
                            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, NOW(), NOW()) RETURNING id
                        """, 
                            alert_data.alert_type,
                            alert_data.title,
                            alert_data.date,
                            alert_data.status,
                            document_id
                        )
                        alert_ids.append(alert_id)
                
                logger.info(f"✅ Document saved to database with ID: {document_id}")
                logger.info(f"🚨 Created {len(alert_ids)} alerts")
                
                return document_id
                
            except Exception as e:
                logger.error(f"❌ Error saving document analysis: {str(e)}")
                raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID with related alerts
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data with alerts
        """
        async with self.pool.acquire() as conn:
            try:
                # Get document
                doc_row = await conn.fetchrow("""
                    SELECT * FROM documents WHERE id = $1
                """, document_id)
                
                if not doc_row:
                    return None
                
                # Get alerts
                alert_rows = await conn.fetch("""
                    SELECT * FROM alerts WHERE "documentId" = $1 ORDER BY "createdAt" DESC
                """, document_id)
                
                # Convert to dict
                document = dict(doc_row)
                document['alerts'] = [dict(alert) for alert in alert_rows]
                
                logger.info(f"📄 Retrieved document: {document['originalName']}")
                return document
                
            except Exception as e:
                logger.error(f"❌ Error retrieving document {document_id}: {str(e)}")
                raise
    
    async def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent documents with alerts
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of recent documents
        """
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT d.*, 
                           COUNT(a.id) as alert_count,
                           COUNT(CASE WHEN a.status = 'urgent' AND a."isResolved" = false THEN 1 END) as urgent_alert_count
                    FROM documents d
                    LEFT JOIN alerts a ON d.id = a."documentId"
                    GROUP BY d.id
                    ORDER BY d."createdAt" DESC
                    LIMIT $1
                """, limit)
                
                docs_list = [dict(row) for row in rows]
                logger.info(f"📋 Retrieved {len(docs_list)} recent documents")
                return docs_list
                
            except Exception as e:
                logger.error(f"❌ Error retrieving recent documents: {str(e)}")
                raise
    
    async def get_urgent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get urgent unresolved alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of urgent alerts with document info
        """
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT a.*, d."originalName", d."patientName", d."reportTitle"
                    FROM alerts a
                    JOIN documents d ON a."documentId" = d.id
                    WHERE a.status = 'urgent' AND a."isResolved" = false
                    ORDER BY a."createdAt" DESC
                    LIMIT $1
                """, limit)
                
                alerts_list = [dict(row) for row in rows]
                logger.info(f"🚨 Retrieved {len(alerts_list)} urgent alerts")
                return alerts_list
                
            except Exception as e:
                logger.error(f"❌ Error retrieving urgent alerts: {str(e)}")
                raise
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """
        Mark alert as resolved
        
        Args:
            alert_id: Alert ID
            resolved_by: User ID who resolved the alert
            
        Returns:
            Success status
        """
        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute("""
                    UPDATE alerts 
                    SET "isResolved" = true, "resolvedAt" = $1, "resolvedBy" = $2
                    WHERE id = $3
                """, datetime.now(), resolved_by, alert_id)
                
                success = result.split()[-1] == '1'  # Check if one row was updated
                
                if success:
                    logger.info(f"✅ Alert {alert_id} resolved by {resolved_by}")
                return success
                
            except Exception as e:
                logger.error(f"❌ Error resolving alert {alert_id}: {str(e)}")
                return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for dashboard
        
        Returns:
            Statistics dictionary
        """
        async with self.pool.acquire() as conn:
            try:
                # Get counts
                total_documents = await conn.fetchval("SELECT COUNT(*) FROM documents")
                urgent_alerts = await conn.fetchval("""
                    SELECT COUNT(*) FROM alerts WHERE status = 'urgent' AND "isResolved" = false
                """)
                resolved_alerts = await conn.fetchval("SELECT COUNT(*) FROM alerts WHERE \"isResolved\" = true")
                
                # Get recent activity (last 7 days)
                seven_days_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                seven_days_ago = seven_days_ago.replace(day=seven_days_ago.day - 7)
                
                recent_documents = await conn.fetchval("""
                    SELECT COUNT(*) FROM documents WHERE "createdAt" >= $1
                """, seven_days_ago)
                
                stats = {
                    'total_documents': total_documents,
                    'urgent_alerts': urgent_alerts,
                    'resolved_alerts': resolved_alerts,
                    'recent_documents': recent_documents,
                    'last_updated': datetime.now().isoformat()
                }
                
                logger.info(f"📊 Database statistics: {stats}")
                return stats
                
            except Exception as e:
                logger.error(f"❌ Error getting statistics: {str(e)}")
                return {
                    'total_documents': 0,
                    'urgent_alerts': 0,
                    'resolved_alerts': 0,
                    'recent_documents': 0,
                    'error': str(e)
                }
    
    async def get_all_alerts(self, limit: int = 50, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """
        Get all alerts with optional filtering
        
        Args:
            limit: Maximum number of alerts to return
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List of alerts
        """
        async with self.pool.acquire() as conn:
            try:
                where_clause = "" if include_resolved else 'WHERE a."isResolved" = false'
                
                rows = await conn.fetch(f"""
                    SELECT a.*, d."patientName", d."reportTitle", d."originalName"
                    FROM alerts a
                    LEFT JOIN documents d ON a."documentId" = d.id
                    {where_clause}
                    ORDER BY a."createdAt" DESC
                    LIMIT $1
                """, limit)
                
                alerts_list = [dict(row) for row in rows]
                logger.info(f"🚨 Retrieved {len(alerts_list)} alerts")
                return alerts_list
                
            except Exception as e:
                logger.error(f"❌ Error retrieving alerts: {str(e)}")
                raise
    
    async def get_document_alerts(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all alerts for a specific document
        
        Args:
            document_id: Document UUID
            
        Returns:
            List of alerts for the document
        """
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT * FROM alerts 
                    WHERE "documentId" = $1
                    ORDER BY "createdAt" DESC
                """, document_id)
                
                alerts_list = [dict(row) for row in rows]
                logger.info(f"📋 Retrieved {len(alerts_list)} alerts for document {document_id}")
                return alerts_list
                
            except Exception as e:
                logger.error(f"❌ Error retrieving document alerts: {str(e)}")
                raise
    
    async def search_documents(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search documents by patient name, claim number, or report title
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        async with self.pool.acquire() as conn:
            try:
                search_pattern = f"%{query}%"
                rows = await conn.fetch("""
                    SELECT d.*, 
                           COUNT(a.id) as alert_count,
                           COUNT(CASE WHEN a.status = 'urgent' AND a."isResolved" = false THEN 1 END) as urgent_alert_count
                    FROM documents d
                    LEFT JOIN alerts a ON d.id = a."documentId"
                    WHERE LOWER(d."patientName") LIKE LOWER($1)
                       OR LOWER(d."claimNumber") LIKE LOWER($1)
                       OR LOWER(d."reportTitle") LIKE LOWER($1)
                       OR LOWER(d."originalName") LIKE LOWER($1)
                    GROUP BY d.id
                    ORDER BY d."createdAt" DESC
                    LIMIT $2
                """, search_pattern, limit)
                
                docs_list = [dict(row) for row in rows]
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