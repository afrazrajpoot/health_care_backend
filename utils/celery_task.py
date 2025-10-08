# utils/celery_task.py (updated: added serialization safety and logging)
from config.celery_config import app as celery_app
from datetime import datetime
import traceback
import requests
import json
from typing import Any

from config.settings import CONFIG
from utils.logger import logger
from services.file_service import FileService

def serialize_payload(payload: dict) -> dict:
    """Convert any non-JSON datetimes to ISO strings for Celery serialization."""
    def convert(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    return convert(payload)

# Celery task for finalizing
@celery_app.task(bind=True, name='finalize_document_task', max_retries=3, retry_backoff=True)
def finalize_document_task(self, webhook_payload: dict):
    """
    Calls webhook after validation/GCS upload.
    """
    start_time = datetime.now()
    file_service = FileService()  # For potential cleanup
    
    try:
        # Ensure serializable (safety net)
        webhook_payload = serialize_payload(webhook_payload)
        logger.info(f"\n🔄 === FINALIZE TASK STARTED (Task ID: {self.request.id}) ===\n")
        filename = webhook_payload.get('filename', 'unknown')
        logger.info(f"📁 Filename: {filename}")
        logger.info(f"☁️ GCS URL: {webhook_payload.get('gcs_url', 'unknown')}")
        if webhook_payload.get('physician_id'):
            logger.info(f"👨‍⚕️ Physician ID: {webhook_payload['physician_id']}")
        
        webhook_url = CONFIG.get("api_base_url", "http://localhost:8000") + "/api/webhook/save-document"
        logger.info(f"🌐 Calling webhook: {webhook_url}")
        
        response = requests.post(webhook_url, json=webhook_payload, timeout=30)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"⏱️ Webhook time: {processing_time:.0f}ms")
        
        if response.status_code == 200:
            response_data = response.json()
            status = response_data.get("status", "unknown")
            if status == "ignored":
                logger.warning(f"⚠️ Ignored in webhook: {response_data.get('reason', '')} for {filename}")
            elif status == "skipped":
                logger.warning(f"⚠️ Skipped in webhook: {response_data.get('reason', '')} for {filename}")
            else:
                logger.info(f"✅ Success in webhook, doc ID: {response_data.get('document_id', 'unknown')}")
            logger.info("✅ === FINALIZE TASK COMPLETED ===\n")
            
            return {
                "status": "success",
                "response": response_data,
                "processing_time_ms": int(processing_time),
                "filename": filename,
                "gcs_url": webhook_payload.get("gcs_url"),
                "document_id": response_data.get("document_id", webhook_payload.get("document_id")),
                "physician_id": webhook_payload.get("physician_id")
            }
        else:
            error_msg = f"Webhook failed (status {response.status_code}): {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    except Exception as e:
        logger.error(f"❌ Finalize error (Task ID: {self.request.id}): {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"🔄 Retrying (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e, countdown=60 ** (self.request.retries + 1))  # Exponential backoff
        else:
            # Final cleanup
            blob_path = webhook_payload.get("blob_path")
            if blob_path:
                try:
                    file_service.delete_from_gcs(blob_path)
                    logger.info(f"🗑️ Final cleanup GCS: {blob_path}")
                except Exception as del_exc:
                    logger.warning(f"⚠️ Cleanup failed: {str(del_exc)} - {blob_path}")
        raise