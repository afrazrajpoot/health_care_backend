# utils/celery_task.py
from config.celery_config import app as celery_app
from datetime import datetime
import traceback
import requests
import json
from typing import Any
from services.progress_service import progress_service
from services.file_service import FileService
from utils.logger import logger
from config.settings import CONFIG
import time  # For artificial delays to simulate real-time progress

def serialize_payload(payload: dict) -> dict:
    """Convert any non-JSON datatypes to ISO strings for Celery serialization."""
    def convert(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    return convert(payload)

@celery_app.task(bind=True, name='finalize_document_task', max_retries=3, retry_backoff=True)
def finalize_document_task(self, webhook_payload: dict, batch_task_id: str = None, file_index: int = None):
    """
    Individual file processing task with detailed progress tracking
    """
    start_time = datetime.now()
    filename = webhook_payload.get('filename', 'unknown')
    file_num = file_index + 1 if file_index is not None else 'N/A'
    max_retries_local = 3
    
    process_msg = f"🔄 Processing file {file_num}: {filename}"
    logger.info(process_msg)
    print(process_msg)
    
    success = False
    for attempt in range(max_retries_local + 1):
        try:
            webhook_payload = serialize_payload(webhook_payload)
            
            # IMMEDIATE progress update when file starts (0% for this file, but overall progresses)
            if batch_task_id and file_index is not None:
                progress_service.update_progress(
                    task_id=batch_task_id,
                    current_step=file_index + 1,
                    current_file=f"{filename} - Starting...",
                    status='processing',
                    file_progress=0  # This file just started
                )
            
            # Quick sequential updates for this file
            progress_steps = [10, 25, 40, 60, 80, 90]
            for progress_step in progress_steps:
                time.sleep(0.5)  # Reduced from 1 second to 0.5 seconds
                
                if batch_task_id and file_index is not None:
                    status_messages = {
                        10: "Uploading to cloud...",
                        25: "Extracting text...", 
                        40: "Analyzing content...",
                        60: "Processing data...",
                        80: "Generating summary...",
                        90: "Finalizing..."
                    }
                    
                    progress_service.update_progress(
                        task_id=batch_task_id,
                        current_step=file_index + 1,
                        current_file=f"{filename} - {status_messages.get(progress_step, 'Processing...')}",
                        status='processing',
                        file_progress=progress_step
                    )
            
            # Webhook call - the actual work
            webhook_url = CONFIG.get("api_base_url", "https://api.kebilo.com") + "/api/webhook/save-document"
            response = requests.post(webhook_url, json=webhook_payload, timeout=30)
            
            if response.status_code != 200:
                raise ValueError(f"Webhook status {response.status_code}: {response.text}")
            
            response_data = response.json()
            
            # File completed successfully - mark as 100% for this file
            if batch_task_id and file_index is not None:
                progress_service.update_progress(
                    task_id=batch_task_id,
                    current_step=file_index + 1,
                    current_file=f"{filename} - Completed",
                    status='success',
                    file_progress=100
                )
            
            # Increment completed count (triggers overall progress calculation)
            if batch_task_id and file_index is not None:
                progress_service.update_progress(
                    task_id=batch_task_id,
                    completed=True
                )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            success_msg = f"✅ File {file_num} succeeded: {filename}"
            logger.info(success_msg)
            print(success_msg)
            success = True
            
            return {
                "status": "success",
                "response": response_data,
                "processing_time_ms": int(processing_time),
                "filename": filename,
                "file_index": file_index
            }
        
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries_local:
                retry_msg = f"🔄 Retry {attempt + 1}/{max_retries_local} for {filename}: {error_str}"
                logger.info(retry_msg)
                print(retry_msg)
                time.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4 seconds
                continue
            else:
                final_error_msg = f"❌ Failed {filename} after {max_retries_local} attempts: {error_str}"
                logger.error(final_error_msg)
                print(final_error_msg)
                
                # Mark file as failed (100% for this file but failed status)
                if batch_task_id and file_index is not None:
                    progress_service.update_progress(
                        task_id=batch_task_id,
                        current_step=file_index + 1,
                        current_file=f"{filename} - Failed",
                        status='failed',
                        file_progress=100,
                        failed_file=filename
                    )
                
                # Still increment completed count for failed file
                if batch_task_id and file_index is not None:
                    progress_service.update_progress(
                        task_id=batch_task_id,
                        completed=True
                    )
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return {
                    "status": "failed",
                    "error": error_str,
                    "processing_time_ms": int(processing_time),
                    "filename": filename,
                    "file_index": file_index
                }
    
    if not success:
        raise Exception("Unexpected error in finalize_document_task")

@celery_app.task(bind=True, name='process_batch_documents', max_retries=3)
def process_batch_documents(self, payloads: list[dict]):
    """
    Master batch processing task: Processes each payload sequentially.
    Handles progress initialization, individual file processing, and final completion.
    """
    try:
        file_service = FileService()
        results = []
        batch_task_id = self.request.id
        
        filenames = [p.get('filename', 'unknown') for p in payloads]
        user_id = payloads[0].get('user_id') if payloads else None
        
        start_msg = f"🚀 Batch start: {len(payloads)} files for task {batch_task_id}, user {user_id}"
        logger.info(start_msg)
        print(start_msg)  # DEBUG PRINT
        
        # Initialize progress tracking for this batch
        progress_service.initialize_task_progress(
            task_id=batch_task_id,
            total_steps=len(payloads),
            filenames=filenames,
            user_id=user_id
        )
        
        # Initial progress update
        progress_service.update_progress(
            task_id=batch_task_id,
            current_step=0,
            current_file="Initializing batch...",
            status='processing',
            file_progress=0
        )
        
        # Process each file sequentially
        for i, payload in enumerate(payloads):
            filename = payload.get('filename', 'unknown')
            seq_msg = f"🔄 Batch file {i+1}/{len(payloads)}: {filename}"
            logger.info(seq_msg)
            print(seq_msg)  # DEBUG PRINT
            
            # Start file processing (0% file progress, but service will calculate overall)
            progress_service.update_progress(
                task_id=batch_task_id,
                current_step=i + 1,
                current_file=filename,
                status='processing'
            )
            
            # Process individual file
            payload = serialize_payload(payload)
            result = finalize_document_task(payload, batch_task_id=batch_task_id, file_index=i)
            results.append(result)
            
            # Log result
            if result["status"] == "success":
                success_msg = f"✅ Batch file {i+1} success: {filename}"
                logger.info(success_msg)
                print(success_msg)  # DEBUG PRINT
            else:
                fail_msg = f"❌ Batch file {i+1} failed: {filename} - {result.get('error', 'Unknown')}"
                logger.error(fail_msg)
                print(fail_msg)  # DEBUG PRINT
                # No extra completed=True here; handled in finalize_document_task for both success and failure
        
        # Final batch complete update (NO file_progress=100; let calculation use completed_steps / total_steps = 100%)
        final_progress = progress_service.update_progress(
            task_id=batch_task_id,
            current_step=len(payloads),
            current_file="Batch completed",
            status='completed'
            # REMOVED file_progress to avoid over 100% - rely on completed_steps calculation
        )
        
        # Calculate success statistics
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = len(payloads) - success_count
        
        complete_msg = f"🏁 Batch complete for {batch_task_id}: {success_count}/{len(payloads)} successful, {failed_count} failed"
        logger.info(complete_msg)
        print(complete_msg)  # DEBUG PRINT
        
        # Return comprehensive batch results
        return {
            "batch_task_id": batch_task_id,
            "total_files": len(payloads),
            "successful_files": success_count,
            "failed_files": failed_count,
            "processing_time_ms": sum(r.get('processing_time_ms', 0) for r in results),
            "results": results,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"❌ Batch task failed: {str(e)}")
        # Update progress to failed status
        if 'batch_task_id' in locals():
            progress_service.update_progress(
                task_id=batch_task_id,
                status='failed',
                current_file=f"Batch failed: {str(e)}"
            )
        raise

@celery_app.task(bind=True, name='process_single_document', max_retries=3, retry_backoff=True)
def process_single_document(self, webhook_payload: dict):
    """
    Process a single document with progress tracking
    """
    start_time = datetime.now()
    filename = webhook_payload.get('filename', 'unknown')
    task_id = self.request.id
    user_id = webhook_payload.get('user_id')
    
    start_msg = f"🚀 Single document start: {filename} for task {task_id}, user {user_id}"
    logger.info(start_msg)
    print(start_msg)
    
    # Initialize progress for single file
    progress_service.initialize_task_progress(
        task_id=task_id,
        total_steps=1,
        filenames=[filename],
        user_id=user_id
    )
    
    try:
        webhook_payload = serialize_payload(webhook_payload)
        
        # Progress updates for single file
        progress_steps = [10, 25, 40, 60, 80, 90, 100]
        for progress_step in progress_steps:
            time.sleep(0.5)
            
            status_messages = {
                10: "Uploading to cloud...",
                25: "Extracting text...", 
                40: "Analyzing content...",
                60: "Processing data...",
                80: "Generating summary...",
                90: "Finalizing...",
                100: "Completed"
            }
            
            progress_service.update_progress(
                task_id=task_id,
                current_step=1,
                current_file=f"{filename} - {status_messages.get(progress_step, 'Processing...')}",
                status='processing',
                file_progress=progress_step
            )
        
        # Webhook call
        webhook_url = CONFIG.get("api_base_url", "https://api.kebilo.com") + "/api/webhook/save-document"
        response = requests.post(webhook_url, json=webhook_payload, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Webhook status {response.status_code}: {response.text}")
        
        response_data = response.json()
        
        # Mark as completed
        progress_service.update_progress(
            task_id=task_id,
            current_step=1,
            current_file=f"{filename} - Completed",
            status='completed',
            file_progress=100,
            completed=True
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        success_msg = f"✅ Single document succeeded: {filename}"
        logger.info(success_msg)
        print(success_msg)
        
        return {
            "status": "success",
            "response": response_data,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "task_id": task_id
        }
        
    except Exception as e:
        error_str = str(e)
        error_msg = f"❌ Single document failed: {filename} - {error_str}"
        logger.error(error_msg)
        print(error_msg)
        
        # Mark as failed
        progress_service.update_progress(
            task_id=task_id,
            current_step=1,
            current_file=f"{filename} - Failed",
            status='failed',
            file_progress=100,
            completed=True,
            failed_file=filename
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return {
            "status": "failed",
            "error": error_str,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "task_id": task_id
        }

@celery_app.task(name='cleanup_old_progress')
def cleanup_old_progress():
    """
    Periodic task to clean up old progress data from Redis
    """
    try:
        cleaned_count = progress_service.cleanup_completed_tasks(older_than_hours=1)
        logger.info(f"🧹 Periodic cleanup: {cleaned_count} old progress entries removed")
        print(f"🧹 Periodic cleanup: {cleaned_count} old progress entries removed")
        return cleaned_count
    except Exception as e:
        logger.error(f"❌ Cleanup task failed: {str(e)}")
        print(f"❌ Cleanup task failed: {str(e)}")
        return 0