from config.celery_config import app as celery_app
from celery import group  # Added for parallel task execution
from datetime import datetime
import traceback
import requests
import json
import time
import uuid
from typing import Any
from services.progress_service import progress_service
from services.file_service import FileService
from utils.logger import logger
from config.settings import CONFIG

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

@celery_app.task(bind=True, name='finalize_document_task', max_retries=0)  # Disabled retries to avoid issues with long-running reasoning
def finalize_document_task(self, webhook_payload: dict, batch_task_id: str = None, file_index: int = None, queue_id: str = None):
    """
    Individual file processing task with deduplication and queue tracking
    """
    start_time = datetime.now()
    filename = webhook_payload.get('filename', 'unknown')
    file_num = file_index + 1 if file_index is not None else 'N/A'
    
    # FIX: Generate task ID safely
    task_id = getattr(self, 'request', None)
    if task_id and hasattr(task_id, 'id'):
        current_task_id = task_id.id
    else:
        # Fallback: generate a unique ID
        current_task_id = str(uuid.uuid4())
        logger.warning(f"⚠️ Using fallback task ID: {current_task_id}")
    
    # Added for visibility in parallel runs
    logger.info(f"🎯 Sub-task {current_task_id} started for file {file_index}: {filename} in batch {batch_task_id}")
    
    process_msg = f"🔄 Processing file {file_num}: {filename}"
    logger.info(process_msg)
    print(process_msg)
    
    # ===== DEDUPLICATION CHECK =====
    document_hash = progress_service.create_document_hash(filename)
    if progress_service.is_processing(document_hash):
        processing_task = progress_service.get_processing_task(document_hash)
        skip_msg = f"⚠️ Document already processing: {filename} by task {processing_task}"
        logger.warning(skip_msg)
        print(skip_msg)
        
        # Still update progress for this task to avoid hanging
        if batch_task_id and file_index is not None:
            progress_service.update_progress(
                task_id=batch_task_id,
                current_step=file_index + 1,
                current_file=f"{filename} - Skipped (already processing)",
                status='skipped',
                file_progress=100,
                completed=True
            )
        
        return {
            "status": "skipped",
            "reason": "already_processing",
            "filename": filename,
            "processing_task": processing_task
        }
    
    # FIX: Mark as processing with safe task ID
    progress_service.mark_processing(document_hash, current_task_id)
    
    try:
        webhook_payload = serialize_payload(webhook_payload)
        
        # ===== QUEUE TRACKING =====
        if queue_id:
            progress_service.move_task_to_active(queue_id, current_task_id)
        
        # IMMEDIATE progress update when file starts (0% for this file, but overall progresses)
        if batch_task_id and file_index is not None:
            progress_service.update_progress(
                task_id=batch_task_id,
                current_step=file_index + 1,
                current_file=f"{filename} - Starting...",
                status='processing',
                file_progress=0  # This file just started
            )
        
        # Quick sequential updates for this file (simulating progress while reasoning happens)
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
        
        # Webhook call - the actual work (increased timeout for long-running reasoning)
        webhook_url = CONFIG.get("api_base_url", "http://localhost:8000") + "/api/webhook/save-document"
        response = requests.post(webhook_url, json=webhook_payload, timeout=300)  # Increased to 5 minutes for reasoning time
        
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
        
        # ===== QUEUE COMPLETION =====
        if queue_id:
            progress_service.complete_task_in_queue(queue_id, current_task_id, success=True)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        success_msg = f"✅ File {file_num} succeeded: {filename}"
        logger.info(success_msg)
        print(success_msg)
        
        # Clear processing mark on success
        progress_service.mark_completed(document_hash)
        
        return {
            "status": "success",
            "response": response_data,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "file_index": file_index,
            "queue_id": queue_id
        }
    
    except Exception as e:
        error_str = str(e)
        final_error_msg = f"❌ Failed {filename}: {error_str}"
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
        
        # ===== QUEUE FAILURE =====
        if queue_id:
            progress_service.complete_task_in_queue(queue_id, current_task_id, success=False)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Clear processing mark on failure
        progress_service.mark_completed(document_hash)
        
        return {
            "status": "failed",
            "error": error_str,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "file_index": file_index,
            "queue_id": queue_id
        }

    # Clear processing mark on unexpected failure
    progress_service.mark_completed(document_hash)
    raise Exception("Unexpected error in finalize_document_task")

@celery_app.task(bind=True, name='process_batch_documents', max_retries=0)  # Disabled retries
def process_batch_documents(self, payloads: list[dict]):
    """
    Master batch processing task with queue tracking and deduplication
    Handles both batches and single documents (as a batch of 1).
    Now dispatches file tasks in parallel using Celery group.
    """
    try:
        file_service = FileService()  # Kept for compatibility; unused in current logic
        results = []
        
        # FIX: Get batch task ID safely
        batch_task_id = getattr(self.request, 'id', None)
        if not batch_task_id:
            batch_task_id = str(uuid.uuid4())
            logger.warning(f"⚠️ Using fallback batch task ID: {batch_task_id}")
        
        filenames = [p.get('filename', 'unknown') for p in payloads]
        user_id = payloads[0].get('user_id') if payloads else None
        
        # ===== QUEUE INITIALIZATION =====
        queue_id = None
        if user_id:
            queue_id = progress_service.get_user_queue(user_id)
            if not queue_id:
                queue_id = progress_service.initialize_queue_progress(user_id)
            logger.info(f"📊 Using queue {queue_id} for user {user_id}")
        
        start_msg = f"🚀 Batch start: {len(payloads)} files for task {batch_task_id}, user {user_id}, queue {queue_id}"
        logger.info(start_msg)
        print(start_msg)
        
        # Initialize progress tracking for this batch with queue reference
        progress_service.initialize_task_progress(
            task_id=batch_task_id,
            total_steps=len(payloads),
            filenames=filenames,
            user_id=user_id,
            queue_id=queue_id
        )
        
        # Initial progress update
        progress_service.update_progress(
            task_id=batch_task_id,
            current_step=0,
            current_file="Initializing batch...",
            status='processing',
            file_progress=0
        )
        
        # ===== PARALLEL PROCESSING: Create group of async tasks =====
        # Serialize payloads upfront for consistency
        serialized_payloads = [serialize_payload(p) for p in payloads]
        
        # Build the group: Each sub-task gets its own args (payload, batch_task_id, index, queue_id)
        # Use .s() for signature (immutable task args)
        task_group = group(
            finalize_document_task.s(payload, batch_task_id, i, queue_id)
            for i, payload in enumerate(serialized_payloads)
        )
        
        # Dispatch the group asynchronously and wait for all results
        # This blocks the batch task until all sub-tasks complete (but sub-tasks run parallel)
        group_results = task_group.apply_async().get(timeout=3600)  # 1-hour timeout for entire batch; adjust as needed
        
        # Collect results from the group (list of AsyncResult objects)
        for i, async_result in enumerate(group_results):
            try:
                # Get the actual result (dict from finalize_document_task)
                result = async_result.get(timeout=60)  # Per-result timeout
                results.append(result)
                
                filename = payloads[i].get('filename', 'unknown')
                if result["status"] == "success":
                    success_msg = f"✅ Batch file {i+1} success: {filename}"
                    logger.info(success_msg)
                    print(success_msg)
                elif result["status"] == "skipped":
                    skip_msg = f"⏭️ Batch file {i+1} skipped: {filename} - {result.get('reason', 'Unknown')}"
                    logger.warning(skip_msg)
                    print(skip_msg)
                else:
                    fail_msg = f"❌ Batch file {i+1} failed: {filename} - {result.get('error', 'Unknown')}"
                    logger.error(fail_msg)
                    print(fail_msg)
            except Exception as sub_err:
                # Handle any sub-task timeout/failure
                error_msg = f"❌ Sub-task {i+1} retrieval failed: {str(sub_err)}"
                logger.error(error_msg)
                print(error_msg)
                results.append({
                    "status": "failed",
                    "error": str(sub_err),
                    "filename": payloads[i].get('filename', 'unknown'),
                    "file_index": i,
                    "queue_id": queue_id
                })
        
        # Final batch complete update (progress_service should auto-calculate based on sub-updates)
        final_progress = progress_service.update_progress(
            task_id=batch_task_id,
            current_step=len(payloads),
            current_file="Batch completed",
            status='completed'
        )
        
        # Calculate success statistics
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = sum(1 for r in results if r.get('status') == 'failed')
        skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
        
        complete_msg = f"🏁 Batch complete for {batch_task_id}: {success_count}/{len(payloads)} successful, {failed_count} failed, {skipped_count} skipped"
        logger.info(complete_msg)
        print(complete_msg)
        
        # Return comprehensive batch results
        return {
            "batch_task_id": batch_task_id,
            "queue_id": queue_id,
            "total_files": len(payloads),
            "successful_files": success_count,
            "failed_files": failed_count,
            "skipped_files": skipped_count,
            "processing_time_ms": sum(r.get('processing_time_ms', 0) for r in results),
            "results": results,
            "user_id": user_id
        }
        
    except Exception as e:
        error_msg = f"❌ Batch task failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        
        # Update progress to failed status
        if 'batch_task_id' in locals():
            progress_service.update_progress(
                task_id=batch_task_id,
                status='failed',
                current_file=f"Batch failed: {str(e)}"
            )
        raise

def finalize_document_task_worker(webhook_payload: dict, batch_task_id: str = None, file_index: int = None, queue_id: str = None, file_task_id: str = None):
    """
    Worker function for finalize_document_task that can be called directly
    This avoids the Celery task context issue (kept for potential synchronous use elsewhere)
    """
    start_time = datetime.now()
    filename = webhook_payload.get('filename', 'unknown')
    file_num = file_index + 1 if file_index is not None else 'N/A'
    
    # Use provided file_task_id or generate one
    current_task_id = file_task_id or str(uuid.uuid4())
    
    process_msg = f"🔄 Processing file {file_num}: {filename}"
    logger.info(process_msg)
    
    # ===== DEDUPLICATION CHECK =====
    document_hash = progress_service.create_document_hash(filename)
    if progress_service.is_processing(document_hash):
        processing_task = progress_service.get_processing_task(document_hash)
        skip_msg = f"⚠️ Document already processing: {filename} by task {processing_task}"
        logger.warning(skip_msg)
        
        # Still update progress for this task to avoid hanging
        if batch_task_id and file_index is not None:
            progress_service.update_progress(
                task_id=batch_task_id,
                current_step=file_index + 1,
                current_file=f"{filename} - Skipped (already processing)",
                status='skipped',
                file_progress=100,
                completed=True
            )
        
        return {
            "status": "skipped",
            "reason": "already_processing",
            "filename": filename,
            "processing_task": processing_task
        }
    
    # Mark as processing
    progress_service.mark_processing(document_hash, current_task_id)
    
    try:
        webhook_payload = serialize_payload(webhook_payload)
        
        # ===== QUEUE TRACKING =====
        if queue_id:
            progress_service.move_task_to_active(queue_id, current_task_id)
        
        # IMMEDIATE progress update when file starts
        if batch_task_id and file_index is not None:
            progress_service.update_progress(
                task_id=batch_task_id,
                current_step=file_index + 1,
                current_file=f"{filename} - Starting...",
                status='processing',
                file_progress=0
            )
        
        # Quick sequential updates for this file
        progress_steps = [10, 25, 40, 60, 80, 90]
        for progress_step in progress_steps:
            time.sleep(0.5)
            
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
        
        # Webhook call - the actual work (increased timeout for long-running reasoning)
        webhook_url = CONFIG.get("api_base_url", "http://localhost:8000") + "/webhook/save-document"
        response = requests.post(webhook_url, json=webhook_payload, timeout=300)  # Increased to 5 minutes
        
        if response.status_code != 200:
            raise ValueError(f"Webhook status {response.status_code}: {response.text}")
        
        response_data = response.json()
        
        # File completed successfully
        if batch_task_id and file_index is not None:
            progress_service.update_progress(
                task_id=batch_task_id,
                current_step=file_index + 1,
                current_file=f"{filename} - Completed",
                status='success',
                file_progress=100
            )
        
        # Increment completed count
        if batch_task_id and file_index is not None:
            progress_service.update_progress(
                task_id=batch_task_id,
                completed=True
            )
        
        # ===== QUEUE COMPLETION =====
        if queue_id:
            progress_service.complete_task_in_queue(queue_id, current_task_id, success=True)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        success_msg = f"✅ File {file_num} succeeded: {filename}"
        logger.info(success_msg)
        
        # Clear processing mark on success
        progress_service.mark_completed(document_hash)
        
        return {
            "status": "success",
            "response": response_data,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "file_index": file_index,
            "queue_id": queue_id
        }
    
    except Exception as e:
        error_str = str(e)
        final_error_msg = f"❌ Failed {filename}: {error_str}"
        logger.error(final_error_msg)
        
        # Mark file as failed
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
        
        # ===== QUEUE FAILURE =====
        if queue_id:
            progress_service.complete_task_in_queue(queue_id, current_task_id, success=False)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Clear processing mark on failure
        progress_service.mark_completed(document_hash)
        
        return {
            "status": "failed",
            "error": error_str,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "file_index": file_index,
            "queue_id": queue_id
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