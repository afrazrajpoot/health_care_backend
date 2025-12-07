"""
FIXED Celery Tasks - Progress Only 100% When ALL Tasks Complete
"""

from config.celery_config import app as celery_app
from celery import group, chord
from datetime import datetime
import traceback
import asyncio
import aiohttp
import uuid
from typing import Any, Dict, List
from services.progress_service import progress_service
from services.file_service import FileService
from utils.logger import logger
from config.settings import CONFIG
from helpers.helpers import serialize_payload


# ============================================================================
# FIXED FINALIZE DOCUMENT TASK (NO PROGRESS UPDATES TO 100%)
# ============================================================================

@celery_app.task(bind=True, name='finalize_document_task', max_retries=0)
def finalize_document_task(self, webhook_payload: dict, batch_task_id: str = None, file_index: int = None, queue_id: str = None):
    """
    FIXED: Individual file processing - NO premature 100% progress updates.
    Only updates progress for the current file, never marks batch as complete.
    """
    return asyncio.run(_async_finalize_document_worker(
        self, webhook_payload, batch_task_id, file_index, queue_id
    ))


async def _async_finalize_document_worker(task_self, webhook_payload: dict, batch_task_id: str, file_index: int, queue_id: str):
    """
    FIXED: Only updates individual file progress, never touches batch completion.
    """
    start_time = datetime.now()
    filename = webhook_payload.get('filename', 'unknown')
    
    # Get task ID safely
    task_id = getattr(task_self, 'request', None)
    if task_id and hasattr(task_id, 'id'):
        current_task_id = task_id.id
    else:
        current_task_id = str(uuid.uuid4())
        logger.warning(f"⚠️ Using fallback task ID: {current_task_id}")
    
    logger.info(f"🎯 Starting async task {current_task_id} for file {file_index}: {filename}")
    
    # ===== DEDUPLICATION CHECK =====
    document_hash = progress_service.create_document_hash(filename)
    if progress_service.is_processing(document_hash):
        processing_task = progress_service.get_processing_task(document_hash)
        logger.warning(f"⚠️ Document already processing: {filename} by task {processing_task}")
        
        # Update progress - but don't mark as completed=True
        if batch_task_id and file_index is not None:
            await _async_update_file_progress(
                batch_task_id, file_index, filename,
                status='skipped', file_progress=100
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
        
        # ✅ FIX: Start at 0% progress for this file only
        if batch_task_id and file_index is not None:
            await _async_update_file_progress(
                batch_task_id, file_index, filename,
                status='processing', file_progress=0,
                message="Starting document processing..."
            )
        
        # OPTIMIZATION: Async webhook call (non-blocking HTTP)
        webhook_url = CONFIG.get("api_base_url", "https://api.kebilo.com") + "/webhook/save-document"
        
        async with aiohttp.ClientSession() as session:
            # ✅ FIX: Update file progress to 25%
            if batch_task_id and file_index is not None:
                await _async_update_file_progress(
                    batch_task_id, file_index, filename,
                    status='processing', file_progress=25,
                    message="Analyzing document structure..."
                )
            
            # ✅ FIX: Update file progress to 50%
            if batch_task_id and file_index is not None:
                await _async_update_file_progress(
                    batch_task_id, file_index, filename,
                    status='processing', file_progress=50,
                    message="Extracting document content..."
                )
            
            # ✅ FIX: Update file progress to 75%
            if batch_task_id and file_index is not None:
                await _async_update_file_progress(
                    batch_task_id, file_index, filename,
                    status='processing', file_progress=75,
                    message="Processing with AI engine..."
                )
            
            # Non-blocking webhook call
            async with session.post(
                webhook_url,
                json=webhook_payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ValueError(f"Webhook status {response.status}: {text}")
                
                response_data = await response.json()
        
        # ✅ FIX: Mark file as 100% complete BUT don't mark batch as completed
        if batch_task_id and file_index is not None:
            await _async_update_file_progress(
                batch_task_id, file_index, filename,
                status='success', file_progress=100,
                message="File processing completed"
            )
        
        # ===== QUEUE COMPLETION =====
        if queue_id:
            progress_service.complete_task_in_queue(queue_id, current_task_id, success=True)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"✅ File {file_index + 1} completed successfully: {filename} ({processing_time:.0f}ms)")
        
        # Clear processing mark
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
        logger.error(f"❌ Failed {filename}: {error_str}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # FAILURE: Mark file as failed but don't mark batch as completed
        if batch_task_id and file_index is not None:
            await _async_update_file_progress(
                batch_task_id, file_index, filename,
                status='failed', file_progress=100,
                message=f"Failed: {error_str}"
            )
        
        # ===== QUEUE FAILURE =====
        if queue_id:
            progress_service.complete_task_in_queue(queue_id, current_task_id, success=False)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Clear processing mark
        progress_service.mark_completed(document_hash)
        
        return {
            "status": "failed",
            "error": error_str,
            "processing_time_ms": int(processing_time),
            "filename": filename,
            "file_index": file_index,
            "queue_id": queue_id
        }


async def _async_update_file_progress(
    batch_task_id: str,
    file_index: int,
    filename: str,
    status: str,
    file_progress: int,
    message: str = None
):
    """
    FIXED: Updates progress for individual file ONLY.
    Never marks the batch as completed.
    """
    loop = asyncio.get_event_loop()
    
    # Build progress message
    if message:
        current_file = f"{filename} - {message}"
    elif status == 'success':
        current_file = f"{filename} - Completed"
    elif status == 'failed':
        current_file = f"{filename} - Failed"
    elif status == 'skipped':
        current_file = f"{filename} - Skipped"
    else:
        current_file = f"{filename} - Processing..."
    
    # ✅ FIX: Use the new update_file_progress_only method
    await loop.run_in_executor(
        None,
        lambda: progress_service.update_file_progress_only(
            task_id=batch_task_id,
            file_index=file_index,
            filename=filename,
            status=status,
            file_progress=file_progress,
            message=message
        )
    )


# ============================================================================
# FIXED BATCH PROCESSING TASK (ONLY CALLBACK MARKS 100%)
# ============================================================================

@celery_app.task(bind=True, name='process_batch_documents', max_retries=0)
def process_batch_documents(self, payloads: list[dict]):
    """
    FIXED: Master batch processing - ONLY callback can mark as 100% complete.
    """
    try:
        # Get batch task ID safely
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
            logger.info(f"📊 Queue {queue_id} for user {user_id}")
        
        logger.info(f"🚀 Batch start: {len(payloads)} files (task {batch_task_id})")
        
        # Initialize progress tracking
        progress_service.initialize_task_progress(
            task_id=batch_task_id,
            total_steps=len(payloads),
            filenames=filenames,
            user_id=user_id,
            queue_id=queue_id
        )
        
        # ✅ FIX: Initial progress at 0% - use original method with completed=False
        # ✅ FIX: Initial progress at 0%
        progress_service.update_status(
            task_id=batch_task_id,
            status='processing',
            message="Initializing batch processing..."
        )
        
        # ===== PARALLEL PROCESSING =====
        serialized_payloads = [serialize_payload(p) for p in payloads]
        
        # Use chord with callback
        task_group = group(
            finalize_document_task.s(payload, batch_task_id, i, queue_id)
            for i, payload in enumerate(serialized_payloads)
        )
        
        callback = _batch_complete_callback.s(batch_task_id, filenames, user_id, queue_id)
        chord_result = chord(task_group)(callback)
        
        logger.info(f"✅ Dispatched {len(payloads)} tasks in parallel (non-blocking)")
        
        return {
            "batch_task_id": batch_task_id,
            "queue_id": queue_id,
            "total_files": len(payloads),
            "user_id": user_id,
            "status": "dispatched",
            "message": f"{len(payloads)} tasks dispatched for parallel processing",
            "chord_id": chord_result.id if hasattr(chord_result, 'id') else None
        }
        
    except Exception as e:
        logger.error(f"❌ Batch task dispatch failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Update progress to failed
        if 'batch_task_id' in locals():
            progress_service.update_status(
                task_id=batch_task_id,
                status='failed',
                message=f"Batch failed: {str(e)}",
                completed=True
            )
        
        raise


@celery_app.task(name='_batch_complete_callback')
def _batch_complete_callback(results: List[Dict], batch_task_id: str, filenames: List[str], user_id: str, queue_id: str):
    """
    FIXED: ONLY this callback can mark batch as 100% complete.
    This runs after ALL individual tasks finish.
    """
    try:
        logger.info(f"📊 Batch callback for task {batch_task_id}: {len(results)} results")
        
        # Calculate statistics
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = sum(1 for r in results if r.get('status') == 'failed')
        skipped_count = sum(1 for r in results if r.get('status') == 'skipped')
        
        # Log individual results
        for i, result in enumerate(results):
            filename = filenames[i] if i < len(filenames) else 'unknown'
            if result.get("status") == "success":
                logger.info(f"✅ Batch file {i+1} success: {filename}")
            elif result.get("status") == "skipped":
                logger.warning(f"⏭️ Batch file {i+1} skipped: {filename}")
            else:
                logger.error(f"❌ Batch file {i+1} failed: {filename}")
        
        # ✅ FIX: ONLY HERE we mark as 100% complete using update_batch_progress
        final_status = 'completed' if failed_count == 0 else 'completed_with_errors'
        progress_service.update_batch_progress(
            task_id=batch_task_id,
            results=results,
            status=final_status,
            completed=True
        )
        
        logger.info(f"🏁 Batch complete: {success_count}/{len(results)} successful, {failed_count} failed, {skipped_count} skipped")
        
        return {
            "batch_task_id": batch_task_id,
            "queue_id": queue_id,
            "total_files": len(results),
            "successful_files": success_count,
            "failed_files": failed_count,
            "skipped_files": skipped_count,
            "processing_time_ms": sum(r.get('processing_time_ms', 0) for r in results),
            "results": results,
            "user_id": user_id,
            "status": final_status
        }
        
    except Exception as e:
        logger.error(f"❌ Batch callback failed: {str(e)}")
        
        # Even on callback failure, mark as failed with 100%
        # Even on callback failure, mark as failed with 100%
        progress_service.update_status(
            task_id=batch_task_id,
            status='failed',
            message=f"Batch callback failed: {str(e)}",
            completed=True
        )
        
        return {
            "batch_task_id": batch_task_id,
            "status": "callback_failed",
            "error": str(e)
        }

# ============================================================================
# OTHER TASKS (UNCHANGED)
# ============================================================================

@celery_app.task(name='cleanup_old_progress')
def cleanup_old_progress():
    """Periodic task to clean up old progress data from Redis."""
    try:
        cleaned_count = progress_service.cleanup_completed_tasks(older_than_hours=1)
        logger.info(f"🧹 Periodic cleanup: {cleaned_count} old progress entries removed")
        return cleaned_count
    except Exception as e:
        logger.error(f"❌ Cleanup task failed: {str(e)}")
        return 0


@celery_app.task(name='health_check')
def health_check():
    """Health check task to verify Celery worker is running."""
    try:
        redis_status = progress_service.check_health()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "redis": redis_status,
            "worker": "celery@afraz-Latitude-7490"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# TASK REGISTRY
# ============================================================================

TASKS = {
    'finalize_document_task': finalize_document_task,
    'process_batch_documents': process_batch_documents,
    '_batch_complete_callback': _batch_complete_callback,
    'cleanup_old_progress': cleanup_old_progress,
    'health_check': health_check,
}

def register_tasks():
    """Register all tasks with Celery."""
    for name, task in TASKS.items():
        celery_app.tasks.register(task)
    logger.info(f"✅ Registered {len(TASKS)} Celery tasks")

register_tasks()