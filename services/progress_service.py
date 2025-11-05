# services/progress_service.py
import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime
import redis
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from config.settings import CONFIG
from utils.logger import logger

# Import your existing socket manager
from utils.socket_manager import sio

class ProgressService:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=CONFIG.get('redis_host', 'localhost'),
                port=CONFIG.get('redis_port', 6379),
                db=0,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully for progress tracking")
            print("✅ Redis connected successfully for progress tracking")
            
            # Initialize queue tracking
            self._init_queue_tracking()
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {str(e)}")
            print(f"❌ Redis connection failed: {str(e)}")
            raise
    
    def _init_queue_tracking(self):
        """Initialize queue tracking structures"""
        try:
            # Global queue tracking
            if not self.redis_client.exists("global:queue_tasks"):
                self.redis_client.set("global:queue_tasks", json.dumps([]))
            logger.info("📊 Queue tracking initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize queue tracking: {str(e)}")

    # ===== DEDUPLICATION METHODS =====
    
    def create_document_hash(self, filename: str, content_hash: str = None) -> str:
        """Create a unique hash for document deduplication"""
        if content_hash:
            return hashlib.md5(f"{filename}:{content_hash}".encode()).hexdigest()
        return hashlib.md5(filename.encode()).hexdigest()

    def is_processing(self, document_hash: str) -> bool:
        """Check if a document is already being processed"""
        key = f"processing:{document_hash}"
        return self.redis_client.exists(key)
    
    def mark_processing(self, document_hash: str, task_id: str, timeout: int = 300):
        """Mark a document as being processed"""
        key = f"processing:{document_hash}"
        self.redis_client.setex(key, timeout, task_id)
        logger.debug(f"🔒 Marked as processing: {document_hash} for task {task_id}")
    
    def mark_completed(self, document_hash: str):
        """Remove processing mark"""
        key = f"processing:{document_hash}"
        deleted = self.redis_client.delete(key)
        if deleted:
            logger.debug(f"🔓 Removed processing mark: {document_hash}")
    
    def get_processing_task(self, document_hash: str) -> Optional[str]:
        """Get the task ID that's processing a document"""
        key = f"processing:{document_hash}"
        return self.redis_client.get(key)

    # ===== QUEUE-LEVEL PROGRESS TRACKING =====
    
    def initialize_queue_progress(self, user_id: str, queue_id: str = None) -> str:
        """Initialize progress tracking for a user's entire queue"""
        if not queue_id:
            queue_id = f"user_queue:{user_id}:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        queue_data = {
            'queue_id': queue_id,
            'user_id': user_id,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_tasks': {},
            'queued_tasks': [],
            'completed_task_ids': [],
            'start_time': datetime.now().isoformat(),
            'overall_progress': 0,
            'status': 'active'
        }
        
        self.redis_client.setex(
            f"queue:{queue_id}",
            7200,  # 2 hours
            json.dumps(queue_data)
        )
        
        # Also set user's current queue
        self.redis_client.setex(
            f"user_current_queue:{user_id}",
            7200,
            queue_id
        )
        
        logger.info(f"📊 Queue progress initialized: {queue_id} for user {user_id}")
        return queue_id
    
    def add_task_to_queue(self, queue_id: str, task_id: str, task_type: str, filename: str) -> bool:
        """Add a task to the queue"""
        queue_data = self._get_queue_data(queue_id)
        if not queue_data:
            return False
        
        task_info = {
            'task_id': task_id,
            'type': task_type,
            'filename': filename,
            'status': 'queued',
            'added_time': datetime.now().isoformat(),
            'progress': 0
        }
        
        # Add to queued tasks if not already present
        existing_tasks = [t['task_id'] for t in queue_data['queued_tasks']]
        if task_id not in existing_tasks:
            queue_data['queued_tasks'].append(task_info)
            queue_data['total_tasks'] += 1
        
        self._save_queue_data(queue_id, queue_data)
        self._update_queue_progress(queue_id)
        
        logger.info(f"📥 Task added to queue {queue_id}: {filename} ({task_id})")
        return True
    
    def move_task_to_active(self, queue_id: str, task_id: str) -> bool:
        """Move a task from queued to active"""
        queue_data = self._get_queue_data(queue_id)
        if not queue_data:
            return False
        
        # Find task in queued
        task_index = None
        task_info = None
        for i, task in enumerate(queue_data['queued_tasks']):
            if task['task_id'] == task_id:
                task_index = i
                task_info = task
                break
        
        if task_index is not None:
            # Remove from queued
            task_info = queue_data['queued_tasks'].pop(task_index)
            task_info['status'] = 'processing'
            task_info['start_time'] = datetime.now().isoformat()
            
            # Add to active
            queue_data['active_tasks'][task_id] = task_info
            
            self._save_queue_data(queue_id, queue_data)
            self._update_queue_progress(queue_id)
            
            logger.info(f"🔄 Task moved to active: {task_id} in queue {queue_id}")
            return True
        
        return False
    
    def update_task_progress_in_queue(self, queue_id: str, task_id: str, progress: int, status: str = None) -> bool:
        """Update progress of a specific task in the queue"""
        queue_data = self._get_queue_data(queue_id)
        if not queue_data:
            return False
        
        # Update in active tasks
        if task_id in queue_data['active_tasks']:
            queue_data['active_tasks'][task_id]['progress'] = progress
            if status:
                queue_data['active_tasks'][task_id]['status'] = status
            if progress >= 100:
                queue_data['active_tasks'][task_id]['end_time'] = datetime.now().isoformat()
        
        self._save_queue_data(queue_id, queue_data)
        self._update_queue_progress(queue_id)
        
        return True
    
    def complete_task_in_queue(self, queue_id: str, task_id: str, success: bool = True) -> bool:
        """Mark a task as completed in the queue"""
        queue_data = self._get_queue_data(queue_id)
        if not queue_data:
            return False
        
        task_info = None
        
        # Remove from active tasks
        if task_id in queue_data['active_tasks']:
            task_info = queue_data['active_tasks'].pop(task_id)
        
        # Update counters
        if task_info:
            if success:
                queue_data['completed_tasks'] += 1
                task_info['status'] = 'completed'
            else:
                queue_data['failed_tasks'] += 1
                task_info['status'] = 'failed'
            
            task_info['end_time'] = datetime.now().isoformat()
            task_info['progress'] = 100
            
            queue_data['completed_task_ids'].append(task_id)
        
        self._save_queue_data(queue_id, queue_data)
        self._update_queue_progress(queue_id)
        
        logger.info(f"✅ Task completed in queue {queue_id}: {task_id} (success: {success})")
        return True
    
    def _update_queue_progress(self, queue_id: str):
        """Calculate and update overall queue progress"""
        queue_data = self._get_queue_data(queue_id)
        if not queue_data or queue_data['total_tasks'] == 0:
            return
        
        # Calculate weighted progress
        total_weight = queue_data['total_tasks']
        completed_weight = queue_data['completed_tasks'] + queue_data['failed_tasks']
        
        # Add progress from active tasks
        active_progress = 0
        for task_id, task_info in queue_data['active_tasks'].items():
            active_progress += task_info['progress'] / 100  # Convert to fraction
        
        # Overall progress calculation
        overall_progress = ((completed_weight + active_progress) / total_weight) * 100
        queue_data['overall_progress'] = min(100, overall_progress)
        
        # Check if queue is complete
        if (queue_data['completed_tasks'] + queue_data['failed_tasks']) >= queue_data['total_tasks']:
            queue_data['status'] = 'completed'
            queue_data['end_time'] = datetime.now().isoformat()
            logger.info(f"🏁 Queue {queue_id} completed: {queue_data['completed_tasks']} successful, {queue_data['failed_tasks']} failed")
        
        self._save_queue_data(queue_id, queue_data)
        
        # Emit queue progress update
        self._emit_queue_progress_background(queue_id, queue_data)
    
    def _emit_queue_progress_background(self, queue_id: str, queue_data: Dict):
        """Emit queue progress update via socket"""
        try:
            def emit_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._emit_queue_progress_async(queue_id, queue_data))
                finally:
                    loop.close()

            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(emit_in_thread)
        except Exception as e:
            logger.error(f"❌ Failed to emit queue progress: {str(e)}")
    
    async def _emit_queue_progress_async(self, queue_id: str, queue_data: Dict):
        """Emit queue progress update"""
        try:
            user_id = queue_data.get('user_id')
            if user_id:
                emit_data = {
                    'queue_id': queue_id,
                    'overall_progress': queue_data['overall_progress'],
                    'total_tasks': queue_data['total_tasks'],
                    'completed_tasks': queue_data['completed_tasks'],
                    'failed_tasks': queue_data['failed_tasks'],
                    'active_tasks': len(queue_data['active_tasks']),
                    'status': queue_data['status']
                }
                
                await sio.emit('queue_progress', emit_data, room=f"user_{user_id}")
                logger.debug(f"📊 Queue progress emitted: {queue_data['overall_progress']}% for {queue_id}")
        except Exception as e:
            logger.error(f"❌ Failed to emit queue progress: {str(e)}")
    
    def get_queue_progress(self, queue_id: str) -> Optional[Dict]:
        """Get current queue progress"""
        return self._get_queue_data(queue_id)
    
    def get_user_queue(self, user_id: str) -> Optional[str]:
        """Get user's current queue ID"""
        return self.redis_client.get(f"user_current_queue:{user_id}")
    
    def _get_queue_data(self, queue_id: str) -> Optional[Dict]:
        """Get queue data from Redis"""
        try:
            data = self.redis_client.get(f"queue:{queue_id}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"❌ Failed to get queue data: {str(e)}")
            return None
    
    def _save_queue_data(self, queue_id: str, data: Dict):
        """Save queue data to Redis"""
        try:
            self.redis_client.setex(
                f"queue:{queue_id}",
                7200,
                json.dumps(data)
            )
        except Exception as e:
            logger.error(f"❌ Failed to save queue data: {str(e)}")

    # ===== TASK-LEVEL PROGRESS METHODS =====
    
    def initialize_task_progress(self, task_id: str, total_steps: int, filenames: List[str], user_id: str = None, queue_id: str = None):
        """Initialize progress for a batch task - SYNC for reliability in API/Celery"""
        try:
            progress_data = {
                'task_id': task_id,
                'total_steps': total_steps,
                'current_step': 0,
                'completed_steps': 0,
                'status': 'processing',
                'start_time': datetime.now().isoformat(),
                'current_file': '',
                'processed_files': [],
                'failed_files': [],
                'filenames': filenames,
                'progress_percentage': 0,
                'current_file_progress': 0,
                'user_id': user_id,
                'queue_id': queue_id  # Store queue reference
            }
            
            # Sync Redis set
            self.redis_client.setex(
                f"progress:{task_id}", 
                3600,
                json.dumps(progress_data)
            )
            logger.info(f"📊 Progress initialized for task {task_id}: {total_steps} files, user: {user_id}, queue: {queue_id}")
            
            # Register with queue if provided
            if queue_id and user_id and filenames:
                for filename in filenames:
                    self.add_task_to_queue(queue_id, task_id, 'batch_processing', filename)
            
            # Use background thread for async operations in Celery context
            self._emit_batch_started_background(task_id, filenames, total_steps, user_id)
            
            return progress_data
        except Exception as e:
            logger.error(f"❌ Failed to initialize progress for task {task_id}: {str(e)}")
            raise
    
    def _emit_batch_started_background(self, task_id: str, filenames: List[str], total_files: int, user_id: str = None):
        """Background emit using thread for sync Celery compatibility"""
        try:
            def emit_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._emit_batch_started_async(task_id, filenames, total_files, user_id))
                finally:
                    loop.close()

            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(emit_in_thread)
            logger.info(f"📡 Background batch_started emit queued for task {task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to queue batch_started emit for task {task_id}: {str(e)}")

    async def _emit_batch_started_async(self, task_id: str, filenames: List[str], total_files: int, user_id: str = None):
        """Internal async method to emit batch started event"""
        try:
            emit_data = {
                'task_id': task_id,
                'filenames': filenames,
                'total_files': total_files,
                'user_id': user_id
            }
            
            logger.info(f"📡 Emitting batch_started for task {task_id} to user {user_id}")
            
            if user_id:
                await sio.emit('batch_started', emit_data, room=f"user_{user_id}")
                logger.info(f"✅ batch_started sent to user room: user_{user_id}")
            else:
                await sio.emit('batch_started', emit_data)
                logger.info("✅ batch_started broadcast to all users")
                
        except Exception as e:
            logger.error(f"❌ Failed to emit batch_started for task {task_id}: {str(e)}")

    def update_progress(
        self, 
        task_id: str, 
        current_step: int = None,
        current_file: str = None,
        status: str = None,
        completed: bool = False,
        failed_file: Optional[str] = None,
        file_progress: Optional[int] = None
    ):
        """Simplified progress calculation - each file contributes equally"""
        try:
            progress_key = f"progress:{task_id}"
            progress_data = self.redis_client.get(progress_key)
            
            if not progress_data:
                error_msg = f"❌ Progress data not found for task {task_id}"
                logger.warning(error_msg)
                return None
            
            progress = json.loads(progress_data)
            
            # Update basic fields
            if current_step is not None:
                progress['current_step'] = current_step
            if current_file is not None:
                progress['current_file'] = current_file
            if status is not None:
                progress['status'] = status
            
            # Handle file-level progress
            if file_progress is not None:
                progress['current_file_progress'] = min(100, max(0, file_progress))
            
            # Handle failed file
            if failed_file:
                if failed_file not in progress['failed_files']:
                    progress['failed_files'].append(failed_file)
                progress['current_file_progress'] = 100
            
            # Handle completion of current file
            if completed:
                if not failed_file and current_file and current_file not in progress['processed_files']:
                    progress['processed_files'].append(current_file)
                
                old_completed = progress['completed_steps']
                progress['completed_steps'] += 1
                progress['current_file_progress'] = 0
                
                logger.info(f"✅ File processed: {current_file or 'unknown'} | Completed: {old_completed} -> {progress['completed_steps']}")
            
            # SIMPLIFIED PROGRESS CALCULATION:
            # Each file contributes equally to overall progress
            total_steps = progress['total_steps']
            completed_steps = progress['completed_steps']
            
            # Base progress from completed files
            base_progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
            
            # Add current file progress as fraction
            current_file_contribution = (progress.get('current_file_progress', 0) / 100) * (1 / total_steps) * 100
            
            # Total progress
            new_percentage = min(100, base_progress + current_file_contribution)
            old_percentage = progress.get('progress_percentage', 0)
            progress['progress_percentage'] = new_percentage
            
            calc_msg = f"📊 Progress: {completed_steps}/{total_steps} files + {progress.get('current_file_progress', 0)}% current = {new_percentage:.1f}% overall"
            logger.info(calc_msg)
            
            # Update queue progress if this task is part of one
            queue_id = progress.get('queue_id')
            if queue_id:
                self.update_task_progress_in_queue(queue_id, task_id, new_percentage, progress['status'])
            
            # Check if all files are completed
            if progress['completed_steps'] >= total_steps and progress['status'] != 'completed':
                progress['status'] = 'completed'
                progress['end_time'] = datetime.now().isoformat()
                progress['progress_percentage'] = 100  # Force 100% when all files done
                complete_msg = f"🏁 Task {task_id} completed: {progress['completed_steps']}/{total_steps} files"
                logger.info(complete_msg)
                
                # Mark as completed in queue
                if queue_id:
                    success = len(progress['failed_files']) == 0
                    self.complete_task_in_queue(queue_id, task_id, success)
                
                # Schedule reset after completion
                self._schedule_reset_task(task_id, progress)
                
                self._emit_task_complete_background(task_id, progress)
            
            # Save updated progress
            self.redis_client.setex(progress_key, 3600, json.dumps(progress))
            
            update_msg = f"📊 Saved: {progress['progress_percentage']:.1f}% | Step: {progress['current_step']} | Status: {progress['status']} | File: '{progress['current_file']}'"
            logger.info(update_msg)
            
            # Emit progress update
            self._emit_progress_update_background(task_id, progress)
            
            return progress
        except Exception as e:
            error_msg = f"❌ Failed to update progress for task {task_id}: {str(e)}"
            logger.error(error_msg)
            return None

    def _schedule_reset_task(self, task_id: str, progress: Dict):
        """Schedule progress reset after completion"""
        try:
            # Schedule reset after 30 seconds (give time for frontend to receive final updates)
            reset_delay = 30  # seconds
            
            def delayed_reset():
                import time
                time.sleep(reset_delay)
                self.reset_task_progress(task_id)
            
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(delayed_reset)
            
            logger.info(f"⏰ Reset scheduled for task {task_id} in {reset_delay} seconds")
        except Exception as e:
            logger.error(f"❌ Failed to schedule reset for task {task_id}: {str(e)}")

    def reset_task_progress(self, task_id: str):
        """Reset/remove progress data for a completed task"""
        try:
            progress_key = f"progress:{task_id}"
            
            # Get progress data before deletion for emitting reset event
            progress_data = self.redis_client.get(progress_key)
            if progress_data:
                progress = json.loads(progress_data)
                user_id = progress.get('user_id')
                
                # Delete the progress data
                deleted = self.redis_client.delete(progress_key)
                
                if deleted:
                    logger.info(f"🔄 Progress reset for task {task_id}")
                    
                    # Emit reset event
                    self._emit_progress_reset_background(task_id, user_id)
                else:
                    logger.warning(f"⚠️ No progress found to reset for task {task_id}")
            else:
                logger.warning(f"⚠️ No progress data found for task {task_id} during reset")
                
        except Exception as e:
            logger.error(f"❌ Failed to reset progress for task {task_id}: {str(e)}")

    def _emit_progress_reset_background(self, task_id: str, user_id: str = None):
        """Background emit for progress_reset event"""
        try:
            def emit_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._emit_progress_reset_async(task_id, user_id))
                finally:
                    loop.close()

            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(emit_in_thread)
            logger.info(f"🔄 Background progress_reset emit queued for task {task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to queue progress_reset emit for task {task_id}: {str(e)}")

    async def _emit_progress_reset_async(self, task_id: str, user_id: str = None):
        """Emit progress_reset event when task progress is cleared"""
        try:
            reset_data = {
                'task_id': task_id,
                'message': 'Progress reset after completion'
            }
            
            if user_id:
                await sio.emit('progress_reset', reset_data, room=f"user_{user_id}")
                logger.info(f"🔄 progress_reset emitted to user_{user_id} for task {task_id}")
            else:
                await sio.emit('progress_reset', reset_data)
                logger.info(f"🔄 progress_reset broadcast for task {task_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to emit progress_reset for task {task_id}: {str(e)}")

    def _emit_progress_update_background(self, task_id: str, progress: Dict):
        """Background emit using thread for sync Celery compatibility"""
        try:
            def emit_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._emit_progress_update_async(task_id, progress))
                finally:
                    loop.close()

            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(emit_in_thread)
            logger.info(f"📡 Background progress emit queued for task {task_id} at {progress['progress_percentage']}%")
        except Exception as e:
            logger.error(f"❌ Failed to queue progress emit for task {task_id}: {str(e)}")

    async def _emit_progress_update_async(self, task_id: str, progress: Dict):
        """Internal async method to emit progress updates via socket"""
        try:
            user_id = progress.get('user_id')
            emit_data = {
                'task_id': task_id,
                'progress': progress
            }
            if user_id:
                await sio.emit('progress_update', emit_data, room=f"user_{user_id}")
                logger.info(f"📡 Progress update emitted to user_{user_id} for task {task_id}: {progress['progress_percentage']}%")
            else:
                await sio.emit('progress_update', emit_data)
                logger.info(f"📡 Progress update broadcast for task {task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to emit progress update for task {task_id}: {str(e)}")

    def _emit_task_complete_background(self, task_id: str, progress: Dict):
        """Background emit for task_complete event when 100% reached"""
        try:
            def emit_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._emit_task_complete_async(task_id, progress))
                finally:
                    loop.close()

            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(emit_in_thread)
            logger.info(f"🏁 Background task_complete emit queued for task {task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to queue task_complete emit for task {task_id}: {str(e)}")

    async def _emit_task_complete_async(self, task_id: str, progress: Dict):
        """Emit special task_complete event at 100%"""
        try:
            user_id = progress.get('user_id')
            complete_data = {
                'task_id': task_id,
                'status': 'completed',
                'summary': {
                    'total_files': progress['total_steps'],
                    'processed': progress['completed_steps'],
                    'successful': len(progress['processed_files']),
                    'failed': len(progress['failed_files'])
                }
            }
            if user_id:
                await sio.emit('task_complete', complete_data, room=f"user_{user_id}")
                logger.info(f"🏁 task_complete emitted to user_{user_id} for task {task_id}")
            else:
                await sio.emit('task_complete', complete_data)
                logger.info(f"🏁 task_complete broadcast for task {task_id}")
        except Exception as e:
            logger.error(f"❌ Failed to emit task_complete for task {task_id}: {str(e)}")

    def cleanup_completed_tasks(self, older_than_hours: int = 1):
        """Clean up progress data for completed tasks older than specified hours"""
        try:
            pattern = "progress:*"
            keys = self.redis_client.keys(pattern)
            cleaned_count = 0
            
            for key in keys:
                try:
                    progress_data = self.redis_client.get(key)
                    if progress_data:
                        progress = json.loads(progress_data)
                        
                        # Check if task is completed and older than threshold
                        if progress.get('status') == 'completed':
                            end_time_str = progress.get('end_time')
                            if end_time_str:
                                end_time = datetime.fromisoformat(end_time_str)
                                time_diff = datetime.now() - end_time
                                
                                if time_diff.total_seconds() > (older_than_hours * 3600):
                                    self.redis_client.delete(key)
                                    cleaned_count += 1
                                    logger.info(f"🧹 Cleaned up old progress: {key}")
                except Exception as e:
                    logger.error(f"❌ Error processing key {key}: {str(e)}")
                    continue
            
            logger.info(f"🧹 Cleanup completed: {cleaned_count} old progress entries removed")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup completed tasks: {str(e)}")
            return 0

    async def get_progress(self, task_id: str) -> Optional[Dict]:
        """Get current progress for a task - ASYNC for API"""
        try:
            progress_data = self.redis_client.get(f"progress:{task_id}")
            if progress_data:
                progress = json.loads(progress_data)
                debug_msg = f"📊 Retrieved progress for task {task_id}: {progress['progress_percentage']}% | Status: {progress['status']} | File: '{progress['current_file']}' | File prog: {progress['current_file_progress']}% | Completed: {progress['completed_steps']}/{progress['total_steps']}"
                logger.debug(debug_msg)
                return progress
            else:
                warn_msg = f"📊 No progress found for task {task_id}"
                logger.warning(warn_msg)
                return None
        except Exception as e:
            error_msg = f"❌ Failed to get progress for task {task_id}: {str(e)}"
            logger.error(error_msg)
            return None
    def update_file_progress_only(self, task_id: str, file_index: int, filename: str, status: str, file_progress: int):
        """
        Updates progress for individual file ONLY - never marks batch as complete.
        Calculates overall progress as average of all file progresses.
        """
        progress_data = self.get_progress(task_id)
        if not progress_data:
            return
        
        # Initialize file progress tracking
        if 'files_progress' not in progress_data:
            progress_data['files_progress'] = {}
        
        # Update this file's progress
        progress_data['files_progress'][str(file_index)] = {
            'filename': filename,
            'progress': file_progress,
            'status': status
        }
        
        # Calculate overall progress (average of all files)
        total_files = progress_data.get('total_steps', 1)
        if progress_data['files_progress']:
            total_progress = sum(f['progress'] for f in progress_data['files_progress'].values())
            overall_progress = total_progress / total_files
        else:
            overall_progress = 0
        
        # NEVER set to 100% here - cap at 99%
        progress_data['file_progress_percent'] = min(overall_progress, 99)
        progress_data['current_step'] = file_index + 1
        progress_data['current_file'] = f"{filename} - {status}"
        progress_data['status'] = 'processing'  # Always 'processing' until callback
        
        self._save_progress(task_id, progress_data, completed=False)

    def update_batch_progress(self, task_id: str, current_file: str, status: str, overall_progress: int, completed: bool = False):
        """
        Updates batch-level progress. Only this method can mark as completed.
        """
        progress_data = self.get_progress(task_id)
        if not progress_data:
            return
        
        progress_data['current_file'] = current_file
        progress_data['status'] = status
        progress_data['file_progress_percent'] = overall_progress
        
        # Only mark as completed if explicitly requested (from callback)
        self._save_progress(task_id, progress_data, completed=completed)
# Singleton instance
progress_service = ProgressService()