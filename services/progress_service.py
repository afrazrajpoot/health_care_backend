# services/progress_service.py
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import redis
import json
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
            print("✅ Redis connected successfully for progress tracking")  # DEBUG PRINT
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {str(e)}")
            print(f"❌ Redis connection failed: {str(e)}")  # DEBUG PRINT
            raise
    
    def initialize_task_progress(self, task_id: str, total_steps: int, filenames: List[str], user_id: str = None):
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
                'user_id': user_id
            }
            
            # Sync Redis set
            self.redis_client.setex(
                f"progress:{task_id}", 
                3600,
                json.dumps(progress_data)
            )
            logger.info(f"📊 Progress initialized for task {task_id}: {total_steps} files, user: {user_id}")
            print(f"📊 Progress initialized for task {task_id}: {total_steps} files, user: {user_id}")  # DEBUG PRINT
            
            # Use background thread for async operations in Celery context
            self._emit_batch_started_background(task_id, filenames, total_steps, user_id)
            
            return progress_data
        except Exception as e:
            logger.error(f"❌ Failed to initialize progress for task {task_id}: {str(e)}")
            print(f"❌ Failed to initialize progress for task {task_id}: {str(e)}")  # DEBUG PRINT
            raise
    
    def _emit_batch_started_background(self, task_id: str, filenames: List[str], total_files: int, user_id: str = None):
        """Background emit using thread for sync Celery compatibility"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            import asyncio

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
            print(f"📡 Emitting batch_started for task {task_id} to user {user_id}")  # DEBUG PRINT
            
            if user_id:
                await sio.emit('batch_started', emit_data, room=f"user_{user_id}")
                logger.info(f"✅ batch_started sent to user room: user_{user_id}")
                print(f"✅ batch_started sent to user room: user_{user_id}")  # DEBUG PRINT
            else:
                await sio.emit('batch_started', emit_data)
                logger.info("✅ batch_started broadcast to all users")
                print("✅ batch_started broadcast to all users")  # DEBUG PRINT
                
        except Exception as e:
            logger.error(f"❌ Failed to emit batch_started for task {task_id}: {str(e)}")
            print(f"❌ Failed to emit batch_started for task {task_id}: {str(e)}")  # DEBUG PRINT

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
                print(error_msg)
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
                print(f"✅ File processed: {current_file or 'unknown'} | Completed: {old_completed} -> {progress['completed_steps']}")
            
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
            print(calc_msg)
            
            # Check if all files are completed
            if progress['completed_steps'] >= total_steps and progress['status'] != 'completed':
                progress['status'] = 'completed'
                progress['end_time'] = datetime.now().isoformat()
                progress['progress_percentage'] = 100  # Force 100% when all files done
                complete_msg = f"🏁 Task {task_id} completed: {progress['completed_steps']}/{total_steps} files"
                logger.info(complete_msg)
                print(complete_msg)
                
                # Schedule reset after completion
                self._schedule_reset_task(task_id, progress)
                
                self._emit_task_complete_background(task_id, progress)
            
            # Save updated progress
            self.redis_client.setex(progress_key, 3600, json.dumps(progress))
            
            update_msg = f"📊 Saved: {progress['progress_percentage']:.1f}% | Step: {progress['current_step']} | Status: {progress['status']} | File: '{progress['current_file']}'"
            logger.info(update_msg)
            print(update_msg)
            
            # Emit progress update
            self._emit_progress_update_background(task_id, progress)
            
            return progress
        except Exception as e:
            error_msg = f"❌ Failed to update progress for task {task_id}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
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
            
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(delayed_reset)
            
            logger.info(f"⏰ Reset scheduled for task {task_id} in {reset_delay} seconds")
            print(f"⏰ Reset scheduled for task {task_id} in {reset_delay} seconds")
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
                    print(f"🔄 Progress reset for task {task_id}")
                    
                    # Emit reset event
                    self._emit_progress_reset_background(task_id, user_id)
                else:
                    logger.warning(f"⚠️ No progress found to reset for task {task_id}")
                    print(f"⚠️ No progress found to reset for task {task_id}")
            else:
                logger.warning(f"⚠️ No progress data found for task {task_id} during reset")
                print(f"⚠️ No progress data found for task {task_id} during reset")
                
        except Exception as e:
            logger.error(f"❌ Failed to reset progress for task {task_id}: {str(e)}")
            print(f"❌ Failed to reset progress for task {task_id}: {str(e)}")

    def _emit_progress_reset_background(self, task_id: str, user_id: str = None):
        """Background emit for progress_reset event"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            import asyncio

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
                print(f"🔄 progress_reset emitted to user_{user_id} for task {task_id}")
            else:
                await sio.emit('progress_reset', reset_data)
                logger.info(f"🔄 progress_reset broadcast for task {task_id}")
                print(f"🔄 progress_reset broadcast for task {task_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to emit progress_reset for task {task_id}: {str(e)}")

    def _emit_progress_update_background(self, task_id: str, progress: Dict):
        """Background emit using thread for sync Celery compatibility"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            import asyncio

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
                print(f"📡 Progress update emitted to user_{user_id} for task {task_id}: {progress['progress_percentage']}")  # DEBUG PRINT
            else:
                await sio.emit('progress_update', emit_data)
                logger.info(f"📡 Progress update broadcast for task {task_id}")
                print(f"📡 Progress update broadcast for task {task_id}")  # DEBUG PRINT
        except Exception as e:
            logger.error(f"❌ Failed to emit progress update for task {task_id}: {str(e)}")
            print(f"❌ Failed to emit progress update for task {task_id}: {str(e)}")  # DEBUG PRINT

    def _emit_task_complete_background(self, task_id: str, progress: Dict):
        """Background emit for task_complete event when 100% reached"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            import asyncio

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
                print(f"🏁 task_complete emitted to user_{user_id} for task {task_id}")  # DEBUG PRINT
            else:
                await sio.emit('task_complete', complete_data)
                logger.info(f"🏁 task_complete broadcast for task {task_id}")
                print(f"🏁 task_complete broadcast for task {task_id}")  # DEBUG PRINT
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
                print(debug_msg)  # DEBUG PRINT
                return progress
            else:
                warn_msg = f"📊 No progress found for task {task_id}"
                logger.warning(warn_msg)
                print(warn_msg)  # DEBUG PRINT
                return None
        except Exception as e:
            error_msg = f"❌ Failed to get progress for task {task_id}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)  # DEBUG PRINT
            return None

# Singleton instance
progress_service = ProgressService()