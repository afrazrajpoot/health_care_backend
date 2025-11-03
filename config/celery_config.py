# celery_config.py (updated)
from celery import Celery
import os
import logging

logger = logging.getLogger(__name__)

broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

app = Celery(
    'ai_backend',
    broker=broker_url,
    backend=result_backend,
    include=['main']
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    broker_connection_retry_on_startup=True,
    result_expires=86400,
    
    # Updated for parallelism: Allow prefetching multiple tasks
    task_acks_late=True,
    worker_prefetch_multiplier=4,  # Increase this (e.g., 4-16) based on your worker concurrency and task size
)

# Test Redis connection (unchanged)
from redis import Redis

def test_broker_connection():
    try:
        redis_client = Redis.from_url(broker_url)
        redis_client.ping()
        logger.info("✅ Redis broker connection successful")
    except Exception as e:
        logger.error(f"❌ Redis broker connection failed: {str(e)}")

def test_backend_connection():
    try:
        redis_client = Redis.from_url(result_backend)
        redis_client.ping()
        logger.info("✅ Redis result backend connection successful")
    except Exception as e:
        logger.error(f"❌ Redis result backend connection failed: {str(e)}")

test_broker_connection()
test_backend_connection()
logger.info(f"Celery configured with broker: {broker_url}, backend: {result_backend}")