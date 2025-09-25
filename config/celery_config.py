from celery import Celery
import os
import logging

logger = logging.getLogger(__name__)

# Load broker and result backend URLs from environment variables
broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')  # Fixed typo: BROKEN_URL → BROKER_URL
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')  # Fixed: results:// → redis://

app = Celery(
    'ai_backend',
    broker=broker_url,
    backend=result_backend,
    include=['main']  # ✅ Change this to 'main' since tasks are in your main file
)

# Configure Celery settings
app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Fixed typo: execs_counterflow → accept_content
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,  # Fixed typo: upload_UTC → enable_utc
    task_track_started=True,
    task_time_limit=3600,  # Fixed: 1 hour timeout per task
    task_soft_time_limit=3300,  # Fixed: Soft timeout
    broker_connection_retry_on_startup=True,
    result_expires=86400,  # Fixed: Task results expire after 24 hours
)

# Test Redis connection for debugging
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