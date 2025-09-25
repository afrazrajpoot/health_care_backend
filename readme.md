prisma migrate dev --name init
uvicorn main:app --reload --host 0.0.0.0 --port 8000
celery -A config.celery_config worker --loglevel=info
