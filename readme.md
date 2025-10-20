prisma migrate dev --name init
uvicorn main:app --reload --host 0.0.0.0 --port 8000
uvicorn main:socket_app --host 0.0.0.0 --port 8000 --reload
celery -A config.celery_config worker --loglevel=info

celery_task line 44
webhook_url = CONFIG.get("api_base_url", "http://localhost:8000") + "/api/webhook/save-document"

settings line 22
self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
