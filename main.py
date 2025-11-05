import os
import asyncio
import json
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prisma import Prisma
from controllers.document_controller import router as document_router
from controllers.rebutle_controller import router as agent_router
from controllers.auth_controller import router as auth_router, get_current_user, verify_token
from config.settings import CONFIG
from utils.logger import setup_logging
from utils.socket_manager import sio
import socketio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Optional
from services.followup_service import check_all_overdue_tasks  # 🆕 Imported from separate service

# 🧠 Setup logging
setup_logging()

# ⚙️ FastAPI app
app = FastAPI(
    title="Document AI Extraction API",
    description="Extract text, entities, tables, and form fields from documents + Kebilo AI Agent",
    version="1.0.2"
)

# 🧩 Prisma ORM instance
db = Prisma()

# 🌐 CORS setup
NEXTAUTH_URL = os.getenv("NEXTAUTH_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[NEXTAUTH_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# 🔐 Enhanced Authentication dependency
async def authenticate_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Global authentication dependency for all routes"""
    public_paths = [
        "/health",
        "/docs",
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/sync-login",
        "/api/auth/refresh",
        "/redoc",
        "/api/agent/progress/",  # 🆕 ADDED PUBLIC PROGRESS ROUTE
"/webhook/save-document"  # 🆕 FULL PATH WITH PREFIX
    ]
   
    # Check if path is public (enhanced pattern matching)
    is_public = any(
        request.url.path == path or 
        request.url.path.startswith(path)  # Handles /api/agent/progress/123
        for path in public_paths
    )
   
    if is_public:
        return None
   
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please create Bearer token."
        )
   
    try:
        # ✅ Use the pure function without Depends for manual invocation
        user = await verify_token(credentials.credentials, db)
        print(f"🔐 Authenticated user: {user.email} accessing {request.method} {request.url.path}")
        return user
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# 📦 Include routers
app.include_router(
    auth_router,
    prefix="/api/auth",
    tags=["authentication"]
)

# Protected routes - ALL APIs require authentication
app.include_router(
    document_router,
    prefix="/api/documents",
    tags=["documents"],
    dependencies=[Depends(authenticate_request)]
)

app.include_router(
    agent_router,
    prefix="/api/agent",
    tags=["agent"],
    dependencies=[Depends(authenticate_request)]
)

# 📁 Ensure upload directory exists
os.makedirs(CONFIG["upload_dir"], exist_ok=True)

# 🚀 Mount Socket.IO ASGI app
socket_app = socketio.ASGIApp(
    sio,
    other_asgi_app=app
)

# ⚙️ Azure OpenAI LLM setup (global, but used in service)
llm = AzureChatOpenAI(
    azure_deployment=CONFIG["azure_openai_deployment"],
    azure_endpoint=CONFIG["azure_openai_endpoint"],
    api_key=CONFIG["azure_openai_api_key"],
    api_version=CONFIG["azure_openai_api_version"],
    temperature=0.3,
    model_name="gpt-4o-mini"
)

# 🆕 PUBLIC Progress Route (No Authentication)

from services.progress_service import ProgressService
progress_service = ProgressService()

@app.get("/api/agent/progress/{task_id}")
async def get_progress(task_id: str):
    """Get current progress for a task"""
    try:
        progress = await progress_service.get_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task progress not found")
        return progress
    except Exception as e:
        error_msg = f"❌ Error getting progress for task {task_id}: {str(e)}"
        # logger.error(error_msg)
        print(error_msg)  # DEBUG PRINT
        raise HTTPException(status_code=500, detail=f"Error retrieving progress: {str(e)}")
# 🩺 Health check (public)



from services.webhook_service import WebhookService
from services.database_service import get_database_service
import logging
logger = logging.getLogger("document_ai")
@app.post("/webhook/save-document")
async def save_document_webhook(request: Request):
    """
    Main webhook handler: Uses WebhookService to process the request.
    """
    try:
        data = await request.json()
        logger.info(f"📥 Webhook received for document save: {data.get('document_id', 'unknown')}")

        db_service = await get_database_service()
        service = WebhookService()
        result = await service.handle_webhook(data, db_service)

        return result

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"❌ Webhook save failed: {str(e)}", exc_info=True)

        # Save to FailDocs on general exception
        if 'data' in locals():
            blob_path = data.get("blob_path", "") ,
            physician_id = data.get("physician_id")
            reason = f"Webhook processing failed: {str(e)}"

            # Emit error event
            try:
                await sio.emit('task_error', {
                    'document_id': data.get('document_id', 'unknown'),
                    'filename': data.get('filename', 'unknown'),
                    'error': str(e),
                    'gcs_url': data.get('gcs_url', 'unknown'),
                    'physician_id': physician_id,
                    'blob_path': blob_path
                })
            except:
                pass  # Ignore emit failure

        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "processor": CONFIG["processor_id"]}

# 🔐 SECURED Protected API routes
@app.get("/api/user/tasks")
async def get_user_tasks(current_user = Depends(authenticate_request)):
    """Get tasks for current user - REQUIRES AUTHENTICATION"""
    try:
        tasks = await db.task.find_many(
            where={"physicianId": current_user.physicianId}
        )
        return {
            "user": {
                "id": current_user.id,
                "email": current_user.email,
                "name": f"{current_user.firstName} {current_user.lastName}"
            },
            "tasks": tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/documents")
async def get_user_documents(current_user = Depends(authenticate_request)):
    """Get documents for current user - REQUIRES AUTHENTICATION"""
    try:
        documents = await db.document.find_many(
            where={"userId": current_user.id}
        )
        return {
            "user": {
                "id": current_user.id,
                "email": current_user.email
            },
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/profile")
async def get_user_profile(current_user = Depends(authenticate_request)):
    """Get current user profile - REQUIRES AUTHENTICATION"""
    return {
        "user": current_user,
        "authenticated": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/admin/overdue-stats")
async def get_overdue_stats(current_user = Depends(authenticate_request)):
    """Get overdue tasks statistics - REQUIRES AUTHENTICATION"""
    try:
        now = datetime.now(timezone.utc)
       
        total_overdue = await db.task.count(
            where={
                "dueDate": {"lt": now},
                "status": {"not": "Completed"}
            }
        )
       
        physician_overdue = await db.task.count(
            where={
                "dueDate": {"lt": now},
                "status": {"not": "Completed"},
                "physicianId": current_user.physicianId
            }
        )
       
        return {
            "total_overdue_tasks": total_overdue,
            "your_overdue_tasks": physician_overdue,
            "user": current_user.email
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ⚙️ Database lifecycle
@app.on_event("startup")
async def startup():
    await db.connect()
    print("✅ Connected to PostgreSQL")
    # 🕓 Start enhanced cron scheduler - runs every 1 minute FOR TESTING
    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_all_overdue_tasks, "interval", minutes=1)
    scheduler.start()
    print("🕒 Enhanced scheduler started — runs every 1 minute (FOR TESTING)")
    print("🔒 All API routes are secured with JWT authentication")
    print("🌐 Public route available: /api/agent/progress/{task_id}")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    print("🔌 Disconnected from DB")

# 🧾 Enhanced Audit middleware
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    user_id = None
    email = None
    user_agent = request.headers.get("user-agent", "Unknown")
    ip_address = request.client.host if request.client else "Unknown"
   
    # Skip audit for health checks
    if request.url.path == "/health":
        return await call_next(request)
   
    try:
        # Try to get user from JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            current_user = await verify_token(token, db)
            if current_user:
                user_id = current_user.id
                email = current_user.email
    except:
        pass # Silent fail for audit logging
   
    # Log the request
    print(f"📝 Audit: {request.method} {request.url.path} - User: {email or 'Anonymous'} - IP: {ip_address}")
   
    try:
        await db.auditlog.create(
            data={
                "userId": user_id,
                "email": email,
                "action": f"{request.method} {request.url.path}",
                "ipAddress": ip_address,
                "userAgent": user_agent,
                "path": request.url.path,
                "method": request.method,
            }
        )
    except Exception as e:
        print("⚠️ Failed to save audit log:", e)
    response = await call_next(request)
    return response

# ▶️ Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host=CONFIG["host"], port=CONFIG["port"])    