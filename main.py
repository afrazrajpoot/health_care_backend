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
            detail="Authentication required. Please provide Bearer token."
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

# ⚙️ Azure OpenAI LLM setup
llm = AzureChatOpenAI(
    azure_deployment=CONFIG["azure_openai_deployment"],
    azure_endpoint=CONFIG["azure_openai_endpoint"],
    api_key=CONFIG["azure_openai_api_key"],
    api_version=CONFIG["azure_openai_api_version"],
    temperature=0.3,
    model_name="gpt-4o-mini"
)

# 🧠 Enhanced Follow-up task generation
followup_prompt_template = PromptTemplate.from_template("""
You are a medical task management AI. Analyze the following overdue tasks and generate appropriate follow-up tasks.
OVERDUE TASKS:
{overdue_tasks}
CRITICAL INFORMATION:
- Current Date: {current_date}
- Total Overdue Tasks: {total_tasks}
INSTRUCTIONS:
1. Analyze EACH overdue task and create ONE logical follow-up
2. Consider medical urgency and workflow dependencies
3. Set reasonable due dates (1-7 days from now)
4. Assign appropriate departments based on task context
RESPONSE FORMAT - JSON array:
[
  {{
    "description": "Specific follow-up action",
    "department": "Relevant department",
    "reason": "Why this follow-up is needed based on the overdue task",
    "due_in_days": 2,
    "related_physician_id": "physician_id_if_available",
    "priority": "High/Medium/Low",
    "original_task_id": "reference_to_original_task"
  }}
]
EXAMPLES:
- Overdue: "Review patient lab results" → Follow-up: "Contact patient with lab results update"
- Overdue: "Submit insurance claim" → Follow-up: "Follow up on insurance claim status"
- Overdue: "Patient follow-up appointment" → Follow-up: "Schedule make-up appointment"
""")

async def generate_followup_tasks(overdue_tasks):
    """Generate automatic follow-up tasks for ALL overdue items"""
    if not overdue_tasks:
        return []
    formatted_tasks = "\n".join([
        f"- Task ID: {task.id} | Description: {task.description} | "
        f"Dept: {task.department} | Due: {task.dueDate} | "
        f"Physician: {task.physicianId or 'Not assigned'} | Patient: {task.patient}"
        for task in overdue_tasks
    ])
    try:
        response = llm.invoke(followup_prompt_template.format(
            overdue_tasks=formatted_tasks,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            total_tasks=len(overdue_tasks)
        ))
        if hasattr(response, 'content'):
            content = response.content.replace('```json', '').replace('```', '').strip()
            followup_suggestions = json.loads(content)
           
            validated_suggestions = []
            for suggestion in followup_suggestions:
                if isinstance(suggestion, dict):
                    # Find the original task for reference
                    original_task = next((t for t in overdue_tasks if t.id == suggestion.get("original_task_id")), overdue_tasks[0])
                   
                    validated_suggestion = {
                        "description": suggestion.get("description", f"Follow-up for: {original_task.description}"),
                        "department": suggestion.get("department", original_task.department),
                        "reason": suggestion.get("reason", f"Auto-generated follow-up for overdue task: {original_task.description}"),
                        "due_in_days": max(1, min(7, suggestion.get("due_in_days", 2))),
                        "related_physician_id": suggestion.get("related_physician_id", original_task.physicianId),
                        "priority": suggestion.get("priority", "Medium"),
                        "patient": original_task.patient,
                        "status": "Pending",
                        "sourceDocument": original_task.sourceDocument,
                        "claimNumber": original_task.claimNumber
                    }
                    validated_suggestions.append(validated_suggestion)
           
            return validated_suggestions
    except Exception as e:
        print(f"❌ Follow-up task generation failed: {e}")
    return []

# ⏰ ENHANCED Cron job: check ALL overdue tasks
async def check_all_overdue_tasks():
    """Check ALL overdue tasks and generate follow-ups for EVERY overdue task"""
    print("=" * 60)
    print("🔄 AUTO FOLLOW-UP SCHEDULED JOB STARTED")
    print(f" ⏰ Time: {datetime.now()}")
    print("=" * 60)
   
    try:
        now = datetime.now(timezone.utc)
       
        # Find ALL overdue tasks regardless of physician
        overdue_tasks = await db.task.find_many(
            where={
                "dueDate": {"lt": now},
                "status": {"not": "Completed"}
            },
            include={
                "document": True
            }
        )
        if not overdue_tasks:
            print("✅ No overdue tasks found in the system.")
            return

        # Filter to only tasks with physicianId
        tasks_with_assignee = [task for task in overdue_tasks if task.physicianId is not None]
        if not tasks_with_assignee:
            print("✅ No overdue tasks with assigned physician found. Skipping LLM generation.")
            print(f" 📊 Total unassigned overdue tasks: {len(overdue_tasks)}")
            return

        print(f"⚠️ Found {len(overdue_tasks)} total overdue tasks in the system")
        print(f"📋 Processing {len(tasks_with_assignee)} tasks with assigned physician")
       
        # Print overdue tasks summary (only for tasks with assignee)
        for i, task in enumerate(tasks_with_assignee, 1):
            overdue_days = (now - task.dueDate).days
            print(f" {i}. [{task.patient}] {task.description} - "
                  f"Overdue: {overdue_days} days - Physician: {task.physicianId}")
        
        # Generate follow-up tasks only for tasks with assignee
        followup_suggestions = await generate_followup_tasks(tasks_with_assignee)
       
        if not followup_suggestions:
            print("❌ No follow-up tasks generated")
            return
        created_count = 0
        for suggestion in followup_suggestions:
            try:
                # Determine assigned physician ID based on role
                original_task_id = suggestion.get("original_task_id")
                assigned_id = suggestion.get("related_physician_id")
                if original_task_id:
                    original_task = next((t for t in tasks_with_assignee if t.id == original_task_id), None)
                    if original_task and original_task.physicianId:
                        assignee = await db.user.find_unique(where={"id": original_task.physicianId})
                        if assignee and assignee.role != "Physician" and assignee.physicianId:
                            assigned_id = assignee.physicianId
                        # else keep as is (user.id if physician, or original if none)
                suggestion["related_physician_id"] = assigned_id

                due_date = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=suggestion["due_in_days"])
               
                await db.task.create(
                    data={
                        "description": suggestion["description"],
                        "department": suggestion["department"],
                        "reason": suggestion["reason"],
                        "dueDate": due_date,
                        "patient": suggestion["patient"],
                        "status": "Pending",
                        "physicianId": suggestion["related_physician_id"],
                        "sourceDocument": suggestion["sourceDocument"],
                        "claimNumber": suggestion["claimNumber"],
                        "quickNotes": {
                            "status_update": f"Auto-generated follow-up",
                            "details": f"Priority: {suggestion['priority']} - Generated from overdue task",
                            "one_line_note": f"Auto follow-up created on {datetime.now().strftime('%Y-%m-%d')}"
                        }
                    }
                )
                created_count += 1
                print(f" ✅ Created: {suggestion['description']}")
               
            except Exception as e:
                print(f" ❌ Failed to create follow-up task: {e}")
        print("=" * 60)
        print(f"✅ SCHEDULED JOB COMPLETED")
        print(f" 📊 Created {created_count} follow-up tasks")
        print(f" 📊 Processed {len(tasks_with_assignee)} overdue tasks with assignee")
        print("=" * 60)
           
    except Exception as e:
        print(f"❌ Error in scheduled job: {e}")

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
    # 🕓 Start enhanced cron scheduler - runs every 30 minutes
    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_all_overdue_tasks, "interval", minutes=30)
    scheduler.start()
    print("🕒 Enhanced scheduler started — runs every 30 minutes")
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