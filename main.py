from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os

from controllers.document_controller import router as document_router
from config.settings import CONFIG
from utils.logger import setup_logging

from prisma import Prisma  # 👈 Import Prisma client

# Setup logging
setup_logging()

# FastAPI app
app = FastAPI(
    title="Document AI Extraction API",
    description="Extract text, entities, tables, and form fields from documents using Google Document AI",
    version="1.0.0"
)

# Prisma client
db = Prisma()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(document_router, prefix="/api", tags=["document"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-09-20T10:00:00Z",
        "processor": CONFIG["processor_id"]
    }

# 🔥 Connect Prisma when app starts
@app.on_event("startup")
async def startup():
    await db.connect()
    print("✅ Connected to PostgreSQL with Prisma")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    print("🔌 Disconnected from database")


# 📊 Audit middleware - save user visits/actions
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    # 👇 Get user info directly from headers
    user_id = request.headers.get("x-user-id")
    user_email = request.headers.get("x-user-email")
    user_name = request.headers.get("x-user-name")

    # 📝 Save audit log
    try:
        await db.auditlog.create(
            data={
                "userId": user_id,
                "email": user_email,
                "action": "Uploaded document",
                "ipAddress": request.client.host,
                "userAgent": request.headers.get("user-agent"),
                "path": request.url.path,
                "method": request.method,
            }
        )
    except Exception as e:
        print("⚠️ Failed to save audit log:", e)

    response = await call_next(request)
    return response


# 📁 Ensure upload directory exists
os.makedirs(CONFIG["upload_dir"], exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
