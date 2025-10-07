# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from prisma import Prisma
from controllers.document_controller import router as document_router
from config.settings import CONFIG
from utils.logger import setup_logging

# 👇 Import socket manager
# from socket_manager import sio
from utils.socket_manager import sio
import socketio

setup_logging()

app = FastAPI(
    title="Document AI Extraction API",
    description="Extract text, entities, tables, and form fields from documents",
    version="1.0.0"
)

# Prisma
db = Prisma()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(document_router, prefix="/api", tags=["document"])

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "processor": CONFIG["processor_id"]}

# Connect DB
@app.on_event("startup")
async def startup():
    await db.connect()
    print("✅ Connected to PostgreSQL")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()
    print("🔌 Disconnected from DB")

# Middleware (Audit log)
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    try:
        await db.auditlog.create(
            data={
                "userId": request.headers.get("x-user-id"),
                "email": request.headers.get("x-user-email"),
                "action": "Uploaded document",
                "ipAddress": request.client.host,
                "userAgent": request.headers.get("user-agent"),
                "path": request.url.path,
                "method": request.method,
            }
        )
    except Exception as e:
        print("⚠️ Failed to save audit log:", e)

    return await call_next(request)

# 📁 Ensure upload directory
os.makedirs(CONFIG["upload_dir"], exist_ok=True)

# 🚀 Mount Socket.IO ASGI app on top of FastAPI
# socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
# main.py
socket_app = socketio.ASGIApp(
    sio,
    other_asgi_app=app  # ✅ keep only this
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
