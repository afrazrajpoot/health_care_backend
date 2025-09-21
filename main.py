from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os

from controllers.document_controller import router as document_router
from config.settings import CONFIG
from utils.logger import setup_logging

# Setup logging
setup_logging()

# FastAPI app
app = FastAPI(
    title="Document AI Extraction API",
    description="Extract text, entities, tables, and form fields from documents using Google Document AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
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

# Ensure upload directory exists
os.makedirs(CONFIG["upload_dir"], exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)