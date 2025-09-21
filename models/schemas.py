from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ExtractionResult(BaseModel):
    text: str
    pages: int
    entities: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    formFields: List[Dict[str, Any]]
    confidence: float
    success: bool
    error: Optional[str] = None
    fileInfo: Optional[Any] = None
    summary: str = ""  # New field for the summary

class ErrorResponse(BaseModel):
    success: bool = False
    error: str = ""

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    processor: str

class FileInfo(BaseModel):
    originalName: str = ""
    size: int = 0
    mimeType: str = ""