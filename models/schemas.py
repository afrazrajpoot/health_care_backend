from pydantic import BaseModel,Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class PatientInfo(BaseModel):
    patient_name: Optional[str] = None
    patient_email: Optional[str] = None
    claim_no: Optional[str] = None
    report_title: Optional[str] = None
    time_day: Optional[str] = None
    status: str = "normal"  # "urgent", "low", "normal"

class WorkStatusAlert(BaseModel):
    alert_type: str  # "Work Status Review", "Medical Urgency", "Follow-Up Required"
    title: str
    date: str
    status: str  # "urgent", "low priority", "normal"

class ComprehensiveAnalysis(BaseModel):
    original_report: str
    report_json: PatientInfo
    summary: List[str]
    work_status_alert: List[WorkStatusAlert] = []

class ExtractionResult(BaseModel):
    text: str = ""
    pages: int = 0
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    formFields: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    success: bool = False
    error: Optional[str] = None
    fileInfo: Optional[Any] = None
    summary: str = ""
    gcs_file_link: str = ""
    comprehensive_analysis: Optional[ComprehensiveAnalysis] = None
    document_id: Optional[str] = None
    database_error: Optional[str] = None

class ErrorResponse(BaseModel):
    error: bool = True
    message: str
    guidance: List[str] = []

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    processor: str

class FileInfo(BaseModel):
    originalName: str = ""
    size: int = 0
    mimeType: str = ""


class TaskResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None