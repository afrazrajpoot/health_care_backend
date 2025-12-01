# models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PatientInfo(BaseModel):
    patient_name: Optional[str] = None
    patient_email: Optional[str] = None
    claim_no: Optional[str] = None
    report_title: Optional[str] = None
    time_day: Optional[str] = None
    status: Optional[str] = "normal"



class WorkStatusAlert(BaseModel):
    alert_type: str
    title: str
    date: str
    status: str

class AlertModel(BaseModel):
    """Deterministic alert produced by RuleEngine (persisted)."""
    alert_type: str
    title: str
    date: Optional[str] = None
    status: str = "normal"
    source: Optional[str] = None
    rule_id: Optional[str] = None


class ActionModel(BaseModel):
    """Deterministic action/task to be created (persisted or shown on dashboard)."""
    action_id: Optional[str] = None
    type: str  # e.g., "referral", "rebuttal", "request_missing_info", "check_authorization"
    reason: str
    due_date: Optional[str] = None
    assignee: Optional[str] = None
    source_doc_id: Optional[str] = None
    deterministic_rule_id: Optional[str] = None
    suggestion: bool = False  # False = deterministic, True = LLM-suggestion
    auto_create: bool = True
    created_at: Optional[str] = None







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



class ComplianceNudge(BaseModel):
    nudge_id: str
    type: str  # e.g. "missing_documentation", "authorization_check"
    reason: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    created_at: Optional[str] = None
    rule_id: Optional[str] = None
    source: Optional[str] = None

class Referral(BaseModel):
    referral_id: str
    specialty: Optional[str] = None
    reason: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    created_at: Optional[str] = None
    rule_id: Optional[str] = None
    source: Optional[str] = None














class ComprehensiveAnalysis(BaseModel):
    original_report: str = ""
    report_json: PatientInfo = Field(default_factory=PatientInfo)
    summary: List[str] = Field(default_factory=list)
    work_status_alert: List[WorkStatusAlert] = Field(default_factory=list)

class ExtractionResult(BaseModel):
    text: str = ""
    raw_text: Optional[str] = ""  # NEW: Original flat text
    llm_text: Optional[str] = Field(default=None, description="LLM-optimized text with explicit section annotations")
    page_zones: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Per-page zones: header, body, footer, signature"
    )
    pages: int = 0
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    formFields: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    success: bool = False
    gcs_file_link: str = ""
    fileInfo: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    comprehensive_analysis: Optional[ComprehensiveAnalysis] = None
    document_id: str = ""
    error: Optional[str] = None
    database_error: Optional[str] = None
    symbols: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata including checkboxes and handwritten text")
    