"""
Data models for document extraction
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypedDict
from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(Enum):
    """All supported document types"""
    MRI = "MRI"
    CT = "CT"
    XRAY = "X-ray"
    ULTRASOUND = "Ultrasound"
    EMG = "EMG"
    LABS = "Labs"
    PR2 = "PR-2"
    PR4 = "PR-4"
    DFR = "DFR"
    CONSULT = "Consult"
    RFA = "RFA"
    UR = "UR"
    AUTHORIZATION = "Authorization"
    PEER_TO_PEER = "Peer-to-Peer"
    QME = "QME"
    AME = "AME"
    IME = "IME"
    ADJUSTER = "Adjuster"
    ATTORNEY = "Attorney"
    NCM = "NCM"
    SIGNATURE_REQUEST = "Signature Request"
    REFERRAL = "Referral"
    DISCHARGE = "Discharge"
    MED_REFILL = "Med Refill"
    UNKNOWN = "Unknown"


@dataclass
class ExtractionResult:
    """Structured output for all extractions"""
    document_type: str
    document_date: str
    summary_line: str
    examiner_name: Optional[str] = None
    specialty: Optional[str] = None
    body_parts: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)






# ============================================================================
# PYDANTIC MODELS (Enhanced with verification fields)
# ============================================================================

class DateReasoning(BaseModel):
    """Structured reasoning about dates found in document"""
    extracted_dates: List[str] = Field(..., description="All dates found in the document in YYYY-MM-DD format")
    date_contexts: Dict[str, str] = Field(..., description="Context around each date found")
    reasoning: str = Field(..., description="Step-by-step reasoning for date assignments")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each date assignment (0.0-1.0)")
    predicted_assignments: Dict[str, str] = Field(..., description="Predicted date assignments")

class BodyPartAnalysis(BaseModel):
    """Analysis for a specific body part"""
    body_part: str = Field(..., description="Specific body part involved if it's a workers comp report else disease/condition")
    diagnosis: str = Field(..., description="Diagnosis for this body part")
    key_concern: str = Field(..., description="Key concern for this body part")
    clinical_summary: str = Field(..., description="Clinical summary of important findings for this body part")
    treatment_plan: str = Field(..., description="Treatment plan specific to this body part")
    extracted_recommendation: str = Field(..., description="Recommendations for this body part")
    adls_affected: str = Field(..., description="ADLs affected for this body part")
    work_restrictions: str = Field(..., description="Work restrictions for this body part")

class DocumentAnalysis(BaseModel):
    """Structured analysis of medical document matching database schema (enhanced with verification)"""
    patient_name: str = Field(..., description="Full name of the patient")
    claim_number: str = Field(..., description="Claim number. Use 'Not specified' if not found")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    doi: str = Field(..., description="Date of injury in YYYY-MM-DD format")
    status: str = Field(..., description="Current status: normal, urgent, critical, etc.")
    rd: str = Field(..., description="Report date in YYYY-MM-DD format")
    
    # Body parts analysis
    body_part: str = Field(..., description="Primary body part involved if it's a workers comp report else disease/condition")
    body_parts_analysis: List[BodyPartAnalysis] = Field(default=[], description="Detailed analysis for each body part")
    
    diagnosis: str = Field(..., description="Primary diagnosis and key findings")
    key_concern: str = Field(..., description="Main clinical concern in 2-3 words")
    extracted_recommendation: str = Field(..., description="Extracted key recommendation keywords/phrases")
    extracted_decision: str = Field(..., description="Extracted key decision/judgment keywords/phrases")
    ur_decision: str = Field(..., description="Extracted UR decision keywords/phrases")
    ur_denial_reason: Optional[str] = Field(None, description="UR denial reason if applicable")
    adls_affected: str = Field(..., description="Activities affected in 2-3 words")
    work_restrictions: str = Field(..., description="Work restrictions in 2-3 words")
    consulting_doctor: str = Field(default="Not specified", description="Name of consultant doctor")
    referral_doctor: str = Field(default="Not specified", description="Name of referral doctor if available")
    ai_outcome: str = Field(..., description="AI-generated key outcome prediction keywords/phrases")
    document_type: str = Field(..., description="Type of document")
    summary_points: List[str] = Field(..., description="3-5 key points, each 2-3 words")
    date_reasoning: Optional[DateReasoning] = Field(None, description="Reasoning behind date assignments")
    is_task_needed: bool = Field(default=False, description="If analysis determines any tasks are needed based on pending actions")
    
    # NEW: Verification metadata
    extraction_confidence: float = Field(default=0.0, description="Overall confidence score (0.0-1.0)")
    verified: bool = Field(default=False, description="Whether extraction has been verified")
    verification_notes: List[str] = Field(default=[], description="Any issues found during verification")

class BriefSummary(BaseModel):
    """Structured brief summary of the report"""
    brief_summary: str = Field(..., description="A concise 1-2 sentence summary of the entire report")


class VerificationResult(BaseModel):
    """Structured verification result"""
    is_valid: bool = Field(..., description="Whether extraction passed validation")
    confidence_score: float = Field(..., description="Overall confidence score (0.0-1.0)")
    issues_found: List[str] = Field(default=[], description="List of validation issues")
    corrections_made: Dict[str, Any] = Field(default={}, description="Fields that were corrected")
    needs_review: bool = Field(..., description="Whether manual review is recommended")


# ============================================================================
# STATE SCHEMA (unchanged)
# ============================================================================

class ReasoningState(TypedDict, total=False):
    document_text: str
    document_type: str
    current_date: str
    regex_dates: list
    llm_date_analysis: dict
    extraction_complete: bool
    all_dates: dict
    date_clues: dict
    context_analysis_complete: bool
    date_reasoning: dict
    reasoning_complete: bool
    final_date_assignments: dict
    date_reasoning_complete: bool
    validated_date_assignments: dict

