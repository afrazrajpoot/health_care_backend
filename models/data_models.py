"""
Data models for document extraction
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypedDict
from enum import Enum
from pydantic import BaseModel, Field


from enum import Enum

class DocumentType(Enum):
    """All supported document types - 70+ document types"""
    
    # Imaging Reports
    MRI = "MRI"
    CT = "CT"
    XRAY = "X-ray"
    ULTRASOUND = "Ultrasound"
    EMG = "EMG"
    MAMMOGRAM = "Mammogram"
    PET_SCAN = "PET Scan"
    BONE_SCAN = "Bone Scan"
    DEXA_SCAN = "DEXA Scan"
    FLUOROSCOPY = "Fluoroscopy"
    ANGIOGRAM = "Angiogram"
    
    # Laboratory & Diagnostics
    LABS = "Labs"
    PATHOLOGY = "Pathology"
    BIOPSY = "Biopsy"
    GENETIC_TESTING = "Genetic Testing"
    TOXICOLOGY = "Toxicology"
    ALLERGY_TESTING = "Allergy Testing"
    
    # Progress Reports & Evaluations
    PR2 = "PR-2"
    PR4 = "PR-4"
    DFR = "DFR"
    CONSULT = "Consult"
    PROGRESS_NOTE = "Progress Note"
    OFFICE_VISIT = "Office Visit"
    CLINIC_NOTE = "Clinic Note"
    TELEMEDICINE = "Telemedicine"
    
    # Medical Evaluations
    QME = "QME"
    AME = "AME"
    IME = "IME"
    IMR = "IMR"
    FCE = "FCE"
    PEER_REVIEW = "Peer Review"
    INDEPENDENT_REVIEW = "Independent Review"
    
    # Authorization & Utilization
    RFA = "RFA"
    UR = "UR"
    AUTHORIZATION = "Authorization"
    PEER_TO_PEER = "Peer-to-Peer"
    TREATMENT_AUTH = "Treatment Authorization"
    PROCEDURE_AUTH = "Procedure Authorization"
    
    # Therapy & Treatment Notes
    PHYSICAL_THERAPY = "Physical Therapy"
    OCCUPATIONAL_THERAPY = "Occupational Therapy"
    CHIROPRACTIC = "Chiropractic"
    ACUPUNCTURE = "Acupuncture"
    MASSAGE_THERAPY = "Massage Therapy"
    PAIN_MANAGEMENT = "Pain Management"
    
    # NEW: Clinical Notes - Nursing and additional types
    NURSING = "Nursing"
    NURSING_NOTE = "Nursing Note"
    VITAL_SIGNS = "Vital Signs"
    MEDICATION_ADMINISTRATION = "Medication Administration"
    
    # Surgical Documents
    SURGERY_REPORT = "Surgery Report"
    OPERATIVE_NOTE = "Operative Note"
    ANESTHESIA_REPORT = "Anesthesia Report"
    PRE_OP = "Pre-Op"
    POST_OP = "Post-Op"
    DISCHARGE = "Discharge"
    
    # Hospital Documents
    ADMISSION_NOTE = "Admission Note"
    HOSPITAL_COURSE = "Hospital Course"
    ER_REPORT = "ER Report"
    EMERGENCY_ROOM = "Emergency Room"
    HOSPITAL_PROGRESS = "Hospital Progress"
    
    # Specialty Reports
    CARDIOLOGY = "Cardiology"
    NEUROLOGY = "Neurology"
    ORTHOPEDICS = "Orthopedics"
    PSYCHIATRY = "Psychiatry"
    PSYCHOLOGY = "Psychology"
    PSYCHOTHERAPY = "Psychotherapy"
    BEHAVIORAL_HEALTH = "Behavioral Health"
    RHEUMATOLOGY = "Rheumatology"
    ENDOCRINOLOGY = "Endocrinology"
    GASTROENTEROLOGY = "Gastroenterology"
    PULMONOLOGY = "Pulmonology"
    
    # Diagnostic Studies
    SLEEP_STUDY = "Sleep Study"
    EKG = "EKG"
    ECG = "ECG"
    HOLTER_MONITOR = "Holter Monitor"
    ECHO = "Echocardiogram"
    STRESS_TEST = "Stress Test"
    PULMONARY_FUNCTION = "Pulmonary Function"
    NERVE_CONDUCTION = "Nerve Conduction"
    
    # Medications & Pharmacy
    MED_REFILL = "Med Refill"
    PRESCRIPTION = "Prescription"
    PHARMACY = "Pharmacy"
    MEDICATION_LIST = "Medication List"
    PRIOR_AUTH = "Prior Authorization"
    
    # Administrative & Correspondence
    ADJUSTER = "Adjuster"
    ATTORNEY = "Attorney"
    NCM = "NCM"
    SIGNATURE_REQUEST = "Signature Request"
    REFERRAL = "Referral"
    CORRESPONDENCE = "Correspondence"
    APPEAL = "Appeal"
    DENIAL_LETTER = "Denial Letter"
    APPROVAL_LETTER = "Approval Letter"
    
    # Work Status & Disability
    WORK_STATUS = "Work Status"
    WORK_RESTRICTIONS = "Work Restrictions"
    RETURN_TO_WORK = "Return to Work"
    DISABILITY = "Disability"
    CLAIM_FORM = "Claim Form"
    
    # Legal Documents
    DEPOSITION = "Deposition"
    INTERROGATORY = "Interrogatory"
    SUBPOENA = "Subpoena"
    AFFIDAVIT = "Affidavit"
    
    # Employer & Vocational
    EMPLOYER_REPORT = "Employer Report"
    VOCATIONAL_REHAB = "Vocational Rehab"
    JOB_ANALYSIS = "Job Analysis"
    WORK_CAPACITY = "Work Capacity"
    
    # Generic Fallbacks
    CLINICAL_NOTE = "Clinical Note"
    MEDICAL_REPORT = "Medical Report"
    ADMINISTRATIVE = "Administrative"
    UNKNOWN = "Unknown"

@dataclass
class ExtractionResult:
    """Structured output for all extractions"""
    document_type: str
    document_date: str
    summary_line: str  # Long detailed summary
    short_summary: str = ""  # NEW: Add this line for concise summary
    examiner_name: Optional[str] = None
    specialty: Optional[str] = None
    body_parts: List[str] = field(default_factory=list)
    medications: Optional[Dict[str, Any]] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)






# ============================================================================
# PYDANTIC MODELS (Enhanced with verification fields and WC/GM mode support)
# ============================================================================

class DateReasoning(BaseModel):
    """Structured reasoning about dates found in document"""
    extracted_dates: List[str] = Field(..., description="All dates found in the document in YYYY-MM-DD format")
    date_contexts: Dict[str, str] = Field(..., description="Context around each date found")
    reasoning: str = Field(..., description="Step-by-step reasoning for date assignments")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each date assignment (0.0-1.0)")
    predicted_assignments: Dict[str, str] = Field(..., description="Predicted date assignments")

class BodyPartAnalysis(BaseModel):
    """Analysis for a specific body part with WC/GM mode support"""
    body_part: str = Field(..., description="Specific body part involved if it's a workers comp report else disease/condition")
    diagnosis: str = Field(..., description="Diagnosis for this body part")
    key_concern: str = Field(..., description="Key concern for this body part")
    clinical_summary: str = Field(..., description="Clinical summary of important findings for this body part")
    treatment_plan: Optional[str] = Field(None, description="Treatment plan specific to this body part")
    extracted_recommendation: Optional[str] = Field(None, description="Recommendations for this body part")
    adls_affected: str = Field(..., description="ADLs affected for this body part")
    work_restrictions: str = Field(..., description="Work restrictions for this body part")
    
    # 🆕 WC-SPECIFIC FIELDS
    injury_type: Optional[str] = Field(None, description="sprain/strain/fracture etc. (WC mode)")
    work_relatedness: Optional[str] = Field(None, description="Confirmed/probable/possible (WC mode)")
    permanent_impairment: Optional[str] = Field(None, description="Permanent disability rating (WC mode)")
    mmi_status: Optional[str] = Field(None, description="Maximum Medical Improvement status (WC mode)")
    return_to_work_plan: Optional[str] = Field(None, description="RTW plan/timing (WC mode)")
    
    # 🆕 GM-SPECIFIC FIELDS
    condition_severity: Optional[str] = Field(None, description="mild/moderate/severe, acute/chronic (GM mode)")
    symptoms: Optional[str] = Field(None, description="Key symptoms reported (GM mode)")
    medications: Optional[str] = Field(None, description="Current/prescribed medications (GM focus)")
    chronic_condition: Optional[bool] = Field(None, description="Is this a chronic condition? (GM mode)")
    comorbidities: Optional[str] = Field(None, description="Other existing conditions (GM mode)")
    lifestyle_recommendations: Optional[str] = Field(None, description="Diet/exercise/smoking cessation etc. (GM mode)")
    
    # 🆕 QUALITY OF LIFE IMPACT (both modes, different emphasis)
    pain_level: Optional[str] = Field(None, description="Pain scale 0-10 or description")
    functional_limitations: Optional[str] = Field(None, description="General functional limitations")

class DocumentAnalysis(BaseModel):
    """Structured analysis of medical document matching database schema with WC/GM mode support"""
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
    ur_decision: str = Field('', description="Extracted UR decision keywords/phrases")
    ur_denial_reason: Optional[str] = Field(None, description="UR denial reason if applicable")
    adls_affected: str = Field(..., description="Activities affected in 2-3 words")
    work_restrictions: str = Field(..., description="Work restrictions in 2-3 words")
    consulting_doctor: str = Field(default="Not specified", description="Name of consultant doctor")
    all_doctors: List[str] = Field(default=[], description="List of all doctors mentioned in the document")
    referral_doctor: str = Field(default="Not specified", description="Name of referral doctor if available")
    ai_outcome: str = Field(..., description="AI-generated key outcome prediction keywords/phrases")
    document_type: str = Field(..., description="Type of document")
    summary_points: List[str] = Field(..., description="3-5 key points, each 2-3 words")
    date_reasoning: Optional[DateReasoning] = Field(None, description="Reasoning behind date assignments")
    is_task_needed: bool = Field(default=False, description="If analysis determines any tasks are needed based on pending actions")
    
    # 🆕 WC-SPECIFIC FIELDS
    work_impact: Optional[str] = Field(None, description="Specific work impact details (WC focus)")
    physical_demands: Optional[str] = Field(None, description="Physical demands affected (WC focus)")
    work_capacity: Optional[str] = Field(None, description="Work capacity assessment (WC focus)")
    
    # 🆕 GM-SPECIFIC FIELDS
    daily_living_impact: Optional[str] = Field(None, description="Impact on daily living activities (GM focus)")
    functional_limitations: Optional[str] = Field(None, description="General functional limitations (GM focus)")
    symptom_impact: Optional[str] = Field(None, description="Impact of symptoms on function (GM focus)")
    quality_of_life: Optional[str] = Field(None, description="Overall quality of life impact (GM focus)")
    
    # 🆕 MODE-AWARE FIELDS
    patient_id: Optional[str] = Field(None, description="Medical record number or patient identifier (GM mode)")
    onset_date: Optional[str] = Field(None, description="Symptom onset or condition start date (GM mode)")
    
    # NEW: Verification metadata
    extraction_confidence: float = Field(default=0.0, description="Overall confidence score (0.0-1.0)")
    verified: bool = Field(default=False, description="Whether extraction has been verified")
    verification_notes: List[str] = Field(default=[], description="Any issues found during verification")
    
    # 🆕 FORMATTED SUMMARY FOR BOTH MODES
    formatted_summary: Optional[str] = Field(None, description="Formatted one-line summary appropriate for the mode")

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