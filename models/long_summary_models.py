"""
Pydantic Models for Long Summary Generation
Ensures consistent, structured output without hallucination.
"""
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator


def normalize_claim_number(value: Any) -> Optional[str]:
    """Convert claim_number to string if it's a list/array."""
    if value is None:
        return None
    if isinstance(value, list):
        # Join multiple claim numbers with comma and space
        return ", ".join(str(v) for v in value if v)
    return str(value) if value else None


# ============================================================================
# MEDICAL LONG SUMMARY MODELS
# ============================================================================

class MedicalDocumentOverview(BaseModel):
    """Overview section for medical documents"""
    document_type: str = Field(default="", description="Type of the medical document")
    report_date: str = Field(default="", description="Date of the report in MM/DD/YYYY format. Use '00/00/0000' if not found")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present, otherwise null. If multiple, join with comma.")
    patient_name: str = Field(default="", description="Full name of the patient")
    provider: str = Field(default="", description="Healthcare provider or facility name")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class DoctorInfo(BaseModel):
    """Information about a doctor mentioned in the document"""
    name: str = Field(description="Full name of the doctor")
    title: str = Field(default="", description="Title or credentials (MD, DO, NP, etc.)")
    role: str = Field(default="", description="Role in treatment (treating, consulting, referring, etc.)")


class PatientClinicalInformation(BaseModel):
    """Patient and clinical information section"""
    name: str = Field(default="", description="Patient's full name")
    dob: str = Field(default="", description="Date of birth")
    chief_complaint: str = Field(default="", description="Main reason for visit or chief complaint")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="List of all doctors mentioned in the document")


class DiagnosesAssessments(BaseModel):
    """Diagnoses and assessments section"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis from the document")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="List of secondary diagnoses (up to 3)")
    lab_results: List[str] = Field(default_factory=list, description="Key lab results with values/ranges (up to 5)")
    imaging_findings: List[str] = Field(default_factory=list, description="Key imaging observations")


class TreatmentObservations(BaseModel):
    """Treatment and observations section"""
    current_medications: List[str] = Field(default_factory=list, description="List of current medications with doses if stated")
    clinical_observations: List[str] = Field(default_factory=list, description="Vital signs, exam findings, clinical observations")
    procedures_treatments: List[str] = Field(default_factory=list, description="Recent or ongoing procedures/treatments")


class StatusRecommendations(BaseModel):
    """Status and recommendations section"""
    work_status: str = Field(default="", description="Current work status")
    mmi: str = Field(default="", description="Maximum Medical Improvement status")
    recommendations: List[str] = Field(default_factory=list, description="List of recommendations/next steps (up to 5)")


class MedicalLongSummary(BaseModel):
    """Complete structured medical long summary"""
    content_type: Literal["medical"] = Field(default="medical", description="Type of content")
    document_overview: MedicalDocumentOverview = Field(default_factory=MedicalDocumentOverview, description="Document overview section")
    patient_clinical_info: PatientClinicalInformation = Field(default_factory=PatientClinicalInformation, description="Patient and clinical information")
    critical_findings: List[str] = Field(default_factory=list, description="Up to 5 critical findings from the document")
    diagnoses_assessments: DiagnosesAssessments = Field(default_factory=DiagnosesAssessments, description="Diagnoses and assessments")
    treatment_observations: TreatmentObservations = Field(default_factory=TreatmentObservations, description="Treatment and observations")
    status_recommendations: StatusRecommendations = Field(default_factory=StatusRecommendations, description="Status and recommendations")


# ============================================================================
# ADMINISTRATIVE LONG SUMMARY MODELS
# ============================================================================

class AuthorInfo(BaseModel):
    """Author/signature information - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, utilization reviewers, or other officials mentioned in the document. Leave empty if no signature found.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature if present")


class AdministrativeDocumentOverview(BaseModel):
    """Overview section for administrative documents"""
    document_type: str = Field(default="", description="Type of the administrative document")
    document_date: str = Field(default="", description="Date of the document in MM/DD/YYYY format. Use '00/00/0000' if not found")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present, otherwise null. If multiple, join with comma.")
    purpose: str = Field(default="", description="Purpose of the document")
    author: AuthorInfo = Field(default_factory=AuthorInfo, description="Author/signature information")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class KeyParties(BaseModel):
    """Key parties involved in the document"""
    patient: str = Field(default="", description="Patient name")
    provider: str = Field(default="", description="Provider name/facility")
    referring_party: str = Field(default="", description="Referring party if applicable")


class KeyInformation(BaseModel):
    """Key information section for administrative documents"""
    important_dates: str = Field(default="", description="Important dates mentioned")
    reference_numbers: str = Field(default="", description="Reference numbers (claim numbers, case numbers, etc.)")
    administrative_details: str = Field(default="", description="Other administrative details")


class ActionItems(BaseModel):
    """Action items section"""
    required_actions: List[str] = Field(default_factory=list, description="List of required actions (up to 5)")
    deadlines: str = Field(default="", description="Any deadlines mentioned")


class ContactFollowUp(BaseModel):
    """Contact and follow-up section"""
    contact_information: str = Field(default="", description="Contact information if provided")
    next_steps: str = Field(default="", description="Next steps or follow-up actions")


class AdministrativeLongSummary(BaseModel):
    """Complete structured administrative long summary"""
    content_type: Literal["administrative"] = Field(default="administrative", description="Type of content")
    document_overview: AdministrativeDocumentOverview = Field(default_factory=AdministrativeDocumentOverview, description="Document overview section")
    key_parties: KeyParties = Field(default_factory=KeyParties, description="Key parties involved")
    key_information: KeyInformation = Field(default_factory=KeyInformation, description="Key information")
    action_items: ActionItems = Field(default_factory=ActionItems, description="Action items and deadlines")
    contact_followup: ContactFollowUp = Field(default_factory=ContactFollowUp, description="Contact and follow-up information")


# ============================================================================
# UNIVERSAL LONG SUMMARY MODEL (combines both types)
# ============================================================================

class UniversalLongSummary(BaseModel):
    """
    Universal long summary that can represent either medical or administrative content.
    Uses Optional fields to handle both types flexibly.
    """
    content_type: Literal["medical", "administrative", "unknown"] = Field(
        default="unknown", 
        description="Type of content: 'medical' for clinical documents, 'administrative' for non-clinical"
    )
    
    # === COMMON FIELDS (applicable to both types) ===
    document_type: str = Field(default="", description="Type of the document")
    document_date: str = Field(default="", description="Date of the document. Use '00/00/0000' if not found")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    patient_name: str = Field(default="", description="Patient's full name")
    
    # === MEDICAL-SPECIFIC FIELDS ===
    provider: str = Field(default="", description="Healthcare provider or facility name")
    dob: str = Field(default="", description="Patient's date of birth")
    chief_complaint: str = Field(default="", description="Main reason for visit")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 5)")
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 3)")
    lab_results: List[str] = Field(default_factory=list, description="Key lab results (up to 5)")
    imaging_findings: List[str] = Field(default_factory=list, description="Key imaging findings")
    current_medications: List[str] = Field(default_factory=list, description="Current medications with doses")
    clinical_observations: List[str] = Field(default_factory=list, description="Vital signs, exam findings")
    procedures_treatments: List[str] = Field(default_factory=list, description="Recent/ongoing procedures")
    work_status: str = Field(default="", description="Work status")
    mmi: str = Field(default="", description="Maximum Medical Improvement status")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations (up to 5)")
    
    # === ADMINISTRATIVE-SPECIFIC FIELDS ===
    purpose: str = Field(default="", description="Purpose of the document")
    author_signature: str = Field(default="", description="Name/title of the person who SIGNED the report. Must be the actual signer with physical or electronic signature - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")
    referring_party: str = Field(default="", description="Referring party")
    important_dates: str = Field(default="", description="Important dates mentioned")
    reference_numbers: str = Field(default="", description="Reference numbers")
    administrative_details: str = Field(default="", description="Administrative details")
    required_actions: List[str] = Field(default_factory=list, description="Required actions (up to 5)")
    deadlines: str = Field(default="", description="Deadlines mentioned")
    contact_information: str = Field(default="", description="Contact information")
    next_steps: str = Field(default="", description="Next steps")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


# ============================================================================
# HELPER FUNCTIONS FOR FORMATTING
# ============================================================================

def format_medical_long_summary(summary: UniversalLongSummary) -> str:
    """Format a medical long summary into the expected text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 MEDICAL DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_type:
        lines.append(f"Document Type: {summary.document_type}")
    if summary.document_date:
        lines.append(f"Report Date: {summary.document_date}")
    if summary.claim_number:
        lines.append(f"Claim Number: {summary.claim_number}")
    if summary.patient_name:
        lines.append(f"Patient Name: {summary.patient_name}")
    if summary.author_signature:
        lines.append(f"Author Signature: {summary.author_signature}")
    lines.append("")
    
    # Patient & Clinical Information
    lines.append("👤 PATIENT & CLINICAL INFORMATION")
    lines.append("-" * 50)
    if summary.patient_name:
        lines.append(f"Name: {summary.patient_name}")
    if summary.dob:
        lines.append(f"DOB: {summary.dob}")
    if summary.chief_complaint:
        lines.append(f"Chief Complaint: {summary.chief_complaint}")
    if summary.clinical_history:
        lines.append(f"Clinical History: {summary.clinical_history}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:5]:
            lines.append(f"• {finding}")
        lines.append("")
    
    # Diagnoses & Assessments
    lines.append("🏥 DIAGNOSES & ASSESSMENTS")
    lines.append("-" * 50)
    if summary.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.primary_diagnosis}")
    if summary.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.secondary_diagnoses[:3]:
            lines.append(f"• {dx}")
    if summary.lab_results:
        lines.append("Lab Results:")
        for result in summary.lab_results[:5]:
            lines.append(f"• {result}")
    if summary.imaging_findings:
        lines.append("Imaging Findings:")
        for finding in summary.imaging_findings:
            lines.append(f"• {finding}")
    lines.append("")
    
    # Treatment & Observations
    lines.append("💊 TREATMENT & OBSERVATIONS")
    lines.append("-" * 50)
    if summary.current_medications:
        lines.append("Current Medications:")
        for med in summary.current_medications:
            lines.append(f"• {med}")
    if summary.clinical_observations:
        lines.append("Clinical Observations:")
        for obs in summary.clinical_observations:
            lines.append(f"• {obs}")
    if summary.procedures_treatments:
        lines.append("Procedures/Treatments:")
        for proc in summary.procedures_treatments:
            lines.append(f"• {proc}")
    lines.append("")
    
    # Status & Recommendations
    lines.append("💼 STATUS & RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.work_status:
        lines.append(f"Work Status: {summary.work_status}")
    if summary.mmi:
        lines.append(f"MMI: {summary.mmi}")
    if summary.recommendations:
        lines.append("Recommendations:")
        for rec in summary.recommendations[:5]:
            lines.append(f"• {rec}")
    
    return "\n".join(lines)


def format_administrative_long_summary(summary: UniversalLongSummary) -> str:
    """Format an administrative long summary into the expected text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 ADMINISTRATIVE DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_type:
        lines.append(f"Document Type: {summary.document_type}")
    if summary.document_date:
        lines.append(f"Document Date: {summary.document_date}")
    if summary.claim_number:
        lines.append(f"Claim Number: {summary.claim_number}")
    if summary.purpose:
        lines.append(f"Purpose: {summary.purpose}")
    if summary.author_signature:
        sig_type = f" ({summary.signature_type})" if summary.signature_type else ""
        lines.append(f"Author:")
        lines.append(f"• Signature: {summary.author_signature}{sig_type}")
    lines.append("")
    
    # Key Parties
    lines.append("👥 KEY PARTIES")
    lines.append("-" * 50)
    if summary.patient_name:
        lines.append(f"Patient: {summary.patient_name}")
    if summary.provider:
        lines.append(f"Provider: {summary.provider}")
    if summary.referring_party:
        lines.append(f"Referring Party: {summary.referring_party}")
    lines.append("")
    
    # Key Information
    lines.append("📄 KEY INFORMATION")
    lines.append("-" * 50)
    if summary.important_dates:
        lines.append(f"Important Dates: {summary.important_dates}")
    if summary.reference_numbers:
        lines.append(f"Reference Numbers: {summary.reference_numbers}")
    if summary.administrative_details:
        lines.append(f"Administrative Details: {summary.administrative_details}")
    lines.append("")
    
    # Action Items
    lines.append("✅ ACTION ITEMS")
    lines.append("-" * 50)
    if summary.required_actions:
        lines.append("Required Actions:")
        for action in summary.required_actions[:5]:
            lines.append(f"• {action}")
    if summary.deadlines:
        lines.append(f"Deadlines: {summary.deadlines}")
    lines.append("")
    
    # Contact & Follow-Up
    lines.append("📞 CONTACT & FOLLOW-UP")
    lines.append("-" * 50)
    if summary.contact_information:
        lines.append(f"Contact Information: {summary.contact_information}")
    if summary.next_steps:
        lines.append(f"Next Steps: {summary.next_steps}")
    
    return "\n".join(lines)


def format_universal_long_summary(summary: UniversalLongSummary) -> str:
    """
    Format a universal long summary into text based on content type.
    """
    if summary.content_type == "medical":
        return format_medical_long_summary(summary)
    elif summary.content_type == "administrative":
        return format_administrative_long_summary(summary)
    else:
        # Default to medical format if unknown
        return format_medical_long_summary(summary)


def create_fallback_long_summary(doc_type: str, fallback_date: str) -> UniversalLongSummary:
    """Create a fallback long summary when extraction fails."""
    return UniversalLongSummary(
        content_type="unknown",
        document_type=doc_type,
        document_date=fallback_date,
        patient_name="",
        claim_number=None
    )


# ============================================================================
# PR-2 PROGRESS REPORT SPECIFIC MODELS
# ============================================================================

class PR2ReportOverview(BaseModel):
    """Report overview section for PR-2 Progress Reports"""
    document_type: str = Field(default="PR-2 Progress Report", description="Type of document")
    report_date: str = Field(default="", description="Date of the report. Use fallback date if not found")
    visit_date: str = Field(default="", description="Date of the patient visit")
    treating_physician: str = Field(default="", description="Name of the treating physician")
    specialty: str = Field(default="", description="Physician's specialty")
    time_since_injury: str = Field(default="", description="Time elapsed since injury")
    time_since_last_visit: str = Field(default="", description="Time since last visit")
    author_signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class PR2PatientInformation(BaseModel):
    """Patient information section for PR-2"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    age: str = Field(default="", description="Patient's age")
    date_of_injury: str = Field(default="", description="Date of injury")
    occupation: str = Field(default="", description="Patient's occupation")
    employer: str = Field(default="", description="Patient's employer")
    claims_administrator: str = Field(default="", description="Claims administrator name")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class PR2ChiefComplaint(BaseModel):
    """Chief complaint section for PR-2"""
    primary_complaint: str = Field(default="", description="Primary complaint")
    location: str = Field(default="", description="Location of complaint/pain")
    description: str = Field(default="", description="Description of complaint")


class PR2SubjectiveAssessment(BaseModel):
    """Subjective assessment section for PR-2"""
    current_pain_score: str = Field(default="", description="Current pain score (0-10)")
    previous_pain_score: str = Field(default="", description="Previous pain score")
    symptom_changes: str = Field(default="", description="Changes in symptoms")
    functional_status_patient_reported: str = Field(default="", description="Patient reported functional status")
    patient_compliance: str = Field(default="", description="Patient compliance with treatment")


class PR2ObjectiveFindings(BaseModel):
    """Objective findings section for PR-2"""
    physical_exam_findings: str = Field(default="", description="Physical examination findings")
    rom_measurements: str = Field(default="", description="Range of motion measurements")
    strength_testing: str = Field(default="", description="Strength testing results")
    gait_assessment: str = Field(default="", description="Gait assessment findings")
    neurological_findings: str = Field(default="", description="Neurological findings")
    functional_limitations_observed: List[str] = Field(default_factory=list, description="Observed functional limitations (up to 5)")


class PR2Diagnosis(BaseModel):
    """Diagnosis section for PR-2"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis with ICD-10 if available")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 3)")


class PR2Medications(BaseModel):
    """Medications section for PR-2"""
    current_medications: List[str] = Field(default_factory=list, description="Current medications with doses (up to 8)")
    new_medications: List[str] = Field(default_factory=list, description="Newly prescribed medications (up to 3)")
    dosage_changes: List[str] = Field(default_factory=list, description="Medication dosage changes (up to 3)")


class PR2TreatmentEffectiveness(BaseModel):
    """Treatment effectiveness section for PR-2"""
    patient_response: str = Field(default="", description="Patient's response to treatment")
    functional_gains: str = Field(default="", description="Functional gains achieved")
    objective_improvements: List[str] = Field(default_factory=list, description="Objective improvements noted (up to 5)")
    barriers_to_progress: str = Field(default="", description="Barriers to progress")


class PR2TreatmentAuthorizationRequest(BaseModel):
    """Treatment authorization request section for PR-2"""
    primary_request: str = Field(default="", description="Primary treatment authorization request")
    secondary_requests: List[str] = Field(default_factory=list, description="Secondary requests (up to 3)")
    requested_frequency: str = Field(default="", description="Requested treatment frequency")
    requested_duration: str = Field(default="", description="Requested treatment duration")
    medical_necessity_rationale: str = Field(default="", description="Medical necessity rationale")


class PR2WorkStatus(BaseModel):
    """Work status section for PR-2"""
    current_status: str = Field(default="", description="Current work status (TTD, Modified Duty, Full Duty, etc.)")
    status_effective_date: str = Field(default="", description="Date work status is effective")
    work_limitations: List[str] = Field(default_factory=list, description="Work limitations/restrictions (up to 8)")
    work_status_rationale: str = Field(default="", description="Rationale for work status")
    changes_from_previous_status: str = Field(default="", description="Changes from previous work status")
    expected_return_to_work_date: str = Field(default="", description="Expected return to work date")


class PR2FollowUpPlan(BaseModel):
    """Follow-up plan section for PR-2"""
    next_appointment_date: str = Field(default="", description="Date of next appointment")
    purpose_of_next_visit: str = Field(default="", description="Purpose of next visit")
    specialist_referrals_requested: List[str] = Field(default_factory=list, description="Specialist referrals requested (up to 3)")
    mmi_ps_anticipated_date: str = Field(default="", description="Anticipated MMI/P&S date")
    return_sooner_if: str = Field(default="", description="Conditions to return sooner")


class PR2LongSummary(BaseModel):
    """
    Complete structured PR-2 Progress Report long summary.
    Designed for Workers' Compensation claims processing.
    """
    content_type: Literal["pr2"] = Field(default="pr2", description="Content type for PR-2 documents")
    
    # Main sections
    report_overview: PR2ReportOverview = Field(default_factory=PR2ReportOverview, description="Report overview")
    patient_information: PR2PatientInformation = Field(default_factory=PR2PatientInformation, description="Patient information")
    chief_complaint: PR2ChiefComplaint = Field(default_factory=PR2ChiefComplaint, description="Chief complaint")
    subjective_assessment: PR2SubjectiveAssessment = Field(default_factory=PR2SubjectiveAssessment, description="Subjective assessment")
    objective_findings: PR2ObjectiveFindings = Field(default_factory=PR2ObjectiveFindings, description="Objective findings")
    diagnosis: PR2Diagnosis = Field(default_factory=PR2Diagnosis, description="Diagnosis")
    medications: PR2Medications = Field(default_factory=PR2Medications, description="Medications")
    treatment_effectiveness: PR2TreatmentEffectiveness = Field(default_factory=PR2TreatmentEffectiveness, description="Treatment effectiveness")
    treatment_authorization_request: PR2TreatmentAuthorizationRequest = Field(default_factory=PR2TreatmentAuthorizationRequest, description="Treatment authorization request")
    work_status: PR2WorkStatus = Field(default_factory=PR2WorkStatus, description="Work status")
    follow_up_plan: PR2FollowUpPlan = Field(default_factory=PR2FollowUpPlan, description="Follow-up plan")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 5)")


def format_pr2_long_summary(summary: PR2LongSummary) -> str:
    """Format a PR-2 long summary into the expected text format."""
    lines = []
    
    # Report Overview
    lines.append("📋 REPORT OVERVIEW")
    lines.append("-" * 50)
    if summary.report_overview.document_type:
        lines.append(f"Document Type: {summary.report_overview.document_type}")
    if summary.report_overview.report_date:
        lines.append(f"Report Date: {summary.report_overview.report_date}")
    if summary.report_overview.visit_date:
        lines.append(f"Visit Date: {summary.report_overview.visit_date}")
    if summary.report_overview.treating_physician:
        lines.append(f"Treating Physician: {summary.report_overview.treating_physician}")
    if summary.report_overview.specialty:
        lines.append(f"Specialty: {summary.report_overview.specialty}")
    if summary.report_overview.time_since_injury:
        lines.append(f"Time Since Injury: {summary.report_overview.time_since_injury}")
    if summary.report_overview.time_since_last_visit:
        lines.append(f"Time Since Last Visit: {summary.report_overview.time_since_last_visit}")
    if summary.report_overview.author_signature:
        sig_type = f" ({summary.report_overview.signature_type})" if summary.report_overview.signature_type else ""
        lines.append(f"Author:")
        lines.append(f"• Signature: {summary.report_overview.author_signature}{sig_type}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_information.name:
        lines.append(f"Name: {summary.patient_information.name}")
    if summary.patient_information.date_of_birth:
        lines.append(f"Date of Birth: {summary.patient_information.date_of_birth}")
    if summary.patient_information.age:
        lines.append(f"Age: {summary.patient_information.age}")
    if summary.patient_information.date_of_injury:
        lines.append(f"Date of Injury: {summary.patient_information.date_of_injury}")
    if summary.patient_information.occupation:
        lines.append(f"Occupation: {summary.patient_information.occupation}")
    if summary.patient_information.employer:
        lines.append(f"Employer: {summary.patient_information.employer}")
    if summary.patient_information.claims_administrator:
        lines.append(f"Claims Administrator: {summary.patient_information.claims_administrator}")
    if summary.patient_information.claim_number:
        lines.append(f"Claim Number: {summary.patient_information.claim_number}")
    if summary.patient_information.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.patient_information.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Chief Complaint
    lines.append("🎯 CHIEF COMPLAINT")
    lines.append("-" * 50)
    if summary.chief_complaint.primary_complaint:
        lines.append(f"Primary Complaint: {summary.chief_complaint.primary_complaint}")
    if summary.chief_complaint.location:
        lines.append(f"Location: {summary.chief_complaint.location}")
    if summary.chief_complaint.description:
        lines.append(f"Description: {summary.chief_complaint.description}")
    lines.append("")
    
    # Subjective Assessment
    lines.append("💬 SUBJECTIVE ASSESSMENT")
    lines.append("-" * 50)
    if summary.subjective_assessment.current_pain_score:
        lines.append(f"Current Pain Score: {summary.subjective_assessment.current_pain_score}")
    if summary.subjective_assessment.previous_pain_score:
        lines.append(f"Previous Pain Score: {summary.subjective_assessment.previous_pain_score}")
    if summary.subjective_assessment.symptom_changes:
        lines.append(f"Symptom Changes: {summary.subjective_assessment.symptom_changes}")
    if summary.subjective_assessment.functional_status_patient_reported:
        lines.append(f"Functional Status (Patient Reported): {summary.subjective_assessment.functional_status_patient_reported}")
    if summary.subjective_assessment.patient_compliance:
        lines.append(f"Patient Compliance: {summary.subjective_assessment.patient_compliance}")
    lines.append("")
    
    # Objective Findings
    lines.append("🔬 OBJECTIVE FINDINGS")
    lines.append("-" * 50)
    if summary.objective_findings.physical_exam_findings:
        lines.append(f"Physical Exam Findings: {summary.objective_findings.physical_exam_findings}")
    if summary.objective_findings.rom_measurements:
        lines.append(f"ROM Measurements: {summary.objective_findings.rom_measurements}")
    if summary.objective_findings.strength_testing:
        lines.append(f"Strength Testing: {summary.objective_findings.strength_testing}")
    if summary.objective_findings.gait_assessment:
        lines.append(f"Gait Assessment: {summary.objective_findings.gait_assessment}")
    if summary.objective_findings.neurological_findings:
        lines.append(f"Neurological Findings: {summary.objective_findings.neurological_findings}")
    if summary.objective_findings.functional_limitations_observed:
        lines.append("Functional Limitations Observed:")
        for limitation in summary.objective_findings.functional_limitations_observed[:5]:
            lines.append(f"• {limitation}")
    lines.append("")
    
    # Diagnosis
    lines.append("🏥 DIAGNOSIS")
    lines.append("-" * 50)
    if summary.diagnosis.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.diagnosis.primary_diagnosis}")
    if summary.diagnosis.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.diagnosis.secondary_diagnoses[:3]:
            lines.append(f"• {dx}")
    lines.append("")
    
    # Medications
    lines.append("💊 MEDICATIONS")
    lines.append("-" * 50)
    if summary.medications.current_medications:
        lines.append("Current Medications:")
        for med in summary.medications.current_medications[:8]:
            lines.append(f"• {med}")
    if summary.medications.new_medications:
        lines.append("New Medications:")
        for med in summary.medications.new_medications[:3]:
            lines.append(f"• {med}")
    if summary.medications.dosage_changes:
        lines.append("Dosage Changes:")
        for change in summary.medications.dosage_changes[:3]:
            lines.append(f"• {change}")
    lines.append("")
    
    # Treatment Effectiveness
    lines.append("📈 TREATMENT EFFECTIVENESS")
    lines.append("-" * 50)
    if summary.treatment_effectiveness.patient_response:
        lines.append(f"Patient Response: {summary.treatment_effectiveness.patient_response}")
    if summary.treatment_effectiveness.functional_gains:
        lines.append(f"Functional Gains: {summary.treatment_effectiveness.functional_gains}")
    if summary.treatment_effectiveness.objective_improvements:
        lines.append("Objective Improvements:")
        for improvement in summary.treatment_effectiveness.objective_improvements[:5]:
            lines.append(f"• {improvement}")
    if summary.treatment_effectiveness.barriers_to_progress:
        lines.append(f"Barriers to Progress: {summary.treatment_effectiveness.barriers_to_progress}")
    lines.append("")
    
    # Treatment Authorization Request
    lines.append("✅ TREATMENT AUTHORIZATION REQUEST")
    lines.append("-" * 50)
    if summary.treatment_authorization_request.primary_request:
        lines.append(f"Primary Request: {summary.treatment_authorization_request.primary_request}")
    if summary.treatment_authorization_request.secondary_requests:
        lines.append("Secondary Requests:")
        for req in summary.treatment_authorization_request.secondary_requests[:3]:
            lines.append(f"• {req}")
    if summary.treatment_authorization_request.requested_frequency:
        lines.append(f"Requested Frequency: {summary.treatment_authorization_request.requested_frequency}")
    if summary.treatment_authorization_request.requested_duration:
        lines.append(f"Requested Duration: {summary.treatment_authorization_request.requested_duration}")
    if summary.treatment_authorization_request.medical_necessity_rationale:
        lines.append(f"Medical Necessity Rationale: {summary.treatment_authorization_request.medical_necessity_rationale}")
    lines.append("")
    
    # Work Status
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    if summary.work_status.current_status:
        lines.append(f"Current Status: {summary.work_status.current_status}")
    if summary.work_status.status_effective_date:
        lines.append(f"Status Effective Date: {summary.work_status.status_effective_date}")
    if summary.work_status.work_limitations:
        lines.append("Work Limitations:")
        for limitation in summary.work_status.work_limitations[:8]:
            lines.append(f"• {limitation}")
    if summary.work_status.work_status_rationale:
        lines.append(f"Work Status Rationale: {summary.work_status.work_status_rationale}")
    if summary.work_status.changes_from_previous_status:
        lines.append(f"Changes from Previous Status: {summary.work_status.changes_from_previous_status}")
    if summary.work_status.expected_return_to_work_date:
        lines.append(f"Expected Return to Work Date: {summary.work_status.expected_return_to_work_date}")
    lines.append("")
    
    # Follow-Up Plan
    lines.append("📅 FOLLOW-UP PLAN")
    lines.append("-" * 50)
    if summary.follow_up_plan.next_appointment_date:
        lines.append(f"Next Appointment Date: {summary.follow_up_plan.next_appointment_date}")
    if summary.follow_up_plan.purpose_of_next_visit:
        lines.append(f"Purpose of Next Visit: {summary.follow_up_plan.purpose_of_next_visit}")
    if summary.follow_up_plan.specialist_referrals_requested:
        lines.append("Specialist Referrals Requested:")
        for referral in summary.follow_up_plan.specialist_referrals_requested[:3]:
            lines.append(f"• {referral}")
    if summary.follow_up_plan.mmi_ps_anticipated_date:
        lines.append(f"MMI/P&S Anticipated Date: {summary.follow_up_plan.mmi_ps_anticipated_date}")
    if summary.follow_up_plan.return_sooner_if:
        lines.append(f"Return Sooner If: {summary.follow_up_plan.return_sooner_if}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:5]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_pr2_summary(doc_type: str, fallback_date: str) -> PR2LongSummary:
    """Create a fallback PR-2 long summary when extraction fails."""
    return PR2LongSummary(
        content_type="pr2",
        report_overview=PR2ReportOverview(
            document_type=doc_type,
            report_date=fallback_date
        )
    )


# ============================================================================
# ADMINISTRATIVE/LEGAL DOCUMENT SPECIFIC MODELS
# ============================================================================

class AdminDocumentOverview(BaseModel):
    """Document overview section for administrative/legal documents"""
    document_type: str = Field(default="", description="Type of administrative document")
    document_date: str = Field(default="", description="Date of the document")
    subject: str = Field(default="", description="Subject of the document")
    purpose: str = Field(default="", description="Purpose of the document")
    document_id: str = Field(default="", description="Document ID from headers/footers")


class AdminPartyInfo(BaseModel):
    """Information about a party (sender/recipient) in the document"""
    name: str = Field(default="", description="Name of the party")
    organization: str = Field(default="", description="Organization name")
    title: str = Field(default="", description="Title/position")


class AdminLegalRepresentation(BaseModel):
    """Legal representation information"""
    representative: str = Field(default="", description="Legal representative name")
    firm: str = Field(default="", description="Law firm name")


class AdminPartiesInvolved(BaseModel):
    """All parties involved in the administrative document"""
    patient_details: str = Field(default="", description="Patient details if applicable")
    from_party: AdminPartyInfo = Field(default_factory=AdminPartyInfo, description="Sender information")
    to_party: AdminPartyInfo = Field(default_factory=AdminPartyInfo, description="Recipient information")
    author_signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, insurance representatives, or other officials.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")
    legal_representation: AdminLegalRepresentation = Field(default_factory=AdminLegalRepresentation, description="Legal representation")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class AdminKeyDatesDeadlines(BaseModel):
    """Key dates and deadlines section"""
    response_deadline: str = Field(default="", description="Response deadline")
    hearing_date: str = Field(default="", description="Hearing date if applicable")
    appointment_date: str = Field(default="", description="Appointment date if applicable")
    time_sensitive_requirements: List[str] = Field(default_factory=list, description="Time-sensitive requirements (up to 3)")


class AdminContent(BaseModel):
    """Administrative content section"""
    primary_subject: str = Field(default="", description="Primary subject matter")
    key_points: str = Field(default="", description="Key points from the document")
    current_status: str = Field(default="", description="Current status")
    incident_details: str = Field(default="", description="Incident details if applicable (truncated to 200 chars)")


class AdminActionItemsRequirements(BaseModel):
    """Action items and requirements section"""
    required_responses: List[str] = Field(default_factory=list, description="Required responses (up to 5)")
    documentation_required: List[str] = Field(default_factory=list, description="Documentation required (up to 5)")
    specific_actions: List[str] = Field(default_factory=list, description="Specific actions needed (up to 5)")


class AdminLegalProceduralElements(BaseModel):
    """Legal and procedural elements section"""
    legal_demands: List[str] = Field(default_factory=list, description="Legal demands (up to 3)")
    next_steps: List[str] = Field(default_factory=list, description="Next steps (up to 3)")
    consequences_of_non_compliance: str = Field(default="", description="Consequences of non-compliance")


class AdminMedicalClaimInfo(BaseModel):
    """Medical and claim information section for administrative documents"""
    claim_number: Optional[str] = Field(default=None, description="Claim number. If multiple, join with comma.")
    case_number: str = Field(default="", description="Case number")
    work_status: str = Field(default="", description="Work status")
    disability_information: str = Field(default="", description="Disability information")
    treatment_authorizations: List[str] = Field(default_factory=list, description="Treatment authorizations (up to 3)")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class AdminContactFollowUp(BaseModel):
    """Contact and follow-up section for administrative documents"""
    contact_person: str = Field(default="", description="Contact person name")
    phone: str = Field(default="", description="Phone number")
    email: str = Field(default="", description="Email address")
    submission_address: str = Field(default="", description="Submission/mailing address")
    response_format: str = Field(default="", description="Required response format")


class AdminLongSummary(BaseModel):
    """
    Complete structured Administrative/Legal document long summary.
    Designed for attorney letters, NCM notes, employer reports, disability forms, legal correspondence.
    """
    content_type: Literal["administrative"] = Field(default="administrative", description="Content type for administrative documents")
    
    # Main sections
    document_overview: AdminDocumentOverview = Field(default_factory=AdminDocumentOverview, description="Document overview")
    parties_involved: AdminPartiesInvolved = Field(default_factory=AdminPartiesInvolved, description="Parties involved")
    key_dates_deadlines: AdminKeyDatesDeadlines = Field(default_factory=AdminKeyDatesDeadlines, description="Key dates and deadlines")
    administrative_content: AdminContent = Field(default_factory=AdminContent, description="Administrative content")
    action_items_requirements: AdminActionItemsRequirements = Field(default_factory=AdminActionItemsRequirements, description="Action items and requirements")
    legal_procedural_elements: AdminLegalProceduralElements = Field(default_factory=AdminLegalProceduralElements, description="Legal and procedural elements")
    medical_claim_info: AdminMedicalClaimInfo = Field(default_factory=AdminMedicalClaimInfo, description="Medical and claim information")
    contact_followup: AdminContactFollowUp = Field(default_factory=AdminContactFollowUp, description="Contact and follow-up")
    critical_administrative_findings: List[str] = Field(default_factory=list, description="Critical administrative findings (up to 8)")


def format_admin_long_summary(summary: AdminLongSummary) -> str:
    """Format an administrative long summary into the expected text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 ADMINISTRATIVE DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_overview.document_type:
        lines.append(f"Document Type: {summary.document_overview.document_type}")
    if summary.document_overview.document_date:
        lines.append(f"Document Date: {summary.document_overview.document_date}")
    if summary.document_overview.subject:
        lines.append(f"Subject: {summary.document_overview.subject}")
    if summary.document_overview.purpose:
        lines.append(f"Purpose: {summary.document_overview.purpose}")
    if summary.document_overview.document_id:
        lines.append(f"Document ID: {summary.document_overview.document_id}")
    lines.append("")
    
    # Parties Involved
    lines.append("👥 PARTIES INVOLVED")
    lines.append("-" * 50)
    if summary.parties_involved.patient_details:
        lines.append(f"Patient Details: {summary.parties_involved.patient_details}")
    lines.append("")
    if summary.parties_involved.from_party.name or summary.parties_involved.from_party.organization:
        lines.append("From:")
        if summary.parties_involved.from_party.name:
            lines.append(f"  Name: {summary.parties_involved.from_party.name}")
        if summary.parties_involved.from_party.organization:
            lines.append(f"  Organization: {summary.parties_involved.from_party.organization}")
        if summary.parties_involved.from_party.title:
            lines.append(f"  Title: {summary.parties_involved.from_party.title}")
    if summary.parties_involved.to_party.name or summary.parties_involved.to_party.organization:
        lines.append("To:")
        if summary.parties_involved.to_party.name:
            lines.append(f"  Name: {summary.parties_involved.to_party.name}")
        if summary.parties_involved.to_party.organization:
            lines.append(f"  Organization: {summary.parties_involved.to_party.organization}")
    if summary.parties_involved.author_signature:
        sig_type = f" ({summary.parties_involved.signature_type})" if summary.parties_involved.signature_type else ""
        lines.append("Author:")
        lines.append(f"• Signature: {summary.parties_involved.author_signature}{sig_type}")
    if summary.parties_involved.legal_representation.representative or summary.parties_involved.legal_representation.firm:
        lines.append("Legal Representation:")
        if summary.parties_involved.legal_representation.representative:
            lines.append(f"  Representative: {summary.parties_involved.legal_representation.representative}")
        if summary.parties_involved.legal_representation.firm:
            lines.append(f"  Firm: {summary.parties_involved.legal_representation.firm}")
    if summary.parties_involved.claim_number:
        lines.append(f"Claim Number: {summary.parties_involved.claim_number}")
    if summary.parties_involved.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.parties_involved.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Key Dates & Deadlines
    lines.append("📅 KEY DATES & DEADLINES")
    lines.append("-" * 50)
    if summary.key_dates_deadlines.response_deadline:
        lines.append(f"Response Deadline: {summary.key_dates_deadlines.response_deadline}")
    if summary.key_dates_deadlines.hearing_date:
        lines.append(f"Hearing Date: {summary.key_dates_deadlines.hearing_date}")
    if summary.key_dates_deadlines.appointment_date:
        lines.append(f"Appointment Date: {summary.key_dates_deadlines.appointment_date}")
    if summary.key_dates_deadlines.time_sensitive_requirements:
        lines.append("Time-Sensitive Requirements:")
        for req in summary.key_dates_deadlines.time_sensitive_requirements[:3]:
            lines.append(f"• {req}")
    lines.append("")
    
    # Administrative Content
    lines.append("📄 ADMINISTRATIVE CONTENT")
    lines.append("-" * 50)
    if summary.administrative_content.primary_subject:
        lines.append(f"Primary Subject: {summary.administrative_content.primary_subject}")
    if summary.administrative_content.key_points:
        lines.append(f"Key Points: {summary.administrative_content.key_points}")
    if summary.administrative_content.current_status:
        lines.append(f"Current Status: {summary.administrative_content.current_status}")
    if summary.administrative_content.incident_details:
        lines.append(f"Incident Details: {summary.administrative_content.incident_details}")
    lines.append("")
    
    # Action Items & Requirements
    lines.append("✅ ACTION ITEMS & REQUIREMENTS")
    lines.append("-" * 50)
    if summary.action_items_requirements.required_responses:
        lines.append("Required Responses:")
        for resp in summary.action_items_requirements.required_responses[:5]:
            lines.append(f"• {resp}")
    if summary.action_items_requirements.documentation_required:
        lines.append("Documentation Required:")
        for doc in summary.action_items_requirements.documentation_required[:5]:
            lines.append(f"• {doc}")
    if summary.action_items_requirements.specific_actions:
        lines.append("Specific Actions:")
        for action in summary.action_items_requirements.specific_actions[:5]:
            lines.append(f"• {action}")
    lines.append("")
    
    # Legal & Procedural Elements
    lines.append("⚖️ LEGAL & PROCEDURAL ELEMENTS")
    lines.append("-" * 50)
    if summary.legal_procedural_elements.legal_demands:
        lines.append("Legal Demands:")
        for demand in summary.legal_procedural_elements.legal_demands[:3]:
            lines.append(f"• {demand}")
    if summary.legal_procedural_elements.next_steps:
        lines.append("Next Steps:")
        for step in summary.legal_procedural_elements.next_steps[:3]:
            lines.append(f"• {step}")
    if summary.legal_procedural_elements.consequences_of_non_compliance:
        lines.append(f"Consequences of Non-Compliance: {summary.legal_procedural_elements.consequences_of_non_compliance}")
    lines.append("")
    
    # Medical & Claim Information
    lines.append("🏥 MEDICAL & CLAIM INFORMATION")
    lines.append("-" * 50)
    if summary.medical_claim_info.claim_number:
        lines.append(f"Claim Number: {summary.medical_claim_info.claim_number}")
    if summary.medical_claim_info.case_number:
        lines.append(f"Case Number: {summary.medical_claim_info.case_number}")
    if summary.medical_claim_info.work_status:
        lines.append(f"Work Status: {summary.medical_claim_info.work_status}")
    if summary.medical_claim_info.disability_information:
        lines.append(f"Disability Information: {summary.medical_claim_info.disability_information}")
    if summary.medical_claim_info.treatment_authorizations:
        lines.append("Treatment Authorizations:")
        for auth in summary.medical_claim_info.treatment_authorizations[:3]:
            lines.append(f"• {auth}")
    lines.append("")
    
    # Contact & Follow-Up
    lines.append("📞 CONTACT & FOLLOW-UP")
    lines.append("-" * 50)
    if summary.contact_followup.contact_person:
        lines.append(f"Contact Person: {summary.contact_followup.contact_person}")
    if summary.contact_followup.phone:
        lines.append(f"Phone: {summary.contact_followup.phone}")
    if summary.contact_followup.email:
        lines.append(f"Email: {summary.contact_followup.email}")
    if summary.contact_followup.submission_address:
        lines.append(f"Submission Address: {summary.contact_followup.submission_address}")
    if summary.contact_followup.response_format:
        lines.append(f"Response Format: {summary.contact_followup.response_format}")
    lines.append("")
    
    # Critical Administrative Findings
    if summary.critical_administrative_findings:
        lines.append("🚨 CRITICAL ADMINISTRATIVE FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_administrative_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_admin_summary(doc_type: str, fallback_date: str) -> AdminLongSummary:
    """Create a fallback administrative long summary when extraction fails."""
    return AdminLongSummary(
        content_type="administrative",
        document_overview=AdminDocumentOverview(
            document_type=doc_type,
            document_date=fallback_date
        )
    )

# ============================================================================
# CLINICAL PROGRESS NOTE / THERAPY REPORT SPECIFIC MODELS
# ============================================================================

class ClinicalEncounterOverview(BaseModel):
    """Clinical encounter overview section"""
    note_type: str = Field(default="", description="Type of clinical note (Progress Note, PT, OT, Chiropractic, etc.)")
    visit_date: str = Field(default="", description="Date of the visit")
    visit_type: str = Field(default="", description="Type of visit (initial, follow-up, re-evaluation, etc.)")
    duration: str = Field(default="", description="Duration of the visit")
    facility: str = Field(default="", description="Facility/clinic name")


class ClinicalProviderInfo(BaseModel):
    """Provider information section"""
    treating_provider: str = Field(default="", description="Treating provider name")
    credentials: str = Field(default="", description="Provider credentials (PT, OT, DC, MD, etc.)")
    specialty: str = Field(default="", description="Provider specialty")


class ClinicalSubjectiveFindings(BaseModel):
    """Subjective findings section"""
    chief_complaint: str = Field(default="", description="Main reason for visit/chief complaint")
    pain_description: str = Field(default="", description="Pain description including location, intensity, character")
    functional_limitations: List[str] = Field(default_factory=list, description="Functional limitations reported by patient (up to 5)")


class ClinicalObjectiveExamination(BaseModel):
    """Objective examination findings section"""
    range_of_motion: List[str] = Field(default_factory=list, description="Range of motion findings (up to 5)")
    manual_muscle_testing: List[str] = Field(default_factory=list, description="Manual muscle testing results (up to 5)")
    special_tests: List[str] = Field(default_factory=list, description="Special tests performed with results (up to 3)")


class ClinicalTreatmentProvided(BaseModel):
    """Treatment provided section"""
    treatment_techniques: List[str] = Field(default_factory=list, description="Treatment techniques used with CPT codes if available (up to 5)")
    therapeutic_exercises: List[str] = Field(default_factory=list, description="Therapeutic exercises performed (up to 5)")
    modalities_used: List[str] = Field(default_factory=list, description="Modalities used with parameters (up to 5)")


class ClinicalAssessment(BaseModel):
    """Clinical assessment section"""
    assessment: str = Field(default="", description="Clinical assessment")
    progress: str = Field(default="", description="Progress towards goals")
    clinical_impression: str = Field(default="", description="Clinical impression")
    prognosis: str = Field(default="", description="Prognosis for recovery")


class ClinicalTreatmentPlan(BaseModel):
    """Treatment plan section"""
    short_term_goals: List[str] = Field(default_factory=list, description="Short-term goals (up to 3)")
    home_exercise_program: List[str] = Field(default_factory=list, description="Home exercise program (up to 3)")
    frequency_duration: str = Field(default="", description="Recommended treatment frequency and duration")
    next_appointment: str = Field(default="", description="Next appointment date/time")


class ClinicalWorkStatus(BaseModel):
    """Work status and functional capacity section"""
    current_status: str = Field(default="", description="Current work status (full duty, modified, off work, etc.)")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions (up to 5)")
    functional_capacity: str = Field(default="", description="Functional capacity assessment")


class ClinicalOutcomeMeasures(BaseModel):
    """Outcome measures and progress tracking section"""
    pain_scale: str = Field(default="", description="Pain scale rating (e.g., 7/10)")
    functional_scores: List[str] = Field(default_factory=list, description="Functional scores/outcome measures (up to 3)")


class ClinicalSignatureAuthor(BaseModel):
    """Signature and author section - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class ClinicalLongSummary(BaseModel):
    """
    Complete structured Clinical Progress Note / Therapy Report long summary.
    Designed for PT, OT, Chiropractic, Pain Management, Psychiatry, Nursing notes.
    """
    content_type: Literal["clinical"] = Field(default="clinical", description="Content type for clinical documents")
    
    # Main sections matching the clinical extractor prompt
    encounter_overview: ClinicalEncounterOverview = Field(default_factory=ClinicalEncounterOverview, description="Clinical encounter overview")
    provider_info: ClinicalProviderInfo = Field(default_factory=ClinicalProviderInfo, description="Provider information")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")
    subjective_findings: ClinicalSubjectiveFindings = Field(default_factory=ClinicalSubjectiveFindings, description="Subjective findings")
    objective_examination: ClinicalObjectiveExamination = Field(default_factory=ClinicalObjectiveExamination, description="Objective examination findings")
    treatment_provided: ClinicalTreatmentProvided = Field(default_factory=ClinicalTreatmentProvided, description="Treatment provided")
    clinical_assessment: ClinicalAssessment = Field(default_factory=ClinicalAssessment, description="Clinical assessment")
    treatment_plan: ClinicalTreatmentPlan = Field(default_factory=ClinicalTreatmentPlan, description="Treatment plan")
    work_status: ClinicalWorkStatus = Field(default_factory=ClinicalWorkStatus, description="Work status and functional capacity")
    outcome_measures: ClinicalOutcomeMeasures = Field(default_factory=ClinicalOutcomeMeasures, description="Outcome measures")
    signature_author: ClinicalSignatureAuthor = Field(default_factory=ClinicalSignatureAuthor, description="Signature and author")
    critical_clinical_findings: List[str] = Field(default_factory=list, description="Critical clinical findings (up to 8)")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


def format_clinical_long_summary(summary: ClinicalLongSummary) -> str:
    """Format a clinical long summary into the expected text format."""
    lines = []
    
    # Clinical Encounter Overview
    lines.append("📋 CLINICAL ENCOUNTER OVERVIEW")
    lines.append("-" * 50)
    if summary.encounter_overview.note_type:
        lines.append(f"Note Type: {summary.encounter_overview.note_type}")
    if summary.encounter_overview.visit_date:
        lines.append(f"Visit Date: {summary.encounter_overview.visit_date}")
    if summary.encounter_overview.visit_type:
        lines.append(f"Visit Type: {summary.encounter_overview.visit_type}")
    if summary.encounter_overview.duration:
        lines.append(f"Duration: {summary.encounter_overview.duration}")
    if summary.encounter_overview.facility:
        lines.append(f"Facility: {summary.encounter_overview.facility}")
    lines.append("")
    
    # Provider Information
    lines.append("👨‍⚕️ PROVIDER INFORMATION")
    lines.append("-" * 50)
    if summary.provider_info.treating_provider:
        lines.append(f"Treating Provider: {summary.provider_info.treating_provider}")
    if summary.provider_info.credentials:
        lines.append(f"Credentials: {summary.provider_info.credentials}")
    if summary.provider_info.specialty:
        lines.append(f"Specialty: {summary.provider_info.specialty}")
    if summary.claim_number:
        lines.append(f"Claim Number: {summary.claim_number}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Subjective Findings
    lines.append("🗣️ SUBJECTIVE FINDINGS")
    lines.append("-" * 50)
    if summary.subjective_findings.chief_complaint:
        lines.append(f"Chief Complaint: {summary.subjective_findings.chief_complaint}")
    if summary.subjective_findings.pain_description:
        lines.append(f"Pain: {summary.subjective_findings.pain_description}")
    if summary.subjective_findings.functional_limitations:
        lines.append("Functional Limitations:")
        for limitation in summary.subjective_findings.functional_limitations[:5]:
            lines.append(f"• {limitation}")
    lines.append("")
    
    # Objective Examination
    lines.append("🔍 OBJECTIVE EXAMINATION")
    lines.append("-" * 50)
    if summary.objective_examination.range_of_motion:
        lines.append("Range of Motion:")
        for rom in summary.objective_examination.range_of_motion[:5]:
            lines.append(f"• {rom}")
    if summary.objective_examination.manual_muscle_testing:
        lines.append("Manual Muscle Testing:")
        for mmt in summary.objective_examination.manual_muscle_testing[:5]:
            lines.append(f"• {mmt}")
    if summary.objective_examination.special_tests:
        lines.append("Special Tests:")
        for test in summary.objective_examination.special_tests[:3]:
            lines.append(f"• {test}")
    lines.append("")
    
    # Treatment Provided
    lines.append("💆 TREATMENT PROVIDED")
    lines.append("-" * 50)
    if summary.treatment_provided.treatment_techniques:
        lines.append("Treatment Techniques:")
        for tech in summary.treatment_provided.treatment_techniques[:5]:
            lines.append(f"• {tech}")
    if summary.treatment_provided.therapeutic_exercises:
        lines.append("Therapeutic Exercises:")
        for exercise in summary.treatment_provided.therapeutic_exercises[:5]:
            lines.append(f"• {exercise}")
    if summary.treatment_provided.modalities_used:
        lines.append("Modalities Used:")
        for mod in summary.treatment_provided.modalities_used[:5]:
            lines.append(f"• {mod}")
    lines.append("")
    
    # Clinical Assessment
    lines.append("🏥 CLINICAL ASSESSMENT")
    lines.append("-" * 50)
    if summary.clinical_assessment.assessment:
        lines.append(f"Assessment: {summary.clinical_assessment.assessment}")
    if summary.clinical_assessment.progress:
        lines.append(f"Progress: {summary.clinical_assessment.progress}")
    if summary.clinical_assessment.clinical_impression:
        lines.append(f"Clinical Impression: {summary.clinical_assessment.clinical_impression}")
    if summary.clinical_assessment.prognosis:
        lines.append(f"Prognosis: {summary.clinical_assessment.prognosis}")
    lines.append("")
    
    # Treatment Plan
    lines.append("🎯 TREATMENT PLAN")
    lines.append("-" * 50)
    if summary.treatment_plan.short_term_goals:
        lines.append("Short-term Goals:")
        for goal in summary.treatment_plan.short_term_goals[:3]:
            lines.append(f"• {goal}")
    if summary.treatment_plan.home_exercise_program:
        lines.append("Home Exercise Program:")
        for exercise in summary.treatment_plan.home_exercise_program[:3]:
            lines.append(f"• {exercise}")
    if summary.treatment_plan.frequency_duration:
        lines.append(f"Frequency/Duration: {summary.treatment_plan.frequency_duration}")
    if summary.treatment_plan.next_appointment:
        lines.append(f"Next Appointment: {summary.treatment_plan.next_appointment}")
    lines.append("")
    
    # Work Status
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    if summary.work_status.current_status:
        lines.append(f"Current Status: {summary.work_status.current_status}")
    if summary.work_status.work_restrictions:
        lines.append("Work Restrictions:")
        for restriction in summary.work_status.work_restrictions[:5]:
            lines.append(f"• {restriction}")
    if summary.work_status.functional_capacity:
        lines.append(f"Functional Capacity: {summary.work_status.functional_capacity}")
    lines.append("")
    
    # Outcome Measures
    lines.append("📊 OUTCOME MEASURES")
    lines.append("-" * 50)
    if summary.outcome_measures.pain_scale:
        lines.append(f"Pain Scale: {summary.outcome_measures.pain_scale}")
    if summary.outcome_measures.functional_scores:
        lines.append("Functional Scores:")
        for score in summary.outcome_measures.functional_scores[:3]:
            lines.append(f"• {score}")
    lines.append("")
    
    # Signature & Author
    lines.append("✍️ SIGNATURE & AUTHOR")
    lines.append("-" * 50)
    if summary.signature_author.signature:
        sig_type = f" ({summary.signature_author.signature_type})" if summary.signature_author.signature_type else ""
        lines.append("Author:")
        lines.append(f"• Signature: {summary.signature_author.signature}{sig_type}")
    lines.append("")
    
    # Critical Clinical Findings
    if summary.critical_clinical_findings:
        lines.append("🚨 CRITICAL CLINICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_clinical_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_clinical_summary(doc_type: str, fallback_date: str) -> ClinicalLongSummary:
    """Create a fallback clinical long summary when extraction fails."""
    return ClinicalLongSummary(
        content_type="clinical",
        encounter_overview=ClinicalEncounterOverview(
            note_type=doc_type,
            visit_date=fallback_date
        )
    )


# ============================================================================
# SPECIALIST CONSULTATION REPORT SPECIFIC MODELS
# ============================================================================

class ConsultOverview(BaseModel):
    """Consultation overview section"""
    document_type: str = Field(default="", description="Type of consultation document")
    consultation_date: str = Field(default="", description="Date of the consultation")
    consulting_physician: str = Field(default="", description="Name of the consulting physician")
    specialty: str = Field(default="", description="Specialty of the consulting physician")
    referring_physician: str = Field(default="", description="Name of the referring physician")


class ConsultPatientInfo(BaseModel):
    """Patient information section for consultation"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    date_of_injury: str = Field(default="", description="Date of injury")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class ConsultChiefComplaint(BaseModel):
    """Chief complaint section"""
    primary_complaint: str = Field(default="", description="Primary complaint from the patient")
    location: str = Field(default="", description="Location of the complaint/pain")
    duration: str = Field(default="", description="Duration of the complaint")
    radiation_pattern: str = Field(default="", description="Radiation pattern of the pain if applicable")


class ConsultDiagnosis(BaseModel):
    """Diagnosis and assessment section"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    icd_10_code: str = Field(default="", description="ICD-10 code for primary diagnosis")
    certainty: str = Field(default="", description="Diagnostic certainty level")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 5)")
    causation: str = Field(default="", description="Causation statement linking injury to diagnosis")


class ConsultClinicalHistory(BaseModel):
    """Clinical history and symptoms section"""
    pain_quality: str = Field(default="", description="Quality/character of pain")
    pain_location: str = Field(default="", description="Location of pain")
    radiation: str = Field(default="", description="Radiation pattern of pain")
    aggravating_factors: List[str] = Field(default_factory=list, description="Factors that aggravate symptoms (up to 5)")
    alleviating_factors: List[str] = Field(default_factory=list, description="Factors that alleviate symptoms (up to 5)")


class ConsultPriorTreatment(BaseModel):
    """Prior treatment and efficacy section"""
    prior_treatments: List[str] = Field(default_factory=list, description="Prior treatments received (up to 8)")
    level_of_relief: List[str] = Field(default_factory=list, description="Level of relief from each treatment (up to 5)")
    treatment_failure_statement: str = Field(default="", description="Statement about failure of conservative care")


class ConsultObjectiveFindings(BaseModel):
    """Objective findings section"""
    physical_examination: List[str] = Field(default_factory=list, description="Physical examination findings (up to 8)")
    imaging_review: List[str] = Field(default_factory=list, description="Imaging review findings (up to 5)")


class ConsultTreatmentRecommendations(BaseModel):
    """Treatment recommendations section - most critical for authorization"""
    injections_requested: List[str] = Field(default_factory=list, description="Injections requested with justification (up to 5)")
    procedures_requested: List[str] = Field(default_factory=list, description="Procedures requested with reasons (up to 5)")
    surgery_recommended: List[str] = Field(default_factory=list, description="Surgery recommended with urgency (up to 3)")
    diagnostics_ordered: List[str] = Field(default_factory=list, description="Diagnostics ordered with reasons (up to 5)")
    medication_changes: List[str] = Field(default_factory=list, description="Medication changes with dosages (up to 5)")
    therapy_recommendations: List[str] = Field(default_factory=list, description="Therapy recommendations with frequency (up to 5)")


class ConsultWorkStatus(BaseModel):
    """Work status and impairment section"""
    current_work_status: str = Field(default="", description="Current work status (full duty, modified, off work, etc.)")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions (up to 8)")
    restriction_duration: str = Field(default="", description="Duration of restrictions")
    return_to_work_plan: str = Field(default="", description="Plan for return to work")


class ConsultSignatureAuthor(BaseModel):
    """Signature and author section - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class ConsultLongSummary(BaseModel):
    """
    Complete structured Specialist Consultation Report long summary.
    Designed for Workers' Compensation consultation analysis with 8 critical fields.
    """
    content_type: Literal["consultation"] = Field(default="consultation", description="Content type for consultation documents")
    
    # Main sections matching the consult extractor prompt
    consultation_overview: ConsultOverview = Field(default_factory=ConsultOverview, description="Consultation overview")
    patient_info: ConsultPatientInfo = Field(default_factory=ConsultPatientInfo, description="Patient information")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")
    chief_complaint: ConsultChiefComplaint = Field(default_factory=ConsultChiefComplaint, description="Chief complaint")
    signature_author: ConsultSignatureAuthor = Field(default_factory=ConsultSignatureAuthor, description="Signature and author")
    diagnosis_assessment: ConsultDiagnosis = Field(default_factory=ConsultDiagnosis, description="Diagnosis and assessment")
    clinical_history: ConsultClinicalHistory = Field(default_factory=ConsultClinicalHistory, description="Clinical history and symptoms")
    prior_treatment: ConsultPriorTreatment = Field(default_factory=ConsultPriorTreatment, description="Prior treatment and efficacy")
    objective_findings: ConsultObjectiveFindings = Field(default_factory=ConsultObjectiveFindings, description="Objective findings")
    treatment_recommendations: ConsultTreatmentRecommendations = Field(default_factory=ConsultTreatmentRecommendations, description="Treatment recommendations")
    work_status: ConsultWorkStatus = Field(default_factory=ConsultWorkStatus, description="Work status and impairment")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 8)")


def format_consult_long_summary(summary: ConsultLongSummary) -> str:
    """Format a consultation long summary into the expected text format."""
    lines = []
    
    # Consultation Overview
    lines.append("📋 CONSULTATION OVERVIEW")
    lines.append("-" * 50)
    if summary.consultation_overview.document_type:
        lines.append(f"Document Type: {summary.consultation_overview.document_type}")
    if summary.consultation_overview.consultation_date:
        lines.append(f"Consultation Date: {summary.consultation_overview.consultation_date}")
    if summary.consultation_overview.consulting_physician:
        lines.append(f"Consulting Physician: {summary.consultation_overview.consulting_physician}")
    if summary.consultation_overview.specialty:
        lines.append(f"Specialty: {summary.consultation_overview.specialty}")
    if summary.consultation_overview.referring_physician:
        lines.append(f"Referring Physician: {summary.consultation_overview.referring_physician}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"Name: {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"Date of Birth: {summary.patient_info.date_of_birth}")
    if summary.patient_info.date_of_injury:
        lines.append(f"Date of Injury: {summary.patient_info.date_of_injury}")
    if summary.patient_info.claim_number:
        lines.append(f"Claim Number: {summary.patient_info.claim_number}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Chief Complaint
    lines.append("🎯 CHIEF COMPLAINT")
    lines.append("-" * 50)
    if summary.chief_complaint.primary_complaint:
        lines.append(f"Primary Complaint: {summary.chief_complaint.primary_complaint}")
    if summary.chief_complaint.location:
        lines.append(f"Location: {summary.chief_complaint.location}")
    if summary.chief_complaint.duration:
        lines.append(f"Duration: {summary.chief_complaint.duration}")
    if summary.chief_complaint.radiation_pattern:
        lines.append(f"Radiation Pattern: {summary.chief_complaint.radiation_pattern}")
    if summary.signature_author.signature:
        sig_type = f" ({summary.signature_author.signature_type})" if summary.signature_author.signature_type else ""
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.signature_author.signature}{sig_type}")
    lines.append("")
    
    # Diagnosis & Assessment
    lines.append("🏥 DIAGNOSIS & ASSESSMENT")
    lines.append("-" * 50)
    if summary.diagnosis_assessment.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.diagnosis_assessment.primary_diagnosis}")
    if summary.diagnosis_assessment.icd_10_code:
        lines.append(f"- ICD-10: {summary.diagnosis_assessment.icd_10_code}")
    if summary.diagnosis_assessment.certainty:
        lines.append(f"- Certainty: {summary.diagnosis_assessment.certainty}")
    if summary.diagnosis_assessment.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.diagnosis_assessment.secondary_diagnoses[:5]:
            lines.append(f"• {dx}")
    if summary.diagnosis_assessment.causation:
        lines.append(f"Causation: {summary.diagnosis_assessment.causation}")
    lines.append("")
    
    # Clinical History & Symptoms
    lines.append("🔬 CLINICAL HISTORY & SYMPTOMS")
    lines.append("-" * 50)
    if summary.clinical_history.pain_quality:
        lines.append(f"Pain Quality: {summary.clinical_history.pain_quality}")
    if summary.clinical_history.pain_location:
        lines.append(f"Pain Location: {summary.clinical_history.pain_location}")
    if summary.clinical_history.radiation:
        lines.append(f"Radiation: {summary.clinical_history.radiation}")
    if summary.clinical_history.aggravating_factors:
        lines.append("Aggravating Factors:")
        for factor in summary.clinical_history.aggravating_factors[:5]:
            lines.append(f"• {factor}")
    if summary.clinical_history.alleviating_factors:
        lines.append("Alleviating Factors:")
        for factor in summary.clinical_history.alleviating_factors[:5]:
            lines.append(f"• {factor}")
    lines.append("")
    
    # Prior Treatment & Efficacy
    lines.append("💊 PRIOR TREATMENT & EFFICACY")
    lines.append("-" * 50)
    if summary.prior_treatment.prior_treatments:
        lines.append("Prior Treatments Received:")
        for treatment in summary.prior_treatment.prior_treatments[:8]:
            lines.append(f"• {treatment}")
    if summary.prior_treatment.level_of_relief:
        lines.append("Level of Relief:")
        for relief in summary.prior_treatment.level_of_relief[:5]:
            lines.append(f"• {relief}")
    if summary.prior_treatment.treatment_failure_statement:
        lines.append(f"Treatment Failure Statement: {summary.prior_treatment.treatment_failure_statement}")
    lines.append("")
    
    # Objective Findings
    lines.append("📊 OBJECTIVE FINDINGS")
    lines.append("-" * 50)
    if summary.objective_findings.physical_examination:
        lines.append("Physical Examination:")
        for finding in summary.objective_findings.physical_examination[:8]:
            lines.append(f"• {finding}")
    if summary.objective_findings.imaging_review:
        lines.append("Imaging Review:")
        for finding in summary.objective_findings.imaging_review[:5]:
            lines.append(f"• {finding}")
    lines.append("")
    
    # Treatment Recommendations
    lines.append("🎯 TREATMENT RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.treatment_recommendations.injections_requested:
        lines.append("Injections Requested:")
        for injection in summary.treatment_recommendations.injections_requested[:5]:
            lines.append(f"• {injection}")
    if summary.treatment_recommendations.procedures_requested:
        lines.append("Procedures Requested:")
        for procedure in summary.treatment_recommendations.procedures_requested[:5]:
            lines.append(f"• {procedure}")
    if summary.treatment_recommendations.surgery_recommended:
        lines.append("Surgery Recommended:")
        for surgery in summary.treatment_recommendations.surgery_recommended[:3]:
            lines.append(f"• {surgery}")
    if summary.treatment_recommendations.diagnostics_ordered:
        lines.append("Diagnostics Ordered:")
        for diagnostic in summary.treatment_recommendations.diagnostics_ordered[:5]:
            lines.append(f"• {diagnostic}")
    if summary.treatment_recommendations.medication_changes:
        lines.append("Medication Changes:")
        for med in summary.treatment_recommendations.medication_changes[:5]:
            lines.append(f"• {med}")
    if summary.treatment_recommendations.therapy_recommendations:
        lines.append("Therapy Recommendations:")
        for therapy in summary.treatment_recommendations.therapy_recommendations[:5]:
            lines.append(f"• {therapy}")
    lines.append("")
    
    # Work Status & Impairment
    lines.append("💼 WORK STATUS & IMPAIRMENT")
    lines.append("-" * 50)
    if summary.work_status.current_work_status:
        lines.append(f"Current Work Status: {summary.work_status.current_work_status}")
    if summary.work_status.work_restrictions:
        lines.append("Work Restrictions:")
        for restriction in summary.work_status.work_restrictions[:8]:
            lines.append(f"• {restriction}")
    if summary.work_status.restriction_duration:
        lines.append(f"Restriction Duration: {summary.work_status.restriction_duration}")
    if summary.work_status.return_to_work_plan:
        lines.append(f"Return to Work Plan: {summary.work_status.return_to_work_plan}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_consult_summary(doc_type: str, fallback_date: str) -> ConsultLongSummary:
    """Create a fallback consultation long summary when extraction fails."""
    return ConsultLongSummary(
        content_type="consultation",
        consultation_overview=ConsultOverview(
            document_type=doc_type,
            consultation_date=fallback_date
        )
    )


# ============================================================================
# FORMAL MEDICAL REPORT SPECIFIC MODELS
# (Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, Endoscopy, Genetics, Discharge)
# ============================================================================

class FormalMedicalPatientInfo(BaseModel):
    """Patient information section for formal medical reports"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    date_of_injury: str = Field(default="", description="Date of injury")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    employer: str = Field(default="", description="Employer name if applicable")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class FormalMedicalProviders(BaseModel):
    """Healthcare providers section"""
    performing_physician: str = Field(default="", description="Name of performing physician/surgeon")
    specialty: str = Field(default="", description="Physician's specialty")
    ordering_physician: str = Field(default="", description="Name of ordering/referring physician")
    anesthesiologist: str = Field(default="", description="Name of anesthesiologist if applicable")
    assistant_surgeon: str = Field(default="", description="Name of assistant surgeon if applicable")


class FormalMedicalSignatureAuthor(BaseModel):
    """Signature and author section - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class FormalMedicalClinicalIndications(BaseModel):
    """Clinical indications and pre-procedure information"""
    pre_procedure_diagnosis: str = Field(default="", description="Pre-procedure/pre-operative diagnosis")
    chief_complaint: str = Field(default="", description="Chief complaint or presenting symptoms")
    indications: List[str] = Field(default_factory=list, description="Indications for the procedure (up to 5)")
    relevant_history: str = Field(default="", description="Relevant medical history")


class FormalMedicalProcedureDetails(BaseModel):
    """Procedure details section"""
    procedure_name: str = Field(default="", description="Name of the procedure performed")
    procedure_date: str = Field(default="", description="Date of the procedure")
    cpt_codes: List[str] = Field(default_factory=list, description="CPT codes for procedures (up to 5)")
    technique: str = Field(default="", description="Technique or approach used")
    duration: str = Field(default="", description="Duration of the procedure")
    anesthesia_type: str = Field(default="", description="Type of anesthesia used")
    implants_devices: List[str] = Field(default_factory=list, description="Implants or devices used (up to 5)")


class FormalMedicalFindings(BaseModel):
    """Findings section (intraoperative, study results, etc.)"""
    intraoperative_findings: List[str] = Field(default_factory=list, description="Intraoperative or study findings (up to 8)")
    specimens_collected: List[str] = Field(default_factory=list, description="Specimens collected (up to 5)")
    estimated_blood_loss: str = Field(default="", description="Estimated blood loss if applicable")
    complications: str = Field(default="", description="Any complications noted")
    condition_at_completion: str = Field(default="", description="Patient's condition at end of procedure")


class FormalMedicalDiagnosis(BaseModel):
    """Diagnosis section"""
    final_diagnosis: str = Field(default="", description="Final or post-procedure diagnosis")
    icd_10_codes: List[str] = Field(default_factory=list, description="ICD-10 codes (up to 5)")
    pathological_diagnosis: str = Field(default="", description="Pathological diagnosis if applicable")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 5)")


class FormalMedicalRecommendations(BaseModel):
    """Post-procedure recommendations section"""
    post_procedure_care: List[str] = Field(default_factory=list, description="Post-procedure care instructions (up to 5)")
    medications: List[str] = Field(default_factory=list, description="Medications prescribed (up to 8)")
    activity_restrictions: List[str] = Field(default_factory=list, description="Activity restrictions (up to 5)")
    follow_up: str = Field(default="", description="Follow-up appointment/instructions")
    referrals: List[str] = Field(default_factory=list, description="Referrals made (up to 3)")
    additional_procedures: List[str] = Field(default_factory=list, description="Additional procedures recommended (up to 3)")


class FormalMedicalWorkStatus(BaseModel):
    """Work status section for formal medical reports"""
    current_work_status: str = Field(default="", description="Current work status")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions (up to 5)")
    restriction_duration: str = Field(default="", description="Duration of restrictions")
    return_to_work_date: str = Field(default="", description="Expected return to work date")


class FormalMedicalLongSummary(BaseModel):
    """
    Complete structured Formal Medical Report long summary.
    Designed for Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, 
    Endoscopy, Genetics, Discharge Summaries and similar comprehensive medical reports.
    """
    content_type: Literal["formal_medical"] = Field(default="formal_medical", description="Content type for formal medical documents")
    
    # Report identification
    report_type: str = Field(default="", description="Type of medical report (Surgery, EMG, Pathology, etc.)")
    report_date: str = Field(default="", description="Date of the report")
    facility: str = Field(default="", description="Facility/hospital name")
    
    # Main sections
    patient_info: FormalMedicalPatientInfo = Field(default_factory=FormalMedicalPatientInfo, description="Patient information")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")
    providers: FormalMedicalProviders = Field(default_factory=FormalMedicalProviders, description="Healthcare providers")
    signature_author: FormalMedicalSignatureAuthor = Field(default_factory=FormalMedicalSignatureAuthor, description="Signature and author")
    clinical_indications: FormalMedicalClinicalIndications = Field(default_factory=FormalMedicalClinicalIndications, description="Clinical indications")
    procedure_details: FormalMedicalProcedureDetails = Field(default_factory=FormalMedicalProcedureDetails, description="Procedure details")
    findings: FormalMedicalFindings = Field(default_factory=FormalMedicalFindings, description="Findings")
    diagnosis: FormalMedicalDiagnosis = Field(default_factory=FormalMedicalDiagnosis, description="Diagnosis")
    recommendations: FormalMedicalRecommendations = Field(default_factory=FormalMedicalRecommendations, description="Recommendations")
    work_status: FormalMedicalWorkStatus = Field(default_factory=FormalMedicalWorkStatus, description="Work status")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 8)")


def format_formal_medical_long_summary(summary: FormalMedicalLongSummary) -> str:
    """Format a formal medical long summary into the expected text format."""
    lines = []
    
    # Report Overview
    lines.append("📋 FORMAL MEDICAL REPORT OVERVIEW")
    lines.append("-" * 50)
    if summary.report_type:
        lines.append(f"Report Type: {summary.report_type}")
    if summary.report_date:
        lines.append(f"Report Date: {summary.report_date}")
    if summary.facility:
        lines.append(f"Facility: {summary.facility}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"Name: {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"DOB: {summary.patient_info.date_of_birth}")
    if summary.patient_info.date_of_injury:
        lines.append(f"DOI: {summary.patient_info.date_of_injury}")
    if summary.patient_info.claim_number:
        lines.append(f"Claim Number: {summary.patient_info.claim_number}")
    if summary.patient_info.employer:
        lines.append(f"Employer: {summary.patient_info.employer}")
    lines.append("")
    
    # Healthcare Providers
    lines.append("👨‍⚕️ HEALTHCARE PROVIDERS")
    lines.append("-" * 50)
    if summary.providers.performing_physician:
        lines.append(f"Performing Physician: {summary.providers.performing_physician}")
    if summary.providers.specialty:
        lines.append(f"Specialty: {summary.providers.specialty}")
    if summary.providers.ordering_physician:
        lines.append(f"Ordering Physician: {summary.providers.ordering_physician}")
    if summary.providers.anesthesiologist:
        lines.append(f"Anesthesiologist: {summary.providers.anesthesiologist}")
    if summary.providers.assistant_surgeon:
        lines.append(f"Assistant Surgeon: {summary.providers.assistant_surgeon}")
    if summary.signature_author.signature:
        sig_type = f" ({summary.signature_author.signature_type})" if summary.signature_author.signature_type else ""
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.signature_author.signature}{sig_type}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Clinical Indications
    lines.append("🎯 CLINICAL INDICATIONS")
    lines.append("-" * 50)
    if summary.clinical_indications.pre_procedure_diagnosis:
        lines.append(f"Pre-Procedure Diagnosis: {summary.clinical_indications.pre_procedure_diagnosis}")
    if summary.clinical_indications.chief_complaint:
        lines.append(f"Chief Complaint: {summary.clinical_indications.chief_complaint}")
    if summary.clinical_indications.indications:
        lines.append("Indications:")
        for indication in summary.clinical_indications.indications[:5]:
            lines.append(f"• {indication}")
    if summary.clinical_indications.relevant_history:
        lines.append(f"Relevant History: {summary.clinical_indications.relevant_history}")
    lines.append("")
    
    # Procedure Details
    lines.append("🔧 PROCEDURE DETAILS")
    lines.append("-" * 50)
    if summary.procedure_details.procedure_name:
        lines.append(f"Procedure: {summary.procedure_details.procedure_name}")
    if summary.procedure_details.procedure_date:
        lines.append(f"Procedure Date: {summary.procedure_details.procedure_date}")
    if summary.procedure_details.cpt_codes:
        lines.append(f"CPT Codes: {', '.join(summary.procedure_details.cpt_codes[:5])}")
    if summary.procedure_details.technique:
        lines.append(f"Technique: {summary.procedure_details.technique}")
    if summary.procedure_details.duration:
        lines.append(f"Duration: {summary.procedure_details.duration}")
    if summary.procedure_details.anesthesia_type:
        lines.append(f"Anesthesia: {summary.procedure_details.anesthesia_type}")
    if summary.procedure_details.implants_devices:
        lines.append("Implants/Devices:")
        for device in summary.procedure_details.implants_devices[:5]:
            lines.append(f"• {device}")
    lines.append("")
    
    # Findings
    lines.append("🔬 FINDINGS")
    lines.append("-" * 50)
    if summary.findings.intraoperative_findings:
        lines.append("Intraoperative/Study Findings:")
        for finding in summary.findings.intraoperative_findings[:8]:
            lines.append(f"• {finding}")
    if summary.findings.specimens_collected:
        lines.append("Specimens Collected:")
        for specimen in summary.findings.specimens_collected[:5]:
            lines.append(f"• {specimen}")
    if summary.findings.estimated_blood_loss:
        lines.append(f"Estimated Blood Loss: {summary.findings.estimated_blood_loss}")
    if summary.findings.complications:
        lines.append(f"Complications: {summary.findings.complications}")
    if summary.findings.condition_at_completion:
        lines.append(f"Condition at Completion: {summary.findings.condition_at_completion}")
    lines.append("")
    
    # Diagnosis
    lines.append("🏥 DIAGNOSIS")
    lines.append("-" * 50)
    if summary.diagnosis.final_diagnosis:
        lines.append(f"Final Diagnosis: {summary.diagnosis.final_diagnosis}")
    if summary.diagnosis.icd_10_codes:
        lines.append(f"ICD-10 Codes: {', '.join(summary.diagnosis.icd_10_codes[:5])}")
    if summary.diagnosis.pathological_diagnosis:
        lines.append(f"Pathological Diagnosis: {summary.diagnosis.pathological_diagnosis}")
    if summary.diagnosis.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.diagnosis.secondary_diagnoses[:5]:
            lines.append(f"• {dx}")
    lines.append("")
    
    # Recommendations
    lines.append("📋 RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.recommendations.post_procedure_care:
        lines.append("Post-Procedure Care:")
        for care in summary.recommendations.post_procedure_care[:5]:
            lines.append(f"• {care}")
    if summary.recommendations.medications:
        lines.append("Medications:")
        for med in summary.recommendations.medications[:8]:
            lines.append(f"• {med}")
    if summary.recommendations.activity_restrictions:
        lines.append("Activity Restrictions:")
        for restriction in summary.recommendations.activity_restrictions[:5]:
            lines.append(f"• {restriction}")
    if summary.recommendations.follow_up:
        lines.append(f"Follow-Up: {summary.recommendations.follow_up}")
    if summary.recommendations.referrals:
        lines.append("Referrals:")
        for referral in summary.recommendations.referrals[:3]:
            lines.append(f"• {referral}")
    if summary.recommendations.additional_procedures:
        lines.append("Additional Procedures Recommended:")
        for proc in summary.recommendations.additional_procedures[:3]:
            lines.append(f"• {proc}")
    lines.append("")
    
    # Work Status
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    if summary.work_status.current_work_status:
        lines.append(f"Current Work Status: {summary.work_status.current_work_status}")
    if summary.work_status.work_restrictions:
        lines.append("Work Restrictions:")
        for restriction in summary.work_status.work_restrictions[:5]:
            lines.append(f"• {restriction}")
    if summary.work_status.restriction_duration:
        lines.append(f"Restriction Duration: {summary.work_status.restriction_duration}")
    if summary.work_status.return_to_work_date:
        lines.append(f"Return to Work Date: {summary.work_status.return_to_work_date}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_formal_medical_summary(doc_type: str, fallback_date: str) -> FormalMedicalLongSummary:
    """Create a fallback formal medical long summary when extraction fails."""
    return FormalMedicalLongSummary(
        content_type="formal_medical",
        report_type=doc_type,
        report_date=fallback_date
    )


# ============================================================================
# IMAGING LONG SUMMARY MODELS
# ============================================================================

class ImagingAuthorInfo(BaseModel):
    """Author/signature information for imaging reports - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT referring physicians, ordering providers, technologists, or other officials mentioned in the document.")


class ImagingRadiologistInfo(BaseModel):
    """Radiologist information"""
    name: str = Field(default="", description="Radiologist's name")
    credentials: str = Field(default="", description="Credentials (MD, DO, etc.)")
    specialty: str = Field(default="Radiology", description="Specialty")


class ImagingDoctorInfo(BaseModel):
    """Doctor information for imaging reports"""
    name: str = Field(default="", description="Doctor's name")
    title: str = Field(default="", description="Title or credentials")
    role: str = Field(default="", description="Role (radiologist, referring, ordering)")


class ImagingOverview(BaseModel):
    """Imaging overview section (Field 1)"""
    document_type: str = Field(default="", description="Type of imaging document")
    exam_date: str = Field(default="", description="Date of the imaging exam")
    exam_type: str = Field(default="", description="Type of exam (MRI, CT, X-ray, Ultrasound)")
    radiologist: str = Field(default="", description="Radiologist name")
    imaging_center: str = Field(default="", description="Imaging center/facility")
    referring_physician: str = Field(default="", description="Referring physician name")
    author: ImagingAuthorInfo = Field(default_factory=ImagingAuthorInfo, description="Author/signature info")


class ImagingPatientInfo(BaseModel):
    """Patient information section"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    date_of_injury: str = Field(default="", description="Date of injury")
    employer: str = Field(default="", description="Employer name")
    all_doctors_involved: List[ImagingDoctorInfo] = Field(default_factory=list, description="All doctors mentioned")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class ImagingClinicalIndication(BaseModel):
    """Clinical indication section (Field 2)"""
    clinical_indication: str = Field(default="", description="Reason for the imaging study")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    chief_complaint: str = Field(default="", description="Chief complaint")
    specific_questions: str = Field(default="", description="Specific clinical questions")


class ImagingTechnicalDetails(BaseModel):
    """Technical details section (Field 3)"""
    study_type: str = Field(default="", description="Type of study")
    body_part_imaged: str = Field(default="", description="Body part being imaged")
    laterality: str = Field(default="", description="Left, right, or bilateral")
    contrast_used: str = Field(default="", description="Whether contrast was used (with/without)")
    contrast_type: str = Field(default="", description="Type of contrast if used")
    prior_studies_available: str = Field(default="", description="Prior studies available for comparison")
    technical_quality: str = Field(default="", description="Technical quality of the study")
    limitations: str = Field(default="", description="Any technical limitations")


class ImagingPrimaryFinding(BaseModel):
    """Primary finding details"""
    description: str = Field(default="", description="Description of primary finding")
    location: str = Field(default="", description="Anatomical location")
    size: str = Field(default="", description="Size/dimensions")
    characteristics: str = Field(default="", description="Imaging characteristics")
    acuity: str = Field(default="", description="Acuity (acute, subacute, chronic)")


class ImagingKeyFindings(BaseModel):
    """Key findings section (Field 4)"""
    primary_finding: ImagingPrimaryFinding = Field(default_factory=ImagingPrimaryFinding, description="Primary finding")
    secondary_findings: List[str] = Field(default_factory=list, description="Secondary findings")
    normal_findings: List[str] = Field(default_factory=list, description="Normal findings (up to 5)")


class ImagingImpression(BaseModel):
    """Impression and conclusion section (Field 5)"""
    overall_impression: str = Field(default="", description="Radiologist's overall impression")
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    final_diagnostic_statement: str = Field(default="", description="Final diagnostic statement")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Differential diagnoses (up to 3)")
    clinical_correlation: str = Field(default="", description="Clinical correlation statement")


class ImagingRecommendations(BaseModel):
    """Recommendations and follow-up section (Field 6)"""
    follow_up_recommended: str = Field(default="", description="Whether follow-up is recommended")
    follow_up_modality: str = Field(default="", description="Recommended follow-up imaging modality")
    follow_up_timing: str = Field(default="", description="Timing for follow-up")
    clinical_correlation_needed: str = Field(default="", description="Whether clinical correlation is needed")
    specialist_consultation: str = Field(default="", description="Specialist consultation recommendation")


class ImagingLongSummary(BaseModel):
    """Complete structured imaging long summary"""
    content_type: Literal["imaging"] = Field(default="imaging", description="Type of content")
    imaging_overview: ImagingOverview = Field(default_factory=ImagingOverview, description="Imaging overview section")
    patient_info: ImagingPatientInfo = Field(default_factory=ImagingPatientInfo, description="Patient information")
    clinical_indication: ImagingClinicalIndication = Field(default_factory=ImagingClinicalIndication, description="Clinical indication")
    technical_details: ImagingTechnicalDetails = Field(default_factory=ImagingTechnicalDetails, description="Technical details")
    key_findings: ImagingKeyFindings = Field(default_factory=ImagingKeyFindings, description="Key findings")
    impression: ImagingImpression = Field(default_factory=ImagingImpression, description="Impression and conclusion")
    recommendations: ImagingRecommendations = Field(default_factory=ImagingRecommendations, description="Recommendations and follow-up")


def format_imaging_long_summary(summary: ImagingLongSummary) -> str:
    """Format ImagingLongSummary Pydantic model to readable text format."""
    lines = []
    
    # Imaging Overview
    lines.append("📋 IMAGING OVERVIEW")
    lines.append("-" * 50)
    if summary.imaging_overview.document_type:
        lines.append(f"Document Type: {summary.imaging_overview.document_type}")
    if summary.imaging_overview.exam_date:
        lines.append(f"Exam Date: {summary.imaging_overview.exam_date}")
    if summary.imaging_overview.exam_type:
        lines.append(f"Exam Type: {summary.imaging_overview.exam_type}")
    if summary.imaging_overview.radiologist:
        lines.append(f"Radiologist: {summary.imaging_overview.radiologist}")
    if summary.imaging_overview.imaging_center:
        lines.append(f"Imaging Center: {summary.imaging_overview.imaging_center}")
    if summary.imaging_overview.referring_physician:
        lines.append(f"Referring Physician: {summary.imaging_overview.referring_physician}")
    if summary.imaging_overview.author.signature:
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.imaging_overview.author.signature}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"Name: {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"Date of Birth: {summary.patient_info.date_of_birth}")
    if summary.patient_info.claim_number:
        lines.append(f"Claim Number: {summary.patient_info.claim_number}")
    if summary.patient_info.date_of_injury:
        lines.append(f"Date of Injury: {summary.patient_info.date_of_injury}")
    if summary.patient_info.employer:
        lines.append(f"Employer: {summary.patient_info.employer}")
    if summary.patient_info.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.patient_info.all_doctors_involved[:5]:
            role_info = f" ({doctor.role})" if doctor.role else ""
            title_info = f", {doctor.title}" if doctor.title else ""
            lines.append(f"• {doctor.name}{title_info}{role_info}")
    lines.append("")
    
    # Clinical Indication
    lines.append("🎯 CLINICAL INDICATION")
    lines.append("-" * 50)
    if summary.clinical_indication.clinical_indication:
        lines.append(f"Clinical Indication: {summary.clinical_indication.clinical_indication}")
    if summary.clinical_indication.clinical_history:
        lines.append(f"Clinical History: {summary.clinical_indication.clinical_history}")
    if summary.clinical_indication.chief_complaint:
        lines.append(f"Chief Complaint: {summary.clinical_indication.chief_complaint}")
    if summary.clinical_indication.specific_questions:
        lines.append(f"Specific Questions: {summary.clinical_indication.specific_questions}")
    lines.append("")
    
    # Technical Details
    lines.append("🔧 TECHNICAL DETAILS")
    lines.append("-" * 50)
    if summary.technical_details.study_type:
        lines.append(f"Study Type: {summary.technical_details.study_type}")
    if summary.technical_details.body_part_imaged:
        lines.append(f"Body Part Imaged: {summary.technical_details.body_part_imaged}")
    if summary.technical_details.laterality:
        lines.append(f"Laterality: {summary.technical_details.laterality}")
    if summary.technical_details.contrast_used:
        lines.append(f"Contrast Used: {summary.technical_details.contrast_used}")
    if summary.technical_details.contrast_type:
        lines.append(f"Contrast Type: {summary.technical_details.contrast_type}")
    if summary.technical_details.prior_studies_available:
        lines.append(f"Prior Studies Available: {summary.technical_details.prior_studies_available}")
    if summary.technical_details.technical_quality:
        lines.append(f"Technical Quality: {summary.technical_details.technical_quality}")
    if summary.technical_details.limitations:
        lines.append(f"Limitations: {summary.technical_details.limitations}")
    lines.append("")
    
    # Key Findings
    lines.append("📊 KEY FINDINGS")
    lines.append("-" * 50)
    pf = summary.key_findings.primary_finding
    has_primary = any([pf.description, pf.location, pf.size, pf.characteristics, pf.acuity])
    if has_primary:
        lines.append("Primary Finding:")
        if pf.description:
            lines.append(f"• Description: {pf.description}")
        if pf.location:
            lines.append(f"• Location: {pf.location}")
        if pf.size:
            lines.append(f"• Size: {pf.size}")
        if pf.characteristics:
            lines.append(f"• Characteristics: {pf.characteristics}")
        if pf.acuity:
            lines.append(f"• Acuity: {pf.acuity}")
    if summary.key_findings.secondary_findings:
        lines.append("")
        lines.append("Secondary Findings:")
        for finding in summary.key_findings.secondary_findings[:5]:
            lines.append(f"• {finding}")
    if summary.key_findings.normal_findings:
        lines.append("")
        lines.append("Normal Findings:")
        for finding in summary.key_findings.normal_findings[:5]:
            lines.append(f"• {finding}")
    lines.append("")
    
    # Impression & Conclusion
    lines.append("💡 IMPRESSION & CONCLUSION")
    lines.append("-" * 50)
    if summary.impression.overall_impression:
        lines.append(f"Overall Impression: {summary.impression.overall_impression}")
    if summary.impression.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.impression.primary_diagnosis}")
    if summary.impression.final_diagnostic_statement:
        lines.append(f"Final Diagnostic Statement: {summary.impression.final_diagnostic_statement}")
    if summary.impression.differential_diagnoses:
        lines.append("")
        lines.append("Differential Diagnoses:")
        for dx in summary.impression.differential_diagnoses[:3]:
            lines.append(f"• {dx}")
    if summary.impression.clinical_correlation:
        lines.append(f"Clinical Correlation: {summary.impression.clinical_correlation}")
    lines.append("")
    
    # Recommendations & Follow-Up
    lines.append("📋 RECOMMENDATIONS & FOLLOW-UP")
    lines.append("-" * 50)
    if summary.recommendations.follow_up_recommended:
        lines.append(f"Follow-up Recommended: {summary.recommendations.follow_up_recommended}")
    if summary.recommendations.follow_up_modality:
        lines.append(f"Follow-up Modality: {summary.recommendations.follow_up_modality}")
    if summary.recommendations.follow_up_timing:
        lines.append(f"Follow-up Timing: {summary.recommendations.follow_up_timing}")
    if summary.recommendations.clinical_correlation_needed:
        lines.append(f"Clinical Correlation Needed: {summary.recommendations.clinical_correlation_needed}")
    if summary.recommendations.specialist_consultation:
        lines.append(f"Specialist Consultation: {summary.recommendations.specialist_consultation}")
    
    return "\n".join(lines)


def create_fallback_imaging_summary(doc_type: str, fallback_date: str) -> ImagingLongSummary:
    """Create a fallback imaging long summary when extraction fails."""
    return ImagingLongSummary(
        content_type="imaging",
        imaging_overview=ImagingOverview(
            document_type=doc_type,
            exam_date=fallback_date,
            exam_type=doc_type
        ),
        technical_details=ImagingTechnicalDetails(
            study_type=doc_type
        )
    )


# ============================================================================
# QME/AME/IME LONG SUMMARY MODELS
# ============================================================================

class QMEAuthorInfo(BaseModel):
    """Author/signature information for QME reports - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the QME report (physical or electronic signature). Must be the actual signer/evaluating physician who signed - NOT providers, claim adjusters, requesting physicians, defense attorneys, applicant attorneys, or other officials mentioned in the document.")


class QMEDoctorInfo(BaseModel):
    """Doctor information for QME reports"""
    name: str = Field(default="", description="Doctor's full name")
    credentials: str = Field(default="", description="Credentials (MD, DO, DC, etc.)")
    role: str = Field(default="", description="Role (evaluating, treating, consulting)")


class QMEPatientInfo(BaseModel):
    """Patient information section for QME reports"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    date_of_injury: str = Field(default="", description="Date of injury")
    employer: str = Field(default="", description="Employer name")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class QMEReportDetails(BaseModel):
    """Report details section for QME reports"""
    report_type: str = Field(default="", description="Type of report (QME/AME/IME)")
    report_date: str = Field(default="", description="Date of the report")
    evaluating_physician: str = Field(default="", description="Name of evaluating physician")
    author: QMEAuthorInfo = Field(default_factory=QMEAuthorInfo, description="Author/signature info")
    all_doctors_involved: List[QMEDoctorInfo] = Field(default_factory=list, description="All doctors mentioned in report")


class QMEDiagnosis(BaseModel):
    """Diagnosis section for QME reports"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    icd_10_codes: List[str] = Field(default_factory=list, description="ICD-10 codes")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses")
    body_parts_affected: List[str] = Field(default_factory=list, description="Body parts affected")


class QMEPhysicalExamFindings(BaseModel):
    """Physical examination findings section"""
    general_findings: List[str] = Field(default_factory=list, description="General clinical findings")
    range_of_motion: List[str] = Field(default_factory=list, description="Range of motion measurements")
    strength_testing: List[str] = Field(default_factory=list, description="Strength/motor testing results")
    sensory_findings: List[str] = Field(default_factory=list, description="Sensory examination findings")
    special_tests: List[str] = Field(default_factory=list, description="Special tests performed and results")


class QMEClinicalStatus(BaseModel):
    """Clinical status section"""
    current_condition: str = Field(default="", description="Current clinical condition")
    pain_level: str = Field(default="", description="Pain level/score if documented")
    functional_limitations: List[str] = Field(default_factory=list, description="Functional limitations")
    subjective_complaints: List[str] = Field(default_factory=list, description="Patient's subjective complaints")


class QMEMedications(BaseModel):
    """Medications section for QME reports"""
    current_medications: List[str] = Field(default_factory=list, description="Current medications with dosages")
    previous_medications: List[str] = Field(default_factory=list, description="Previous medications")
    future_medications: List[str] = Field(default_factory=list, description="Recommended future medications")


class QMEMedicalLegalConclusions(BaseModel):
    """Medical-legal conclusions section - CRITICAL for QME"""
    mmi_status: str = Field(default="", description="Maximum Medical Improvement status")
    mmi_date: str = Field(default="", description="Date of MMI if reached")
    wpi_rating: str = Field(default="", description="Whole Person Impairment rating/percentage")
    apportionment: str = Field(default="", description="Apportionment determination")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions")
    work_status: str = Field(default="", description="Current work status")
    causation_opinion: str = Field(default="", description="Causation opinion")
    future_medical_care: str = Field(default="", description="Need for future medical care")


class QMERecommendations(BaseModel):
    """Recommendations section for QME reports"""
    treatment_recommendations: List[str] = Field(default_factory=list, description="Treatment recommendations")
    diagnostic_recommendations: List[str] = Field(default_factory=list, description="Diagnostic test recommendations")
    specialist_referrals: List[str] = Field(default_factory=list, description="Specialist referral recommendations")
    follow_up: str = Field(default="", description="Follow-up recommendations")


class QMELongSummary(BaseModel):
    """Complete structured QME/AME/IME long summary"""
    content_type: Literal["qme"] = Field(default="qme", description="Type of content")
    patient_info: QMEPatientInfo = Field(default_factory=QMEPatientInfo, description="Patient information")
    report_details: QMEReportDetails = Field(default_factory=QMEReportDetails, description="Report details")
    diagnosis: QMEDiagnosis = Field(default_factory=QMEDiagnosis, description="Diagnosis information")
    physical_exam_findings: QMEPhysicalExamFindings = Field(default_factory=QMEPhysicalExamFindings, description="Physical exam findings")
    clinical_status: QMEClinicalStatus = Field(default_factory=QMEClinicalStatus, description="Clinical status")
    medications: QMEMedications = Field(default_factory=QMEMedications, description="Medications")
    medical_legal_conclusions: QMEMedicalLegalConclusions = Field(default_factory=QMEMedicalLegalConclusions, description="Medical-legal conclusions")
    recommendations: QMERecommendations = Field(default_factory=QMERecommendations, description="Recommendations")


def format_qme_long_summary(summary: QMELongSummary) -> str:
    """Format QMELongSummary Pydantic model to readable text format."""
    lines = []
    
    # Patient Information
    lines.append("## PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"- **Name:** {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"- **Date of Birth:** {summary.patient_info.date_of_birth}")
    if summary.patient_info.claim_number:
        lines.append(f"- **Claim Number:** {summary.patient_info.claim_number}")
    if summary.patient_info.date_of_injury:
        lines.append(f"- **Date of Injury:** {summary.patient_info.date_of_injury}")
    if summary.patient_info.employer:
        lines.append(f"- **Employer:** {summary.patient_info.employer}")
    lines.append("")
    
    # Report Details
    lines.append("## REPORT DETAILS")
    lines.append("-" * 50)
    if summary.report_details.report_type:
        lines.append(f"- **Report Type:** {summary.report_details.report_type}")
    if summary.report_details.report_date:
        lines.append(f"- **Report Date:** {summary.report_details.report_date}")
    if summary.report_details.evaluating_physician:
        lines.append(f"- **Evaluating Physician:** {summary.report_details.evaluating_physician}")
    if summary.report_details.author.signature:
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.report_details.author.signature}")
    lines.append("")
    
    # All Doctors Involved
    if summary.report_details.all_doctors_involved:
        lines.append("## All Doctors Involved")
        lines.append("-" * 50)
        for doctor in summary.report_details.all_doctors_involved[:10]:
            creds = f", {doctor.credentials}" if doctor.credentials else ""
            role = f" ({doctor.role})" if doctor.role else ""
            lines.append(f"• {doctor.name}{creds}{role}")
        lines.append("")
    
    # Diagnosis
    lines.append("## DIAGNOSIS")
    lines.append("-" * 50)
    if summary.diagnosis.primary_diagnosis:
        lines.append(f"- **Primary Diagnosis:** {summary.diagnosis.primary_diagnosis}")
    if summary.diagnosis.icd_10_codes:
        lines.append(f"- **ICD-10 Codes:** {', '.join(summary.diagnosis.icd_10_codes[:5])}")
    if summary.diagnosis.secondary_diagnoses:
        lines.append("- **Secondary Diagnoses:**")
        for dx in summary.diagnosis.secondary_diagnoses[:5]:
            lines.append(f"  • {dx}")
    if summary.diagnosis.body_parts_affected:
        lines.append(f"- **Body Parts Affected:** {', '.join(summary.diagnosis.body_parts_affected[:5])}")
    lines.append("")
    
    # Physical Examination Findings
    lines.append("## PHYSICAL EXAMINATION FINDINGS")
    lines.append("-" * 50)
    if summary.physical_exam_findings.general_findings:
        lines.append("**General Findings:**")
        for finding in summary.physical_exam_findings.general_findings[:5]:
            lines.append(f"• {finding}")
    if summary.physical_exam_findings.range_of_motion:
        lines.append("**Range of Motion:**")
        for rom in summary.physical_exam_findings.range_of_motion[:5]:
            lines.append(f"• {rom}")
    if summary.physical_exam_findings.strength_testing:
        lines.append("**Strength Testing:**")
        for strength in summary.physical_exam_findings.strength_testing[:5]:
            lines.append(f"• {strength}")
    if summary.physical_exam_findings.sensory_findings:
        lines.append("**Sensory Findings:**")
        for sensory in summary.physical_exam_findings.sensory_findings[:5]:
            lines.append(f"• {sensory}")
    if summary.physical_exam_findings.special_tests:
        lines.append("**Special Tests:**")
        for test in summary.physical_exam_findings.special_tests[:5]:
            lines.append(f"• {test}")
    lines.append("")
    
    # Clinical Status
    lines.append("## CLINICAL STATUS")
    lines.append("-" * 50)
    if summary.clinical_status.current_condition:
        lines.append(f"- **Current Condition:** {summary.clinical_status.current_condition}")
    if summary.clinical_status.pain_level:
        lines.append(f"- **Pain Level:** {summary.clinical_status.pain_level}")
    if summary.clinical_status.functional_limitations:
        lines.append("- **Functional Limitations:**")
        for limitation in summary.clinical_status.functional_limitations[:5]:
            lines.append(f"  • {limitation}")
    if summary.clinical_status.subjective_complaints:
        lines.append("- **Subjective Complaints:**")
        for complaint in summary.clinical_status.subjective_complaints[:5]:
            lines.append(f"  • {complaint}")
    lines.append("")
    
    # Medications
    lines.append("## MEDICATIONS")
    lines.append("-" * 50)
    if summary.medications.current_medications:
        lines.append("- **Current Medications:**")
        for med in summary.medications.current_medications[:10]:
            lines.append(f"  • {med}")
    if summary.medications.previous_medications:
        lines.append("- **Previous Medications:**")
        for med in summary.medications.previous_medications[:5]:
            lines.append(f"  • {med}")
    if summary.medications.future_medications:
        lines.append("- **Future/Recommended Medications:**")
        for med in summary.medications.future_medications[:5]:
            lines.append(f"  • {med}")
    lines.append("")
    
    # Medical-Legal Conclusions (CRITICAL)
    lines.append("## MEDICAL-LEGAL CONCLUSIONS")
    lines.append("-" * 50)
    if summary.medical_legal_conclusions.mmi_status:
        lines.append(f"- **MMI Status:** {summary.medical_legal_conclusions.mmi_status}")
    if summary.medical_legal_conclusions.mmi_date:
        lines.append(f"- **MMI Date:** {summary.medical_legal_conclusions.mmi_date}")
    if summary.medical_legal_conclusions.wpi_rating:
        lines.append(f"- **WPI Rating:** {summary.medical_legal_conclusions.wpi_rating}")
    if summary.medical_legal_conclusions.apportionment:
        lines.append(f"- **Apportionment:** {summary.medical_legal_conclusions.apportionment}")
    if summary.medical_legal_conclusions.work_status:
        lines.append(f"- **Work Status:** {summary.medical_legal_conclusions.work_status}")
    if summary.medical_legal_conclusions.work_restrictions:
        lines.append("- **Work Restrictions:**")
        for restriction in summary.medical_legal_conclusions.work_restrictions[:5]:
            lines.append(f"  • {restriction}")
    if summary.medical_legal_conclusions.causation_opinion:
        lines.append(f"- **Causation Opinion:** {summary.medical_legal_conclusions.causation_opinion}")
    if summary.medical_legal_conclusions.future_medical_care:
        lines.append(f"- **Future Medical Care:** {summary.medical_legal_conclusions.future_medical_care}")
    lines.append("")
    
    # Recommendations
    lines.append("## RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.recommendations.treatment_recommendations:
        lines.append("- **Treatment Recommendations:**")
        for rec in summary.recommendations.treatment_recommendations[:5]:
            lines.append(f"  • {rec}")
    if summary.recommendations.diagnostic_recommendations:
        lines.append("- **Diagnostic Recommendations:**")
        for rec in summary.recommendations.diagnostic_recommendations[:5]:
            lines.append(f"  • {rec}")
    if summary.recommendations.specialist_referrals:
        lines.append("- **Specialist Referrals:**")
        for ref in summary.recommendations.specialist_referrals[:3]:
            lines.append(f"  • {ref}")
    if summary.recommendations.follow_up:
        lines.append(f"- **Follow-Up:** {summary.recommendations.follow_up}")
    
    return "\n".join(lines)


def create_fallback_qme_summary(doc_type: str, fallback_date: str) -> QMELongSummary:
    """Create a fallback QME long summary when extraction fails."""
    return QMELongSummary(
        content_type="qme",
        report_details=QMEReportDetails(
            report_type=doc_type,
            report_date=fallback_date
        )
    )


# ============================================================================
# UR/IMR DECISION DOCUMENT LONG SUMMARY MODELS
# ============================================================================

class URAuthorInfo(BaseModel):
    """Author/signature information for UR decision documents - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the UR decision document (physical or electronic signature). Must be the actual signer/reviewing physician who signed - NOT requesting providers, claim adjusters, utilization review organizations, insurance representatives, or other officials mentioned in the document.")


class URDoctorInfo(BaseModel):
    """Doctor information for UR documents"""
    name: str = Field(default="", description="Doctor's full name")
    title: str = Field(default="", description="Title or credentials (MD, DO, etc.)")
    role: str = Field(default="", description="Role (requesting, reviewing, consulting)")


class URDocumentOverview(BaseModel):
    """Document overview section for UR decision documents"""
    document_type: str = Field(default="", description="Type of decision document")
    document_date: str = Field(default="", description="Date of the document")
    decision_date: str = Field(default="", description="Date of the decision")
    document_id: str = Field(default="", description="Document identifier")
    claim_case_number: Optional[str] = Field(default=None, description="Claim or case number")
    jurisdiction: str = Field(default="", description="Jurisdiction")
    author: URAuthorInfo = Field(default_factory=URAuthorInfo, description="Author/signature info")


class URPatientInfo(BaseModel):
    """Patient information for UR documents"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    member_id: str = Field(default="", description="Member ID")


class URRequestingProvider(BaseModel):
    """Requesting provider information"""
    name: str = Field(default="", description="Provider's name")
    specialty: str = Field(default="", description="Provider's specialty")
    npi: str = Field(default="", description="NPI number")


class URReviewingEntity(BaseModel):
    """Reviewing entity information"""
    name: str = Field(default="", description="Reviewing entity name")
    reviewer: str = Field(default="", description="Reviewer name")
    credentials: str = Field(default="", description="Reviewer credentials")


class URPartiesInvolved(BaseModel):
    """All parties involved in the UR decision"""
    patient: URPatientInfo = Field(default_factory=URPatientInfo, description="Patient information")
    requesting_provider: URRequestingProvider = Field(default_factory=URRequestingProvider, description="Requesting provider")
    reviewing_entity: URReviewingEntity = Field(default_factory=URReviewingEntity, description="Reviewing entity")
    claims_administrator: str = Field(default="", description="Claims administrator name")
    all_doctors_involved: List[URDoctorInfo] = Field(default_factory=list, description="All doctors mentioned")


class URRequestDetails(BaseModel):
    """Request details section"""
    date_of_service_requested: str = Field(default="", description="Date of service requested")
    request_received: str = Field(default="", description="Date request was received")
    requested_services: List[str] = Field(default_factory=list, description="List of requested services with CPT codes")
    clinical_reason: str = Field(default="", description="Clinical reason for request")


class URPartialDecision(BaseModel):
    """Partial decision breakdown item"""
    service: str = Field(default="", description="Service name")
    decision: str = Field(default="", description="Decision (approved/denied)")
    quantity: str = Field(default="", description="Quantity approved/denied")


class URDecisionOutcome(BaseModel):
    """Decision outcome section"""
    overall_decision: str = Field(default="", description="Overall decision (APPROVED/DENIED/PARTIALLY APPROVED/PENDING)")
    decision_details: str = Field(default="", description="Details of the decision")
    partial_decision_breakdown: List[URPartialDecision] = Field(default_factory=list, description="Breakdown for partial decisions")
    effective_dates: str = Field(default="", description="Effective start/end dates")


class URMedicalNecessity(BaseModel):
    """Medical necessity determination section"""
    medical_necessity: str = Field(default="", description="Medical necessity determination")
    criteria_applied: str = Field(default="", description="Criteria applied (ODG, MTUS, ACOEM, etc.)")
    clinical_rationale: str = Field(default="", description="Clinical rationale for decision")
    guidelines_referenced: List[str] = Field(default_factory=list, description="Guidelines referenced")


class URReviewerAnalysis(BaseModel):
    """Reviewer analysis section"""
    clinical_summary_reviewed: str = Field(default="", description="Clinical summary that was reviewed")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from review")
    documentation_gaps: List[str] = Field(default_factory=list, description="Documentation gaps noted")


class URAppealInformation(BaseModel):
    """Appeal information section"""
    appeal_deadline: str = Field(default="", description="Deadline for appeal")
    appeal_procedures: str = Field(default="", description="Appeal procedures")
    required_documentation: List[str] = Field(default_factory=list, description="Required documentation for appeal")
    timeframe_for_response: str = Field(default="", description="Timeframe for response")


class URLongSummary(BaseModel):
    """Complete structured UR/IMR decision document long summary"""
    content_type: Literal["ur_decision"] = Field(default="ur_decision", description="Type of content")
    document_overview: URDocumentOverview = Field(default_factory=URDocumentOverview, description="Document overview")
    parties_involved: URPartiesInvolved = Field(default_factory=URPartiesInvolved, description="Parties involved")
    request_details: URRequestDetails = Field(default_factory=URRequestDetails, description="Request details")
    decision_outcome: URDecisionOutcome = Field(default_factory=URDecisionOutcome, description="Decision outcome")
    medical_necessity: URMedicalNecessity = Field(default_factory=URMedicalNecessity, description="Medical necessity determination")
    reviewer_analysis: URReviewerAnalysis = Field(default_factory=URReviewerAnalysis, description="Reviewer analysis")
    appeal_information: URAppealInformation = Field(default_factory=URAppealInformation, description="Appeal information")
    critical_actions_required: List[str] = Field(default_factory=list, description="Critical time-sensitive actions")


def format_ur_long_summary(summary: URLongSummary) -> str:
    """Format URLongSummary Pydantic model to readable text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 DECISION DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_overview.document_type:
        lines.append(f"Document Type: {summary.document_overview.document_type}")
    if summary.document_overview.document_date:
        lines.append(f"Document Date: {summary.document_overview.document_date}")
    if summary.document_overview.decision_date:
        lines.append(f"Decision Date: {summary.document_overview.decision_date}")
    if summary.document_overview.document_id:
        lines.append(f"Document ID: {summary.document_overview.document_id}")
    if summary.document_overview.claim_case_number:
        lines.append(f"Claim/Case Number: {summary.document_overview.claim_case_number}")
    if summary.document_overview.jurisdiction:
        lines.append(f"Jurisdiction: {summary.document_overview.jurisdiction}")
    if summary.document_overview.author.signature:
        lines.append("Author:")
        lines.append(f"• Signature: {summary.document_overview.author.signature}")
    lines.append("")
    
    # Parties Involved
    lines.append("👥 PARTIES INVOLVED")
    lines.append("-" * 50)
    patient = summary.parties_involved.patient
    if patient.name:
        lines.append(f"Patient: {patient.name}")
        if patient.date_of_birth:
            lines.append(f"  DOB: {patient.date_of_birth}")
        if patient.member_id:
            lines.append(f"  Member ID: {patient.member_id}")
    
    req_provider = summary.parties_involved.requesting_provider
    if req_provider.name:
        lines.append(f"Requesting Provider: {req_provider.name}")
        if req_provider.specialty:
            lines.append(f"  Specialty: {req_provider.specialty}")
        if req_provider.npi:
            lines.append(f"  NPI: {req_provider.npi}")
    
    rev_entity = summary.parties_involved.reviewing_entity
    if rev_entity.name:
        lines.append(f"Reviewing Entity: {rev_entity.name}")
        if rev_entity.reviewer:
            lines.append(f"  Reviewer: {rev_entity.reviewer}")
        if rev_entity.credentials:
            lines.append(f"  Credentials: {rev_entity.credentials}")
    
    if summary.parties_involved.claims_administrator:
        lines.append(f"Claims Administrator: {summary.parties_involved.claims_administrator}")
    
    if summary.parties_involved.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.parties_involved.all_doctors_involved[:10]:
            title = f", {doctor.title}" if doctor.title else ""
            role = f" ({doctor.role})" if doctor.role else ""
            lines.append(f"• {doctor.name}{title}{role}")
    lines.append("")
    
    # Request Details
    lines.append("📋 REQUEST DETAILS")
    lines.append("-" * 50)
    if summary.request_details.date_of_service_requested:
        lines.append(f"Date of Service Requested: {summary.request_details.date_of_service_requested}")
    if summary.request_details.request_received:
        lines.append(f"Request Received: {summary.request_details.request_received}")
    if summary.request_details.requested_services:
        lines.append("Requested Services:")
        for service in summary.request_details.requested_services[:10]:
            lines.append(f"• {service}")
    if summary.request_details.clinical_reason:
        lines.append(f"Clinical Reason: {summary.request_details.clinical_reason}")
    lines.append("")
    
    # Decision Outcome
    lines.append("⚖️ DECISION OUTCOME")
    lines.append("-" * 50)
    if summary.decision_outcome.overall_decision:
        lines.append(f"Overall Decision: {summary.decision_outcome.overall_decision}")
    if summary.decision_outcome.decision_details:
        lines.append(f"Decision Details: {summary.decision_outcome.decision_details}")
    if summary.decision_outcome.partial_decision_breakdown:
        lines.append("Partial Decision Breakdown:")
        for item in summary.decision_outcome.partial_decision_breakdown[:5]:
            qty = f" ({item.quantity})" if item.quantity else ""
            lines.append(f"• {item.service}: {item.decision}{qty}")
    if summary.decision_outcome.effective_dates:
        lines.append(f"Effective Dates: {summary.decision_outcome.effective_dates}")
    lines.append("")
    
    # Medical Necessity
    lines.append("🏥 MEDICAL NECESSITY DETERMINATION")
    lines.append("-" * 50)
    if summary.medical_necessity.medical_necessity:
        lines.append(f"Medical Necessity: {summary.medical_necessity.medical_necessity}")
    if summary.medical_necessity.criteria_applied:
        lines.append(f"Criteria Applied: {summary.medical_necessity.criteria_applied}")
    if summary.medical_necessity.clinical_rationale:
        lines.append(f"Clinical Rationale: {summary.medical_necessity.clinical_rationale}")
    if summary.medical_necessity.guidelines_referenced:
        lines.append("Guidelines Referenced:")
        for guideline in summary.medical_necessity.guidelines_referenced[:5]:
            lines.append(f"• {guideline}")
    lines.append("")
    
    # Reviewer Analysis
    lines.append("🔍 REVIEWER ANALYSIS")
    lines.append("-" * 50)
    if summary.reviewer_analysis.clinical_summary_reviewed:
        lines.append(f"Clinical Summary Reviewed: {summary.reviewer_analysis.clinical_summary_reviewed}")
    if summary.reviewer_analysis.key_findings:
        lines.append("Key Findings:")
        for finding in summary.reviewer_analysis.key_findings[:5]:
            lines.append(f"• {finding}")
    if summary.reviewer_analysis.documentation_gaps:
        lines.append("Documentation Gaps:")
        for gap in summary.reviewer_analysis.documentation_gaps[:3]:
            lines.append(f"• {gap}")
    lines.append("")
    
    # Appeal Information
    lines.append("🔄 APPEAL INFORMATION")
    lines.append("-" * 50)
    if summary.appeal_information.appeal_deadline:
        lines.append(f"Appeal Deadline: {summary.appeal_information.appeal_deadline}")
    if summary.appeal_information.appeal_procedures:
        lines.append(f"Appeal Procedures: {summary.appeal_information.appeal_procedures}")
    if summary.appeal_information.required_documentation:
        lines.append("Required Documentation:")
        for doc in summary.appeal_information.required_documentation[:5]:
            lines.append(f"• {doc}")
    if summary.appeal_information.timeframe_for_response:
        lines.append(f"Timeframe for Response: {summary.appeal_information.timeframe_for_response}")
    lines.append("")
    
    # Critical Actions Required
    if summary.critical_actions_required:
        lines.append("🚨 CRITICAL ACTIONS REQUIRED")
        lines.append("-" * 50)
        for action in summary.critical_actions_required[:8]:
            lines.append(f"• {action}")
    
    return "\n".join(lines)


def create_fallback_ur_summary(doc_type: str, fallback_date: str) -> URLongSummary:
    """Create a fallback UR long summary when extraction fails."""
    return URLongSummary(
        content_type="ur_decision",
        document_overview=URDocumentOverview(
            document_type=doc_type,
            document_date=fallback_date
        )
    )
