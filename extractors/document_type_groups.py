"""
Document type groups - organized by category for better maintainability
"""

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


class DocumentTypeGroups:
    """Organized groups of document types for easy management"""
    
    # ==================== QME & EVALUATION TYPES ====================
    MEDICAL_EVALUATIONS = {
        DocumentType.QME.value: "QME Prompts",
        DocumentType.AME.value: "QME Prompts", 
        DocumentType.IME.value: "QME Prompts",
        DocumentType.IMR.value: "QME Prompts",
        DocumentType.FCE.value: "QME Prompts",
        DocumentType.PEER_REVIEW.value: "QME Prompts",
        DocumentType.INDEPENDENT_REVIEW.value: "QME Prompts",
    }
    
    # ==================== UR & AUTHORIZATION TYPES ====================
    AUTHORIZATION_DOCS = {
        DocumentType.UR.value: "UR Prompts",
        DocumentType.RFA.value: "UR Prompts",
        DocumentType.AUTHORIZATION.value: "UR Prompts",
        DocumentType.PEER_TO_PEER.value: "UR Prompts",
        DocumentType.TREATMENT_AUTH.value: "UR Prompts",
        DocumentType.PROCEDURE_AUTH.value: "UR Prompts",
        DocumentType.APPEAL.value: "UR Prompts",
        DocumentType.DENIAL_LETTER.value: "UR Prompts",
        DocumentType.APPROVAL_LETTER.value: "UR Prompts",
        DocumentType.PRIOR_AUTH.value: "UR Prompts",
    }
    
    # ==================== IMAGING TYPES ====================
    IMAGING_DOCS = {
        DocumentType.MRI.value: "Imaging Prompts",
        DocumentType.CT.value: "Imaging Prompts",
        DocumentType.XRAY.value: "Imaging Prompts",
        DocumentType.ULTRASOUND.value: "Imaging Prompts",
        DocumentType.MAMMOGRAM.value: "Imaging Prompts",
        DocumentType.PET_SCAN.value: "Imaging Prompts",
        DocumentType.BONE_SCAN.value: "Imaging Prompts",
        DocumentType.DEXA_SCAN.value: "Imaging Prompts",
        DocumentType.FLUOROSCOPY.value: "Imaging Prompts",
        DocumentType.ANGIOGRAM.value: "Imaging Prompts",
        DocumentType.EMG.value: "Imaging Prompts",
    }
    
    # ==================== SURGICAL TYPES ====================
    SURGICAL_DOCS = {
        DocumentType.SURGERY_REPORT.value: "Surgical Prompts",
        DocumentType.OPERATIVE_NOTE.value: "Surgical Prompts",
        DocumentType.ANESTHESIA_REPORT.value: "Surgical Prompts",
        DocumentType.PRE_OP.value: "Surgical Prompts",
        DocumentType.POST_OP.value: "Surgical Prompts",
        DocumentType.DISCHARGE.value: "Surgical Prompts",
    }
    
    # ==================== THERAPY TYPES ====================
    THERAPY_DOCS = {
        DocumentType.PHYSICAL_THERAPY.value: "Therapy Prompts",
        DocumentType.OCCUPATIONAL_THERAPY.value: "Therapy Prompts",
        DocumentType.CHIROPRACTIC.value: "Therapy Prompts",
        DocumentType.ACUPUNCTURE.value: "Therapy Prompts",
        DocumentType.MASSAGE_THERAPY.value: "Therapy Prompts",
        DocumentType.PAIN_MANAGEMENT.value: "Therapy Prompts",
    }
    
    # ==================== PROGRESS NOTES ====================
    PROGRESS_NOTES = {
        DocumentType.PR2.value: "Progress Prompts",
        DocumentType.PR4.value: "Progress Prompts",
        DocumentType.DFR.value: "Progress Prompts",
        DocumentType.CONSULT.value: "Progress Prompts",
        DocumentType.PROGRESS_NOTE.value: "Progress Prompts",
        DocumentType.OFFICE_VISIT.value: "Progress Prompts",
        DocumentType.CLINIC_NOTE.value: "Progress Prompts",
        DocumentType.TELEMEDICINE.value: "Progress Prompts",
    }
    
    # ==================== HOSPITAL DOCS ====================
    HOSPITAL_DOCS = {
        DocumentType.ADMISSION_NOTE.value: "Hospital Prompts",
        DocumentType.HOSPITAL_COURSE.value: "Hospital Prompts",
        DocumentType.ER_REPORT.value: "Hospital Prompts",
        DocumentType.EMERGENCY_ROOM.value: "Hospital Prompts",
        DocumentType.HOSPITAL_PROGRESS.value: "Hospital Prompts",
    }
    
    # ==================== LABORATORY DOCS ====================
    LABORATORY_DOCS = {
        DocumentType.LABS.value: "Lab Prompts",
        DocumentType.PATHOLOGY.value: "Lab Prompts",
        DocumentType.BIOPSY.value: "Lab Prompts",
        DocumentType.GENETIC_TESTING.value: "Lab Prompts",
        DocumentType.TOXICOLOGY.value: "Lab Prompts",
        DocumentType.ALLERGY_TESTING.value: "Lab Prompts",
    }
    
    # ==================== SPECIALTY REPORTS ====================
    SPECIALTY_REPORTS = {
        DocumentType.CARDIOLOGY.value: "Specialty Prompts",
        DocumentType.NEUROLOGY.value: "Specialty Prompts",
        DocumentType.ORTHOPEDICS.value: "Specialty Prompts",
        DocumentType.PSYCHIATRY.value: "Specialty Prompts",
        DocumentType.PSYCHOLOGY.value: "Specialty Prompts",
        DocumentType.PSYCHOTHERAPY.value: "Specialty Prompts",
        DocumentType.BEHAVIORAL_HEALTH.value: "Specialty Prompts",
        DocumentType.RHEUMATOLOGY.value: "Specialty Prompts",
        DocumentType.ENDOCRINOLOGY.value: "Specialty Prompts",
        DocumentType.GASTROENTEROLOGY.value: "Specialty Prompts",
        DocumentType.PULMONOLOGY.value: "Specialty Prompts",
    }
    
    # ==================== DIAGNOSTIC STUDIES ====================
    DIAGNOSTIC_STUDIES = {
        DocumentType.SLEEP_STUDY.value: "Diagnostic Prompts",
        DocumentType.EKG.value: "Diagnostic Prompts",
        DocumentType.ECG.value: "Diagnostic Prompts",
        DocumentType.HOLTER_MONITOR.value: "Diagnostic Prompts",
        DocumentType.ECHO.value: "Diagnostic Prompts",
        DocumentType.STRESS_TEST.value: "Diagnostic Prompts",
        DocumentType.PULMONARY_FUNCTION.value: "Diagnostic Prompts",
        DocumentType.NERVE_CONDUCTION.value: "Diagnostic Prompts",
    }
    
    # ==================== ALL GROUPS COMBINED ====================
    @classmethod
    def get_all_groups(cls):
        """Get all document type groups"""
        return {
            "MEDICAL_EVALUATIONS": cls.MEDICAL_EVALUATIONS,
            "AUTHORIZATION_DOCS": cls.AUTHORIZATION_DOCS,
            "IMAGING_DOCS": cls.IMAGING_DOCS,
            "SURGICAL_DOCS": cls.SURGICAL_DOCS,
            "THERAPY_DOCS": cls.THERAPY_DOCS,
            "PROGRESS_NOTES": cls.PROGRESS_NOTES,
            "HOSPITAL_DOCS": cls.HOSPITAL_DOCS,
            "LABORATORY_DOCS": cls.LABORATORY_DOCS,
            "SPECIALTY_REPORTS": cls.SPECIALTY_REPORTS,
            "DIAGNOSTIC_STUDIES": cls.DIAGNOSTIC_STUDIES,
        }