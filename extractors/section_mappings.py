"""
Professional Section Mappings for ALL Document Types
"""

from typing import List, Tuple

class SectionMappings:
    """Professional section mappings organized by document type groups"""
    
    # Base sections for unknown document types
    BASE_SECTIONS = [
        ('DOCUMENT OVERVIEW', 'key_findings'),
        ('CLINICAL ASSESSMENT', 'clinical_findings'),
        ('DIAGNOSIS AND FINDINGS', 'diagnosis'),
        ('TREATMENT AND RECOMMENDATIONS', 'treatment'),
        ('FOLLOW-UP PLAN', 'recommendations'),
    ]
    
    # ==================== MEDICAL EVALUATIONS ====================
    MEDICAL_EVALUATION_SECTIONS = [
        ('EVALUATION CONTEXT', 'evaluation_context'),
        ('CLINICAL HISTORY', 'clinical_history'),
        ('SUBJECTIVE COMPLAINTS', 'subjective_complaints'),
        ('PHYSICAL EXAMINATION', 'physical_examination'),
        ('DIAGNOSTIC STUDIES', 'diagnostic_studies'),
        ('DIAGNOSIS AND IMPRESSION', 'diagnosis'),
        ('TREATMENT RECOMMENDATIONS', 'treatment'),
        ('MEDICATION MANAGEMENT', 'medications'),
        ('WORK CAPACITY ASSESSMENT', 'work_status'),
        ('FUNCTIONAL LIMITATIONS', 'restrictions'),
        ('PROGNOSIS AND OUTLOOK', 'prognosis'),
        ('FOLLOW-UP RECOMMENDATIONS', 'recommendations'),
    ]
    
    # ==================== AUTHORIZATION & UTILIZATION ====================
    AUTHORIZATION_SECTIONS = [
        ('REVIEW DECISION', 'decision'),
        ('TREATMENT REQUEST', 'treatment_reviewed'),
        ('CLINICAL RATIONALE', 'clinical_criteria'),
        ('UTILIZATION CRITERIA', 'denial_reasons'),
        ('REVIEWER ASSESSMENT', 'reviewer_info'),
        ('APPEAL PROCESS', 'appeal_process'),
        ('ALTERNATIVE RECOMMENDATIONS', 'alternative_recommendations'),
    ]
    
    # ==================== IMAGING REPORTS ====================
    IMAGING_SECTIONS = [
        ('STUDY PROTOCOL', 'study_type'),
        ('TECHNICAL ASPECTS', 'technique'),
        ('COMPARISON STUDIES', 'comparison'),
        ('RADIOLOGIC FINDINGS', 'findings'),
        ('CLINICAL CORRELATION', 'impression'),
        ('RECOMMENDATIONS', 'recommendations'),
    ]
    
    # ==================== SURGICAL REPORTS ====================
    SURGICAL_SECTIONS = [
        ('PREOPERATIVE ASSESSMENT', 'pre_op_diagnosis'),
        ('OPERATIVE PROCEDURE', 'procedure'),
        ('INTRAOPERATIVE FINDINGS', 'findings'),
        ('POSTOPERATIVE DIAGNOSIS', 'post_op_diagnosis'),
        ('SURGICAL COMPLICATIONS', 'complications'),
        ('POSTOPERATIVE CARE', 'treatment'),
    ]
    
    # ==================== THERAPY NOTES ====================
    THERAPY_SECTIONS = [
        ('TREATMENT SESSION', 'treatment_plan'),
        ('FUNCTIONAL ASSESSMENT', 'functional_status'),
        ('THERAPEUTIC PROGRESS', 'progress'),
        ('TREATMENT GOALS', 'goals'),
        ('HOME PROGRAM', 'home_program'),
        ('RECOMMENDATIONS', 'recommendations'),
    ]
    
    # ==================== PROGRESS NOTES ====================
    PROGRESS_NOTE_SECTIONS = [
        ('SUBJECTIVE ASSESSMENT', 'subjective'),
        ('OBJECTIVE FINDINGS', 'objective'),
        ('CLINICAL ASSESSMENT', 'assessment'),
        ('TREATMENT PLAN', 'plan'),
        ('MEDICATION REVIEW', 'medications'),
        ('FOLLOW-UP PLAN', 'recommendations'),
    ]
    
    # ==================== HOSPITAL DOCUMENTS ====================
    HOSPITAL_SECTIONS = [
        ('ADMISSION ASSESSMENT', 'admission_diagnosis'),
        ('HOSPITAL COURSE', 'hospital_course'),
        ('DISCHARGE SUMMARY', 'discharge_diagnosis'),
        ('DISCHARGE MEDICATIONS', 'discharge_medications'),
        ('DISCHARGE INSTRUCTIONS', 'discharge_instructions'),
        ('FOLLOW-UP CARE', 'follow_up'),
    ]
    
    # ==================== LABORATORY REPORTS ====================
    LABORATORY_SECTIONS = [
        ('LABORATORY RESULTS', 'test_results'),
        ('REFERENCE VALUES', 'reference_ranges'),
        ('ABNORMAL FINDINGS', 'abnormal_findings'),
        ('CLINICAL INTERPRETATION', 'interpretation'),
        ('RECOMMENDATIONS', 'recommendations'),
    ]
    
    # ==================== SPECIALTY CONSULTATIONS ====================
    SPECIALTY_SECTIONS = [
        ('CONSULTATION REASON', 'clinical_findings'),
        ('SPECIALTY EXAMINATION', 'objective_findings'),
        ('DIAGNOSTIC IMPRESSION', 'diagnosis'),
        ('SPECIALTY RECOMMENDATIONS', 'treatment'),
        ('FOLLOW-UP PLAN', 'recommendations'),
    ]
    
    # ==================== WORK & VOCATIONAL ====================
    WORK_VOCATIONAL_SECTIONS = [
        ('FUNCTIONAL CAPACITY', 'work_status'),
        ('WORK RESTRICTIONS', 'restrictions'),
        ('JOB ANALYSIS', 'key_findings'),
        ('VOCATIONAL ASSESSMENT', 'clinical_findings'),
        ('RETURN-TO-WORK PLAN', 'treatment'),
    ]
    
    # ==================== ADMINISTRATIVE & LEGAL ====================
    ADMINISTRATIVE_SECTIONS = [
        ('CASE OVERVIEW', 'key_findings'),
        ('CORRESPONDENCE DETAILS', 'clinical_findings'),
        ('DECISIONS AND ACTIONS', 'decision'),
        ('NEXT STEPS', 'recommendations'),
        ('TIMELINES AND DEADLINES', 'treatment'),
    ]
    
    # Map document groups to their section mappings
    GROUP_SECTION_MAPPINGS = {
        "MEDICAL_EVALUATIONS": MEDICAL_EVALUATION_SECTIONS,
        "AUTHORIZATION_DOCS": AUTHORIZATION_SECTIONS,
        "IMAGING_DOCS": IMAGING_SECTIONS,
        "SURGICAL_DOCS": SURGICAL_SECTIONS,
        "THERAPY_DOCS": THERAPY_SECTIONS,
        "PROGRESS_NOTES": PROGRESS_NOTE_SECTIONS,
        "HOSPITAL_DOCS": HOSPITAL_SECTIONS,
        "LABORATORY_DOCS": LABORATORY_SECTIONS,
        "SPECIALTY_REPORTS": SPECIALTY_SECTIONS,
        "DIAGNOSTIC_STUDIES": IMAGING_SECTIONS,
        "WORK_VOCATIONAL": WORK_VOCATIONAL_SECTIONS,
        "ADMINISTRATIVE_LEGAL": ADMINISTRATIVE_SECTIONS,
    }
    
    @classmethod
    def get_sections_for_group(cls, group_name: str) -> List[Tuple[str, str]]:
        """Get section mapping for a document group"""
        return cls.GROUP_SECTION_MAPPINGS.get(group_name, cls.BASE_SECTIONS)