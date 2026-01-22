"""
Fail Document Update Service - Handles updating and reprocessing failed documents
Extracted from webhook_service.py for better modularity
"""
from datetime import datetime
from typing import Any, List, Optional, Dict
from fastapi import HTTPException
from pydantic import BaseModel, Field
from models.data_models import DocumentAnalysis
from services.report_analyzer import ReportAnalyzer
from utils.logger import logger
from utils.document_detector import detect_document_type
import asyncio
import json
import re
from enum import Enum
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from config.settings import CONFIG

# ============== Document Type Field Mappings ==============
# Essential Clinical Data Points by Document Type - Physician Requirements

DOCUMENT_FIELD_MAPPINGS = {
    # === Medical Reports ===
    "RFA": {
        "critical_fields": [
            "Requested Procedure/Treatment/Service",
            "Medical Necessity Justification",
            "Previous Failed Treatments",
            "Injury Relationship/Causation",
            "Urgency Level"
        ],
        "important_fields": [
            "Provider Credentials",
            "Supporting Objective Findings",
            "Duration/Frequency",
            "Expected Outcomes"
        ],
        "impact_factors": ["Prior authorization history", "Contraindications", "Alternative treatments considered"]
    },
    "PR2": {
        "critical_fields": [
            "Subjective Complaints",
            "Objective Findings",
            "Treatment Response",
            "Work Status",
            "P&S Date Estimate"
        ],
        "important_fields": [
            "Current Treatment Modality",
            "Causation Statement",
            "Future Treatment Plan"
        ],
        "impact_factors": ["Medication changes", "Therapy compliance", "New symptoms", "Work attempts"]
    },
    "PR4": {
        "critical_fields": [
            "MMI Declaration",
            "Permanent Impairment Rating",
            "Permanent Work Restrictions",
            "Apportionment",
            "Causation Analysis"
        ],
        "important_fields": [
            "Future Medical Care",
            "Body Parts Affected",
            "AMA Guidelines Used"
        ],
        "impact_factors": ["All prior treatments", "Imaging findings", "Baseline pre-injury status", "FCE results"]
    },
    "DFR": {
        "critical_fields": [
            "Injury Mechanism",
            "Body Parts Injured",
            "Objective Findings",
            "Initial Diagnosis",
            "Work Status"
        ],
        "important_fields": [
            "Treatment Provided",
            "Disability Dates",
            "Causation"
        ],
        "impact_factors": ["Patient's description of incident", "Timing of symptom onset", "Pre-existing conditions"]
    },
    "QME": {
        "critical_fields": [
            "Causation Opinion",
            "MMI Status",
            "Impairment Rating (WPI)",
            "Diagnosis",
            "Permanent Work Restrictions"
        ],
        "important_fields": [
            "History Review",
            "Physical Examination",
            "Diagnostic Test Review",
            "Future Medical Care",
            "Apportionment"
        ],
        "impact_factors": ["All medical records reviewed", "Consistency of complaints", "Waddell signs", "Validity testing"]
    },
    "AME": {
        "critical_fields": [
            "Causation Opinion",
            "MMI Status",
            "Impairment Rating (WPI)",
            "Diagnosis",
            "Permanent Work Restrictions"
        ],
        "important_fields": [
            "History Review",
            "Physical Examination",
            "Diagnostic Test Review",
            "Future Medical Care",
            "Apportionment"
        ],
        "impact_factors": ["All medical records reviewed", "Consistency of complaints", "Waddell signs", "Validity testing"]
    },
    "IME": {
        "critical_fields": [
            "Causation Opinion",
            "MMI Status",
            "Impairment Rating (WPI)",
            "Diagnosis",
            "Permanent Work Restrictions"
        ],
        "important_fields": [
            "History Review",
            "Physical Examination",
            "Diagnostic Test Review",
            "Future Medical Care",
            "Apportionment"
        ],
        "impact_factors": ["All medical records reviewed", "Consistency of complaints", "Waddell signs", "Validity testing"]
    },
    "IMR": {
        "critical_fields": [
            "Final Decision",
            "Medical Necessity Determination",
            "UR Decision Being Reviewed",
            "Rationale"
        ],
        "important_fields": [
            "Medical Evidence Submitted",
            "Reviewer's Credentials",
            "Evidence-Based Guidelines Cited"
        ],
        "impact_factors": ["Treatment guidelines", "Clinical studies cited", "Alternative treatments"]
    },
    "UR": {
        "critical_fields": [
            "Decision",
            "Rationale",
            "Medical Necessity Assessment",
            "Appeal Rights"
        ],
        "important_fields": [
            "Guidelines Referenced",
            "Alternative Treatments",
            "Physician Reviewer Credentials",
            "Timeframe"
        ],
        "impact_factors": ["Medical records reviewed", "Treatment duration", "Evidence of efficacy"]
    },
    
    # === Clinical Documentation ===
    "CONSULT": {
        "critical_fields": [
            "Chief Complaint",
            "Assessment/Diagnosis",
            "Plan",
            "Work Status Update"
        ],
        "important_fields": [
            "History of Present Illness",
            "Physical Exam Findings",
            "Medication Changes",
            "Follow-up Plan"
        ],
        "impact_factors": ["Vital signs", "Diagnostic test orders", "Specialist referrals"]
    },
    "PROGRESS_NOTE": {
        "critical_fields": [
            "Clinical Course",
            "Pain Scale",
            "Functional Status",
            "Treatment Modifications"
        ],
        "important_fields": [
            "Interval History",
            "Treatment Compliance",
            "Side Effects",
            "Objective Measurements"
        ],
        "impact_factors": ["Home exercise compliance", "Work activity level", "New injuries"]
    },
    "TREATMENT_PLAN": {
        "critical_fields": [
            "Goals",
            "Treatment Frequency",
            "Treatment Duration",
            "Discharge Criteria"
        ],
        "important_fields": [
            "Modalities",
            "Measurable Outcomes",
            "Home Program"
        ],
        "impact_factors": ["Baseline functional status", "Patient motivation", "Equipment needs"]
    },
    "WORK_STATUS": {
        "critical_fields": [
            "Work Status",
            "Restrictions",
            "Duration of Restrictions",
            "Accommodations Needed"
        ],
        "important_fields": [
            "Next Review Date"
        ],
        "impact_factors": ["Job description comparison", "FCE results", "Employer capabilities"]
    },
    "DISCHARGE_SUMMARY": {
        "critical_fields": [
            "Discharge Diagnosis",
            "Hospital Course",
            "Complications",
            "Discharge Medications"
        ],
        "important_fields": [
            "Admission Diagnosis",
            "Follow-up Instructions",
            "Activity Restrictions"
        ],
        "impact_factors": ["Surgical procedures", "ICU stay", "Consults obtained", "Pathology results"]
    },
    "ED_REPORT": {
        "critical_fields": [
            "Chief Complaint and Triage",
            "ED Diagnosis",
            "Treatment Provided",
            "Disposition"
        ],
        "important_fields": [
            "Vital Signs",
            "Physical Exam",
            "Diagnostic Testing",
            "Discharge Instructions"
        ],
        "impact_factors": ["Mechanism of injury", "Time to presentation", "Trauma activation"]
    },
    
    # === Imaging & Diagnostics ===
    "MRI": {
        "critical_fields": [
            "Impression",
            "Key Findings",
            "Pathology Severity",
            "Clinical Correlation Needed"
        ],
        "important_fields": [
            "Indication",
            "Technique",
            "Comparison to Prior"
        ],
        "impact_factors": ["Age-related degeneration", "Acute vs chronic findings", "Mass effect"]
    },
    "CT": {
        "critical_fields": [
            "Impression",
            "Key Findings",
            "Measurements"
        ],
        "important_fields": [
            "Indication",
            "Technique",
            "Comparison to Prior"
        ],
        "impact_factors": ["Trauma protocol used", "IV contrast timing", "Artifact presence"]
    },
    "XRAY": {
        "critical_fields": [
            "Impression",
            "Bone Alignment",
            "Joint Spaces"
        ],
        "important_fields": [
            "Views Obtained",
            "Soft Tissue Findings",
            "Hardware",
            "Comparison"
        ],
        "impact_factors": ["Weight-bearing vs non-weight-bearing", "Stress views"]
    },
    "EMG": {
        "critical_fields": [
            "Conclusion",
            "Abnormal Findings",
            "Localization",
            "Severity"
        ],
        "important_fields": [
            "Indication",
            "Nerves Tested",
            "Denervation",
            "Clinical Correlation"
        ],
        "impact_factors": ["Medications affecting results", "Temperature", "Patient effort"]
    },
    "LAB": {
        "critical_fields": [
            "Critical Values",
            "Flagged Abnormalities",
            "Clinical Significance"
        ],
        "important_fields": [
            "Test Ordered",
            "Reference Ranges",
            "Trends"
        ],
        "impact_factors": ["Fasting status", "Timing of collection", "Medication effects"]
    },
    "PATHOLOGY": {
        "critical_fields": [
            "Diagnosis",
            "Margins",
            "Grade/Stage"
        ],
        "important_fields": [
            "Specimen Type",
            "Gross Description",
            "Microscopic Findings",
            "Additional Testing"
        ],
        "impact_factors": ["Adequacy of sample", "Frozen section correlation"]
    },
    
    # === Procedure Reports ===
    "SURGERY": {
        "critical_fields": [
            "Procedure Performed",
            "Findings",
            "Complications",
            "Postoperative Diagnosis"
        ],
        "important_fields": [
            "Preoperative Diagnosis",
            "Technique",
            "Specimens Sent",
            "Estimated Blood Loss",
            "Hardware Used"
        ],
        "impact_factors": ["Anesthesia type", "Positioning", "Antibiotic prophylaxis", "Operative time"]
    },
    "OPERATIVE_REPORT": {
        "critical_fields": [
            "Procedure Performed",
            "Findings",
            "Complications",
            "Postoperative Diagnosis"
        ],
        "important_fields": [
            "Preoperative Diagnosis",
            "Technique",
            "Specimens Sent",
            "Estimated Blood Loss",
            "Hardware Used"
        ],
        "impact_factors": ["Anesthesia type", "Positioning", "Antibiotic prophylaxis", "Operative time"]
    },
    
    # === Specialty Reports ===
    "CARDIOLOGY": {
        "critical_fields": [
            "Impression",
            "Findings",
            "Ejection Fraction",
            "Ischemia"
        ],
        "important_fields": [
            "Test Type",
            "Indication",
            "Rhythm",
            "Valve Function",
            "Recommendations"
        ],
        "impact_factors": ["Medications during test", "Symptoms during stress", "Cardiac enzymes"]
    },
    "PAIN_MANAGEMENT": {
        "critical_fields": [
            "Pain Location and Quality",
            "Pain Scale",
            "Medication Regimen (MME)",
            "Treatment Plan"
        ],
        "important_fields": [
            "Injection Procedures",
            "Physical Exam",
            "Diagnostic Blocks",
            "Opioid Risk Assessment",
            "Functional Goals"
        ],
        "impact_factors": ["Urine drug screens", "Prescription monitoring", "Opioid agreements"]
    },
    "PSYCH": {
        "critical_fields": [
            "Psychiatric Diagnoses",
            "Mental Status Exam",
            "Suicide Risk",
            "Causation Opinion"
        ],
        "important_fields": [
            "Psychosocial Stressors",
            "Psychological Testing",
            "Treatment Recommendations",
            "Medication Management"
        ],
        "impact_factors": ["Litigation stress", "Chronic pain impact", "Secondary gain assessment"]
    },
    "PT_OT_CHIRO": {
        "critical_fields": [
            "Objective Progress",
            "Patient Response",
            "Functional Goals Status",
            "Discharge Planning"
        ],
        "important_fields": [
            "Baseline Measurements",
            "Treatment Provided",
            "Compliance",
            "Plan Modifications"
        ],
        "impact_factors": ["Equipment used", "Exercise repetitions", "Treatment duration"]
    },
    "FCE": {
        "critical_fields": [
            "Validity of Effort",
            "Lift Capacity",
            "Physical Demand Level",
            "Work Restrictions",
            "Return to Work Recommendation"
        ],
        "important_fields": [
            "Positional Tolerances",
            "Functional Deficits",
            "Reliability"
        ],
        "impact_factors": ["Pain behaviors", "Submaximal effort indicators", "Job description match"]
    },
    
    # === Administrative & Legal ===
    "PEER_REVIEW": {
        "critical_fields": [
            "Recommendation",
            "Medical Necessity Opinion",
            "Standard of Care Assessment",
            "Rationale"
        ],
        "important_fields": [
            "Reviewer Credentials",
            "Records Reviewed",
            "Clinical Question",
            "Alternative Treatments"
        ],
        "impact_factors": ["Treatment guidelines cited", "Literature references", "Regional practice patterns"]
    },
    "MEDICATION": {
        "critical_fields": [
            "Current Medications",
            "Changes Made",
            "Drug Interactions",
            "Side Effects"
        ],
        "important_fields": [
            "Indication",
            "Adherence",
            "Formulary Status"
        ],
        "impact_factors": ["Prescription monitoring data", "Opioid MME", "Controlled substance agreements"]
    },
    "LEGAL_CORRESPONDENCE": {
        "critical_fields": [
            "Legal Question Posed",
            "Medical Opinion Provided",
            "Causation Analysis",
            "Permanent Disability"
        ],
        "important_fields": [
            "Supporting Evidence",
            "Deposition Testimony"
        ],
        "impact_factors": ["Conflicting medical opinions", "Record gaps", "Pre-existing conditions"]
    },
    "DISABILITY_FORM": {
        "critical_fields": [
            "Disability Type",
            "Disability Dates",
            "Work Capacity",
            "Causation",
            "Impairment Rating"
        ],
        "important_fields": [
            "Return to Work Prognosis"
        ],
        "impact_factors": ["Job demands analysis", "Vocational assessment", "Treating physician opinions"]
    },
    
    # === Default for unknown types ===
    "DEFAULT": {
        "critical_fields": [
            "Key Findings",
            "Diagnosis",
            "Recommendations",
            "Work Status"
        ],
        "important_fields": [
            "History",
            "Physical Exam",
            "Treatment Plan",
            "Follow-up"
        ],
        "impact_factors": ["Clinical context", "Prior records", "Patient presentation"]
    }
}

# Cross-document integration points - physicians prioritize seeing
CROSS_DOCUMENT_PRIORITIES = [
    "Temporal relationships - symptom onset relative to injury/treatment",
    "Treatment response patterns - efficacy across modalities",
    "Objective findings correlation - exam findings matching imaging",
    "Medication effectiveness/tolerance - therapeutic response and side effects",
    "Functional trajectory - improvement, plateau, or decline over time",
    "Causation consistency - all providers agreeing on injury relationship",
    "Guideline compliance - adherence to evidence-based protocols",
    "Work status evolution - progression toward RTW",
    "Red flags - serious pathology indicators requiring urgent action",
    "Conflicting information - discrepancies requiring resolution"
]

# ============== Dynamic Summary Models ==============

class SimpleClinicalSummary(BaseModel):
    """Simple key-value clinical summary - just the essential data."""
    data: Dict[str, List[str]] = Field(
        description="Key-value pairs where key is the field name and value is list of bullet points"
    )


# ============== Helper Functions ==============

def get_document_type_mapping(document_type: str) -> Dict[str, Any]:
    """Get the field mapping for a specific document type."""
    doc_type_upper = document_type.upper().strip()
    
    # Try exact match first
    if doc_type_upper in DOCUMENT_FIELD_MAPPINGS:
        return DOCUMENT_FIELD_MAPPINGS[doc_type_upper]
    
    # Try partial matching for common variations
    type_aliases = {
        "REQUEST FOR AUTHORIZATION": "RFA",
        "PROGRESS REPORT": "PR2",
        "PERMANENT AND STATIONARY": "PR4",
        "P&S REPORT": "PR4",
        "DOCTOR'S FIRST REPORT": "DFR",
        "QUALIFIED MEDICAL EVALUATION": "QME",
        "AGREED MEDICAL EVALUATION": "AME",
        "INDEPENDENT MEDICAL EVALUATION": "IME",
        "INDEPENDENT MEDICAL REVIEW": "IMR",
        "UTILIZATION REVIEW": "UR",
        "CONSULTATION": "CONSULT",
        "OFFICE VISIT": "CONSULT",
        "TREATMENT PLAN": "TREATMENT_PLAN",
        "WORK STATUS REPORT": "WORK_STATUS",
        "RETURN TO WORK": "WORK_STATUS",
        "RTW": "WORK_STATUS",
        "DISCHARGE SUMMARY": "DISCHARGE_SUMMARY",
        "EMERGENCY DEPARTMENT": "ED_REPORT",
        "ED REPORT": "ED_REPORT",
        "ER REPORT": "ED_REPORT",
        "MRI REPORT": "MRI",
        "CT SCAN": "CT",
        "X-RAY": "XRAY",
        "RADIOGRAPH": "XRAY",
        "EMG/NCS": "EMG",
        "NERVE CONDUCTION": "EMG",
        "LABORATORY": "LAB",
        "LAB REPORT": "LAB",
        "PATHOLOGY REPORT": "PATHOLOGY",
        "BIOPSY": "PATHOLOGY",
        "SURGERY REPORT": "SURGERY",
        "OPERATIVE REPORT": "OPERATIVE_REPORT",
        "CARDIOLOGY REPORT": "CARDIOLOGY",
        "ECHO": "CARDIOLOGY",
        "EKG": "CARDIOLOGY",
        "PAIN MANAGEMENT": "PAIN_MANAGEMENT",
        "PSYCHOLOGICAL": "PSYCH",
        "PSYCHIATRIC": "PSYCH",
        "PHYSICAL THERAPY": "PT_OT_CHIRO",
        "OCCUPATIONAL THERAPY": "PT_OT_CHIRO",
        "CHIROPRACTIC": "PT_OT_CHIRO",
        "FUNCTIONAL CAPACITY": "FCE",
        "PEER REVIEW": "PEER_REVIEW",
        "MEDICATION REPORT": "MEDICATION",
        "PHARMACY": "MEDICATION",
        "LEGAL": "LEGAL_CORRESPONDENCE",
        "ATTORNEY": "LEGAL_CORRESPONDENCE",
        "DISABILITY CLAIM": "DISABILITY_FORM",
        "CLAIM FORM": "DISABILITY_FORM"
    }
    
    for alias, mapped_type in type_aliases.items():
        if alias in doc_type_upper or doc_type_upper in alias:
            return DOCUMENT_FIELD_MAPPINGS.get(mapped_type, DOCUMENT_FIELD_MAPPINGS["DEFAULT"])
    
    return DOCUMENT_FIELD_MAPPINGS["DEFAULT"]


def normalize_document_type(document_type: str) -> str:
    """Normalize document type string to match our mapping keys."""
    doc_type_upper = document_type.upper().strip()
    
    if doc_type_upper in DOCUMENT_FIELD_MAPPINGS:
        return doc_type_upper
    
    # Common normalizations
    normalizations = {
        "REQUEST FOR AUTHORIZATION": "RFA",
        "PROGRESS REPORT": "PR2",
        "PERMANENT AND STATIONARY": "PR4",
        "QUALIFIED MEDICAL EVALUATION": "QME",
        "AGREED MEDICAL EVALUATION": "AME",
        "INDEPENDENT MEDICAL EVALUATION": "IME",
        "CONSULTATION": "CONSULT",
        "OFFICE VISIT": "CONSULT",
        "MRI REPORT": "MRI",
        "CT SCAN": "CT",
        "PHYSICAL THERAPY": "PT_OT_CHIRO",
        "OCCUPATIONAL THERAPY": "PT_OT_CHIRO",
    }
    
    for pattern, normalized in normalizations.items():
        if pattern in doc_type_upper:
            return normalized
    
    return "DEFAULT"


# ============== Main Summary Generator ==============

async def generate_concise_brief_summary(
    structured_short_summary: Dict[str, Any],
    document_type: str = "Medical Document",
    llm_executor=None
) -> Dict[str, List[str]]:
    """
    Generate a concise, physician-friendly summary by FILTERING and PRIORITIZING
    the output of generate_structured_short_summary.
    
    This is a PURE REDUCER function:
    - No new extraction
    - No new reasoning
    - No schema changes
    - No hallucination or fabrication
    
    Args:
        structured_short_summary: Output from generate_structured_short_summary
            Format: {
                "header": {...},
                "summary": {"items": [{"field": "...", "collapsed": "...", "expanded": "..."}]}
            }
        document_type: Type of document (for prioritization)
        
    Returns:
        Dict[str, List[str]] - Simple key-value pairs with bullet points
        Example: {"Findings": ["finding 1", "finding 2"], "Diagnosis": ["dx 1"]}
    """
    
    # Validate input
    if not structured_short_summary:
        return {"Error": ["No structured summary provided"]}
    
    summary_items = structured_short_summary.get("summary", {}).get("items", [])
    
    if not summary_items:
        return {"Error": ["No summary items found in structured summary"]}
    
    logger.info(f"Reducing structured summary for {document_type}: {len(summary_items)} items")
    
    # Get document-type-specific field mapping for prioritization
    field_mapping = get_document_type_mapping(document_type)
    critical_fields = set(f.lower().replace(" ", "_").replace("/", "_") for f in field_mapping.get("critical_fields", []))
    
    result = {}
    critical_items = []
    
    # Process each item from the structured short summary
    for item in summary_items:
        field = item.get("field", "").strip()
        collapsed = item.get("collapsed", "").strip()
        expanded = item.get("expanded", "").strip()
        
        if not field:
            continue
        
        # Extract bullet points from expanded content
        bullets = []
        if expanded:
            for line in expanded.split('\n'):
                line = line.strip()
                if line:
                    # Remove bullet prefix if present
                    clean_line = re.sub(r'^[•\-\*]\s*', '', line).strip()
                    # Only include meaningful sentences (at least 3 words)
                    if clean_line and len(clean_line.split()) >= 3:
                        bullets.append(clean_line)
        
        # If no expanded bullets, use collapsed as a single bullet
        if not bullets and collapsed and len(collapsed.split()) >= 3:
            bullets = [collapsed]
        
        if not bullets:
            continue
        
        # Normalize field name for display
        display_field = field.replace("_", " ").title()
        
        # Check if this is a critical field
        field_normalized = field.lower().replace(" ", "_").replace("/", "_")
        is_critical = any(cf in field_normalized or field_normalized in cf for cf in critical_fields)
        
        # Check for critical keywords in content
        critical_keywords = ['critical', 'urgent', 'acute', 'severe', 'abnormal', 
                           'denied', 'permanent', 'impairment', 'restriction']
        has_critical_content = any(kw in ' '.join(bullets).lower() for kw in critical_keywords)
        
        if is_critical or has_critical_content:
            critical_items.extend(bullets)
        else:
            if display_field not in result:
                result[display_field] = []
            result[display_field].extend(bullets)
    
    # Add critical items at the top
    if critical_items:
        # Deduplicate while preserving order
        seen = set()
        unique_critical = []
        for item in critical_items:
            if item not in seen:
                seen.add(item)
                unique_critical.append(item)
        result = {"Critical Findings": unique_critical[:10], **result}
    
    # Limit bullets per field to avoid overwhelming output
    for field in result:
        if len(result[field]) > 8:
            result[field] = result[field][:8]
    
    logger.info(f"Reduced to {len(result)} fields with prioritized content")
    
    return result


async def generate_simple_key_value_summary(
    llm: AzureChatOpenAI,
    raw_text: str,
    doc_type: str,
    field_mapping: Dict[str, Any],
    normalized_type: str
) -> Dict[str, List[str]]:
    """
    Extract clinical data as simple key-value pairs.
    Critical/abnormal findings come first in the output.
    
    Returns:
        Dict[str, List[str]] - e.g., {"Findings": ["bullet1", "bullet2"], "Diagnosis": ["dx1"]}
    """
    
    parser = PydanticOutputParser(pydantic_object=SimpleClinicalSummary)
    
    # Build the field instructions based on document type
    critical_fields = field_mapping.get("critical_fields", [])
    important_fields = field_mapping.get("important_fields", [])
    all_fields = critical_fields + important_fields
    
    fields_list = "\n".join([f"   - {f}" for f in all_fields])
    
    system_prompt = SystemMessagePromptTemplate.from_template("""
You are a clinical data extractor. Extract relevant information into simple KEY-VALUE pairs.

DOCUMENT TYPE: {doc_type}

=== FIELDS TO EXTRACT (if present in text) ===
{fields_list}

=== OUTPUT FORMAT ===

Return ONLY a simple JSON object with key-value pairs:
- KEY = Field name (e.g., "Findings", "Diagnosis", "Work Status")
- VALUE = Array of bullet point strings

EXAMPLE:
{{
    "data": {{
        "Critical Findings": ["Severe disc herniation at L4-L5", "Nerve root compression"],
        "Diagnosis": ["Lumbar radiculopathy", "DDD L4-L5"],
        "Work Status": ["Off work", "No lifting over 10 lbs"],
        "Recommendations": ["MRI follow-up in 6 weeks", "Physical therapy 2x/week"]
    }}
}}

=== RULES ===

1. **CRITICAL/ABNORMAL FIRST**: Put abnormal or urgent findings in "Critical Findings" key
2. **USE FIELD NAMES**: Use the exact field names from the list above when data matches
3. **BULLET POINTS**: Each array item is one complete bullet point
4. **NO FABRICATION**: Only extract what's explicitly in the text
5. **NO METADATA**: Do not add timestamps, document types, or any extra fields
6. **SIMPLE OUTPUT**: Just the data object with key-value pairs, nothing else

{format_instructions}
""")
    
    user_prompt = HumanMessagePromptTemplate.from_template("""
SOURCE TEXT:

{raw_text}

---

Extract the relevant clinical data into simple key-value pairs.
Put critical/abnormal findings first. Only include fields that have data in the text.
""")
    
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    
    try:
        chain = chat_prompt | llm
        response = await asyncio.to_thread(lambda: chain.invoke({
            "doc_type": doc_type,
            "fields_list": fields_list,
            "raw_text": raw_text,
            "format_instructions": parser.get_format_instructions()
        }))
        
        response_content = response.content.strip()
        
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            response_content = response_content[start_idx:end_idx+1]
        
        # Clean non-printable characters
        response_content = "".join(ch for ch in response_content if ch >= ' ' or ch in '\n\r\t')
        
        # Parse with Pydantic
        summary = parser.parse(response_content)
        
        # Return just the data dict
        return summary.data
        
    except Exception as e:
        logger.error(f"❌ Simple summary generation failed: {e}")
        # Try direct JSON parse as fallback
        try:
            import json
            parsed = json.loads(response_content)
            if "data" in parsed:
                return parsed["data"]
            return parsed
        except:
            raise e


def generate_simple_fallback(raw_text: str, doc_type: str, field_mapping: Dict[str, Any] = None) -> Dict[str, List[str]]:
    """Fallback: Generate simple key-value summary without LLM."""
    
    logger.info("📝 Creating simple fallback summary...")
    
    if field_mapping is None:
        field_mapping = get_document_type_mapping(doc_type)
    
    # Clean text
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    cleaned_lines = []
    
    noise_indicators = ['click here', 'see full', 'document summary', '...', '---']
    for line in lines:
        if not any(noise in line.lower() for noise in noise_indicators):
            if len(line) > 15:
                cleaned = re.sub(r'^[•\-\*\d\.]+\s*', '', line).strip()
                if cleaned:
                    cleaned_lines.append(cleaned)
    
    result = {}
    
    # Find critical items
    critical_keywords = ['critical', 'urgent', 'acute', 'severe', 'abnormal', 'positive', 
                        'elevated', 'decreased', 'impaired', 'restricted', 'denied', 'permanent']
    
    critical_items = []
    remaining = []
    
    for line in cleaned_lines:
        if any(kw in line.lower() for kw in critical_keywords):
            critical_items.append(line)
        else:
            remaining.append(line)
    
    if critical_items:
        result["Critical Findings"] = critical_items[:10]
    
    # Try to categorize remaining by field keywords
    critical_fields = field_mapping.get("critical_fields", [])
    important_fields = field_mapping.get("important_fields", [])
    
    for field in critical_fields + important_fields:
        field_lower = field.lower()
        keywords = field_lower.replace("/", " ").replace("-", " ").split()
        
        matched = []
        for line in remaining:
            if any(kw in line.lower() for kw in keywords if len(kw) > 3):
                matched.append(line)
        
        if matched:
            result[field] = matched[:5]
            remaining = [l for l in remaining if l not in matched]
    
    # Add any uncategorized as "Additional Information"
    if remaining and len(result) < 3:
        result["Additional Information"] = remaining[:10]
    
    logger.info(f"✅ Fallback summary: {len(result)} fields")
    
    return result


# ============== Quality Validation ==============

def validate_summary_quality(summary: Dict[str, List[str]], source_text: str) -> Dict[str, Any]:
    """
    Validate that summary doesn't fabricate information.
    Works with simple key-value format: {"Field": ["bullet1", "bullet2"]}
    """
    validation = {
        "passed": True,
        "warnings": [],
        "source_coverage": 0.0,
        "item_count": 0
    }
    
    # Handle error response
    if "Error" in summary:
        validation["passed"] = False
        validation["warnings"].append("Summary generation failed")
        return validation
    
    # Collect all values from the simple dict format
    all_content = []
    for key, values in summary.items():
        if isinstance(values, list):
            all_content.extend(values)
    
    validation["item_count"] = len(all_content)
    
    if not all_content:
        validation["passed"] = False
        validation["warnings"].append("No content extracted")
        return validation
    
    # Check source coverage
    source_lower = source_text.lower()
    covered_items = 0
    
    for item in all_content:
        item_lower = item.lower()
        
        # Check if substantial parts exist in source
        words = [w for w in item_lower.split() if len(w) > 4][:10]
        if words:
            matches = sum(1 for w in words if w in source_lower)
            if matches >= 2:
                covered_items += 1
            elif len(item) > 50:
                validation["warnings"].append({
                    "type": "potential_fabrication",
                    "item_preview": item[:100],
                    "match_words": matches
                })
    
    if all_content:
        validation["source_coverage"] = covered_items / len(all_content)
    
    if validation["source_coverage"] < 0.5:
        validation["passed"] = False
        validation["warnings"].append(f"Low source coverage: {validation['source_coverage']:.1%}")
    
    return validation



async def update_fail_document(
    fail_doc: Any,
    updated_fields: dict,
    user_id: str,
    db_service: Any,
    patient_lookup,
    save_document_func,
    create_tasks_func,
    llm_executor=None
) -> dict:
    """
    Updates and processes a failed document using the complete webhook-like logic.
    
    Args:
        fail_doc: The failed document object from database
        updated_fields: Fields to update (patient_name, dob, doi, claim_number, author, document_text)
        user_id: User ID making the update
        db_service: Database service instance
        patient_lookup: Patient lookup service instance
        save_document_func: Function to save document
        create_tasks_func: Function to create tasks
        llm_executor: ThreadPoolExecutor for LLM operations
        
    Returns:
        dict with save result
    """
    # Use updated values if provided, otherwise fallback to fail_doc values
    document_text = updated_fields.get("document_text") or fail_doc.documentText
    dob_str = updated_fields.get("dob") or fail_doc.dob
    doi = updated_fields.get("doi") or fail_doc.doi
    claim_number = updated_fields.get("claim_number") or fail_doc.claimNumber
    patient_name = updated_fields.get("patient_name") or fail_doc.patientName
    author = updated_fields.get("author") or fail_doc.author
    physician_id = fail_doc.physicianId
    filename = fail_doc.fileName
    gcs_url = fail_doc.gcsFileLink
    blob_path = fail_doc.blobPath
    file_hash = fail_doc.fileHash
    mode = "wc"  # Default mode
    
    # ✅ Use aiSummarizerText if available (actual Document AI Summarizer output)
    # This is preferred over documentText for summary generation
    ai_summarizer_text = getattr(fail_doc, 'aiSummarizerText', None)
    if ai_summarizer_text and len(ai_summarizer_text) > 50:
        logger.info(f"📋 Using aiSummarizerText for processing ({len(ai_summarizer_text)} chars)")
        # Use aiSummarizerText as the primary text for document type detection and summary generation
        summarizer_output = ai_summarizer_text
    else:
        logger.info(f"📋 aiSummarizerText not available, using documentText")
        summarizer_output = document_text

    # Construct webhook-like data
    result_data = {
        "text": document_text,
        "pages": 0,
        "entities": [],
        "tables": [],
        "formFields": [],
        "confidence": 0.0,
        "success": False,
        "gcs_file_link": gcs_url,
        "fileInfo": {},
        "comprehensive_analysis": None,
        "document_id": f"update_fail_{fail_doc.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    try:
        # Step 1: Detect document type and check if valid for summary card
        logger.info(f"🔍 Detecting document type for failed document: {fail_doc.id}")
        
        doc_type_result = await asyncio.to_thread(
            lambda: detect_document_type(summarizer_output=summarizer_output, raw_text=document_text)
        )
        
        detected_doc_type = doc_type_result.get('doc_type', 'Unknown')
        is_valid_for_summary_card = doc_type_result.get('is_valid_for_summary_card', True)  # Default True for safety
        summary_card_reasoning = doc_type_result.get('summary_card_reasoning', '')
        
        logger.info(f"📋 Detected Document Type: {detected_doc_type}")
        logger.info(f"🎯 Summary Card Eligibility: {is_valid_for_summary_card}")
        logger.info(f"   Reasoning: {summary_card_reasoning[:100]}..." if len(summary_card_reasoning) > 100 else f"   Reasoning: {summary_card_reasoning}")
        
        # Initialize variables
        long_summary = ""
        short_summary = ""
        report_result = {}
        
        # Step 2: Process document based on summary card eligibility
        if is_valid_for_summary_card:
            # ✅ FULL EXTRACTION: Document requires physician review - generate summaries
            logger.info("📋 Document requires Summary Card - running full LLM extraction...")
            
            report_analyzer = ReportAnalyzer(mode)
            report_result = await asyncio.to_thread(
                report_analyzer.extract_document,
                summarizer_output,  # Use aiSummarizerText (Document AI Summarizer output)
                document_text,  # Use documentText as raw_text parameter
                doc_type_result  # Pass pre-detected doc_type
            )
            
            # ✅ STORE THE ACTUAL REPORT ANALYZER RESULT
            long_summary = report_result.get("long_summary", "")
            short_summary = report_result.get("short_summary", "")
            logger.info(f"✅ ReportAnalyzer completed, author field: {author}")
            
            # ✅ If author provided by user, replace or inject it into long_summary as "Signature:" field
            if author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
                # Replace existing signature line or add new one
                signature_pattern = r'•\s*Signature:.*?(?=\n•|\n\n|$)'
                signature_line = f"• Signature: {author.strip()}"
                
                if re.search(signature_pattern, long_summary, re.IGNORECASE | re.DOTALL):
                    # Replace existing signature
                    long_summary = re.sub(signature_pattern, signature_line, long_summary, flags=re.IGNORECASE | re.DOTALL)
                    logger.info(f"✅ Replaced existing signature with: {author}")
                else:
                    # Add new signature line
                    long_summary = long_summary + f"\n\n{signature_line}"
                    logger.info(f"✅ Injected author into long_summary: {author}")
                
                # Update the report_result dictionary to reflect the modified long_summary
                report_result["long_summary"] = long_summary
                
                # ✅ Also update the author field in short_summary header
                if isinstance(short_summary, dict) and 'header' in short_summary:
                    short_summary['header']['author'] = author.strip()
                    report_result["short_summary"] = short_summary
                    logger.info(f"✅ Updated short_summary header author: {author}")
            
            logger.info(f"✅ Generated long summary: {len(long_summary)} chars")
            logger.info(f"✅ Generated short summary: {type(short_summary)}")
        else:
            # ⏭️ TASK-ONLY MODE: Document is administrative - skip expensive LLM extraction
            logger.info("📌 Document is TASK-ONLY (administrative) - skipping LLM extraction for summaries")
            logger.info(f"   Document type: {detected_doc_type}")
            logger.info(f"   Reason: {summary_card_reasoning}")
            
            # Create minimal summary for task generation (just use raw_text as reference)
            long_summary = f"[TASK-ONLY DOCUMENT]\nType: {detected_doc_type}\nReason: {summary_card_reasoning}\n\nThis document is administrative and does not require physician clinical review. Tasks have been generated for staff action."
            short_summary = ""  # No short summary for task-only docs
            report_result = {
                "long_summary": long_summary,
                "short_summary": short_summary,
                "doc_type": detected_doc_type,
                "is_task_only": True,
                "task_only_reason": summary_card_reasoning
            }
            logger.info("⏭️ Skipped ReportAnalyzer - will proceed to task generation only")
        
        # Helper to convert structured short_summary dict to string
        raw_brief_summary_text = "Summary not available"
        
        # Handle task-only documents differently
        if not is_valid_for_summary_card:
            raw_brief_summary_text = f"{detected_doc_type} - Administrative document for staff action"
        elif short_summary:
            if isinstance(short_summary, dict):
                # Try to extract meaningful text from structured summary
                try:
                    # 1. Try to get items texts
                    items = short_summary.get('summary', {}).get('items', [])
                    text_parts = []
                    for item in items:
                        if isinstance(item, dict):
                            # Prefer expanded text, fall back to collapsed
                            part = item.get('expanded') or item.get('collapsed')
                            if part:
                                text_parts.append(part)
                    
                    if text_parts:
                        raw_brief_summary_text = " ".join(text_parts)
                    elif short_summary.get('header', {}).get('title'):
                         # Fallback to Title if no items
                         raw_brief_summary_text = f"Report: {short_summary['header']['title']}"
                    else:
                        # Fallback to JSON string as last resort
                        raw_brief_summary_text = json.dumps(short_summary)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse structured short_summary: {e}")
                    raw_brief_summary_text = str(short_summary)
            else:
                raw_brief_summary_text = str(short_summary)
        
        # ✅ Process the structured summary through the reducer (only for summary card eligible docs)
        if is_valid_for_summary_card and isinstance(short_summary, dict) and short_summary.get('summary', {}).get('items'):
            brief_summary_text = await generate_concise_brief_summary(
                short_summary,  # Pass the structured summary directly
                detected_doc_type,
                llm_executor
            )
        elif is_valid_for_summary_card and raw_brief_summary_text != "Summary not available":
            # Fallback: create minimal structure from raw text
            brief_summary_text = {"Summary": [raw_brief_summary_text]} if raw_brief_summary_text else {}
        else:
            brief_summary_text = {"Summary": [raw_brief_summary_text]} if raw_brief_summary_text else {}
        
        # Prepare fields for DocumentAnalysis
        da_patient_name = patient_name or "Not specified"
        da_claim_number = claim_number or "Not specified"
        da_dob = dob_str or "0000-00-00" 
        da_doi = doi or "0000-00-00"
        
        # For task-only documents (not valid for summary card), we bypass author/clinic member checks
        # by treating the consulting_doctor as Not specified, ensuring the flow continues for task generation
        if is_valid_for_summary_card:
            da_author = author or "Not specified"
        else:
            da_author = "Not specified"
            if author:
                logger.info(f"ℹ️ Task-only document: Skipping consulting_doctor assignment for author '{author}' to allow processing")
        
        # Manually construct DocumentAnalysis
        document_analysis = DocumentAnalysis(
            patient_name=da_patient_name,
            claim_number=da_claim_number,
            dob=da_dob,
            doi=da_doi,
            status="Not specified",
            rd="0000-00-00", 
            body_part="Not specified",
            body_parts_analysis=[],
            diagnosis="See summary",
            key_concern="Medical evaluation",
            extracted_recommendation="See summary",
            extracted_decision="Not specified",
            ur_decision="",
            ur_denial_reason=None,
            adls_affected="Not specified",
            work_restrictions="Not specified",
            consulting_doctor=da_author,
            all_doctors=[],
            referral_doctor="Not specified",
            ai_outcome="Review required",
            document_type=detected_doc_type,
            summary_points=[],
            brief_summary=brief_summary_text,
            date_reasoning=None,
            is_task_needed=False,
            formatted_summary=brief_summary_text,
            extraction_confidence=1.0 if short_summary else 0.0,
            verified=True,
            verification_notes=["Analysis from basic ReportAnalyzer (Update Fail Doc)"]
        )
        
        brief_summary = document_analysis.brief_summary
        
        # Override with updated fields from the user
        if updated_fields.get("patient_name") and str(updated_fields["patient_name"]).lower() != "not specified":
            document_analysis.patient_name = updated_fields["patient_name"]
            logger.info(f"✅ Overridden patient_name: {updated_fields['patient_name']}")
        
        if updated_fields.get("dob") and str(updated_fields["dob"]).lower() != "not specified":
            document_analysis.dob = updated_fields["dob"]
            logger.info(f"✅ Overridden DOB: {updated_fields['dob']}")
        
        if updated_fields.get("doi") and str(updated_fields["doi"]).lower() != "not specified":
            document_analysis.doi = updated_fields["doi"]
            logger.info(f"✅ Overridden DOI: {updated_fields['doi']}")
        
        if updated_fields.get("claim_number") and str(updated_fields["claim_number"]).lower() != "not specified":
            document_analysis.claim_number = updated_fields["claim_number"]
            logger.info(f"✅ Overridden claim_number: {updated_fields['claim_number']}")
        
        # ✅ Override consulting_doctor (author) if provided by user
        # Only override if document is valid for summary card (medical document)
        if is_valid_for_summary_card and author and str(author).strip().lower() not in ["not specified", "unknown", "none", ""]:
            document_analysis.consulting_doctor = author.strip()
            logger.info(f"✅ Overridden consulting_doctor (author): {author}")

        logger.info(f"Author detected: {author}")

        # Prepare processed_data similar to process_document_data
        processed_data = {
            "document_analysis": document_analysis,
            "brief_summary": brief_summary,
            "text_for_analysis": document_text,
            "raw_text": document_text,  # ✅ Add raw_text for task generation
            "report_analyzer_result": report_result,
            "patient_name": document_analysis.patient_name if document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified" else None,
            "claim_number": document_analysis.claim_number if document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified" else None,
            "dob": document_analysis.dob if hasattr(document_analysis, 'dob') and document_analysis.dob and str(document_analysis.dob).lower() != "not specified" else None,
            "has_patient_name": bool(document_analysis.patient_name and str(document_analysis.patient_name).lower() != "not specified"),
            "has_claim_number": bool(document_analysis.claim_number and str(document_analysis.claim_number).lower() != "not specified"),
            "physician_id": physician_id,
            "user_id": user_id,
            "filename": filename,
            "gcs_url": gcs_url,
            "blob_path": blob_path,
            "file_size": 0,
            "mime_type": "application/octet-stream",
            "processing_time_ms": 0,
            "file_hash": file_hash,
            "result_data": result_data,
            "document_id": str(fail_doc.id),
            "mode": mode,
            "is_valid_for_summary_card": is_valid_for_summary_card,
            "is_task_only": not is_valid_for_summary_card,
            "doc_type_result": doc_type_result
        }

        # Step 2: Perform patient lookup with enhanced fuzzy matching
        logger.info("🔍 Performing patient lookup for updated failed document...")
        lookup_result = await patient_lookup.perform_patient_lookup(db_service, processed_data)
        
        # Step 3: Save document to database
        logger.info("💾 Saving updated document to database...")
        save_result = await save_document_func(db_service, processed_data, lookup_result)
        
        # Step 4: Create tasks if needed
        # ✅ FIX: Pass the actual Document AI Summarizer output (summarizer_output) as document_analysis
        # The task generator expects raw text content, not a DocumentAnalysis object
        tasks_created = 0
        if save_result["document_id"] and save_result["status"] != "failed":
            tasks_created = await create_tasks_func(
                summarizer_output,  # ✅ Pass Document AI Summarizer output (same as direct processing)
                save_result["document_id"],
                processed_data["physician_id"],
                processed_data["filename"],
                processed_data  # Pass full processed_data for patient_name and document_type
            )
            save_result["tasks_created"] = tasks_created

        # Step 5: Delete the FailDoc only if successful
        if save_result["status"] != "failed" and save_result["document_id"]:
            await db_service.delete_fail_doc(fail_doc.id)
            logger.info(f"🗑️ Deleted fail doc {fail_doc.id} after successful update")
            logger.info(f"📡 Success event processed for document: {save_result['document_id']}")
        else:
            logger.warning(f"⚠️ Failed document update unsuccessful, keeping fail doc {fail_doc.id}")
            # Optionally update the fail doc with the new failure reason
            if save_result.get("failure_reason"):
                logger.info(f"📝 Updating fail doc reason: {save_result['failure_reason']}")

        return save_result

    except Exception as e:
        logger.error(f"❌ Failed to update fail document {fail_doc.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Update fail document processing failed: {str(e)}")


async def update_multiple_fail_documents(
    fail_docs_data: List[dict],
    user_id: str,
    db_service: Any,
    patient_lookup,
    save_document_func,
    create_tasks_func,
    llm_executor=None
) -> dict:
    """
    Updates and processes multiple failed documents in batch.
    
    Args:
        fail_docs_data: List of dictionaries containing:
            - fail_doc: The failed document object
            - updated_fields: Fields to update
        user_id: User ID making the update
        db_service: Database service instance
        patient_lookup: Patient lookup service instance
        save_document_func: Function to save document
        create_tasks_func: Function to create tasks
        llm_executor: ThreadPoolExecutor for LLM operations
        
    Returns:
        dict with overall results and individual document results
    """
    results = {
        "total_documents": len(fail_docs_data),
        "successful": 0,
        "failed": 0,
        "documents": []
    }
    
    # Process documents sequentially
    for doc_data in fail_docs_data:
        fail_doc = doc_data.get("fail_doc")
        updated_fields = doc_data.get("updated_fields", {})
        
        if not fail_doc:
            logger.error("❌ Missing fail_doc in batch data")
            results["failed"] += 1
            results["documents"].append({
                "fail_doc_id": "unknown",
                "status": "failed",
                "error": "Missing fail_doc object"
            })
            continue
        
        try:
            # Process individual document
            document_result = await update_fail_document(
                fail_doc=fail_doc,
                updated_fields=updated_fields,
                user_id=user_id,
                db_service=db_service,
                patient_lookup=patient_lookup,
                save_document_func=save_document_func,
                create_tasks_func=create_tasks_func,
                llm_executor=llm_executor
            )
            
            results["successful"] += 1
            results["documents"].append({
                "fail_doc_id": fail_doc.id,
                "status": "success",
                "document_id": document_result.get("document_id"),
                "tasks_created": document_result.get("tasks_created", 0)
            })
            
            logger.info(f"✅ Successfully processed fail document {fail_doc.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process fail document {fail_doc.id}: {str(e)}")
            results["failed"] += 1
            results["documents"].append({
                "fail_doc_id": fail_doc.id,
                "status": "failed",
                "error": str(e)
            })
    
    return results
