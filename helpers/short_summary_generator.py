"""
Optimized Structured Short Summary Generator
Physician-centric with formatted long summary support.
Removed duplicates and unnecessary complexity while preserving core functionality.
"""

import logging
import json
import re
from typing import Dict, List, Literal, Set, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from config.settings import settings

# Import context expansion service for semantic context enrichment
from services.context_expansion_service import expand_summary_with_context, get_context_expansion_service

logger = logging.getLogger("document_ai")


# ============== SINGLE ENHANCED REPORT-TYPE FIELD ELIGIBILITY ==============

REPORT_FIELD_MATRIX = {
    # -------------------------
    # RFA (Request for Authorization) - Physician Focus
    # -------------------------
    "RFA": {
        "allowed": {
            "requested_treatment",      # Exact CPT codes, procedures
            "medical_necessity",        # Diagnosis, supporting justification
            "previous_treatments",      # Conservative care attempts, failures
            "injury_relationship",      # Causation to work injury/claim
            "supporting_findings",      # Imaging, exam findings, test results
            "treatment_plan",           # Duration, frequency, timeline
            "expected_outcomes",        # Functional improvement goals
            "urgency_level"             # Routine vs. expedited
        },
        "physician_priority": ["requested_treatment", "medical_necessity", "injury_relationship", "supporting_findings"]
    },
    "REQUEST FOR AUTHORIZATION": {"inherit": "RFA"},
    
    # -------------------------
    # PR2 (Progress Report) - Physician Focus
    # -------------------------
    "PR-2": {
        "allowed": {
            "subjective_complaints",    # Pain levels, functional limitations
            "objective_findings",       # ROM measurements, strength testing
            "current_treatment",        # Medications, therapy frequency
            "treatment_response",       # Improvement/plateau/decline
            "work_status",              # Off work, modified duty, restrictions
            "pps_estimate",             # P&S date estimate
            "causation_statement",      # Ongoing relationship to injury
            "future_plan"               # Next steps, additional care needed
        },
        "physician_priority": ["treatment_response", "work_status", "objective_findings", "future_plan"]
    },
    "PR2": {"inherit": "PR-2"},
    "PROGRESS REPORT": {"inherit": "PR-2"},
    
    # -------------------------
    # PR4/Permanent & Stationary Report - Physician Focus
    # -------------------------
    "PR-4": {
        "allowed": {
            "mmi_declaration",          # Maximum medical improvement reached
            "impairment_rating",        # Whole person impairment %
            "permanent_restrictions",   # Lift limits, positional restrictions
            "future_medical_care",      # Life care plan components
            "apportionment",            # Prior injuries, degenerative changes
            "causation_analysis",       # Substantial vs. minor factors
            "body_parts_affected",      # All injured areas rated
            "guidelines_referenced"     # AMA Guidelines used
        },
        "physician_priority": ["mmi_declaration", "impairment_rating", "permanent_restrictions", "causation_analysis"]
    },
    "PR4": {"inherit": "PR-4"},
    "PERMANENT AND STATIONARY": {"inherit": "PR-4"},
    "P&S": {"inherit": "PR-4"},
    
    # -------------------------
    # DFR (Doctor's First Report) - Physician Focus
    # -------------------------
    "DFR": {
        "allowed": {
            "injury_mechanism",         # How injury occurred
            "body_parts_injured",       # Primary and secondary areas
            "initial_findings",         # Physical exam abnormalities
            "initial_diagnosis",        # ICD-10 codes
            "treatment_provided",       # Medications, procedures, referrals
            "initial_work_status",      # Ability to return to work
            "disability_dates",         # Off work from/to
            "initial_causation"         # Industrial vs. non-industrial
        },
        "physician_priority": ["injury_mechanism", "initial_diagnosis", "body_parts_injured", "initial_work_status"]
    },
    "DOCTOR'S FIRST REPORT": {"inherit": "DFR"},
    "FIRST REPORT": {"inherit": "DFR"},
    
    # -------------------------
    # QME/AME/IME Reports - Physician Focus
    # -------------------------
    "QME": {
        "allowed": {
            "history_summary",          # Complete medical record summary
            "comprehensive_exam",       # Physical examination with measurements
            "diagnostic_review",        # Interpretation of all imaging/studies
            "diagnoses",                # Primary and secondary with ICD codes
            "causation_opinion",        # Industrial injury relationship
            "mmi_status",               # Current or future estimate
            "impairment_rating_qme",    # WPI calculation with tables
            "future_care_qme",          # Recommended treatments
            "work_restrictions_qme",    # Permanent limitations
            "legal_opinions"            # Depositions/testimony
        },
        "physician_priority": ["diagnoses", "causation_opinion", "impairment_rating_qme", "work_restrictions_qme"]
    },
    "AME": {"inherit": "QME"},
    "PQME": {"inherit": "QME"},
    "IME": {"inherit": "QME"},
    "INDEPENDENT MEDICAL EXAMINATION": {"inherit": "QME"},
    
    # -------------------------
    # IMR (Independent Medical Review) - Physician Focus
    # -------------------------
    "IMR": {
        "allowed": {
            "ur_decision_reviewed",     # Denial/modification details
            "medical_evidence",         # All supporting documentation
            "reviewer_credentials",     # Specialty match
            "necessity_determination",  # Meets standard of care
            "guidelines_cited",         # ODG, ACOEM, etc.
            "final_decision",           # Upheld, overturned, modified
            "clinical_rationale"        # Clinical reasoning for decision
        },
        "physician_priority": ["final_decision", "clinical_rationale", "necessity_determination", "guidelines_cited"]
    },
    "INDEPENDENT MEDICAL REVIEW": {"inherit": "IMR"},
    
    # -------------------------
    # UR (Utilization Review) - Physician Focus
    # -------------------------
    "UR": {
        "allowed": {
            "decision",                 # Approved, denied, modified, delayed
            "rationale",                # Clinical basis for decision
            "guidelines_referenced_ur", # ODG, ACOEM compliance
            "necessity_assessment",     # Meets criteria or not
            "alternative_treatments",   # Suggestions if denied
            "reviewer_credentials_ur",  # Board certification
            "timeframe",                # Expedited vs. routine review
            "appeal_rights"             # IMR eligibility
        },
        "physician_priority": ["decision", "rationale", "necessity_assessment", "alternative_treatments"]
    },
    "UTILIZATION REVIEW": {"inherit": "UR"},
    "PEER REVIEW": {"inherit": "UR"},
    
    # -------------------------
    # CONSULT/Office Visit - Physician Focus
    # -------------------------
    "CONSULT": {
        "allowed": {
            "chief_complaint",          # Patient's primary concern
            "history_of_illness",       # Symptom progression
            "pertinent_exam",           # Positive/negative findings
            "assessment",               # Differential diagnosis
            "plan",                     # Diagnostic workup, treatment
            "medication_changes",       # New prescriptions, discontinuations
            "work_status_update",       # Restrictions modified
            "follow_up_plan"            # Next visit timing
        },
        "physician_priority": ["assessment", "plan", "pertinent_exam", "medication_changes"]
    },
    "OFFICE VISIT": {"inherit": "CONSULT"},
    "CLINIC NOTE": {"inherit": "CONSULT"},
    "PAIN MANAGEMENT": {"inherit": "CONSULT"},
    
    # -------------------------
    # Progress Notes - Physician Focus
    # -------------------------
    "PROGRESS NOTE": {
        "allowed": {
            "interval_history",         # Changes since last visit
            "pain_scale",               # Numerical rating
            "functional_status",        # ADL limitations
            "treatment_compliance",     # Medication adherence
            "side_effects",             # Medication tolerability
            "objective_measurements",   # ROM, strength, edema
            "treatment_modifications",  # Dose adjustments, modality changes
            "clinical_course"           # Improving, stable, worsening
        },
        "physician_priority": ["clinical_course", "pain_scale", "objective_measurements", "treatment_modifications"]
    },
    
    # -------------------------
    # MRI Reports - Physician Focus
    # -------------------------
    "MRI": {
        "allowed": {
            "indication",               # Clinical question being asked
            "technique",                # Sequences used, contrast given
            "comparison",               # Prior studies referenced
            "key_findings",             # Abnormalities with measurements
            "pathology_severity",       # Grading systems
            "clinical_correlation",     # Symptomatic vs. incidental
            "impression"                # Radiologist's summary interpretation
        },
        "physician_priority": ["key_findings", "impression", "clinical_correlation", "pathology_severity"]
    },
    "CT": {"inherit": "MRI"},
    "X-RAY": {"inherit": "MRI"},
    "XRAY": {"inherit": "MRI"},
    "IMAGING": {"inherit": "MRI"},
    
    # -------------------------
    # Surgery/Operative Reports - Physician Focus
    # -------------------------
    "SURGERY REPORT": {
        "allowed": {
            "preop_diagnosis",          # Indication for surgery
            "postop_diagnosis",         # Findings-based diagnosis
            "procedure_performed",      # Exact surgical steps with CPT codes
            "intraoperative_findings",  # Pathology observed
            "technique",                # Approach, instrumentation, fixation
            "complications",            # Intraoperative adverse events
            "specimens_sent",           # Pathology submitted
            "hardware_used"             # Implants, mesh, grafts
        },
        "physician_priority": ["procedure_performed", "intraoperative_findings", "complications", "postop_diagnosis"]
    },
    "OPERATIVE NOTE": {"inherit": "SURGERY REPORT"},
    "POST-OP": {"inherit": "SURGERY REPORT"},
    
    # -------------------------
    # Therapy Reports - Physician Focus
    # -------------------------
    "PHYSICAL THERAPY": {
        "allowed": {
            "baseline_measurements",    # ROM, strength, functional tests
            "treatment_provided_pt",    # Modalities, manual therapy, exercise
            "patient_response_pt",      # Pain changes, functional improvement
            "compliance_pt",            # Attendance, home program adherence
            "objective_progress",       # Measurable changes from baseline
            "functional_goals",         # % achievement
            "plan_modifications_pt",    # Progression of difficulty
            "discharge_planning"        # Readiness assessment
        },
        "physician_priority": ["objective_progress", "treatment_provided_pt", "patient_response_pt", "functional_goals"]
    },
    "THERAPY NOTE": {"inherit": "PHYSICAL THERAPY"},
    "OCCUPATIONAL THERAPY": {"inherit": "PHYSICAL THERAPY"},
    "CHIROPRACTIC": {"inherit": "PHYSICAL THERAPY"},
    
    # -------------------------
    # Labs & Diagnostics - Physician Focus
    # -------------------------
    "LABS": {
        "allowed": {
            "tests_ordered",            # Specific panel
            "critical_values",          # Flagged abnormalities
            "reference_ranges",         # Normal values context
            "trends",                   # Comparison to prior results
            "clinical_significance"     # Impact on diagnosis/treatment
        },
        "physician_priority": ["critical_values", "trends", "clinical_significance"]
    },
    "PATHOLOGY": {"inherit": "LABS"},
    
    # -------------------------
    # Original document types for backward compatibility
    # -------------------------
    "PAIN MANAGEMENT": {"inherit": "CONSULT"},
    "PROGRESS NOTE": {"inherit": "CONSULT"},
    "CLINIC NOTE": {"inherit": "CONSULT"},
    "THERAPY NOTE": {"inherit": "PHYSICAL THERAPY"},
    "OCCUPATIONAL THERAPY": {"inherit": "PHYSICAL THERAPY"},
    "CHIROPRACTIC": {"inherit": "PHYSICAL THERAPY"},
    "EMG": {"inherit": "MRI"},
    "PET SCAN": {"inherit": "MRI"},
    "BONE SCAN": {"inherit": "MRI"},
    "DEXA SCAN": {"inherit": "MRI"},
    
    # -------------------------
    # Default fallback
    # -------------------------
    "DEFAULT": {
        "allowed": {
            "key_findings",
            "recommendations",
            "clinical_summary"
        },
        "physician_priority": ["key_findings", "clinical_summary"]
    }
}


def resolve_allowed_fields(doc_type: str) -> dict:
    """
    Resolve allowed fields and physician priority fields for a document type.
    
    Args:
        doc_type: The document type string
        
    Returns:
        dict with 'allowed' and 'priority' keys
    """
    doc_key = doc_type.upper().replace("-", " ").replace("_", " ").strip()
    
    # Direct lookup
    if doc_key in REPORT_FIELD_MATRIX:
        matrix = REPORT_FIELD_MATRIX[doc_key]
    else:
        # Try partial matching
        matched = None
        for key in REPORT_FIELD_MATRIX:
            if key in doc_key or doc_key in key:
                matched = key
                break
        matrix = REPORT_FIELD_MATRIX.get(matched, REPORT_FIELD_MATRIX["DEFAULT"])
    
    # Handle inheritance
    if "inherit" in matrix:
        parent_key = matrix["inherit"]
        return resolve_allowed_fields(parent_key)
    
    return {
        "allowed": matrix.get("allowed", set()),
        "priority": matrix.get("physician_priority", [])
    }


# ============== PYDANTIC MODELS ==============

class UIFact(BaseModel):
    """
    A UI-clickable summary field with collapsed and expanded views.
    """
    field: str = Field(description="The type of UI field based on document type")
    collapsed: str = Field(description="Short, high-level, one-line summary for collapsed view")
    expanded: str = Field(description="Expanded, bullet-point description for expanded view")


class SummaryHeader(BaseModel):
    """Header information for the structured summary"""
    title: str = Field(description="Document type and body region")
    source_type: str = Field(default="External Medical Document", description="Type of source document")
    author: str = Field(default="", description="Author name with credentials (no Dr. prefix)")
    date: str = Field(default="", description="Document date in YYYY-MM-DD format")
    disclaimer: str = Field(
        default="This summary references an external document and is for workflow purposes only. It does not constitute medical advice.",
        description="Legal disclaimer for the summary"
    )


class SummaryContent(BaseModel):
    """UI-driven summary content with clickable fields"""
    items: List[UIFact] = Field(default_factory=list, description="List of UI-ready fact items")


class StructuredShortSummary(BaseModel):
    """Complete structured short summary for UI display with clickable elements"""
    header: SummaryHeader = Field(description="Header information")
    summary: SummaryContent = Field(description="Summary content with UI-ready items")


# ============== OPTIMIZED HELPER FUNCTIONS ==============

def create_fallback_summary(doc_type: str, priority_fields: List[str] = None) -> dict:
    """
    Create a fallback summary when generation fails.
    """
    doc_type_titles = {
        "RFA": "Request for Authorization",
        "PR-2": "Progress Report",
        "PR-4": "Permanent & Stationary Report",
        "DFR": "Doctor's First Report",
        "QME": "QME/Independent Medical Exam",
        "IME": "Independent Medical Examination",
        "IMR": "Independent Medical Review",
        "UR": "Utilization Review",
        "CONSULT": "Consultation Note",
        "MRI": "MRI/Imaging Report",
        "SURGERY": "Surgery Report"
    }
    
    title = doc_type_titles.get(doc_type, doc_type)
    
    # Create placeholder items
    fallback_items = []
    if priority_fields:
        for field in priority_fields[:3]:  # Top 3 priority fields
            fallback_items.append({
                "field": field,
                "collapsed": f"Clinical information for {field.replace('_', ' ')} requires physician review",
                "expanded": f"• Information extraction for {field.replace('_', ' ')} failed\n• Physician review of original document recommended\n• Clinical assessment needed for this section"
            })
    
    # Always add at least one item
    if not fallback_items:
        fallback_items.append({
            "field": "clinical_summary",
            "collapsed": f"Clinical information from {title} requires physician review",
            "expanded": f"• Unable to extract structured clinical data from the {title}\n• Physician review of original document recommended\n• Key clinical findings may require manual extraction"
        })
    
    return {
        "header": {
            "title": title,
            "source_type": "External Medical Document",
            "author": "",
            "date": "",
            "disclaimer": "This summary references an external document and is for workflow purposes only. It does not constitute medical advice. Clinical data extraction failed - physician review of original document is recommended."
        },
        "summary": {
            "items": fallback_items
        }
    }


def ensure_proper_header(structured_summary: dict, doc_type: str, raw_text: str) -> dict:
    """
    Ensure the structured summary has a properly formatted header.
    Extracts date and author from raw_text if not already present.
    """
    # Document type to title mapping
    doc_type_titles = {
        "RFA": "Request for Authorization",
        "PR-2": "Progress Report",
        "PR2": "Progress Report",
        "PR-4": "Permanent & Stationary Report",
        "PR4": "Permanent & Stationary Report",
        "DFR": "Doctor's First Report",
        "QME": "QME Report",
        "AME": "AME Report",
        "IME": "Independent Medical Examination",
        "IMR": "Independent Medical Review",
        "UR": "Utilization Review",
        "CONSULT": "Consultation Report",
        "MRI": "Imaging Report",
        "CT": "Imaging Report",
        "XRAY": "Imaging Report",
        "X-RAY": "Imaging Report",
        "SURGERY": "Operative Report",
        "PHYSICAL THERAPY": "Physical Therapy Report",
        "LABS": "Laboratory Report",
        "PROGRESS NOTE": "Progress Note"
    }
    
    # Ensure header exists
    if "header" not in structured_summary:
        structured_summary["header"] = {}
    
    header = structured_summary["header"]
    
    # Set title
    if not header.get("title"):
        base_title = doc_type_titles.get(doc_type.upper(), doc_type)
        
        # Try to extract body region
        body_regions = [
            r'(?:lumbar|lumbosacral|l-?spine)',
            r'(?:cervical|c-?spine)',
            r'(?:thoracic|t-?spine)',
            r'(?:shoulder)',
            r'(?:knee)',
            r'(?:hip)',
            r'(?:ankle)',
            r'(?:wrist)',
            r'(?:elbow)'
        ]
        
        body_region = ""
        raw_text_lower = raw_text[:5000].lower()
        for pattern in body_regions:
            if re.search(pattern, raw_text_lower):
                match = re.search(pattern, raw_text_lower)
                if match:
                    body_region = match.group(0).replace("-", " ").title()
                    break
        
        if body_region:
            header["title"] = f"{base_title} - {body_region}"
        else:
            header["title"] = base_title
    
    # Extract date if not present
    if not header.get("date"):
        date_patterns = [
            r'(?:Date|DATE|Report Date|Exam Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, raw_text[:3000])
            if match:
                date_str = match.group(1)
                # Normalize to YYYY-MM-DD
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            if len(year) == 2:
                                year = f"20{year}" if int(year) < 50 else f"19{year}"
                            header["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            break
                    elif '-' in date_str:
                        header["date"] = date_str
                        break
                except:
                    continue
        if not header.get("date"):
            header["date"] = ""
    
    # Extract author if not present
    if not header.get("author"):
        author_patterns = [
            r'(?:Physician|Author|Prepared by|Signed by)[:\s]*([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s*(?:M\.?D\.?|D\.?O\.?|D\.?C\.?)',
        ]
        for pattern in author_patterns:
            match = re.search(pattern, raw_text[:5000])
            if match:
                author = match.group(1)
                # Clean author
                author = re.sub(r'^Dr\.?\s*', '', author, flags=re.IGNORECASE)
                header["author"] = author.strip()
                break
        if not header.get("author"):
            header["author"] = ""
    
    # Ensure required fields
    if not header.get("source_type"):
        header["source_type"] = "External Medical Document"
    
    if not header.get("disclaimer"):
        header["disclaimer"] = "This summary references an external document and is for workflow purposes only. It does not constitute medical advice."
    
    return structured_summary


def filter_disallowed_fields(structured_summary: dict, allowed_fields: Set[str]) -> dict:
    """
    Filter out any UI fields that are not allowed for the document type.
    """
    if "summary" in structured_summary and "items" in structured_summary["summary"]:
        structured_summary["summary"]["items"] = [
            item for item in structured_summary["summary"]["items"]
            if item.get("field") in allowed_fields
        ]
    return structured_summary


def deduplicate_fields(structured_summary: dict) -> dict:
    """
    Ensure each field type appears only once.
    """
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    items = structured_summary["summary"]["items"]
    field_map = {}
    
    for item in items:
        field = item.get("field", "")
        if not field:
            continue
            
        if field not in field_map:
            field_map[field] = item
        else:
            # Merge expanded content
            existing = field_map[field]
            new_expanded = item.get("expanded", "")
            if new_expanded:
                if existing.get("expanded"):
                    # Add new bullets if not already present
                    existing_bullets = set(existing["expanded"].split('\n'))
                    new_bullets = set(new_expanded.split('\n'))
                    combined = existing_bullets.union(new_bullets)
                    existing["expanded"] = '\n'.join(sorted(combined))
                else:
                    existing["expanded"] = new_expanded
    
    structured_summary["summary"]["items"] = list(field_map.values())
    return structured_summary


def validate_summary_items(items: List[dict]) -> List[dict]:
    """
    Simple validation to ensure items have basic required structure.
    """
    validated_items = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
            
        field = item.get("field", "").strip()
        collapsed = item.get("collapsed", "").strip()
        expanded = item.get("expanded", "").strip()
        
        # Skip items without required fields
        if not field or (not collapsed and not expanded):
            continue
        
        # Ensure expanded has bullet format
        if expanded and not any(b.strip().startswith('•') for b in expanded.split('\n')):
            # Convert to bullets
            lines = [line.strip() for line in expanded.split('. ') if line.strip()]
            if lines:
                expanded = '\n'.join([f"• {line}" for line in lines])
                item["expanded"] = expanded
        
        validated_items.append(item)
    
    return validated_items


def prioritize_fields(items: List[dict], priority_fields: List[str]) -> List[dict]:
    """
    Reorder items so priority fields appear first.
    """
    priority_items = []
    non_priority_items = []
    
    for item in items:
        field = item.get("field", "")
        if field in priority_fields:
            priority_items.append(item)
        else:
            non_priority_items.append(item)
    
    # Order priority items by priority list
    ordered_priority = []
    for priority_field in priority_fields:
        for item in priority_items:
            if item.get("field") == priority_field:
                ordered_priority.append(item)
                break
    
    # Add any remaining priority items
    for item in priority_items:
        if item not in ordered_priority:
            ordered_priority.append(item)
    
    return ordered_priority + non_priority_items


# ============== CONTEXT EXPANSION INTEGRATION ==============

def enrich_summary_with_context(structured_summary: dict, document_text: str) -> dict:
    """
    Enrich summary items with supporting context from the original document.
    
    Uses semantic similarity (embeddings + cosine similarity) to find the most
    relevant passages from the source document for each bullet point.
    
    Args:
        structured_summary: The structured short summary dict
        document_text: The full original document text
        
    Returns:
        Enhanced summary with 'context_expansion' field added to each item
    """
    if not structured_summary or not document_text:
        return structured_summary
    
    items = structured_summary.get("summary", {}).get("items", [])
    if not items:
        return structured_summary
    
    try:
        # Use the context expansion service
        enhanced_items = expand_summary_with_context(items, document_text, min_relevance=0.65)
        
        if enhanced_items:
            structured_summary["summary"]["items"] = enhanced_items
            
            # Add metadata about context expansion
            total_expanded = sum(1 for item in enhanced_items if item.get("context_expansion"))
            structured_summary["_context_expansion_metadata"] = {
                "items_with_context": total_expanded,
                "total_items": len(enhanced_items),
                "expansion_enabled": True
            }
            
            logger.info(f"✅ Context expansion complete: {total_expanded}/{len(enhanced_items)} items enriched")
        
        return structured_summary
        
    except Exception as e:
        logger.warning(f"⚠️ Context enrichment failed: {e}")
        structured_summary["_context_expansion_metadata"] = {
            "expansion_enabled": False,
            "error": str(e)
        }
        return structured_summary


# ============== OPTIMIZED MAIN GENERATION FUNCTION ==============

def generate_structured_short_summary(llm: AzureChatOpenAI, raw_text: str, doc_type: str, text: str) -> dict:
    """
    Generate structured summary focused on CRITICAL clinical data points only.
    Minimizes noise and avoids trivial information while preserving context.
    Args:
        llm: The AzureChatOpenAI instance
        raw_text: The Document AI summarizer output (primary context)
        doc_type: Document type
        text: Original full text of the document
    """
    logger.info(f"🎯 Generating critical-focus summary for {doc_type}")
    
    # Resolve allowed fields
    field_info = resolve_allowed_fields(doc_type)
    allowed_fields = field_info["allowed"]
    priority_fields = field_info.get("priority", [])
    
    logger.info(f"📋 Allowed: {len(allowed_fields)} fields, Priority: {len(priority_fields)} fields")
    
    # Document type descriptions for context
    doc_type_descriptions = {
        "RFA": "Request for Authorization",
        "PR-2": "Progress Report",
        "PR-4": "Permanent & Stationary Report",
        "DFR": "Doctor's First Report",
        "QME": "QME/IME Report",
        "IME": "Independent Medical Exam",
        "IMR": "Independent Medical Review",
        "UR": "Utilization Review",
        "CONSULT": "Consultation Note",
        "MRI": "Imaging Report",
        "CT": "Imaging Report",
        "XRAY": "Imaging Report",
        "SURGERY": "Surgery Report",
        "PHYSICAL THERAPY": "Therapy Report",
        "LABS": "Laboratory Report"
    }
        
    # Enhanced system prompt with CRITICAL FOCUS
    system_prompt = SystemMessagePromptTemplate.from_template("""
You are a MEDICAL DOCUMENT RESTATER for PHYSICIAN REVIEW.
Your ONLY function is to RESTATE what was EXPLICITLY documented in past tense.

🚨 **ABSOLUTE PAST-TENSE RULE - NO EXCEPTIONS:**
EVERY statement MUST use ONLY these verb forms:
✅ "was documented", "were documented"
✅ "was reported", "were reported" 
✅ "was noted", "were noted"
✅ "was described", "were described"
✅ "was referenced", "were referenced"
✅ "was stated", "were stated"
✅ "was listed", "were listed"

❌ **STRICTLY FORBIDDEN PRESENT TENSE:**
• "Diagnoses include" → VIOLATION
• "Plan includes" → VIOLATION  
• "Patient has" → VIOLATION
• "Findings show" → VIOLATION
• "Treatment consists of" → VIOLATION
• ANY present tense verbs

🎯 **CORRECT ATTRIBUTION PATTERNS:**

**DIAGNOSES:**
❌ WRONG: "Diagnoses include depression and anxiety"
✅ CORRECT: "The following diagnoses were documented: Major Depressive Disorder, Panic Disorder"

❌ WRONG: "Patient has lumbar disc herniation"
✅ CORRECT: "Lumbar disc herniation was documented"

**TREATMENT PLANS:**
❌ WRONG: "Plan includes physical therapy"
✅ CORRECT: "Physical therapy was referenced in the treatment plan"

❌ WRONG: "Treatment consists of medication management"
✅ CORRECT: "Medication management was described in the treatment plan"

**FINDINGS:**
❌ WRONG: "MRI shows disc herniation"
✅ CORRECT: "Disc herniation was noted on MRI"

❌ WRONG: "Exam reveals limited range of motion"
✅ CORRECT: "Limited range of motion was documented during examination"

🔴 **CRITICAL FILTERING RULES:**

**INCLUDE ONLY (with past tense attribution):**
✅ "MMI was declared with 12% impairment"
✅ "Work restrictions were documented as 'no lifting >20 lbs'"
✅ "Physical therapy was authorized for 12 visits"
✅ "Lumbar disc herniation was noted on MRI"
✅ "Hydrocodone was prescribed at 10mg TID"

**EXCLUDE/SUMMARIZE:**
❌ Normal/negative findings (unless critical)
❌ Routine administrative details  
❌ Generic statements with no clinical value
❌ "No prior studies were available" (trivial negative)
❌ Repetitive information

🧠 **CLINICAL PRIORITIZATION:**
Before including any information, ask:
1. Is this CRITICAL for physician decision-making?
2. Would omitting this lead to clinical risk?
3. Is this actionable information?
4. Is this new or changed from previous status?

**If answer is NO to ALL questions → EXCLUDE or SUMMARIZE BRIEFLY**

🚨 **OUTPUT OPTIMIZATION RULES:**

**COLLAPSED TEXT (ONE LINE):**
• Complete PAST-TENSE sentence
• Example: "MMI was declared with 12% whole person impairment"
• NOT: "Multiple findings were documented" (too generic)

**EXPANDED TEXT (BULLETS):**
• 2-5 bullets MAXIMUM per field
• ONLY include CRITICAL details
• EVERY bullet MUST be PAST TENSE
• Use COMPLETE sentences: "L4-L5 disc herniation was noted on MRI with 5mm protrusion"
• NOT: "Disc herniation", "5mm" (fragments)
• Group related findings: "Cervical spine conditions were documented including: disc degeneration, spondylosis, and radiculopathy"

**FIELD-SPECIFIC PAST-TENSE TEMPLATES:**

**ASSESSMENT/DIAGNOSES:**
❌ "Diagnoses include: depression, anxiety"
✅ "The following diagnoses were documented: Major Depressive Disorder, Panic Disorder"

❌ "Assessment reveals chronic pain syndrome"
✅ "Chronic pain syndrome was documented in the assessment"

**PLAN/TREATMENT:**
❌ "Plan includes: medication management, therapy"
✅ "The treatment plan referenced medication management and therapy"

❌ "Will start physical therapy next week"
✅ "Physical therapy was scheduled to begin the following week"

**FINDINGS:**
❌ "MRI shows L4-L5 herniation"
✅ "L4-L5 disc herniation was noted on MRI"

❌ "Exam reveals tenderness at L4-L5"
✅ "Tenderness at L4-L5 was documented during examination"

**WORK STATUS:**
❌ "Patient is off work"
✅ "Off work status was documented"

❌ "Has lifting restrictions"
✅ "Lifting restrictions were documented as 'no lifting >10 lbs'"

**MEDICATIONS:**
❌ "Taking hydrocodone 10mg TID"
✅ "Hydrocodone 10mg three times daily was documented"

❌ "Prescribed gabapentin 300mg"
✅ "Gabapentin 300mg was prescribed"

📋 **OUTPUT STRUCTURE:**
{{
    "header": {{
        "date": "YYYY-MM-DD or empty",
        "title": "Document Type - Body Region",
        "author": "Author Name, Credentials",
        "disclaimer": "Standard disclaimer",
        "source_type": "External Medical Document"
    }},
    "summary": {{
        "items": [
            {{
                "field": "field_name",
                "collapsed": "One COMPLETE PAST-TENSE sentence",
                "expanded": "• Past tense complete sentence\\n• Past tense complete sentence\\n• Past tense complete sentence"
            }}
        ]
    }}
}}

🚨 **PAST-TENSE VALIDATION CHECKLIST (ALL MUST PASS):**
✅ NO present tense verbs in any output
✅ NO "includes", "has", "shows", "reveals", "consists of"
✅ ONLY "was/were documented/noted/reported/described"
✅ Every sentence has clear attribution
✅ No clinical judgments or interpretations
✅ Decision terms preserved exactly

**EXAMPLES OF CORRECT VS INCORRECT:**

**INCORRECT (Present tense - VIOLATION):**
{{
  "field": "assessment",
  "collapsed": "Diagnoses include Major Depressive Disorder and Panic Disorder",
  "expanded": "• Patient has multiple psychiatric diagnoses\\n• Treatment includes medication management"
}}

**CORRECT (Past tense - COMPLIANT):**
{{
  "field": "assessment",
  "collapsed": "Multiple psychiatric diagnoses were documented including Major Depressive Disorder and Panic Disorder",
  "expanded": "• Major Depressive Disorder was documented\\n• Panic Disorder was noted\\n• Medication management was referenced in the treatment plan"
}}

**INCORRECT (Present tense - VIOLATION):**
{{
  "field": "plan",
  "collapsed": "Plan includes evaluating hydroxyzine response and exploring TMS",
  "expanded": "• Will evaluate hydroxyzine response\\n• May consider TMS or Spravato"
}}

**CORRECT (Past tense - COMPLIANT):**
{{
  "field": "plan",
  "collapsed": "Treatment plan referenced evaluation of hydroxyzine response and consideration of TMS or Spravato",
  "expanded": "• Evaluation of hydroxyzine response was referenced\\n• TMS or Spravato were described as potential treatment options"
}}

**INCORRECT (Fragments - VIOLATION):**
{{
  "field": "findings",
  "collapsed": "MRI findings",
  "expanded": "• Disc herniation\\n• 5mm\\n• Stenosis"
}}

**CORRECT (Complete past tense - COMPLIANT):**
{{
  "field": "findings",
  "collapsed": "L4-L5 disc herniation with stenosis was documented on MRI",
  "expanded": "• L4-L5 disc herniation was noted on MRI\\n• 5mm protrusion was measured\\n• Spinal stenosis was documented at the same level"
}}

Return valid JSON only.
""")

    user_prompt = HumanMessagePromptTemplate.from_template("""
**DOCUMENT TYPE:** {doc_type}
**CRITICAL MISSION:** Restate ONLY what was EXPLICITLY documented in PAST TENSE

**ALLOWED FIELDS (use only these with PAST TENSE):**
{allowed_fields_list}

**PHYSICIAN PRIORITY FIELDS (focus here first):**
{priority_fields_list}

**SOURCE DOCUMENT:**
{raw_text}

**TASK:** Restate CRITICAL clinical information using STRICT PAST-TENSE ATTRIBUTION.

**ABSOLUTE PAST-TENSE REQUIREMENTS:**

1. **ONLY THESE VERBS:**
   • was documented / were documented
   • was reported / were reported
   • was noted / were noted
   • was described / were described
   • was referenced / were referenced
   • was stated / were stated

2. **NEVER USE PRESENT TENSE:**
   ❌ "Diagnoses include..." → ✅ "The following diagnoses were documented..."
   ❌ "Plan includes..." → ✅ "The treatment plan referenced..."
   ❌ "Patient has..." → ✅ "[Condition] was documented..."
   ❌ "MRI shows..." → ✅ "[Finding] was noted on MRI..."
   ❌ "Will start..." → ✅ "[Treatment] was scheduled..."

3. **COMPLETE SENTENCES:**
   • Every bullet must be a complete past-tense sentence
   • Minimum 5 words per bullet
   • Include attribution in every sentence

4. **CRITICAL FILTERING:**
   • Include only information that affects clinical decisions
   • Exclude normal/negative findings unless critical
   • Skip trivial negatives like "No prior studies were available"
   • Group related information

5. **QUALITY OVER QUANTITY:**
   • 2-5 bullets maximum per field
   • Priority fields: up to 8 bullets
   • Non-priority fields: 3-5 bullets

**FIELD TRANSFORMATION EXAMPLES:**

**ASSESSMENT FIELD:**
Source: "Diagnoses: Major Depressive Disorder, Panic Disorder, Generalized Anxiety Disorder"
❌ WRONG: "Diagnoses include depression and anxiety disorders"
✅ CORRECT: "The following psychiatric diagnoses were documented: Major Depressive Disorder, Panic Disorder, and Generalized Anxiety Disorder"

**PLAN FIELD:**
Source: "Plan: Evaluate hydroxyzine response, consider TMS or Spravato"
❌ WRONG: "Plan includes evaluating medication response and exploring TMS"
✅ CORRECT: "The treatment plan referenced evaluation of hydroxyzine response and consideration of TMS or Spravato options"

**FINDINGS FIELD:**
Source: "MRI: L4-L5 disc herniation with 5mm protrusion, mild stenosis"
❌ WRONG: "MRI shows disc herniation and stenosis"
✅ CORRECT: "L4-L5 disc herniation was noted on MRI with 5mm protrusion and mild stenosis"

**WORK STATUS FIELD:**
Source: "Off work, no lifting >10 lbs for 4 weeks"
❌ WRONG: "Patient is off work with lifting restrictions"
✅ CORRECT: "Off work status was documented with lifting restrictions of no more than 10 pounds for 4 weeks"

**MEDICATIONS FIELD:**
Source: "Hydrocodone 10mg TID, gabapentin 300mg TID"
❌ WRONG: "Taking hydrocodone and gabapentin"
✅ CORRECT: "Hydrocodone 10mg three times daily and gabapentin 300mg three times daily were documented"

**OUTPUT:** Valid JSON with STRICT PAST-TENSE compliance.
Every sentence must pass: "This is a restatement of what was explicitly documented."
""")
    
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    
    # Create LLM instance
    from config.settings import CONFIG
    
    summary_llm = AzureChatOpenAI(
        azure_deployment=CONFIG.get("azure_openai_deployment"),
        azure_endpoint=CONFIG.get("azure_openai_endpoint"),
        api_key=CONFIG.get("azure_openai_api_key"),
        api_version=CONFIG.get("azure_openai_api_version"),
        temperature=0.1,  # Low for consistency
        max_tokens=6000,   # Reduced since we want concise output
        timeout=90,
        request_timeout=90,
    )
    
    # Retry mechanism
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Critical-focus generation attempt {attempt + 1}/{max_retries}")
            
            chain = chat_prompt | summary_llm
            response = chain.invoke({
                "doc_type": doc_type,
                "allowed_fields_list": "\n".join([f"• {field}" for field in allowed_fields]),
                "priority_fields_list": "\n".join([f"• {field}" for field in priority_fields]),
                "raw_text": raw_text[:12000]  # Limit for focus
            })
            
            # Extract JSON
            response_content = response.content.strip()
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                response_content = response_content[start_idx:end_idx+1]
            
            # Clean and parse
            response_content = "".join(ch for ch in response_content if ch >= ' ' or ch in '\n\r\t')
            structured_summary = json.loads(response_content, strict=False)
            
            # Apply post-processing with CRITICAL focus
            structured_summary = ensure_proper_header(structured_summary, doc_type, raw_text)
            structured_summary = apply_critical_focus_filtering(structured_summary, priority_fields)
            structured_summary = filter_disallowed_fields(structured_summary, allowed_fields)
            structured_summary = deduplicate_fields(structured_summary)
            
            # Enrich with semantic context expansion from original document
            # This attaches supporting context from the source document to each bullet point
            if text and structured_summary.get("summary", {}).get("items"):
                try:
                    logger.info("🔍 Enriching summary with semantic context expansion...")
                    structured_summary = enrich_summary_with_context(structured_summary, text)
                except Exception as ctx_error:
                    logger.warning(f"⚠️ Context expansion failed (non-critical): {ctx_error}")
                    # Continue without context expansion - non-blocking error
            
            logger.info(f"✅ Generated critical-focus summary with {len(structured_summary.get('summary', {}).get('items', []))} items")
            return structured_summary
            
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {e}"
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
    
    # Fallback
    logger.error(f"❌ All generation attempts failed. Using fallback.")
    return create_fallback_summary(doc_type, priority_fields)


def apply_critical_focus_filtering(structured_summary: dict, priority_fields: List[str]) -> dict:
    """
    Apply additional filtering to ensure only critical information is included.
    Removes trivial bullets and ensures complete sentences.
    """
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    items = structured_summary["summary"]["items"]
    filtered_items = []
    
    for item in items:
        field = item.get("field", "")
        collapsed = item.get("collapsed", "").strip()
        expanded = item.get("expanded", "").strip()
        
        # Skip if both empty
        if not collapsed and not expanded:
            continue
        
        # Check if collapsed is meaningful (not generic)
        if is_trivial_statement(collapsed):
            logger.info(f"🗑️ Skipping field with trivial collapsed: {field} - '{collapsed[:50]}...'")
            continue
        
        # Filter expanded bullets for critical info only
        if expanded:
            bullets = [b.strip() for b in expanded.split('\n') if b.strip().startswith('•')]
            critical_bullets = []
            
            for bullet in bullets:
                # Clean the bullet
                bullet_text = bullet[1:].strip() if bullet.startswith('•') else bullet
                
                # Skip trivial bullets
                if is_trivial_statement(bullet_text):
                    logger.debug(f"🗑️ Skipping trivial bullet in {field}: '{bullet_text[:50]}...'")
                    continue
                
                # Ensure complete sentence
                if not is_complete_sentence(bullet_text):
                    bullet_text = make_complete_sentence(bullet_text, field)
                
                # Check minimum word count (except for medication doses)
                if field != "medications" and len(bullet_text.split()) < 5:
                    # Try to enhance if possible
                    enhanced = enhance_brief_statement(bullet_text, field)
                    if enhanced and len(enhanced.split()) >= 5:
                        bullet_text = enhanced
                    else:
                        logger.debug(f"🗑️ Skipping too-brief bullet in {field}: '{bullet_text}'")
                        continue
                
                critical_bullets.append(f"• {bullet_text}")
            
            # Only include field if we have critical bullets
            if critical_bullets and len(critical_bullets) <= 8:  # Limit to 8 bullets max
                # Limit to 5 bullets for non-priority fields
                if field not in priority_fields and len(critical_bullets) > 5:
                    critical_bullets = critical_bullets[:5]
                    logger.info(f"📏 Limited {field} to 5 bullets (non-priority field)")
                
                item["expanded"] = "\n".join(critical_bullets)
                filtered_items.append(item)
            else:
                logger.info(f"🗑️ Skipping field {field} - no critical bullets after filtering")
        else:
            # Has collapsed but no expanded - keep if collapsed is critical
            if not is_trivial_statement(collapsed):
                filtered_items.append(item)
    
    # Reorder by priority
    filtered_items = prioritize_fields(filtered_items, priority_fields)
    
    structured_summary["summary"]["items"] = filtered_items
    return structured_summary


def is_trivial_statement(text: str) -> bool:
    """
    Check if a statement is trivial or non-critical for clinical decision-making.
    
    Clinical Signal Filter:
    - EXCLUDE: Normal findings, expected baselines, negative results with no clinical value
    - INCLUDE: Abnormal findings, new diagnoses, significant symptoms, risks, treatment changes
    
    This function acts as a clinical relevance gate - only actionable information passes through.
    """
    if not text or len(text.strip()) < 10:
        return True
    
    text_lower = text.lower()
    
    # ============== TRIVIAL PATTERN CATEGORIES ==============
    
    # 1. NEGATIVE/ABSENT FINDINGS (No clinical signal)
    negative_findings_patterns = [
        r'no\s+(?:signs?|evidence|indication|symptoms?)\s+(?:of|for)',  # "No signs of typhoid"
        r'no\s+(?:prior|previous|comparison|available|found|identified|specified|documented)',
        r'not\s+(?:available|found|identified|specified|documented|noted|observed|seen|detected)',
        r'(?:was|were)\s+(?:not\s+)?(?:negative|absent|unremarkable)',
        r'negative\s+(?:for|finding|result|screen|test)',
        r'(?:denies?|denied)\s+(?:any|all|having)',
        r'no\s+(?:acute|significant|notable|abnormal|remarkable)\s+(?:findings?|changes?|abnormalit)',
        r'absence\s+of\s+(?:any|significant|notable)',
        r'without\s+(?:any|evidence|signs?|symptoms?)\s+of',
        r'ruled\s+out',
        r'excluded',
    ]
    
    # 2. NORMAL/BASELINE FINDINGS (Expected, non-actionable)
    normal_baseline_patterns = [
        r'(?:within|in)\s+normal\s+(?:limits?|range|parameters?)',
        r'normal\s+(?:range|findings?|exam|study|results?|appearance|anatomy|function)',
        r'unremarkable\s+(?:exam|findings?|study|appearance|history)',
        r'(?:grossly|essentially|otherwise)\s+(?:normal|unremarkable|intact)',
        r'(?:appeared?|appears?|seems?)\s+(?:normal|unremarkable|stable)',
        r'no\s+(?:obvious|apparent|visible|detectable)\s+(?:abnormalit|patholog|lesion|mass)',
        r'preserved\s+(?:function|anatomy|structure|alignment)',
        r'intact\s+(?:skin|sensation|reflexes?|motor|function)',
        r'symmetrical?\s+(?:findings?|exam|appearance)',
        r'age[\s-]?(?:appropriate|related|expected)',
        r'expected\s+(?:for|given|based\s+on)',
        r'consistent\s+with\s+(?:normal|age|baseline)',
        r'benign\s+(?:finding|appearance|condition)',
        r'stable\s+(?:appearance|finding|condition|from\s+prior)',
        r'unchanged\s+(?:from|compared|since)',
    ]
    
    # 3. NON-ACTIONABLE/ROUTINE STATEMENTS
    routine_patterns = [
        r'routine\s+(?:follow[\s-]?up|appointment|visit|exam|screening)',
        r'as\s+(?:previously|already|earlier)\s+(?:documented|noted|reported|discussed|mentioned)',
        r'(?:patient|pt)\s+(?:was|is)\s+(?:advised|counseled|educated|informed)',
        r'(?:discussed|reviewed)\s+(?:with|options|plan)',
        r'(?:will|to)\s+(?:continue|maintain|monitor|follow[\s-]?up)',
        r'return\s+(?:as\s+needed|prn|if\s+(?:symptoms?|condition))',
        r'no\s+(?:new|additional|further)\s+(?:complaints?|concerns?|issues?)',
        r'(?:remains?|remained)\s+(?:stable|unchanged|the\s+same)',
        r'tolerating\s+(?:well|medication|treatment)',
        r'compliant\s+with\s+(?:medication|treatment|therapy)',
        r'no\s+(?:side\s+effects?|adverse\s+(?:effects?|reactions?|events?))',
        r'(?:good|fair|adequate)\s+(?:progress|response|compliance|tolerance)',
    ]
    
    # 4. GENERIC/VAGUE STATEMENTS (No specific clinical value)
    generic_patterns = [
        r'multiple\s+(?:findings?|items?|things?)\s+were\s+(?:documented|noted|reported)',
        r'various\s+(?:findings?|items?|aspects?)\s+were\s+(?:documented|noted|reported)',
        r'several\s+(?:findings?|items?|points?)\s+were\s+(?:documented|noted|reported)',
        r'information\s+was\s+(?:documented|provided|included|noted)',
        r'details?\s+(?:was|were)\s+(?:provided|included|documented)',
        r'content\s+was\s+(?:included|documented|reviewed)',
        r'(?:the\s+)?(?:report|document|record)\s+(?:contained|included|documented)',
        r'(?:some|certain)\s+(?:findings?|information)\s+(?:was|were)',
        r'general\s+(?:findings?|information|overview)',
        r'(?:standard|typical|usual)\s+(?:findings?|presentation|course)',
    ]
    
    # 5. HEDGING/UNCERTAIN STATEMENTS (Low clinical confidence)
    hedging_patterns = [
        r'(?:may|might|could|possibly|potentially)\s+be\s+(?:normal|benign|insignificant)',
        r'(?:unlikely|improbable)\s+(?:to\s+be|that)',
        r'(?:cannot|can\s+not)\s+(?:exclude|rule\s+out)\s+(?:normal|benign)',
        r'(?:no\s+definite|no\s+definitive|no\s+clear)\s+(?:evidence|finding|abnormality)',
        r'(?:clinically|medically)\s+(?:insignificant|unimportant|irrelevant)',
        r'of\s+(?:no|little|minimal)\s+(?:clinical|diagnostic)\s+(?:significance|importance|relevance)',
        r'incidental\s+(?:finding|note|observation)',
        r'artifact',
    ]
    
    # 6. SPECIFIC NON-INFORMATIVE DISEASE EXCLUSIONS
    # Statements that rule out conditions without adding diagnostic value
    disease_exclusion_patterns = [
        r'no\s+(?:signs?|evidence)\s+of\s+(?:\w+\s+){0,2}(?:were|was)\s+(?:noted|observed|found|seen)',
        r'(?:typhoid|malaria|tuberculosis|tb|hiv|hepatitis|dengue|cholera)\s+(?:was|were)\s+(?:ruled\s+out|negative|not\s+(?:found|detected|present))',
        r'(?:screening|test|tests?)\s+(?:for\s+)?(?:\w+\s+){0,2}(?:was|were)\s+negative',
        r'no\s+(?:fracture|dislocation|subluxation|effusion|mass|lesion|tumor)\s+(?:was|were)\s+(?:identified|seen|noted|detected)',
    ]
    
    # ============== PATTERN MATCHING ==============
    
    all_trivial_patterns = (
        negative_findings_patterns +
        normal_baseline_patterns +
        routine_patterns +
        generic_patterns +
        hedging_patterns +
        disease_exclusion_patterns
    )
    
    for pattern in all_trivial_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # ============== PHRASE-BASED DETECTION ==============
    
    trivial_phrases = [
        # Generic statements
        "multiple findings were documented",
        "various findings were noted",
        "several items were reported",
        "information was documented",
        "details were provided",
        "content was included",
        # Normal findings
        "within normal limits",
        "no abnormality",
        "no abnormalities",
        "grossly normal",
        "essentially normal",
        "appears normal",
        "appeared normal",
        "unremarkable",
        # Negative findings
        "no evidence of",
        "no signs of",
        "no indication of",
        "was negative",
        "were negative",
        "not detected",
        "not identified",
        "not observed",
        "not seen",
        "not found",
        "ruled out",
        "was excluded",
        # Routine/expected
        "as expected",
        "as anticipated",
        "no change",
        "no changes",
        "remains stable",
        "remained stable",
        "unchanged from",
        "no new findings",
        "no acute findings",
        "no significant change",
        "stable condition",
        "stable findings",
        # Non-actionable
        "will continue",
        "will monitor",
        "to follow up",
        "return if",
        "as needed",
        "prn",
    ]
    
    if any(phrase in text_lower for phrase in trivial_phrases):
        return True
    
    # ============== STRUCTURAL TRIVIALITY CHECK ==============
    
    # If statement is primarily about absence/negation without actionable context
    negation_words = ['no', 'not', 'none', 'neither', 'never', 'without', 'absent', 'negative', 'denied', 'denies']
    words = text_lower.split()
    negation_count = sum(1 for word in words if word in negation_words)
    
    # If more than 30% of content words are negations, likely trivial
    if len(words) > 5 and negation_count / len(words) > 0.3:
        # But check if there's actionable content despite negations
        actionable_indicators = [
            'recommend', 'advised', 'prescribed', 'refer', 'urgent', 'emergent',
            'critical', 'abnormal', 'elevated', 'decreased', 'significant',
            'concern', 'risk', 'complication', 'diagnosis', 'treatment'
        ]
        if not any(indicator in text_lower for indicator in actionable_indicators):
            return True
    
    return False


def is_complete_sentence(text: str) -> bool:
    """
    Check if text appears to be a complete sentence.
    Simple heuristic - has subject and verb, ends with punctuation.
    """
    if not text:
        return False
    
    # Check for ending punctuation
    if not text[-1] in '.!?':
        return False
    
    # Check for verb indicators (simple heuristic)
    words = text.split()
    if len(words) < 4:  # Very short sentences are often fragments
        return False
    
    # Check for common verbs in past tense (for non-authorship)
    verb_indicators = ['was', 'were', 'documented', 'reported', 'noted', 
                      'showed', 'revealed', 'indicated', 'found', 'observed']
    
    has_verb = any(word.lower() in verb_indicators for word in words)
    if not has_verb:
        # Check for other verb forms
        has_verb_ending = any(word.lower().endswith(('ed', 'ing')) for word in words)
        if not has_verb_ending:
            return False
    
    return True


def make_complete_sentence(fragment: str, field: str) -> str:
    """
    Try to convert a fragment into a complete sentence.
    """
    fragment = fragment.strip()
    if not fragment:
        return fragment
    
    # Add ending punctuation if missing
    if fragment[-1] not in '.!?':
        fragment += '.'
    
    # Common field-specific completions
    field_completions = {
        "findings": f"The report documented {fragment.lower()}",
        "work_status": f"Work status included {fragment.lower()}",
        "medications": f"Medication regimen included {fragment.lower()}",
        "recommendations": f"Treatment recommendations included {fragment.lower()}",
        "physical_exam": f"Physical examination revealed {fragment.lower()}",
    }
    
    if field in field_completions:
        return field_completions[field]
    
    # Generic completion
    if not fragment[0].isupper():
        fragment = fragment[0].upper() + fragment[1:]
    
    # Check if it already has a verb
    words = fragment.lower().split()
    has_verb = any(word in ['was', 'were', 'documented', 'reported', 'noted'] for word in words)
    
    if not has_verb:
        return f"The report documented {fragment.lower()}"
    
    return fragment


def enhance_brief_statement(text: str, field: str) -> str:
    """
    Try to enhance a very brief statement with more context.
    """
    # Field-specific enhancements
    enhancements = {
        "findings": {
            r'(\d+mm)': r'a \1 protrusion',
            r'([A-Z]\d+-[A-Z]\d+)': r'at the \1 level',
            r'(herniation|protrusion|stenosis)': r'with \1',
        },
        "work_status": {
            r'off\s+work': r'off work with temporary total disability',
            r'no\s+lifting': r'no lifting restrictions',
            r'restrictions': r'work restrictions',
        },
        "medications": {
            r'(\d+mg)': r'\1 dosage',
            r'([A-Z][a-z]+)': r'\1 medication',
        }
    }
    
    if field in enhancements:
        for pattern, replacement in enhancements[field].items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

# ============== COMPATIBILITY FUNCTIONS ==============

def remove_patient_identifiers(structured_summary: dict) -> dict:
    """Compatibility function - returns as-is since PII removal happens elsewhere."""
    return structured_summary


def validate_ui_items(structured_summary: dict) -> dict:
    """Compatibility function - validates summary items."""
    if "summary" in structured_summary and "items" in structured_summary["summary"]:
        structured_summary["summary"]["items"] = validate_summary_items(
            structured_summary["summary"]["items"]
        )
    return structured_summary


def filter_empty_or_generic_fields(structured_summary: dict) -> dict:
    """Compatibility function - filters empty fields."""
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    items = structured_summary["summary"]["items"]
    filtered_items = []
    
    for item in items:
        collapsed = item.get("collapsed", "").strip()
        expanded = item.get("expanded", "").strip()
        
        # Keep if either collapsed or expanded has content
        if collapsed or expanded:
            filtered_items.append(item)
    
    structured_summary["summary"]["items"] = filtered_items
    return structured_summary


def enforce_bullet_format_all_fields(structured_summary: dict) -> dict:
    """Compatibility function - ensures bullet format."""
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    for item in structured_summary["summary"]["items"]:
        expanded = item.get("expanded", "").strip()
        if expanded and not any(b.strip().startswith('•') for b in expanded.split('\n')):
            lines = [line.strip() for line in expanded.split('. ') if line.strip()]
            if lines:
                item["expanded"] = '\n'.join([f"• {line}" for line in lines])
    
    return structured_summary


def consolidate_redundant_bullets(structured_summary: dict) -> dict:
    """Compatibility function - simple consolidation."""
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    return structured_summary  # Minimal consolidation - deduplicate_fields handles basics
