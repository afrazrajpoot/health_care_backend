"""
Structured Short Summary Generator
Reusable helper for generating UI-ready, clickable medical summaries with collapsed/expanded views.
"""
import logging
import json
import re
from typing import Dict, List, Literal, Set
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("document_ai")


# ============== Report-Type Field Eligibility Matrix ==============

REPORT_FIELD_MATRIX: Dict[str, Dict] = {
    # Med-Legal Reports (QME family)
    "QME": {
        "allowed": {
            "findings", "physical_exam",
            "medications", "recommendations", "rationale",
            "mmi_status", "work_status"
        }
    },
    "AME": {"inherit": "QME"},
    "PQME": {"inherit": "QME"},
    "IME": {"inherit": "QME"},
    
    # Consult Reports
    "CONSULT": {
        "allowed": {
            "findings",
            "physical_exam", "medications", "recommendations"
        }
    },
    "PAIN MANAGEMENT": {"inherit": "CONSULT"},
    "PROGRESS NOTE": {"inherit": "CONSULT"},
    "OFFICE VISIT": {"inherit": "CONSULT"},
    "CLINIC NOTE": {"inherit": "CONSULT"},
    
    # Imaging Reports
    "MRI": {
        "allowed": {"findings", }
    },
    "CT": {"inherit": "MRI"},
    "X-RAY": {"inherit": "MRI"},
    "XRAY": {"inherit": "MRI"},
    "ULTRASOUND": {"inherit": "MRI"},
    "EMG": {"inherit": "MRI"},
    "PET SCAN": {"inherit": "MRI"},
    "BONE SCAN": {"inherit": "MRI"},
    "DEXA SCAN": {"inherit": "MRI"},
    
    # Utilization Review
    "UR": {
        "allowed": {"recommendations", "rationale"}
    },
    "IMR": {"inherit": "UR"},
    "PEER REVIEW": {"inherit": "UR"},
    
    # Therapy Notes
    "PHYSICAL THERAPY": {
        "allowed": {"findings", "recommendations"}
    },
    "OCCUPATIONAL THERAPY": {"inherit": "PHYSICAL THERAPY"},
    "CHIROPRACTIC": {"inherit": "PHYSICAL THERAPY"},
    
    # Surgical Reports
    "SURGERY REPORT": {
        "allowed": {"findings", "recommendations"}
    },
    "OPERATIVE NOTE": {"inherit": "SURGERY REPORT"},
    "POST-OP": {"inherit": "SURGERY REPORT"},
    
    # PR-2 Reports
    "PR-2": {
        "allowed": {"findings", "recommendations", "work_status"}
    },
    "PR2": {"inherit": "PR-2"},
    
    # Labs & Diagnostics
    "LABS": {
        "allowed": {"findings", "recommendations"}
    },
    "PATHOLOGY": {"inherit": "LABS"},
    
    # Default for unmatched types
    "DEFAULT": {
        "allowed": {"findings", "recommendations"}
    }
}


def resolve_allowed_fields(doc_type: str) -> Set[str]:
    """
    Resolve the allowed fields for a document type, handling inheritance.
    
    Args:
        doc_type: The document type string
        
    Returns:
        Set of allowed field names
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
    
    return matrix.get("allowed", set())


# ============== Pydantic Models for UI-Ready Summary ==============

class UIFact(BaseModel):
    """
    A UI-clickable summary field with collapsed and expanded views.
    Used for generating clickable UI elements that expand on user interaction.
    """
    field: Literal[
        "findings",
        "physical_exam",
        "vital_signs",
        "medications",
        "recommendations",
        "rationale",
        "mmi_status",
        "work_status"
    ] = Field(description="The type of UI field")
    collapsed: str = Field(description="Short, high-level, one-line summary for collapsed view")
    expanded: str = Field(description="Expanded, still attributed description for expanded view")


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
    """
    UI-driven summary content with clickable fields.
    Each item represents a collapsible UI element.
    """
    items: List[UIFact] = Field(default_factory=list, description="List of UI-ready fact items")


class StructuredShortSummary(BaseModel):
    """Complete structured short summary for UI display with clickable elements"""
    header: SummaryHeader = Field(description="Header information")
    summary: SummaryContent = Field(description="Summary content with UI-ready items")


# ============== Helper Functions ==============

def create_fallback_structured_summary(doc_type: str) -> dict:
    """Create a fallback structured summary when generation fails."""
    return {
        "header": {
            "title": doc_type,
            "source_type": "External Medical Document",
            "author": "",
            "date": "",
            "disclaimer": "This summary references an external document and is for workflow purposes only. It does not constitute medical advice."
        },
        "summary": {
            "items": []
        }
    }


def remove_patient_identifiers(structured_summary: dict) -> dict:
    """
    Remove any patient identifiers that may have slipped through.
    Scans all text fields for PII patterns and removes them.
    """
    # Patterns to detect and remove
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{2}/\d{2}/\d{4}\b',  # DOB format
        r'\bMRN[:\s]*\w+\b',  # MRN
        r'\bClaim[#:\s]*[\w-]+\b',  # Claim numbers
        r'\bPatient[:\s]+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Patient names
    ]
    
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text
        cleaned = text
        for pattern in pii_patterns:
            cleaned = re.sub(pattern, '[REDACTED]', cleaned, flags=re.IGNORECASE)
        # Remove any [REDACTED] placeholders entirely
        cleaned = re.sub(r'\[REDACTED\]\s*', '', cleaned)
        return cleaned.strip()
    
    def clean_dict(d: dict) -> dict:
        if not isinstance(d, dict):
            return d
        cleaned = {}
        for key, value in d.items():
            if isinstance(value, str):
                cleaned[key] = clean_text(value)
            elif isinstance(value, dict):
                cleaned[key] = clean_dict(value)
            elif isinstance(value, list):
                cleaned[key] = clean_list(value)
            else:
                cleaned[key] = value
        return cleaned
    
    def clean_list(lst: list) -> list:
        if not isinstance(lst, list):
            return lst
        cleaned = []
        for item in lst:
            if isinstance(item, str):
                cleaned.append(clean_text(item))
            elif isinstance(item, dict):
                cleaned.append(clean_dict(item))
            elif isinstance(item, list):
                cleaned.append(clean_list(item))
            else:
                cleaned.append(item)
        return cleaned
    
    return clean_dict(structured_summary)


def ensure_header_fields(structured_summary: dict, doc_type: str, raw_text: str) -> dict:
    """
    Ensure all required header fields are present and properly formatted.
    """
    if "header" not in structured_summary:
        structured_summary["header"] = {}
    
    header = structured_summary["header"]
    
    # Ensure title
    if not header.get("title"):
        header["title"] = doc_type
    
    # Ensure source_type
    if not header.get("source_type"):
        header["source_type"] = "External Medical Document"
    
    # Clean author - remove "Dr." prefix if present
    if header.get("author"):
        author = header["author"]
        author = re.sub(r'^Dr\.?\s*', '', author, flags=re.IGNORECASE)
        header["author"] = author.strip()
    
    # Validate date format (YYYY-MM-DD)
    if header.get("date"):
        date_str = header["date"]
        # Try to convert common formats to YYYY-MM-DD
        date_patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # MM/DD/YYYY
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3'),  # Already correct
        ]
        for pattern, replacement in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    # Normalize to YYYY-MM-DD
                    parts = re.sub(pattern, replacement, date_str)
                    header["date"] = parts.split()[0] if ' ' in parts else parts
                    break
                except:
                    pass
    
    # Ensure disclaimer
    if not header.get("disclaimer"):
        header["disclaimer"] = "This summary references an external document and is for workflow purposes only. It does not constitute medical advice."
    
    # Ensure summary section exists with items array
    if "summary" not in structured_summary:
        structured_summary["summary"] = {
            "items": []
        }
    elif "items" not in structured_summary["summary"]:
        structured_summary["summary"]["items"] = []
    
    return structured_summary


def filter_disallowed_fields(structured_summary: dict, allowed_fields: Set[str]) -> dict:
    """
    Filter out any UI fields that are not allowed for the document type.
    This is a defensive measure to ensure compliance.
    
    Args:
        structured_summary: The structured summary dict
        allowed_fields: Set of allowed field names
        
    Returns:
        Filtered structured summary
    """
    if "summary" in structured_summary and "items" in structured_summary["summary"]:
        structured_summary["summary"]["items"] = [
            item for item in structured_summary["summary"]["items"]
            if item.get("field") in allowed_fields
        ]
    return structured_summary


def validate_ui_items(structured_summary: dict) -> dict:
    """
    Validate UI items have required fields (collapsed, expanded).
    Removes invalid items and ensures proper structure.
    """
    valid_fields = {
        "findings", "physical_exam",
        "vital_signs", "medications", "recommendations", "rationale",
        "mmi_status", "work_status"
    }
    
    if "summary" in structured_summary and "items" in structured_summary["summary"]:
        items = structured_summary["summary"]["items"]
        validated_items = []
        
        for item in items:
            if isinstance(item, dict):
                field = item.get("field", "").lower()
                collapsed = item.get("collapsed", "").strip()
                expanded = item.get("expanded", "").strip()
                
                # Skip items without required fields or invalid field types
                if field not in valid_fields:
                    continue
                if not collapsed or not expanded:
                    continue
                
                # Normalize field name
                item["field"] = field
                validated_items.append(item)
        
        structured_summary["summary"]["items"] = validated_items
    
    return structured_summary


def deduplicate_fields(structured_summary: dict) -> dict:
    """
    Ensure each field type appears only once.
    If duplicates exist, consolidate them into a single item.
    """
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    items = structured_summary["summary"]["items"]
    field_map = {}  # field_name -> consolidated item
    
    for item in items:
        field = item.get("field", "")
        if not field:
            continue
            
        if field not in field_map:
            # First occurrence - store as is
            field_map[field] = {
                "field": field,
                "collapsed": item.get("collapsed", ""),
                "expanded": item.get("expanded", "")
            }
        else:
            # Duplicate - consolidate by appending to expanded
            existing = field_map[field]
            new_expanded = item.get("expanded", "")
            if new_expanded and new_expanded not in existing["expanded"]:
                existing["expanded"] = existing["expanded"].rstrip(". ") + ". " + new_expanded
            logger.warning(f"⚠️ Consolidated duplicate field '{field}' into single item")
    
    # Convert back to list, maintaining a logical order
    field_order = [
        "findings", "physical_exam",
        "vital_signs", "medications", "recommendations", "rationale",
        "mmi_status", "work_status"
    ]
    
    deduplicated_items = []
    for field in field_order:
        if field in field_map:
            deduplicated_items.append(field_map[field])
    
    # Add any fields not in the standard order
    for field, item in field_map.items():
        if field not in field_order:
            deduplicated_items.append(item)
    
    structured_summary["summary"]["items"] = deduplicated_items
    return structured_summary


def filter_empty_or_generic_fields(structured_summary: dict) -> dict:
    """
    Filter out fields that have no meaningful content or generic "not found" messages.
    
    Removes items where:
    - Text is too short or empty
    - Contains generic phrases like "No specific X were documented"
    - Contains "not found", "not available", "not specified"
    - Is just a placeholder with no real clinical content
    - Contains incomplete sentences (e.g., "The at MMI", "The from work")
    """
    if "summary" not in structured_summary or "items" not in structured_summary["summary"]:
        return structured_summary
    
    # Generic phrases that indicate no meaningful content
    generic_patterns = [
        r"no\s+specific\s+\w+\s+were?\s+(documented|noted|described|found)",
        r"no\s+\w+\s+were?\s+(documented|noted|described|found)",
        r"not\s+(found|available|specified|documented|noted)",
        r"none\s+(documented|noted|described|found)",
        r"(recommendations?|findings?|medications?|exam)\s+not\s+",
        r"^no\s+\w+\s*\.?$",  # Just "No X" or "No X."
    ]
    
    # Incomplete sentence patterns that indicate malformed text
    incomplete_patterns = [
        r"^the\s+(at|from|to|in|on|for|with|was|is)\s+\w+",  # "The at MMI", "The from work", etc.
        r"\b(patient|report|document)\s+(at|from|to)\s+(the|a)\s*$",  # Incomplete phrases
        r"^(at|from|to|in|on)\s+\w+\s*$",  # Just preposition + word
        r"\bthe\s+the\b",  # Duplicate "the"
        r"^was\s+\w+\s*$",  # Just "was [word]"
        r"^is\s+\w+\s*$",  # Just "is [word]"
    ]
    
    items = structured_summary["summary"]["items"]
    filtered_items = []
    
    for item in items:
        collapsed = item.get("collapsed", "").strip()
        expanded = item.get("expanded", "").strip()
        field = item.get("field", "")
        
        # Check if collapsed or expanded is too short or empty
        if not collapsed or not expanded or len(collapsed) < 10 or len(expanded) < 15:
            logger.info(f"🗑️ Removed empty field '{field}' (too short or empty)")
            continue
        
        # Check for incomplete sentences
        is_incomplete = False
        for text, label in [(collapsed, "collapsed"), (expanded, "expanded")]:
            for pattern in incomplete_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"🗑️ Removed field '{field}' with incomplete {label} text: '{text[:50]}...'")
                    is_incomplete = True
                    break
            if is_incomplete:
                break
        
        if is_incomplete:
            continue
        
        # Check for generic patterns in both collapsed and expanded
        is_generic = False
        text_to_check = f"{collapsed} {expanded}".lower()
        
        for pattern in generic_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                logger.info(f"🗑️ Removed generic field '{field}': contains '{pattern}'")
                is_generic = True
                break
        
        if is_generic:
            continue
        
        # Keep this item
        filtered_items.append(item)
    
    structured_summary["summary"]["items"] = filtered_items
    logger.info(f"✅ Filtered to {len(filtered_items)} meaningful fields (removed {len(items) - len(filtered_items)} empty/generic/incomplete)")
    return structured_summary


def generate_structured_short_summary(llm: AzureChatOpenAI, raw_text: str, doc_type: str, long_summary: str) -> dict:
    """
    Generate a structured, UI-ready summary with clickable collapsed/expanded fields.
    Output is reference-only, past-tense, and EMR-safe.
    
    Args:
        llm: Azure OpenAI LLM instance
        raw_text: The Document AI summarizer output (primary context)
        doc_type: Document type
        long_summary: Detailed reference context
        
    Returns:
        dict: Structured summary with header and UI-ready items
    """
    logger.info("🎯 Generating UI-ready structured summary...")
    
    # Resolve allowed fields for this document type
    allowed_fields = resolve_allowed_fields(doc_type)
    logger.info(f"📋 Allowed fields for {doc_type}: {allowed_fields}")
    
    # Create Pydantic output parser for consistent response structure
    pydantic_parser = PydanticOutputParser(pydantic_object=StructuredShortSummary)

    system_prompt = SystemMessagePromptTemplate.from_template("""
You are a COURT REPORTER, not a clinician.

You extract and reformat content from EXTERNAL medical or legal documents.
You do NOT author medical conclusions.
You do NOT interpret findings.
You do NOT make clinical judgments.

CORE PRINCIPLE:
DocLatch must sound like a court reporter documenting what was said, NOT a clinician concluding what is true.

🚨 ZERO TOLERANCE FOR HALLUCINATION OR FABRICATION:
- ONLY extract information that is EXPLICITLY stated in the source document
- NEVER infer, assume, or fabricate any information
- NEVER add clinical interpretations that are not in the original text
- If something is not clearly stated, DO NOT include it
- When in doubt, leave it out

🔴 CRITICAL DECISION TERMS - MUST BE PRESERVED EXACTLY:
When the document contains specific decision or status terms, you MUST include them verbatim:
- "authorized", "approved", "denied", "deferred", "modified"
- "recommended", "not recommended", "contraindicated"
- "granted", "rejected", "pending review"
- "certified", "supported", "not supported"
- "at MMI", "not at MMI", "permanent and stationary"
- "temporarily disabled", "permanently disabled"

EXAMPLE - Preserving Decision Terms (CORRECT):
Source: "The request for lumbar MRI is DENIED as not medically necessary."
✅ Output: "The report denied the request for lumbar MRI as not medically necessary."

❌ WRONG - Omitting Decision Term:
"The report documented a lumbar MRI request."  ← Missing "DENIED"!

EXAMPLE - Preserving Authorization Status (CORRECT):
Source: "Physical therapy 2x/week for 6 weeks is AUTHORIZED."
✅ Output: "The report authorized physical therapy twice weekly for six weeks."

❌ WRONG - Vague:
"Physical therapy was recommended."  ← Missing "AUTHORIZED"!

🚨 CRITICAL FIELD RULES (HIGHEST PRIORITY):
1. EACH FIELD TYPE CAN ONLY APPEAR ONCE in the output
2. If multiple items belong to the same field type, CONSOLIDATE them into ONE item
3. ONLY use field names from the allowed list provided
4. Use the CORRECT field type for the content:
   - "findings" = Clinical observations, test results, abnormalities, diagnoses
   - "recommendations" = Treatment plans, follow-up, referrals (include authorization status if stated)
   - "medications" = Drugs prescribed or referenced
   - "physical_exam" = Physical examination findings
   - "vital_signs" = Vital sign measurements
   - "rationale" = Clinical reasoning documented (especially for UR denials/approvals)
   - "mmi_status" = Maximum Medical Improvement status
   - "work_status" = Work restrictions or capacity

MANDATORY LANGUAGE RULES (NO EXCEPTIONS):
✅ ALLOWED VERBS ONLY:
- documented, described, referenced, reported, noted, indicated, stated, listed, mentioned

❌ FORBIDDEN VERBS (cause authorship leakage):
- identified, consistent with, demonstrates, confirms, shows, reveals, suggests

✅ ATTRIBUTION PATTERNS:
- "The [document type] documented..."
- "The report described..."
- "[Condition] was noted in the report..."
- "As documented in the [document type]..."

🚨 COMPLETE SENTENCE REQUIREMENTS:
- EVERY sentence MUST be grammatically complete and meaningful
- NEVER write incomplete fragments like "The at MMI" or "The from work"
- ALWAYS include the subject (patient, report, document) AND complete verb phrase
- Examples of COMPLETE sentences:
  ✅ "The patient is at maximum medical improvement (MMI)"
  ✅ "The report documented that the patient is off from work"
  ✅ "The patient cannot return to work at this time"
  ❌ "The at MMI" (INCOMPLETE - missing subject and verb)
  ❌ "The from work" (INCOMPLETE - meaningless fragment)
  ❌ "The return to work at this time" (INCOMPLETE - missing subject and verb)

TENSE & VOICE:
- Past tense only (was documented, were noted, was described)
- Never present tense declarations
- Attribution must be clear in EVERY statement

PRIVACY:
- No patient identifiers (name, DOB, MRN, phone, claim number)
- Dates in YYYY-MM-DD format only

UI FIELD STRUCTURE:
Each field contains:
1. collapsed → One-line summary with attribution (high-signal)
2. expanded → Consolidated detail covering ALL relevant info for that field type

🔹 SPECIAL FORMAT FOR physical_exam AND medications:
These fields use BULLET POINT format in expanded (NOT paragraph prose):
- collapsed = Brief summary line
- expanded = Simple bullet points, one finding/medication per line
- NO paragraph text, NO verbose descriptions
- Each bullet is a short, scannable item

EXAMPLE - physical_exam (BULLET FORMAT):
{{
  "field": "physical_exam",
  "collapsed": "Reduced strength and positive provocative tests were noted",
  "expanded": "• Reduced right shoulder abduction strength\n• Pain with supraspinatus testing\n• Positive Spurling's maneuver on left\n• Limited cervical rotation and extension"
}}

EXAMPLE - medications (BULLET FORMAT):
{{
  "field": "medications",
  "collapsed": "Pain and nerve medications were documented",
  "expanded": "• Gabapentin 300 mg three times daily\n• Meloxicam 15 mg once daily\n• Topical diclofenac twice daily\n• Cyclobenzaprine 10 mg at bedtime"
}}

❌ WRONG (paragraph style for these fields):
"The report documented gabapentin 300 mg three times daily for leg tingling, with NSAIDs noted to cause stomach discomfort..."

✅ CORRECT (bullet points):
"• Gabapentin 300 mg three times daily\n• NSAIDs (stomach discomfort noted)\n• Muscle relaxants for sleep"

CONSOLIDATION EXAMPLE (CORRECT):
If source has multiple findings like disc degeneration, facet arthrosis, and anterolisthesis:
✅ ONE "findings" item consolidating all (CONCISE - max 3 lines):
{{
  "field": "findings",
  "collapsed": "Degenerative disc disease and structural changes were documented",
  "expanded": "The report documented disc degeneration at L2-3, facet arthrosis at L5-S1, and Grade 1 anterolisthesis of L4 on L5. No acute fracture was noted."
}}

🚨 CRITICAL EXPANDED TEXT LENGTH LIMITS:
- findings: MAX 2-3 sentences (40-60 words)
- recommendations: MAX 2-3 sentences  
- physical_exam: Bullet points only (5-8 bullets max)
- medications: Bullet points only (list only)
- rationale: MAX 2-3 sentences
- ALL OTHER FIELDS: MAX 2-3 sentences

EXPANDED WRITING RULES FOR "findings":
✅ CORRECT STYLE (concise, high-level summary):
"The MRI documented meniscal tear, moderate cartilage loss, and ACL degeneration. A large effusion with synovitis was noted."

❌ WRONG - Too verbose with excessive detail:
"The X-ray report documented diffuse disc degeneration at multiple levels, most pronounced at L2-3, along with bilateral facet arthrosis particularly at L5-S1. Additionally, Grade 1 anterolisthesis of L4 on L5 was noted secondary to facet arthrosis. Mild degenerative changes in the SI joints and marginal spurring from L3-L5 were also observed. The vertebral bodies appeared intact with no acute fracture identified."

✅ CORRECT - Concise summary without excessive anatomical detail:
"The report documented disc degeneration at L2-3, facet arthrosis at L5-S1, and Grade 1 anterolisthesis. No acute fracture was noted."

🔹 For imaging reports (MRI, CT, X-Ray):
- Group related findings into categories (e.g., "meniscal pathology", "cartilage changes", "ligament status")
- Use summary language, NOT detailed anatomical descriptions
- Skip negative findings unless clinically significant
- 2-3 sentences maximum

EXAMPLE - MRI Knee (CORRECT concise format):
{{
  "field": "findings",
  "collapsed": "Meniscal tear, cartilage loss, ligament changes, and effusion were documented",
  "expanded": "The MRI documented meniscal tear with post-meniscectomy changes, moderate medial compartment cartilage loss with osteophytes, and ACL degeneration. A large effusion with synovitis was noted."
}}

� CRITICAL: ALL FIELDS REQUIRE COMPLETE, MEANINGFUL SENTENCES
Every field (findings, recommendations, medications, physical_exam, mmi_status, work_status, etc.) MUST have:
- Grammatically complete sentences in BOTH collapsed AND expanded
- Clear subject (patient, report, document, etc.)
- Complete verb phrase
- Meaningful content that makes sense when read aloud

✅ COMPLETE SENTENCE CHECKLIST (APPLIES TO ALL FIELDS):
- Include proper subject: "The patient", "The report documented", "The MRI showed", etc.
- Include complete verb phrase: "documented", "was noted", "were described", etc.
- NEVER write fragments like "The at", "The from", "The with", "The and"
- NEVER omit the subject or verb
- Every sentence must be understandable without context
- Read it aloud - if it sounds incomplete or meaningless, rewrite it

EXAMPLE - findings (CORRECT):
{{
  "field": "findings",
  "collapsed": "Bilateral knee osteoarthritis and meniscal injuries were documented",
  "expanded": "The MRI documented bilateral knee osteoarthritis with moderate cartilage loss. Meniscal tears were noted in both knees."
}}

❌ WRONG - findings (INCOMPLETE):
{{
  "field": "findings",
  "collapsed": "Bilateral knee osteoarthritis and meniscal injuries",  ← INCOMPLETE! Missing verb
  "expanded": "The and meniscal injuries were documented."  ← MEANINGLESS! Missing subject
}}

EXAMPLE - recommendations (CORRECT):
{{
  "field": "recommendations",
  "collapsed": "MRI, injections, and pain management were recommended",
  "expanded": "The report recommended MRI of the knee, corticosteroid injections, and pain management referral. No surgical indication was noted."
}}

❌ WRONG - recommendations (INCOMPLETE):
{{
  "field": "recommendations",
  "collapsed": "MRI, injections, and pain management",  ← INCOMPLETE! Missing verb
  "expanded": "The recommended MRI and injections."  ← INCOMPLETE! Missing subject
}}

EXAMPLE - mmi_status (CORRECT):
{{
  "field": "mmi_status",
  "collapsed": "The patient is at maximum medical improvement (MMI)",
  "expanded": "The report indicated that the patient is at maximum medical improvement (MMI) due to ongoing symptoms and the need for further treatment."
}}

❌ WRONG - mmi_status (INCOMPLETE):
{{
  "field": "mmi_status",
  "collapsed": "The at MMI",  ← INCOMPLETE! Missing "patient is"
  "expanded": "The report indicated that the at maximum medical improvement..."  ← MEANINGLESS!
}}

EXAMPLE - work_status (CORRECT):
{{
  "field": "work_status",
  "collapsed": "The patient is off from work with restrictions",
  "expanded": "The report documented that the patient cannot return to work at this time. Work restrictions included no lifting, climbing, or kneeling."
}}

❌ WRONG - work_status (INCOMPLETE):
{{
  "field": "work_status",
  "collapsed": "The from work",  ← INCOMPLETE! Missing "patient is off"
  "expanded": "The report documented that the return to work at this time."  ← MEANINGLESS!
}}

EXAMPLE - physical_exam (CORRECT with bullets):
{{
  "field": "physical_exam",
  "collapsed": "Reduced strength and positive provocative tests were noted",
  "expanded": "• Reduced right shoulder abduction strength\n• Pain with supraspinatus testing\n• Positive Spurling's maneuver on left\n• Limited cervical rotation and extension"
}}

❌ WRONG - physical_exam (INCOMPLETE):
{{
  "field": "physical_exam",
  "collapsed": "Reduced strength and positive tests",  ← VAGUE! What tests?
  "expanded": "• Reduced right shoulder\n• Pain with testing\n• Positive maneuver"  ← INCOMPLETE! Each bullet must be complete
}}

❌ WRONG - User's verbose example to avoid:
"The MRI report documented truncation and tearing of the medial meniscus body, consistent with partial post-meniscectomy versus persistent tear. Moderate cartilage loss with osteophyte formation was noted in the medial compartment, and mild cartilage loss with osteophytes in the lateral compartment. Increased T2 signal in the anterior cruciate ligament was described, consistent with mucinous degeneration and/or partial tear, unchanged from prior imaging. A moderate to large effusion with synovitis and marked edema in the prepatellar and prepatellar tendon soft tissues were also documented..."

❌ WRONG - Multiple items with same field:
[
  {{"field": "findings", "collapsed": "Disc degeneration noted..."}},
  {{"field": "findings", "collapsed": "Facet arthrosis documented..."}},
  {{"field": "findings", "collapsed": "Anterolisthesis reported..."}}
]

OUTPUT STRUCTURE:
{format_instructions}

HEADER RULES:
- Title must reflect document type and body region
- Author must be name + credentials if present (no "Dr." prefix)
- Date must be in YYYY-MM-DD format. If not found, use empty string
- Disclaimer appears EXACTLY ONCE

Output valid JSON only.
""")

    user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE:
{doc_type}

ALLOWED UI FIELDS FOR THIS REPORT TYPE (use ONLY these field names):
{allowed_fields}

SOURCE DOCUMENT (External):
{raw_text}

REFERENCE CONTEXT:
{long_summary}

TASK:
Generate UI-ready fields following these STRICT rules:

🚨 CRITICAL: ONE ITEM PER FIELD TYPE
- Each field name (findings, recommendations, etc.) can appear AT MOST ONCE
- Consolidate ALL related content into ONE item per field type
- Use ONLY field names from the allowed list above
- ⚠️ IMPORTANT: If a field type has NO meaningful content in the source document, EXCLUDE it entirely from the output
- DO NOT create placeholder items with generic "No specific X were documented" messages
- Only include fields that have actual clinical content to report

FIELD CATEGORIZATION GUIDE:
- "findings" → All clinical observations, test results, abnormalities, imaging findings, diagnoses
- "recommendations" → Treatment plans, follow-up instructions, referrals (ALWAYS include authorization status: approved/denied/authorized)
- "medications" → All drugs referenced with dosages if available
- "physical_exam" → Physical examination findings
- "mmi_status" → MMI determination (med-legal reports only)
- "work_status" → Work restrictions/capacity (med-legal reports only)
- "rationale" → Clinical reasoning for decisions (especially UR approvals/denials)

🚨 PRESERVE CRITICAL DECISION TERMS:
When the source document contains terms like:
- "authorized", "approved", "denied", "rejected", "deferred"
- "recommended", "not recommended", "contraindicated"
- "granted", "supported", "not supported"
- "at MMI", "not at MMI"

YOU MUST include these terms in your output. DO NOT omit or soften them.

FORMAT FOR EACH ITEM:
- collapsed = One-line summary (include ALL key points for that field type)
- expanded = Format depends on field type:

🔹 FOR physical_exam AND medications → USE BULLET POINTS:
  • One item per line
  • Short, scannable entries
  • Include dosages for medications
  • NO paragraph prose
  
🔹 FOR findings, recommendations, rationale, mmi_status, work_status → USE CONCISE PARAGRAPH:
  • Maximum 1-3 lines (50-100 words)
  • High-signal, key points only
  • NO exhaustive detail or repetition
  • Attribution language required
  • Prioritize: diagnosis, key abnormalities, actionable items
  • ALWAYS preserve decision terms (approved/denied/authorized)

EXAMPLE - recommendations with AUTHORIZATION STATUS (CORRECT):
{{
  "field": "recommendations",
  "collapsed": "Physical therapy was authorized and surgery was denied",
  "expanded": "The report authorized physical therapy twice weekly for six weeks. Surgical intervention was denied as not medically necessary at this time."
}}

❌ WRONG - Omitting Decision Status:
{{
  "field": "recommendations",
  "collapsed": "Physical therapy and surgery were discussed",  ← WRONG! Missing approved/denied status
  "expanded": "The report mentioned physical therapy and surgical options."  ← VAGUE! Missing authorization decisions
}}

EXAMPLE - rationale for UR DENIAL (CORRECT):
{{
  "field": "rationale",
  "collapsed": "The request was denied due to lack of conservative treatment",
  "expanded": "The report denied the MRI request as the patient has not completed six weeks of conservative treatment including physical therapy and medication management."
}}

EXAMPLE - findings (CORRECT CONCISE FORMAT):
{{
  "field": "findings",
  "collapsed": "Right foot pain and diminished sensation were documented",
  "expanded": "The consultation documented persistent right foot pain following a work-related injury, with diminished sensation in the superficial peroneal nerve distribution. MRI and X-rays showed no structural pathology."
}}

EXAMPLE - recommendations (CORRECT CONCISE FORMAT):
{{
  "field": "recommendations",
  "collapsed": "Electrodiagnostics and pain management were recommended",
  "expanded": "The report recommended electrodiagnostic studies and pain management referral. No surgical indication was noted."
}}

❌ WRONG - Too verbose findings:
"The consultation report documented diffuse right dorsal and plantar foot pain persisting since a work-related dog attack on 2024-04-18. Occasional radiation of pain down the leg and into the foot was noted. Subjectively diminished sensation in the superficial peroneal nerve distribution over the dorsum of the right foot was described, specifically involving the medial and intermediate cutaneous nerves. Imaging findings included an unremarkable MRI..."

✅ CORRECT - Concise findings:
"The report documented persistent right foot pain with diminished sensation in the peroneal nerve distribution. Imaging showed no structural pathology."

EXAMPLE - physical_exam (CORRECT BULLET FORMAT):
{{
  "field": "physical_exam",
  "collapsed": "Muscle spasm and reduced motion were noted",
  "expanded": "• Cervical spine tightness and spasm\n• Reduced rotation and extension\n• Positive Spurling's maneuver (left arm)\n• Limited lumbar flexion due to pain\n• Positive straight leg raise (left)\n• Reduced sensation L5 dermatome"
}}

EXAMPLE - medications (CORRECT BULLET FORMAT):
{{
  "field": "medications",
  "collapsed": "Gabapentin and NSAIDs were referenced",
  "expanded": "• Gabapentin 300 mg three times daily\n• Pregabalin (potential transition)\n• NSAIDs (stomach discomfort noted)\n• Muscle relaxants for sleep"
}}

EXAMPLE FOR X-RAY (allowed: findings):

CORRECT OUTPUT (CONCISE):
{{
  "items": [
    {{
      "field": "findings",
      "collapsed": "Degenerative changes were documented",
      "expanded": "The X-ray documented disc degeneration at L2-3, facet arthrosis at L5-S1, and Grade 1 anterolisthesis. No acute fracture was noted."
    }}
  ]
}}

WRONG OUTPUT (duplicates):
{{
  "items": [
    {{"field": "findings", "collapsed": "Disc degeneration noted..."}},
    {{"field": "findings", "collapsed": "Facet arthrosis documented..."}},
    {{"field": "findings", "collapsed": "No fracture observed..."}}
  ]
}}

ATTRIBUTION REQUIREMENTS:
✅ Use past tense exclusively
✅ Include attribution language in every statement
✅ Start expanded text with "The [document type]..." pattern

Output JSON only.
""")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    try:
        # Truncate raw_text if too long (keep most relevant content)
        truncated_text = raw_text[:8000] if len(raw_text) > 8000 else raw_text
        truncated_long_summary = long_summary[:4000] if len(long_summary) > 4000 else long_summary
        
        chain = chat_prompt | llm
        response = chain.invoke({
            "raw_text": truncated_text,
            "long_summary": truncated_long_summary,
            "doc_type": doc_type,
            "allowed_fields": list(allowed_fields),
            "format_instructions": pydantic_parser.get_format_instructions()
        })
        
        # Extract JSON from response content
        response_content = response.content.strip()
        
        # Try to parse JSON - handle potential markdown code blocks
        if response_content.startswith("```"):
            # Remove markdown code blocks
            response_content = re.sub(r'^```(?:json)?\n?', '', response_content)
            response_content = re.sub(r'\n?```$', '', response_content)
        
        structured_summary = json.loads(response_content)

        # Hard safety checks (non-LLM)
        structured_summary = remove_patient_identifiers(structured_summary)
        structured_summary = ensure_header_fields(structured_summary, doc_type, raw_text)
        structured_summary = validate_ui_items(structured_summary)
        
        # HARD FILTER — drop disallowed fields (defensive)
        structured_summary = filter_disallowed_fields(structured_summary, allowed_fields)
        
        # DEDUPLICATE — ensure each field type appears only once
        structured_summary = deduplicate_fields(structured_summary)
        
        # FILTER EMPTY/GENERIC FIELDS — remove fields with no meaningful content
        structured_summary = filter_empty_or_generic_fields(structured_summary)
        
        # Post-process to ensure attribution compliance
        structured_summary = enforce_attribution_compliance(structured_summary)

        logger.info(f"✅ UI-ready summary generated with {len(structured_summary.get('summary', {}).get('items', []))} items")
        return structured_summary

    except json.JSONDecodeError as je:
        logger.error(f"❌ JSON parsing failed: {je}")
        logger.error(f"Response content: {response.content[:500] if 'response' in dir() else 'N/A'}")
        return create_fallback_structured_summary(doc_type)
        
    except Exception as e:
        logger.error(f"❌ Structured summary generation failed: {e}")
        return create_fallback_structured_summary(doc_type)


def enforce_attribution_compliance(structured_summary: dict) -> dict:
    """
    Post-LLM enforcement layer to catch any attribution violations.
    This is a defensive safety check.
    """
    forbidden_phrases = [
        'consistent with',
        'identified',
        'demonstrates',
        'confirms',
        'shows',
        'reveals',
        'suggests'
    ]
    
    # Check all text fields recursively
    def check_and_warn(text: str, location: str) -> None:
        if not text:
            return
        text_lower = text.lower()
        for phrase in forbidden_phrases:
            if phrase in text_lower:
                logger.warning(f"⚠️ Attribution violation detected in {location}: '{phrase}'")
                # In production, you might want to reject or auto-fix here
    
    # Check summary items
    if 'summary' in structured_summary and 'items' in structured_summary['summary']:
        for idx, item in enumerate(structured_summary['summary']['items']):
            check_and_warn(item.get('collapsed', ''), f"item[{idx}].collapsed")
            check_and_warn(item.get('expanded', ''), f"item[{idx}].expanded")
    
    return structured_summary




