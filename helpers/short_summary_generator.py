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
        r"^the\s+(at|from|to|in|on|for|with|was|is|yet|temporary|permanent|off)\s+",  # "The at MMI", "The yet at", "The temporary", etc.
        r"^the\s+\w+\s+(at|from|to|in|on)\s+",  # "The patient at", "The report from"
        r"\b(patient|report|document)\s+(at|from|to)\s+(the|a)\s*$",  # Incomplete phrases
        r"^(at|from|to|in|on)\s+\w+\s*$",  # Just preposition + word
        r"\bthe\s+the\b",  # Duplicate "the"
        r"^the\s+(is|are|was|were)\s*$",  # Just "The is/are/was/were"
        r"^the\s+\w+\s+disability",  # "The temporary disability" without verb
        r"^the\s+(not\s+)?yet\s+at",  # "The yet at", "The not yet at"
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
    Output is reference-only, past-tense, and EMR-safe with STRICT non-authorship compliance.
    
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
You do NOT infer causation.
You do NOT provide medical advice or recommendations.

🔴 CORE NON-AUTHORSHIP PRINCIPLE:
Every statement must pass this test: "This is a neutral restatement of what the external document EXPLICITLY said, without interpretation, causation, endorsement, or system judgment."

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

🚨 NON-AUTHORSHIP VIOLATION RULES - AUTOMATIC REJECTION:

❌ FORBIDDEN PATTERN #1: INFERRED CAUSATION
NEVER write: "likely due to", "caused by", "secondary to", "resulted from", "because of"
✅ ALLOWED: "X was referenced", "Y was documented", "Z was noted in the report"

Example Violations:
❌ "Side pain was likely due to muscle spasm secondary to cold weather exposure"
✅ "Side pain for approximately two weeks was documented. Muscle spasm was referenced. Cold weather exposure was referenced."

❌ FORBIDDEN PATTERN #2: DIRECTIVE/INSTRUCTIONAL LANGUAGE
NEVER write: "recommended to", "should", "advised to", "instructed to", "told to"
✅ ALLOWED: "was referenced in the report", "was documented", "measures were described"

Example Violations:
❌ "Conservative management was recommended, including keeping the affected area warm"
✅ "Conservative management measures were referenced in the report. Use of warmth for symptom relief was described."

❌ FORBIDDEN PATTERN #3: COLLAPSED CAUSATION OR REASONING
NEVER combine recommendations with their reasons or outcomes with their causes
✅ REQUIRED: Separate "what was done" from "why it was done"

Example Violations:
❌ "Medication authorization remains pending due to muscle spasm"
✅ FINDINGS: "Muscle spasm was documented"
✅ OPERATIONAL: "Medication authorization was documented as pending"

❌ FORBIDDEN PATTERN #4: HISTORY AS CLINICAL FACT
NEVER state patient-reported information as verified fact without attribution
✅ REQUIRED: "The patient reported...", "was documented as patient-reported", "Use of [X] was documented"

Example Violations:
❌ "The patient uses a cane for mobilization and is on long-term opioid therapy"
✅ "Use of a cane for ambulation was documented. Long-term opioid therapy was documented."
OR: "The patient reported using a cane for ambulation. The patient reported long-term opioid use."

❌ FORBIDDEN PATTERN #5: CLINICAL ENDORSEMENT OR JUDGMENT
NEVER write: "is appropriate", "confirms", "demonstrates", "shows", "reveals", "suggests", "indicates X is necessary"
✅ ALLOWED: "was documented", "was described", "was noted", "was referenced"

Example Violations:
❌ "Long-term opioid therapy is appropriate for this patient"
✅ "Long-term opioid therapy was documented"

❌ FORBIDDEN PATTERN #6: MIXING HISTORY WITH FINDINGS
NEVER blend patient-reported history with clinical findings in the same field
✅ REQUIRED: Separate sections for "Reported History" vs "Findings"

Example Violations:
❌ FINDINGS: "Low back pain severity of 8, side pain for two weeks, muscle spasm, chronic pain syndrome"
✅ REPORTED HISTORY: "Low back pain severity of 8 was documented. Side pain for approximately two weeks was reported."
✅ FINDINGS: "Chronic pain syndrome was documented. Muscle spasm was noted."

🚨 CRITICAL FIELD RULES (HIGHEST PRIORITY):
1. EACH FIELD TYPE CAN ONLY APPEAR ONCE in the output
2. If multiple items belong to the same field type, CONSOLIDATE them into ONE item
3. ONLY use field names from the allowed list provided
4. INCLUDE EVERY FIELD THAT HAS ANY CONTENT - even if only one piece of information
5. DO NOT omit fields just because they have minimal content
6. Use the CORRECT field type for the content:
   - "reported_history" = Patient-reported symptoms, timeline, functional aids, patient-stated medication use
   - "findings" = Clinical observations, test results, abnormalities, diagnoses (verified/documented by provider)
   - "recommendations" = Treatment plans, follow-up, referrals (NEVER include reasons - just what was referenced)
   - "reported_reasons" = Clinical reasoning or rationale AS STATED in document (separate from recommendations)
   - "medications" = Drugs prescribed or referenced
   - "physical_exam" = Physical examination findings
   - "vital_signs" = Vital sign measurements
   - "operational_context" = Administrative items (RFA status, appointment dates, pending actions)
   - "mmi_status" = Maximum Medical Improvement status
   - "work_status" = Work restrictions or capacity

🔴 CRITICAL STRUCTURAL SEPARATION REQUIREMENTS:

REPORTED HISTORY (patient-reported, not verified):
- Symptoms as described by patient
- Duration of complaints
- Functional aids (cane, walker)
- Patient-stated medication use
- ALWAYS include caveat: "as documented in external report" or "patient-reported"

FINDINGS (provider-documented observations):
- Diagnoses
- Clinical observations
- Test results
- Imaging findings
- Physical exam findings
- NEVER mix with patient history

RECOMMENDATIONS (referenced actions only):
- Treatment plans referenced
- Follow-up instructions documented
- Referrals mentioned
- Authorization requests noted
- NEVER include reasons or causation
- ALWAYS preserve decision status (approved/denied/authorized/pending)

REPORTED REASONS (separate from recommendations):
- Clinical rationale AS STATED in document
- Contributing factors AS DOCUMENTED
- NEVER infer or interpret
- Always attribute: "was referenced", "was stated", "was described"

OPERATIONAL CONTEXT (administrative tracking):
- RFA status
- Appointment dates
- Pending actions
- Document review status

MANDATORY LANGUAGE RULES (NO EXCEPTIONS):
✅ ALLOWED VERBS ONLY:
- documented, described, referenced, reported, noted, stated, listed, mentioned

❌ FORBIDDEN VERBS (cause authorship leakage):
- identified, consistent with, demonstrates, confirms, shows, reveals, suggests, indicates, caused, resulted, led to, due to

✅ ATTRIBUTION PATTERNS:
- "The [document type] documented..."
- "The report described..."
- "[Condition] was noted in the report..."
- "As documented in the [document type]..."
- "The patient reported..." (for history only)
- "was referenced in the report"
- "was described as..."

❌ FORBIDDEN ATTRIBUTION PATTERNS:
- "The patient has..." (implies verified fact)
- "The patient requires..." (implies clinical judgment)
- "was recommended to..." (directive)
- "likely caused by..." (causation)
- "should..." (advice)

🚨 COMPLETE SENTENCE REQUIREMENTS (APPLIES TO BOTH COLLAPSED AND EXPANDED):
- EVERY sentence MUST be grammatically complete and meaningful
- NEVER write incomplete fragments like "The at MMI", "The yet at", "The temporary disability"
- ALWAYS include the subject (patient, report, document) AND complete verb phrase
- Collapsed text is a SENTENCE, not a phrase - it needs subject + verb + object

Examples of COMPLETE collapsed sentences:
  ✅ "The patient reported not yet being at maximum medical improvement (MMI)"
  ✅ "Temporary total disability (TTD) status was documented"
  ✅ "The report documented that the patient is off work"
  ✅ "Inability to return to work at this time was noted"
  
Examples of INCOMPLETE collapsed sentences (NEVER DO THIS):
  ❌ "The yet at Maximum Medical Improvement (MMI)" (MISSING "patient is not")
  ❌ "The temporary total disability (TTD)" (MISSING "was documented")
  ❌ "The at MMI" (MISSING "patient is")
  ❌ "The from work" (MEANINGLESS fragment)
  ❌ "The return to work at this time" (MISSING subject and verb)

TENSE & VOICE:
- Past tense only (was documented, were noted, was described, was referenced)
- Never present tense declarations
- Attribution must be clear in EVERY statement
- NEVER use evaluative language

PRIVACY:
- No patient identifiers (name, DOB, MRN, phone, claim number)
- Dates in YYYY-MM-DD format only

🚨 CRITICAL: ALL EXPANDED SECTIONS USE BULLET-POINT FORMAT
- ALL fields (not just physical_exam and medications) use bullet points in expanded view
- collapsed = One-line summary with attribution
- expanded = Simple bullet points, one item per line
- Each bullet is a short, scannable statement
- NO paragraph prose in any expanded section
- Maximum 8-10 bullets per field

EXAMPLE - reported_history (BULLET FORMAT):
{{
  "field": "reported_history",
  "collapsed": "Chronic low back pain and side pain were documented as patient-reported",
  "expanded": "• The patient reported chronic low back pain\n• Side pain for approximately two weeks was reported\n• Use of a cane for ambulation was documented\n• Long-term opioid therapy was documented\n• Note: History is reported as documented and does not represent verified clinical fact"
}}

EXAMPLE - findings (BULLET FORMAT):
{{
  "field": "findings",
  "collapsed": "Chronic pain syndrome, lumbar fusion, and osteoarthritis were documented",
  "expanded": "• Chronic pain syndrome was documented\n• Lumbar fusion was documented\n• Osteoarthritis was documented\n• Sciatica was documented\n• Low back pain severity of 8 was documented"
}}

EXAMPLE - recommendations (BULLET FORMAT, NO REASONS):
{{
  "field": "recommendations",
  "collapsed": "Conservative management measures were referenced in the report",
  "expanded": "• Conservative management measures were referenced in the report\n• Medication authorization requests were referenced as pending"
}}

EXAMPLE - reported_reasons (BULLET FORMAT):
{{
  "field": "reported_reasons",
  "collapsed": "Muscle spasm and cold weather exposure were referenced",
  "expanded": "• Muscle spasm was referenced in the report\n• Cold weather exposure was referenced as a contributing factor\n• Note: Reasons are reported as stated in the source document and are not interpreted or validated"
}}

EXAMPLE - operational_context (BULLET FORMAT):
{{
  "field": "operational_context",
  "collapsed": "Medication RFA pending and follow-up appointment documented",
  "expanded": "• Request for Authorization (RFA) for medications was documented as pending\n• Medications include: Celecoxib, Diclofenac, Hydrocodone, Lidocaine patches, Pregabalin\n• Follow-up appointment dated 2025-12-23 was documented"
}}

EXAMPLE - physical_exam (BULLET FORMAT):
{{
  "field": "physical_exam",
  "collapsed": "Limited range of motion and tenderness were documented",
  "expanded": "• Lumbar flexion limited to 45 degrees was noted\n• Extension limited to 15 degrees was documented\n• Tenderness to palpation at L4-L5 was noted\n• Negative straight leg raise test was documented"
}}

EXAMPLE - medications (BULLET FORMAT):
{{
  "field": "medications",
  "collapsed": "Multiple pain management medications were documented",
  "expanded": "• Hydrocodone 10mg, twice daily\n• Pregabalin 150mg, three times daily\n• Celecoxib 200mg, once daily\n• Lidocaine patches 5%, as needed\n• Diclofenac gel 1%, topical application"
}}

🚨 CRITICAL: CONTENT INCLUSION REQUIREMENTS
- INCLUDE EVERY FIELD THAT HAS ANY VALID CONTENT
- Even if a field has only ONE piece of information, it MUST be included
- DO NOT omit fields because they seem "too short" or "minimal"
- If the source document mentions something relevant to a field, that field MUST appear
- Empty or truly contentless fields (with no information at all) can be omitted
- But ANY field with actual data MUST be present in the output

OUTPUT STRUCTURE:
{format_instructions}

HEADER RULES:
- Title must reflect document type and body region
- Author must be name + credentials if present (no "Dr." prefix)
- Date must be in YYYY-MM-DD format. If not found, use empty string
- Disclaimer appears EXACTLY ONCE

🔒 NON-AUTHORSHIP COMPLIANCE CHECKLIST (verify before output):
□ No causal language ("due to", "caused by", "likely", "secondary to")
□ No directive language ("recommended to", "should", "advised")
□ No collapsed reasoning (recommendations separated from reasons)
□ History separated from findings
□ All statements attributable to source document
□ No clinical judgments or endorsements
□ Past tense attribution language throughout
□ Decision terms preserved exactly as stated
□ Patient-reported info clearly marked
□ ALL expanded sections use bullet-point format
□ ALL fields with any content are included

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
Generate UI-ready fields following STRICT NON-AUTHORSHIP rules:

🚨 CRITICAL: ONE ITEM PER FIELD TYPE
- Each field name can appear AT MOST ONCE
- Consolidate ALL related content into ONE item per field type
- Use ONLY field names from the allowed list above
- ⚠️ INCLUDE ALL FIELDS THAT HAVE ANY CONTENT - even single items
- DO NOT omit fields just because they have minimal information
- Only exclude fields that are TRULY EMPTY (no content at all)

🚨 CRITICAL: ALL EXPANDED SECTIONS USE BULLET POINTS
- Every field's expanded view must be in bullet-point format
- Use "• " prefix for each bullet point
- Separate bullets with "\n"
- NO paragraph prose allowed in expanded sections
- Keep bullets concise and scannable
- Maximum 8-10 bullets per field

🔴 NON-AUTHORSHIP COMPLIANCE REQUIREMENTS:

1. SEPARATE HISTORY FROM FINDINGS:
   - "reported_history" = patient-reported information
   - "findings" = provider-documented observations
   - NEVER mix these

2. SEPARATE RECOMMENDATIONS FROM REASONS:
   - "recommendations" = actions referenced (NO reasons)
   - "reported_reasons" = rationale as stated (separate field)
   - NEVER collapse these together

3. USE ONLY NON-AUTHORSHIP VERBS:
   ✅ documented, described, referenced, reported, noted, stated
   ❌ caused, due to, likely, resulted, shows, confirms, suggests

4. PRESERVE DECISION TERMS EXACTLY:
   - authorized, approved, denied, deferred
   - recommended, not recommended
   - at MMI, not at MMI
   - DO NOT soften or omit these

5. ATTRIBUTION IN EVERY STATEMENT:
   - "The report documented..."
   - "was noted in the report"
   - "The patient reported..."
   - NEVER make unattributed declarations

FIELD CATEGORIZATION GUIDE:
- "reported_history" → Patient-reported symptoms, timeline, functional aids, patient-stated meds
- "findings" → Clinical observations, diagnoses, test results, imaging (provider-documented)
- "recommendations" → Treatment plans, referrals (NO reasons - just what was referenced)
- "reported_reasons" → Clinical rationale AS STATED (separate from recommendations)
- "medications" → All drugs with dosages
- "physical_exam" → Exam findings
- "operational_context" → RFA status, appointments, pending actions
- "mmi_status" → MMI determination
- "work_status" → Work restrictions

FORMAT FOR EACH ITEM:
- collapsed = One-line summary with attribution (complete sentence)
- expanded = BULLET POINTS ONLY (for ALL fields)
  • One item per line
  • Use "• " prefix
  • Separate with "\n"
  • Short, scannable statements
  • Include dosages for medications
  • Maximum 8-10 bullets

EXAMPLE OUTPUT (COMPLIANT - ALL BULLETS):
{{
  "items": [
    {{
      "field": "reported_history",
      "collapsed": "Chronic low back pain and side pain were documented as patient-reported",
      "expanded": "• The patient reported chronic low back pain\n• Side pain for approximately two weeks was reported\n• Use of a cane for ambulation was documented\n• Long-term opioid therapy was documented"
    }},
    {{
      "field": "findings",
      "collapsed": "Chronic pain syndrome and lumbar fusion were documented",
      "expanded": "• Chronic pain syndrome was documented\n• Lumbar fusion was documented\n• Osteoarthritis was documented\n• Sciatica was documented"
    }},
    {{
      "field": "recommendations",
      "collapsed": "Conservative management measures were referenced",
      "expanded": "• Conservative management measures were referenced in the report\n• Medication authorization requests were referenced as pending"
    }},
    {{
      "field": "reported_reasons",
      "collapsed": "Muscle spasm and cold weather exposure were referenced",
      "expanded": "• Muscle spasm was referenced in the report\n• Cold weather exposure was referenced as a contributing factor"
    }},
    {{
      "field": "operational_context",
      "collapsed": "Medication RFA pending and follow-up appointment documented",
      "expanded": "• Request for Authorization (RFA) for medications was documented as pending\n• Follow-up appointment dated 2025-12-23 was documented"
    }}
  ]
}}

🚨 CONTENT INCLUSION REQUIREMENT:
- If a field has ANY information (even one sentence worth), it MUST be included
- Do not skip fields because they seem "too short"
- Better to include all available data than to omit potentially important information
- Example: If there's only one finding, still include the "findings" field
- Example: If there's only one recommendation, still include the "recommendations" field

🚨 REJECTION TEST - Automatically fail these patterns:
❌ "likely due to muscle spasm" → REJECT (inferred causation)
❌ "recommended keeping area warm" → REJECT (directive language)
❌ "uses a cane" → REJECT (history as fact, no attribution)
❌ "pending due to spasm" → REJECT (collapsed reasoning)
❌ "is appropriate" → REJECT (endorsement)
❌ Paragraph text in expanded → REJECT (must be bullets)
❌ Omitting a field that has content → REJECT (must include all data)

✅ ACCEPTANCE TEST - Pass these patterns:
✅ "Muscle spasm was documented"
✅ "Conservative measures were referenced"
✅ "Use of cane was documented"
✅ "was referenced as pending"
✅ "was documented"
✅ All expanded sections use "• " bullet format
✅ All fields with any content are present

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
        
        # MODIFIED: More lenient filtering - only remove truly empty fields
        structured_summary = filter_truly_empty_fields(structured_summary)
        
        # Post-process to ensure attribution compliance
        structured_summary = enforce_attribution_compliance(structured_summary)
        
        # NEW: Enforce non-authorship compliance
        structured_summary = enforce_non_authorship_compliance(structured_summary)
        
        # NEW: Ensure all expanded sections use bullet format
        structured_summary = enforce_bullet_format_all_fields(structured_summary)

        logger.info(f"✅ UI-ready summary generated with {len(structured_summary.get('summary', {}).get('items', []))} items")
        return structured_summary

    except json.JSONDecodeError as je:
        logger.error(f"❌ JSON parsing failed: {je}")
        logger.error(f"Response content: {response.content[:500] if 'response' in dir() else 'N/A'}")
        return create_fallback_structured_summary(doc_type)
        
    except Exception as e:
        logger.error(f"❌ Structured summary generation failed: {e}")
        return create_fallback_structured_summary(doc_type)


def filter_truly_empty_fields(structured_summary: dict) -> dict:
    """
    Filter out only fields that are TRULY empty or have no meaningful content.
    More lenient than previous implementation - keeps fields with any valid content.
    """
    if 'summary' not in structured_summary or 'items' not in structured_summary['summary']:
        return structured_summary
    
    filtered_items = []
    for item in structured_summary['summary']['items']:
        collapsed = item.get('collapsed', '').strip()
        expanded = item.get('expanded', '').strip()
        
        # Only exclude if BOTH collapsed and expanded are empty or just whitespace
        if collapsed or expanded:
            # Has some content - keep it
            filtered_items.append(item)
        else:
            logger.info(f"⚠️ Filtering truly empty field: {item.get('field')}")
    
    structured_summary['summary']['items'] = filtered_items
    return structured_summary


def enforce_bullet_format_all_fields(structured_summary: dict) -> dict:
    """
    Ensure all expanded sections use bullet-point format.
    Converts any paragraph text to bullet points.
    """
    if 'summary' not in structured_summary or 'items' not in structured_summary['summary']:
        return structured_summary
    
    for item in structured_summary['summary']['items']:
        expanded = item.get('expanded', '').strip()
        
        # Skip if already empty
        if not expanded:
            continue
        
        # Check if already in bullet format
        if expanded.startswith('•') or '\n•' in expanded:
            # Already has bullets, just ensure consistency
            continue
        
        # Convert paragraph text to bullets
        # Split by periods or newlines, create bullets
        sentences = [s.strip() for s in expanded.replace('\n', '. ').split('. ') if s.strip()]
        
        if sentences:
            bullet_text = '\n'.join([f"• {s}" if not s.endswith('.') else f"• {s}" for s in sentences])
            item['expanded'] = bullet_text
            logger.info(f"✅ Converted {item.get('field')} to bullet format")
    
    return structured_summary


def enforce_non_authorship_compliance(structured_summary: dict) -> dict:
    """
    Post-processing function to enforce strict non-authorship compliance.
    Scans for forbidden patterns and either fixes or rejects them.
    """
    forbidden_phrases = [
        "likely due to", "caused by", "secondary to", "resulted from", "because of",
        "recommended to", "should", "advised to", "instructed to",
        "is appropriate", "confirms", "demonstrates", "shows", "reveals", "suggests",
        "uses a", "requires a", "has a", "needs a"  # unattributed facts
    ]
    
    items = structured_summary.get('summary', {}).get('items', [])
    compliant_items = []
    
    for item in items:
        collapsed = item.get('collapsed', '').lower()
        expanded = item.get('expanded', '').lower()
        
        # Check for forbidden patterns
        has_violation = False
        for phrase in forbidden_phrases:
            if phrase in collapsed or phrase in expanded:
                logger.warning(f"⚠️ Non-authorship violation detected in {item.get('field')}: '{phrase}'")
                has_violation = True
                break
        
        # Only include items without violations
        if not has_violation:
            compliant_items.append(item)
    
    structured_summary['summary']['items'] = compliant_items
    return structured_summary

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




