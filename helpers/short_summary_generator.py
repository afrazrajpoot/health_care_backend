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
from config.settings import settings

logger = logging.getLogger("document_ai")


# ============== Report-Type Field Eligibility Matrix ==============

REPORT_FIELD_MATRIX: Dict[str, Dict] = {

    # -------------------------
    # Med-Legal Reports (QME family)
    # -------------------------
    "QME": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "physical_exam",
            "medications",
            "recommendations",
            "rationale",
            "mmi_status",
            "work_status"
        }
    },
    "AME": {"inherit": "QME"},
    "PQME": {"inherit": "QME"},
    "IME": {"inherit": "QME"},

    # -------------------------
    # Consult / Clinical Reports
    # -------------------------
    "CONSULT": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "physical_exam",
            "medications",
            "recommendations"
        }
    },
    "PAIN MANAGEMENT": {"inherit": "CONSULT"},
    "PROGRESS NOTE": {"inherit": "CONSULT"},
    "OFFICE VISIT": {"inherit": "CONSULT"},
    "CLINIC NOTE": {"inherit": "CONSULT"},

    # -------------------------
    # Imaging Reports
    # -------------------------
    "MRI": {
        "allowed": {"findings"}
    },
    "CT": {"inherit": "MRI"},
    "X-RAY": {"inherit": "MRI"},
    "XRAY": {"inherit": "MRI"},
    "ULTRASOUND": {"inherit": "MRI"},
    "EMG": {"inherit": "MRI"},
    "PET SCAN": {"inherit": "MRI"},
    "BONE SCAN": {"inherit": "MRI"},
    "DEXA SCAN": {"inherit": "MRI"},

    # -------------------------
    # Utilization Review
    # -------------------------
    "UR": {
        "allowed": {"recommendations", "rationale"}
    },
    "IMR": {"inherit": "UR"},
    "PEER REVIEW": {"inherit": "UR"},

    # -------------------------
    # Therapy Reports
    # -------------------------
    "PHYSICAL THERAPY": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "recommendations"
        }
    },
    "THERAPY NOTE": {"inherit": "PHYSICAL THERAPY"},
    "OCCUPATIONAL THERAPY": {"inherit": "PHYSICAL THERAPY"},
    "CHIROPRACTIC": {"inherit": "PHYSICAL THERAPY"},

    # -------------------------
    # Surgical / Operative Reports
    # -------------------------
    "SURGERY REPORT": {
        "allowed": {"findings"}
    },
    "OPERATIVE NOTE": {"inherit": "SURGERY REPORT"},
    "POST-OP": {"inherit": "SURGERY REPORT"},

    # -------------------------
    # PR-2 Reports
    # -------------------------
    "PR-2": {
        "allowed": {
            "mechanism_of_injury",
            "findings",
            "physical_exam",
            "medications",
            "recommendations",
            "work_status"
        }
    },
    "PR2": {"inherit": "PR-2"},

    # -------------------------
    # Labs & Diagnostics
    # -------------------------
    "LABS": {
        "allowed": {"findings"}
    },
    "PATHOLOGY": {"inherit": "LABS"},

    # -------------------------
    # Legal / Administrative Reports
    # -------------------------
    "ATTORNEY QUESTIONS": {
        "allowed": {"questions", "mmi_status", "work_status"}
    },
    "ADJUSTER QUESTIONS": {"inherit": "ATTORNEY QUESTIONS"},
    "NURSE CASE MANAGER": {"inherit": "ATTORNEY QUESTIONS"},

    # -------------------------
    # Default fallback
    # -------------------------
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
    - Contains incomplete/malformed sentences detected by linguistic analysis
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
        
        # Check for malformed sentences using linguistic validation
        is_malformed = False
        for text, label in [(collapsed, "collapsed"), (expanded, "expanded")]:
            if is_malformed_sentence(text):
                logger.warning(f"🗑️ Removed field '{field}' with malformed {label} text: '{text[:60]}...'")
                is_malformed = True
                break
        
        if is_malformed:
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


def is_malformed_sentence(text: str) -> bool:
    """
    Generic linguistic validation to detect malformed/garbled sentences.
    Uses linguistic rules rather than hardcoded patterns.
    
    Checks:
    1. Sentences starting with "The" must have a valid subject (noun/noun phrase) after it
    2. Function words (prepositions, conjunctions, articles) shouldn't appear in invalid positions
    3. Sentences must have proper semantic content (not just function words)
    4. Detects truncated/garbled word fragments
    
    Returns:
        True if the sentence is malformed, False if it appears valid
    """
    if not text:
        return True
    
    # Clean the text for analysis
    text = text.strip()
    
    # Skip bullet points for analysis (check the actual content)
    if text.startswith('•'):
        text = text[1:].strip()
    
    # If empty after cleaning, it's malformed
    if not text or len(text) < 5:
        return True
    
    words = text.split()
    if len(words) < 2:
        return True
    
    # Define word categories for linguistic analysis
    # Function words that shouldn't directly follow "The" without a noun
    function_words = {
        'a', 'an', 'the',  # articles
        'at', 'to', 'from', 'in', 'on', 'for', 'with', 'by', 'as', 'of',  # prepositions
        'and', 'or', 'but', 'yet', 'so', 'nor',  # conjunctions
        'is', 'are', 'was', 'were', 'be', 'been', 'being',  # be-verbs (alone is odd)
        'not', 'no', 'never',  # negations (shouldn't directly follow "The")
    }
    
    # Words that ARE valid after "The" (nouns, adjectives that modify nouns)
    valid_after_the = {
        'patient', 'report', 'document', 'physician', 'doctor', 'provider',
        'examination', 'assessment', 'evaluation', 'findings', 'results',
        'diagnosis', 'treatment', 'recommendation', 'medication', 'condition',
        'injury', 'pain', 'symptoms', 'history', 'following', 'above', 'below',
        'cervical', 'lumbar', 'thoracic', 'chronic', 'acute', 'bilateral',
        'left', 'right', 'upper', 'lower', 'medical', 'clinical', 'physical',
        'requested', 'recommended', 'documented', 'noted', 'referenced',
        'multiple', 'various', 'several', 'specific', 'primary', 'secondary',
        'temporary', 'permanent', 'total', 'partial',  # These are valid as adjectives before nouns
    }
    
    # Check 1: "The" followed by function word without valid noun structure
    first_word = words[0].lower().rstrip('.,;:')
    if first_word == 'the' and len(words) >= 2:
        second_word = words[1].lower().rstrip('.,;:')
        
        # If second word is a function word, it's likely malformed
        # Exception: some function words can be valid (e.g., "The following")
        if second_word in function_words and second_word not in valid_after_the:
            # Check if there's a valid noun soon after
            has_valid_subject = False
            for i, word in enumerate(words[2:6], start=2):  # Check next few words
                clean_word = word.lower().rstrip('.,;:')
                # Check if it's a content word (not function word) and reasonably long
                if clean_word not in function_words and len(clean_word) > 2:
                    # Check if it looks like a noun (simple heuristic)
                    if not clean_word.endswith(('ly', 'ing')) or clean_word in valid_after_the:
                        has_valid_subject = True
                        break
            
            if not has_valid_subject:
                return True  # Malformed: "The [function word] ..." without valid subject
    
    # Check 2: Detect garbled/truncated words (nonsense fragments)
    nonsense_indicators = 0
    for word in words:
        clean_word = word.lower().rstrip('.,;:!?')
        # Very short words in odd positions (not common function words)
        if len(clean_word) <= 2 and clean_word not in {'a', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}:
            nonsense_indicators += 1
        # Garbled word fragments (uncommon letter combinations)
        if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', clean_word):  # 4+ consonants in a row
            nonsense_indicators += 2
        # Words ending in unusual fragments
        if re.search(r'(aser|asers|ment$|tion$)s{2,}', clean_word):  # doubled endings
            nonsense_indicators += 2
    
    # If too many nonsense indicators relative to sentence length
    if nonsense_indicators > len(words) * 0.3:  # More than 30% nonsense
        return True
    
    # Check 3: Sentence must have at least one verb or verb-like word
    verb_indicators = {'was', 'were', 'is', 'are', 'been', 'being', 'documented', 'noted', 
                       'referenced', 'reported', 'described', 'stated', 'indicated', 'showed',
                       'revealed', 'found', 'observed', 'recorded', 'included', 'recommended'}
    has_verb = any(word.lower().rstrip('.,;:') in verb_indicators or 
                   word.lower().rstrip('.,;:').endswith(('ed', 'ing')) 
                   for word in words)
    
    # Very short sentences without verbs are suspicious
    if len(words) < 5 and not has_verb:
        return True
    
    # Check 4: Detect repeated words (sign of generation error)
    word_list = [w.lower().rstrip('.,;:') for w in words]
    for i in range(len(word_list) - 1):
        if word_list[i] == word_list[i + 1] and word_list[i] not in {'very', 'much', 'so'}:
            return True  # Repeated word like "the the"
    
    # Check 5: Sentence shouldn't start with certain function words
    invalid_starters = {'as', 'at', 'for', 'from', 'in', 'on', 'to', 'with', 'by', 'and', 'or', 'but'}
    if first_word in invalid_starters and len(words) < 6:
        return True
    
    return False  # Sentence appears valid



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
  ❌ "The request for a consultation with a Pulmonologist was apwas documented asd

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

� CRITICAL: SIGNAL CONSOLIDATION (PREVENT REDUNDANCY)
When multiple related items come from the SAME SOURCE SENTENCE:
- CONSOLIDATE into ONE bullet with sub-items OR one comprehensive statement
- DO NOT create separate bullets that repeat the same source text
- Group related findings that share attribution

Example - BEFORE (redundant):
❌ "• Cervical disc degeneration was documented"
❌ "• Status post cervical spinal fusion was documented"  
❌ "• Cervical spondylosis was documented"
❌ "• Cervicalgia was documented"
(All from same sentence = cognitive overload)

Example - AFTER (consolidated):
✅ "• Cervical spine conditions documented include: disc degeneration, status post spinal fusion (C3-C7), spondylosis, and cervicalgia"

OR with sub-bullets:
✅ "• Multiple cervical spine conditions were documented:
  - Cervical disc degeneration
  - Status post cervical spinal fusion (C3-C7)
  - Other cervical spondylosis
  - Cervicalgia"

APPLY THIS TO:
- Multiple diagnoses from same assessment
- Related surgical history items
- Grouped medications from same prescription
- Related exam findings from same examination section

WHEN TO KEEP SEPARATE:
- Items from different source sections
- Findings with different clinical significance
- Items requiring different operational handling

EXAMPLE - findings (CONSOLIDATED BULLET FORMAT):
{{
  "field": "findings",
  "collapsed": "Multiple cervical spine conditions and chronic pain syndrome were documented",
  "expanded": "• Cervical spine conditions documented include: disc degeneration, status post spinal fusion (C3-C7), spondylosis, and cervicalgia\n• Chronic pain syndrome was documented\n• Low back pain severity of 8/10 was documented\n• Sciatica was noted"
}}

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
□ Related items from same source consolidated (no redundant bullets)

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
                                                           
🚫 AUTHORSHIP / JUDGMENT LANGUAGE — STRICTLY FORBIDDEN

Do NOT use any wording that implies judgment, causation, recommendation, assessment, certainty, necessity, or factual patient assertions.

Forbidden terms include (non-exhaustive, exact + variants):

Clinical judgment / opinion:
likely, unlikely, probable, probably, possible, possibly, appears to be, suggests, suggestive of, consistent with, indicative of, concerning for, supports the diagnosis of, points to, favors, rules out, cannot rule out

Causation / attribution:
due to, secondary to, caused by, resulting from, related to, associated with (unless explicitly quoted), attributable to, because of, stemming from, precipitated by

Recommendations / directives:
recommend, recommended, should, advised, advise, plan to, will continue, initiate, discontinue, increase, decrease, start, stop, manage with, treat with, follow up, continue therapy

Assessment / impression language:
assessment, impression, diagnosis is, final diagnosis, primary diagnosis, differential diagnosis, clinical picture, evaluation reveals, findings indicate

Certainty / validation:
confirms, confirmed, demonstrates, establishes, proves, verifies, shows that

Necessity / appropriateness:
necessary, required, appropriate, indicated, justified, warranted, medically necessary

Patient status assertions:
patient has, patient requires, patient needs, patient suffers from, patient is unable to

✅ REQUIRED SAFE NON-AUTHORING LANGUAGE (WHITELIST)

Use attribution-based, passive, or referential phrasing only:

“The patient reported…”
“The report documented…”
was documented, was reported, was referenced
per report, according to the report, as stated in the report
external documentation notes
authorization was requested
no decision documented

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

    # Retry mechanism for robust generation
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Truncate raw_text if too long (keep most relevant content)
            truncated_long_summary = long_summary[:4000] if len(long_summary) > 4000 else long_summary
            
            # Create a new LLM instance with explicit max_tokens to prevent truncation
            # This ensures complete sentences and avoids garbled output
            from langchain_openai import AzureChatOpenAI
            from config.settings import CONFIG
            
            summary_llm = AzureChatOpenAI(
                azure_deployment=CONFIG.get("azure_openai_deployment"),
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=9000,  # Explicit max_tokens to prevent truncation
                timeout=90,  # Longer timeout for complete generation
                request_timeout=90,

            )
            
            chain = chat_prompt | summary_llm
            response = chain.invoke({
                "raw_text": raw_text,
                "long_summary": truncated_long_summary,
                "doc_type": doc_type,
                "allowed_fields": list(allowed_fields),
                "format_instructions": pydantic_parser.get_format_instructions()
            })
            
            # Extract JSON from response content
            response_content = response.content.strip()
            
            # Robust JSON extraction: Find the first outer brace and last outer brace
            # This handles markdown blocks (```json ... ```) AND any conversational filler text
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                response_content = response_content[start_idx:end_idx+1]
            
            # FIX: Clean invalid control characters (common issue with LLM JSON)
            # Remove control characters (0-31) except newline, carriage return, and tab
            response_content = "".join(ch for ch in response_content if ch >= ' ' or ch in '\n\r\t')
            
            # Parse with strict=False to allow control chars like newlines in strings
            structured_summary = json.loads(response_content, strict=False)

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
            
            # Filter out incomplete/garbled sentences
            structured_summary = filter_empty_or_generic_fields(structured_summary)
            
            # NEW: Ensure all expanded sections use bullet format
            structured_summary = enforce_bullet_format_all_fields(structured_summary)
            
            # NEW: Consolidate redundant bullets from same source
            structured_summary = consolidate_redundant_bullets(structured_summary)
            
            # # NEW: Attach citations to summary items if citation feature is enabled
            # if settings.citation_enabled and raw_text:
            #     try:
            #         from services.citation_service import attach_citations_to_short_summary
                    
            #         # Estimate total pages from text length (~2500 chars per page for medical docs)
            #         estimated_pages = max(1, len(raw_text) // 2500 + 1)
                    
            #         structured_summary = attach_citations_to_short_summary(
            #             structured_summary, 
            #             raw_text,
            #             min_confidence=settings.citation_min_confidence,
            #             total_pages=estimated_pages
            #         )
            #         logger.info(f"✅ Citations attached to summary items (estimated {estimated_pages} pages)")
            #     except Exception as citation_error:
            #         logger.warning(f"⚠️ Citation attachment failed (non-critical): {citation_error}")
            #         # Continue without citations - feature is additive, not blocking

            logger.info(f"✅ UI-ready summary generated with {len(structured_summary.get('summary', {}).get('items', []))} items")
            return structured_summary

        except json.JSONDecodeError as je:
            logger.warning(f"⚠️ JSON parsing failed (attempt {attempt+1}/{max_retries}): {je}")
            if attempt == max_retries - 1:
                logger.error(f"Response content: {response.content[:2000] if 'response' in dir() else 'N/A'}")
        
        except Exception as e:
            logger.warning(f"⚠️ Structured summary generation failed (attempt {attempt+1}/{max_retries}): {e}")

    # Fallback if all retries failed
    logger.error("❌ All retries failed. Returning fallback summary.")
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


def consolidate_redundant_bullets(structured_summary: dict) -> dict:
    """
    Post-process to consolidate bullets that likely come from same source.
    Detects bullets with same verb patterns and combines them to reduce redundancy.
    
    This improves signal density by grouping related findings that share attribution,
    reducing cognitive load when scanning summaries.
    """
    if 'summary' not in structured_summary or 'items' not in structured_summary['summary']:
        return structured_summary
    
    items = structured_summary.get('summary', {}).get('items', [])
    
    for item in items:
        expanded = item.get('expanded', '')
        if not expanded or '•' not in expanded:
            continue
            
        bullets = [b.strip() for b in expanded.split('\n') if b.strip().startswith('•')]
        
        # Group bullets with identical attribution patterns
        # Example: "X was documented", "Y was documented", "Z was documented"
        attribution_groups = {}
        non_grouped_bullets = []
        
        for bullet in bullets:
            # Extract attribution pattern (e.g., "was documented", "was noted")
            grouped = False
            for key, patterns in [
                ('documented', [' was documented', ' were documented']),
                ('noted', [' was noted', ' were noted']),
                ('referenced', [' was referenced', ' were referenced']),
            ]:
                for pattern in patterns:
                    if pattern in bullet.lower():
                        if key not in attribution_groups:
                            attribution_groups[key] = []
                        attribution_groups[key].append(bullet)
                        grouped = True
                        break
                if grouped:
                    break
            
            if not grouped:
                non_grouped_bullets.append(bullet)
        
        # If we have 3+ bullets with same attribution, consolidate them
        consolidated_bullets = []
        processed_groups = set()
        
        for attr_key, group_bullets in attribution_groups.items():
            if len(group_bullets) >= 3:
                # Extract the subjects from each bullet
                subjects = []
                for bullet in group_bullets:
                    # Simple extraction: get text before "was documented/noted/referenced"
                    for pattern in [' was documented', ' were documented', ' was noted', ' were noted', ' was referenced', ' were referenced']:
                        if pattern in bullet.lower():
                            # Find pattern case-insensitively
                            idx = bullet.lower().find(pattern)
                            if idx > 0:
                                subject = bullet[2:idx].strip()  # Remove "• " prefix
                                if subject:
                                    subjects.append(subject)
                            break
                
                if len(subjects) >= 3:
                    # Create consolidated bullet
                    consolidated = f"• The following were {attr_key}: {', '.join(subjects)}"
                    consolidated_bullets.append(consolidated)
                    processed_groups.add(attr_key)
                    logger.info(f"✅ Consolidated {len(subjects)} '{attr_key}' bullets into one for {item.get('field')}")
                else:
                    # Not enough valid subjects, keep original bullets
                    consolidated_bullets.extend(group_bullets)
            else:
                # Less than 3 bullets with this attribution - keep separate
                consolidated_bullets.extend(group_bullets)
        
        # Add non-grouped bullets
        consolidated_bullets.extend(non_grouped_bullets)
        
        # Only update if we actually consolidated something
        if processed_groups:
            item['expanded'] = '\n'.join(consolidated_bullets)
    
    return structured_summary
