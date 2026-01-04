"""
Structured Short Summary Generator
Reusable helper for generating UI-ready, severity-classified medical summaries.
"""
import logging
import json
import re
from typing import Dict, List, Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("document_ai")


# ============== Pydantic Models for Structured Short Summary ==============

class Finding(BaseModel):
    """A single clinical finding with severity indicator"""
    label: str = Field(description="Brief label for the finding")
    value: str = Field(description="Description of the finding in past tense")
    indicator: Literal["danger", "warning", "normal"] = Field(
        description="Severity indicator: danger=urgent, warning=notable, normal=reassuring"
    )


class Recommendation(BaseModel):
    """A workflow recommendation (not treatment)"""
    label: str = Field(description="Brief label for the recommendation")
    value: str = Field(description="Workflow action in past tense")


class Status(BaseModel):
    """Contextual status or metadata"""
    label: str = Field(description="Brief label for the status")
    value: str = Field(description="Status value")


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
    """Content section of the structured summary"""
    findings: List[Finding] = Field(default_factory=list, description="Clinical findings with severity")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Workflow recommendations")
    status: List[Status] = Field(default_factory=list, description="Contextual status items")


class StructuredShortSummary(BaseModel):
    """Complete structured short summary for UI display"""
    header: SummaryHeader = Field(description="Header information")
    summary: SummaryContent = Field(description="Summary content with findings, recommendations, status")


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
            "findings": [],
            "recommendations": [],
            "status": []
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
    
    # Ensure summary section exists
    if "summary" not in structured_summary:
        structured_summary["summary"] = {
            "findings": [],
            "recommendations": [],
            "status": []
        }
    
    return structured_summary


def validate_indicators(structured_summary: dict) -> dict:
    """
    Validate and normalize severity indicators in findings.
    Ensures all indicators are one of: danger, warning, normal
    """
    valid_indicators = {"danger", "warning", "normal"}
    
    if "summary" in structured_summary and "findings" in structured_summary["summary"]:
        findings = structured_summary["summary"]["findings"]
        for finding in findings:
            if isinstance(finding, dict):
                indicator = finding.get("indicator", "").lower()
                if indicator not in valid_indicators:
                    # Default to warning for unknown indicators
                    finding["indicator"] = "warning"
    
    return structured_summary


def generate_structured_short_summary(llm: AzureChatOpenAI, raw_text: str, doc_type: str) -> dict:
    """
    Generate a structured, UI-ready summary from raw_text (Document AI summarizer output).
    Output is reference-only, past-tense, severity-classified, and EMR-safe.
    
    Args:
        llm: Azure OpenAI LLM instance
        raw_text: The Document AI summarizer output (primary context)
        doc_type: Document type
        
    Returns:
        dict: Structured summary with header, findings, recommendations, status
    """
    logger.info("🎯 Generating structured summary with Pydantic validation...")
    
    # Create Pydantic output parser for consistent response structure
    pydantic_parser = PydanticOutputParser(pydantic_object=StructuredShortSummary)

    system_prompt = SystemMessagePromptTemplate.from_template("""
You generate STRUCTURED, REFERENCE-ONLY medical summaries for workflow intelligence.

ABSOLUTE RULES — NO EXCEPTIONS:
1. Output MUST be valid JSON only. No prose, no markdown.
2. Use PAST TENSE only (e.g., "was noted", "were reported").
3. ALL content must be clearly REFERENCED from external documents.
4. DO NOT diagnose, prescribe, or issue medical decisions.
5. DO NOT include patient identifiers (name, DOB, MRN, phone, claim number).
6. DO NOT restate full report text — abstract only.
7. No dosages, exam measurements, or orders.
8. Severity is ATTENTION-BASED, not clinical judgment.

SEVERITY INDICATORS (required for each finding):
- "danger" → High-attention or potentially urgent finding
- "warning" → Chronic, degenerative, notable, or follow-up finding
- "normal" → Explicitly negative or reassuring finding

OUTPUT STRUCTURE (STRICT):
{format_instructions}

HEADER RULES:
- Title must reflect document type and body region.
- Author must be name + credentials if present (no "Dr." prefix).
- date must be in YYYY-MM-DD format. If not found, leave empty string.
- Disclaimer must appear EXACTLY ONCE and scope the entire summary.

FINDINGS RULES:
- Include ONLY clinically significant or explicitly stated findings.
- Use short, bullet-ready sentences in past tense.
- Avoid numbers unless explicitly meaningful (e.g., Grade 1).
- Each finding MUST have an indicator (danger/warning/normal).

RECOMMENDATIONS:
- WORKFLOW ONLY (review, follow-up, scheduling context).
- NEVER treatment or prescribing instructions.

STATUS:
- Metadata or contextual facts only (exam type, comparison availability).

FINAL CHECK:
- JSON only
- Past tense
- Reference-only
- No placeholders
- No patient identifiers
""")

    user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT CONTEXT (Primary Source):
{raw_text}

DOCUMENT TYPE: {doc_type}

Generate the structured JSON summary following ALL rules above. Output ONLY valid JSON.
""")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    try:
        # Truncate raw_text if too long (keep most relevant content)
        truncated_text = raw_text[:8000] if len(raw_text) > 8000 else raw_text
        
        chain = chat_prompt | llm
        response = chain.invoke({
            "raw_text": truncated_text,
            "doc_type": doc_type,
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
        structured_summary = validate_indicators(structured_summary)

        logger.info("✅ Structured summary generated successfully")
        return structured_summary

    except json.JSONDecodeError as je:
        logger.error(f"❌ JSON parsing failed: {je}")
        logger.error(f"Response content: {response.content[:500] if 'response' in dir() else 'N/A'}")
        return create_fallback_structured_summary(doc_type)
        
    except Exception as e:
        logger.error(f"❌ Structured summary generation failed: {e}")
        return create_fallback_structured_summary(doc_type)
