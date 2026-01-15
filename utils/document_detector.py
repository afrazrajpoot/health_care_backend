# document_detector_enhanced.py
"""
Enhanced context-aware document type detector with improved fax cover sheet handling.
Uses GPT-4o + LangChain OutputParser (Pydantic) with multi-stage analysis.
"""

import logging
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from config.settings import CONFIG

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --------------------------
# 1. Define the output schema
# --------------------------

class DocumentTypeOut(BaseModel):
    doc_type: str = Field(
        description="The main document type inferred from the text. "
                    "Must be one of the predefined types (RFA, PR2, DFR, QME, IMAGING, CONSULT, UR, etc.), "
                    "OR if none match, use the actual title/heading from the document."
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Short reason why this document type was chosen."
    )
    is_fax_cover: bool = Field(
        description="True if the detected content appears to be from a fax cover sheet rather than the main document.",
        default=False
    )

parser = PydanticOutputParser(pydantic_object=DocumentTypeOut)

# --------------------------
# 2. Enhanced System Prompt
# --------------------------

SYSTEM_PROMPT = """
You are an expert medical document classifier specializing in workers' compensation and healthcare reports.

**CRITICAL INSTRUCTIONS FOR FAX COVER SHEETS:**
1. FAX COVER SHEETS are NOT the actual document - they are transmission pages
2. Common fax cover indicators to IGNORE:
   - "To:", "From:", "Fax:", "Pages:", "Re:", "Date:" headers
   - "Please find attached...", "Enclosed please find..."
   - Transmission timestamps and phone numbers
   - Generic requests like "Requesting medical records" on fax forms
   - Clinic letterheads with routing information only
3. Look PAST the fax cover to find the ACTUAL REPORT CONTENT
4. The real document typically starts after phrases like:
   - "Doctor's First Report", "Progress Report", "Evaluation Report"
   - Medical examination details, patient history, diagnoses
   - Structured medical content sections (History, Examination, Assessment, Plan)

**DOCUMENT TYPE CATEGORIES:**
Medical Reports:
- RFA (Request for Authorization)
- PR2 (Progress Report)
- PR4 (Permanent/Stationary Report)
- DFR (Doctor's First Report)
- QME (Qualified Medical Evaluation)
- AME (Agreed Medical Evaluation)
- IME (Independent Medical Evaluation)
- IMR (Independent Medical Review)
- UR (Utilization Review)

Clinical Documentation:
- CONSULT (Consultation or Office Visit)
- Progress Notes
- Treatment Plans
- Work Status Reports
- Return-to-Work / Restriction Notes
- Discharge Summaries
- Admission Summaries
- Emergency Department Reports
- Nursing Notes

Imaging & Diagnostics:
- MRI (Magnetic Resonance Imaging)
- CT (Computed Tomography)
- X-ray (X-ray Imaging)
- Ultrasound
- EMG (Electromyography / Nerve Study)
- Lab Reports
- Pathology Reports
- Biopsy Reports
- Genetic Testing Reports
- Sleep Study Reports

Procedure Reports:
- Surgery Reports
- Endoscopy / Colonoscopy Reports
- Surgery Pre-Op / Post-Op Notes
- Anesthesia Reports

Specialty Reports:
- Cardiology Reports
- Pain Management Notes
- Psychological / Psychiatric Reports
- PT/OT/Chiro/Acupuncture Notes
- FCE (Functional Capacity Evaluation)

Administrative & Legal:
- Peer Reviews
- UR / IMR Decisions
- Medication / Pharmacy Documents
- Nurse Case Manager Notes
- Attorney Letters
- Disability / Claim Forms
- Appeal / Denial Letters
- ICD/CPT Billing Summaries
- Request for Records
- Legal Correspondence
- Employer Incident Reports

**CLASSIFICATION STRATEGY:**
1. **Ignore fax metadata** - Skip transmission details entirely
2. **Find substantive content** - Look for actual medical/clinical information
3. **Use summarizer context** - The summary often filters out fax noise
4. **Prioritize document structure** - Real reports have clear medical sections
5. **Match predefined types** - Use standard categories when possible
6. **Extract actual titles** - If no match, use the document's real title (NOT fax headers)
7. **Set is_fax_cover=true** ONLY if the entire content appears to be a fax cover with no actual report

**CONFIDENCE LEVELS:**
- 0.9-1.0: Clear document type with definitive medical content
- 0.7-0.89: Strong indicators but some ambiguity
- 0.5-0.69: Multiple possible types, context helps
- Below 0.5: Unclear or insufficient content

**OUTPUT RULES:**
- NEVER return "Fax Report", "Facsimile Cover", "Fax Cover Sheet"
- If predefined type matches → use it
- If no match → extract the ACTUAL document title (e.g., "Orthopedic Consultation", "Spine Surgery Report")
- NEVER use "OTHER" or generic labels
- Be specific and descriptive
"""

HUMAN_PROMPT = """
Analyze this text and determine the PRIMARY document type, ignoring any fax cover sheet information.

**Context Summary (filtered content):**
{summary}

**Analysis Instructions:**
1. First, identify if there's fax cover sheet content (To/From/Re headers, transmission info)
2. Skip past any fax cover content
3. Find the actual medical/clinical document that follows
4. Classify based on the REAL document content, not the fax routing information
5. Use the summary context to understand the document's purpose

Return strict JSON using this schema:
{format_instructions}

**Full Text (for verification):**
{text}
"""

# --------------------------
# 3. Enhanced Detector Function
# --------------------------

HIGH_CONFIDENCE_THRESHOLD = 0.75
FAX_KEYWORDS = [
    "facsimile", "fax cover", "fax transmission", "please find attached",
    "enclosed please find", "pages including cover", "total pages",
    "confidential communication", "intended recipient only"
]

def preprocess_text_for_analysis(text: str) -> tuple[str, bool]:
    """
    Preprocesses text to detect and potentially skip fax cover content.
    Returns: (processed_text, has_fax_cover)
    """
    if not text:
        return text, False
    
    text_lower = text.lower()
    has_fax_indicators = any(kw in text_lower for kw in FAX_KEYWORDS)
    
    # Try to find where actual content starts (common markers)
    content_markers = [
        "doctor's first report", "progress report", "medical evaluation",
        "history of present illness", "chief complaint", "physical examination",
        "consultation report", "operative report", "diagnostic imaging",
        "patient name:", "date of injury:", "diagnosis:", "assessment:",
        "subjective:", "objective:", "impression:", "plan:"
    ]
    
    if has_fax_indicators:
        # Find the earliest content marker
        earliest_pos = len(text)
        for marker in content_markers:
            pos = text_lower.find(marker)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        # If we found a content marker, start from there
        if earliest_pos < len(text):
            # Include some context before the marker (up to 200 chars)
            start_pos = max(0, earliest_pos - 200)
            processed_text = text[start_pos:]
            logger.info(f"Detected fax cover, skipping to position {start_pos}")
            return processed_text, True
    
    return text, has_fax_indicators

def detect_document_type(summarizer_output: str = None, raw_text: str = None) -> dict:
    """
    Enhanced document type detection with improved fax cover sheet handling.
    
    Strategy:
    1. Use summarizer output (already filtered/contextualized) as primary source
    2. Preprocess raw_text to skip fax cover sheets
    3. If low confidence from summarizer, verify with processed raw_text
    4. Compare results and use highest confidence
    5. Flag if fax cover content was detected
    
    Returns dict: {
        "doc_type": str,
        "confidence": float,
        "reasoning": str,
        "source": "summarizer"|"raw_text",
        "is_fax_cover": bool
    }
    """
    if summarizer_output is None and raw_text is None:
        raise ValueError("At least one of summarizer_output or raw_text must be provided")
    
    primary_text = summarizer_output if summarizer_output else raw_text
    fallback_text = raw_text if (summarizer_output and raw_text and summarizer_output != raw_text) else None
    
    logger.info("=" * 80)
    logger.info("ENHANCED DOCUMENT TYPE DETECTION")
    logger.info("=" * 80)
    logger.info(f"Primary text (summarizer) length: {len(primary_text) if primary_text else 0} chars")
    logger.info(f"Fallback text (raw) length: {len(fallback_text) if fallback_text else 0} chars")
    
    # Preprocess raw text to handle fax covers
    processed_raw_text = None
    has_fax_cover = False
    if fallback_text:
        processed_raw_text, has_fax_cover = preprocess_text_for_analysis(fallback_text)
        if has_fax_cover:
            logger.info("⚠️ Fax cover sheet detected in raw text - using processed version")
            logger.info(f"Processed text length: {len(processed_raw_text)} chars")
    
    try:
        model = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )
        logger.info("✓ Azure OpenAI model initialized")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT.strip()),
            ("human", HUMAN_PROMPT.strip())
        ]).partial(format_instructions=parser.get_format_instructions())

        # First attempt: Summarizer output (usually best filtered)
        chain_input = {
            "summary": primary_text[:4000],  # Summary context
            "text": primary_text[:4000]       # Limited for efficiency
        }
        
        logger.info("→ Analyzing with summarizer output...")
        messages = prompt.format_prompt(**chain_input).to_messages()
        response = model.invoke(messages)
        
        logger.info(f"Raw LLM response: {response.content[:500]}...")
        
        result = parser.parse(response.content)
        result = result.model_dump()
        result["source"] = "summarizer"
        
        logger.info(f"Summarizer result: {result['doc_type']} (confidence: {result['confidence']}, is_fax: {result.get('is_fax_cover', False)})")
        
        # Check if we need fallback analysis
        needs_fallback = (
            result["confidence"] < HIGH_CONFIDENCE_THRESHOLD or
            result.get("is_fax_cover", False) or
            result["doc_type"].lower() in ["fax", "fax cover", "facsimile"]
        )
        
        if needs_fallback and processed_raw_text:
            logger.info("→ Low confidence or fax detected - analyzing processed raw text...")
            
            chain_input_raw = {
                "summary": primary_text[:2000],           # Keep summary context
                "text": processed_raw_text[:6000]         # Use more of processed text
            }
            
            messages_raw = prompt.format_prompt(**chain_input_raw).to_messages()
            response_raw = model.invoke(messages_raw)
            
            result_raw = parser.parse(response_raw.content)
            result_raw = result_raw.model_dump()
            result_raw["source"] = "raw_text"
            
            logger.info(f"Raw text result: {result_raw['doc_type']} (confidence: {result_raw['confidence']}, is_fax: {result_raw.get('is_fax_cover', False)})")
            
            # Use better result (higher confidence, not a fax cover)
            if (not result_raw.get("is_fax_cover", False) and 
                (result_raw["confidence"] > result["confidence"] or 
                 result.get("is_fax_cover", False))):
                logger.info(f"✓ Using raw text result (better confidence or avoided fax)")
                result = result_raw
            else:
                logger.info(f"✓ Keeping summarizer result")

        # Final validation: Check if result is still fax-related
        if result["doc_type"].lower() in ["fax", "fax cover", "facsimile", "fax cover sheet"]:
            logger.warning("⚠️ Result still shows fax - marking low confidence")
            result["confidence"] = min(result["confidence"], 0.4)
            result["reasoning"] += " (Warning: May be fax cover sheet)"

        logger.info("=" * 80)
        logger.info("FINAL DETECTION RESULT")
        logger.info("=" * 80)
        logger.info(f'📝 Document Type: {result["doc_type"]}')
        logger.info(f'📊 Confidence: {result["confidence"]:.2f}')
        logger.info(f'💡 Reasoning: {result["reasoning"]}')
        logger.info(f'📄 Source: {result["source"]}')
        logger.info(f'📠 Fax Cover Detected: {result.get("is_fax_cover", False)}')
        logger.info("=" * 80)

        return result
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR IN DOCUMENT TYPE DETECTION")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        logger.error("=" * 80)
        raise


# --------------------------
# 4. Utility Functions
# --------------------------

def validate_document_type(result: dict) -> dict:
    """
    Post-processing validation to catch any remaining issues.
    """
    doc_type = result["doc_type"].lower()
    
    # Block common fax-related misclassifications
    fax_terms = ["fax", "facsimile", "cover sheet", "transmission"]
    if any(term in doc_type for term in fax_terms):
        logger.warning(f"⚠️ Suspicious document type detected: {result['doc_type']}")
        result["confidence"] = min(result["confidence"], 0.3)
        result["reasoning"] += " [FLAGGED: Possible fax cover misclassification]"
    
    return result


def batch_detect_document_types(documents: list[dict]) -> list[dict]:
    """
    Process multiple documents efficiently.
    
    Args:
        documents: List of dicts with 'summarizer_output' and/or 'raw_text' keys
    
    Returns:
        List of detection results with added 'doc_id' if provided
    """
    results = []
    
    for idx, doc in enumerate(documents):
        doc_id = doc.get('doc_id', f'doc_{idx}')
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing document {doc_id} ({idx+1}/{len(documents)})")
        logger.info(f"{'='*80}")
        
        try:
            result = detect_document_type(
                summarizer_output=doc.get('summarizer_output'),
                raw_text=doc.get('raw_text')
            )
            result['doc_id'] = doc_id
            result = validate_document_type(result)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")
            results.append({
                'doc_id': doc_id,
                'doc_type': 'ERROR',
                'confidence': 0.0,
                'reasoning': f'Processing error: {str(e)}',
                'source': 'error',
                'is_fax_cover': False
            })
    
    return results