# document_detector_enhanced.py
"""
Enhanced context-aware document type detector with summary card eligibility.
Uses GPT-4o + LangChain OutputParser (Pydantic) with multi-stage analysis.

Key Addition: Intelligent detection of whether a document requires physician review
(Summary Card) vs. staff action (Task only).
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
    is_valid_for_summary_card: bool = Field(
        description="True if this document requires physician clinical interpretation/decision-making (Summary Card). "
                    "False if it's administrative/task-oriented for staff only. "
                    "Rules: Clinical findings/opinions/recommendations = True. "
                    "Administrative determinations/authorizations/scheduling = False.",
        default=False
    )
    summary_card_reasoning: str = Field(
        description="Explanation for why this document does or does not need a Summary Card for physician review.",
        default=""
    )

parser = PydanticOutputParser(pydantic_object=DocumentTypeOut)

# --------------------------
# 2. Enhanced System Prompt with Summary Card Logic
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**CRITICAL: SUMMARY CARD ELIGIBILITY DETECTION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**CORE PRINCIPLE:**
Summary Cards = Physician-actionable clinical intelligence
Tasks = Staff-actionable administrative items

**THE GOLDEN RULE:**
Set `is_valid_for_summary_card = true` ONLY if ALL of these are true:
1. ✅ Contains clinical facts, findings, or recommendations
2. ✅ Requires physician interpretation or clinical judgment
3. ✅ Could change treatment, restrictions, MMI, impairment, or risk
4. ✅ Staff CANNOT safely act without physician review

If ANY of these are false → `is_valid_for_summary_card = false`

**DOCUMENTS THAT GET SUMMARY CARDS (is_valid_for_summary_card = true):**

✅ **Clinical Reports with Findings:**
- MRI / CT / X-ray / Ultrasound / EMG reports (contain diagnostic findings)
- Lab reports / Pathology / Biopsy results
- QME / AME / IME reports (medical opinions affecting case)
- PR4 / Permanent & Stationary reports (MMI determinations)
- Operative / Surgery reports (procedure outcomes)
- Specialist consultations with clinical recommendations
- Psychological / Psychiatric evaluations
- FCE / Functional Capacity Evaluations (work capacity findings)
- Cardiology reports (Echo, Stress Test, EKG findings)
- Pain management evaluations with clinical findings
- Physical therapy / Chiropractic evaluations with progress assessments

✅ **Why these get Summary Cards:**
- Physician must INTERPRET clinical findings
- Decisions affect patient care, work status, or case trajectory
- Contain medical opinions requiring clinical judgment
- Staff cannot act independently on these

**DOCUMENTS THAT DO NOT GET SUMMARY CARDS (is_valid_for_summary_card = false):**

❌ **Administrative Determinations:**
- UR Authorizations (administrative approval - staff schedules service)
- UR Denials (administrative decision - staff handles appeal prep)
- RFA / Request for Authorization (administrative request - staff processes)
- Prior Authorization approvals/denials
- Scheduling notices / appointment confirmations
- Billing summaries / ICD/CPT codes
- Medication refill approvals
- Records requests

❌ **Legal/Administrative Correspondence:**
- Attorney letters (legal matter - staff routes/responds)
- Adjuster correspondence (claim administration - staff handles)
- Employer incident reports (administrative record - staff files)
- Appeal letters (legal process - staff manages)
- Disability claim forms (administrative - staff completes)
- Legal correspondence / subpoenas

❌ **Pure Progress Notes Without Clinical Changes:**
- PR2 Progress Reports that ONLY contain:
  * Status quo updates ("continues same treatment")
  * Routine follow-ups with no new findings
  * Administrative visit notes
  * BUT: PR2 with NEW findings, changed restrictions, or MMI discussion → True

❌ **Routing/Procedural Documents:**
- Fax cover sheets
- Referral forms (administrative routing)
- Records transmittal letters
- Nurse case manager routine check-ins

❌ **Why these do NOT get Summary Cards:**
- No clinical interpretation needed by physician
- Staff can execute without medical judgment
- Administrative or procedural in nature
- Would create physician cognitive overload if shown

**SPECIAL CASES - CAREFUL JUDGMENT REQUIRED:**

🔄 **Progress Reports (PR2):**
- DEFAULT: `false` (most are routine status updates)
- TRUE only if contains:
  * New diagnostic findings or clinical changes
  * Modified work restrictions
  * MMI/P&S discussion or timeline
  * Significant treatment changes requiring physician review
  * Changed prognosis or new medical opinions

🔄 **Consultation Reports:**
- DEFAULT: `true` (usually contain clinical recommendations)
- FALSE only if purely procedural (e.g., "Patient referred to Dr. X for scheduling")

🔄 **UR Denials:**
- DEFAULT: `false` (staff task: prepare appeal)
- EXCEPTION: Flag for escalation if repeated denials or IMR-risk
  * Still `false` for Summary Card
  * Note in reasoning: "Staff escalation needed for appeal"

**DECISION FRAMEWORK - ASK YOURSELF:**

1. "Can clinic staff act on this without the physician seeing it?"
   - YES → `false`
   - NO → continue to #2

2. "Does this contain medical findings or clinical opinions?"
   - NO → `false`
   - YES → continue to #3

3. "Could this change patient treatment, work status, or case outcome?"
   - NO → `false`
   - YES → `true`

**EXAMPLES WITH REASONING:**

Document: "MRI of lumbar spine - moderate disc bulge L4-L5"
- is_valid_for_summary_card: `true`
- Reasoning: "Contains diagnostic findings requiring physician interpretation to determine impact on treatment plan"

Document: "UR Authorization approved - 6 sessions physical therapy"
- is_valid_for_summary_card: `false`
- Reasoning: "Administrative approval - staff can schedule approved service without physician review"

Document: "QME Report - 10% whole person impairment, causation analysis"
- is_valid_for_summary_card: `true`
- Reasoning: "Contains medical opinions on impairment and causation that directly impact case resolution"

Document: "Attorney letter requesting medical records by 5/15/24"
- is_valid_for_summary_card: `false`
- Reasoning: "Administrative request - staff can process records request and manage deadline"

Document: "PR2 Progress - Patient continues PT, no new symptoms, next visit 4 weeks"
- is_valid_for_summary_card: `false`
- Reasoning: "Routine status update with no clinical changes requiring physician review"

Document: "PR2 Progress - New tingling in left leg, recommending EMG, modified work restrictions"
- is_valid_for_summary_card: `true`
- Reasoning: "New clinical finding with diagnostic recommendation and work status change requires physician review"

**OUTPUT REQUIREMENTS:**
You MUST set both:
1. `is_valid_for_summary_card` (boolean)
2. `summary_card_reasoning` (clear explanation of your decision)

Be conservative: When in doubt, default to `false` to prevent physician overload.
Only set `true` when physician clinical judgment is clearly required.
"""

HUMAN_PROMPT = """
Analyze this text and determine:
1. The PRIMARY document type (ignoring fax cover sheets)
2. Whether this document requires a physician Summary Card

**Context Summary (filtered content):**
{summary}

**Analysis Instructions:**
1. First, identify if there's fax cover sheet content (To/From/Re headers, transmission info)
2. Skip past any fax cover content
3. Find the actual medical/clinical document that follows
4. Classify based on the REAL document content, not the fax routing information
5. Use the summary context to understand the document's purpose
6. **Determine Summary Card eligibility using the GOLDEN RULE:**
   - Does it contain clinical findings/opinions requiring physician judgment? → true
   - Is it administrative/procedural for staff to handle? → false

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
    Enhanced document type detection with Summary Card eligibility.
    
    Strategy:
    1. Use summarizer output (already filtered/contextualized) as primary source
    2. Preprocess raw_text to skip fax cover sheets
    3. If low confidence from summarizer, verify with processed raw_text
    4. Compare results and use highest confidence
    5. Flag if fax cover content was detected
    6. **NEW: Determine if document needs physician Summary Card**
    
    Returns dict: {
        "doc_type": str,
        "confidence": float,
        "reasoning": str,
        "source": "summarizer"|"raw_text",
        "is_fax_cover": bool,
        "is_valid_for_summary_card": bool,
        "summary_card_reasoning": str
    }
    """
    if summarizer_output is None and raw_text is None:
        raise ValueError("At least one of summarizer_output or raw_text must be provided")
    
    primary_text = summarizer_output if summarizer_output else raw_text
    fallback_text = raw_text if (summarizer_output and raw_text and summarizer_output != raw_text) else None
    
    logger.info("=" * 80)
    logger.info("ENHANCED DOCUMENT TYPE DETECTION WITH SUMMARY CARD LOGIC")
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
        
        logger.info(f"Raw LLM response length: {len(response.content)} chars")
        
        result = parser.parse(response.content)
        result = result.model_dump()
        result["source"] = "summarizer"
        
        logger.info(f"Summarizer result: {result['doc_type']} (confidence: {result['confidence']:.2f})")
        logger.info(f"  - Fax cover: {result.get('is_fax_cover', False)}")
        logger.info(f"  - Summary Card: {result.get('is_valid_for_summary_card', False)}")
        logger.info(f"  - Card reasoning: {result.get('summary_card_reasoning', 'N/A')[:100]}")
        
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
            
            logger.info(f"Raw text result: {result_raw['doc_type']} (confidence: {result_raw['confidence']:.2f})")
            logger.info(f"  - Summary Card: {result_raw.get('is_valid_for_summary_card', False)}")
            
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
            # Fax covers should never get summary cards
            result["is_valid_for_summary_card"] = False
            result["summary_card_reasoning"] = "Fax cover sheet - no clinical content"

        logger.info("=" * 80)
        logger.info("FINAL DETECTION RESULT")
        logger.info("=" * 80)
        logger.info(f'📝 Document Type: {result["doc_type"]}')
        logger.info(f'📊 Confidence: {result["confidence"]:.2f}')
        logger.info(f'💡 Reasoning: {result["reasoning"]}')
        logger.info(f'📄 Source: {result["source"]}')
        logger.info(f'📠 Fax Cover: {result.get("is_fax_cover", False)}')
        logger.info(f'🎯 Summary Card Needed: {result.get("is_valid_for_summary_card", False)}')
        logger.info(f'💭 Summary Card Reasoning: {result.get("summary_card_reasoning", "N/A")}')
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
        result["is_valid_for_summary_card"] = False
        if not result.get("summary_card_reasoning"):
            result["summary_card_reasoning"] = "Fax cover - no clinical content"
    
    # Safety check: Ensure summary_card_reasoning exists
    if "summary_card_reasoning" not in result or not result["summary_card_reasoning"]:
        result["summary_card_reasoning"] = "No reasoning provided by classifier"
    
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
                'is_fax_cover': False,
                'is_valid_for_summary_card': False,
                'summary_card_reasoning': f'Error during processing: {str(e)}'
            })
    
    return results


# --------------------------
# 5. Summary Statistics Helper
# --------------------------

def analyze_batch_results(results: list[dict]) -> dict:
    """
    Analyze batch detection results to provide insights.
    
    Returns statistics on:
    - Document type distribution
    - Summary card vs task-only ratio
    - Confidence levels
    - Fax cover detection rate
    """
    stats = {
        'total_documents': len(results),
        'summary_card_count': sum(1 for r in results if r.get('is_valid_for_summary_card', False)),
        'task_only_count': sum(1 for r in results if not r.get('is_valid_for_summary_card', False)),
        'fax_cover_count': sum(1 for r in results if r.get('is_fax_cover', False)),
        'avg_confidence': sum(r.get('confidence', 0) for r in results) / len(results) if results else 0,
        'doc_type_distribution': {},
        'summary_card_by_type': {}
    }
    
    for result in results:
        doc_type = result.get('doc_type', 'UNKNOWN')
        stats['doc_type_distribution'][doc_type] = stats['doc_type_distribution'].get(doc_type, 0) + 1
        
        if result.get('is_valid_for_summary_card', False):
            stats['summary_card_by_type'][doc_type] = stats['summary_card_by_type'].get(doc_type, 0) + 1
    
    stats['summary_card_percentage'] = (stats['summary_card_count'] / stats['total_documents'] * 100) if stats['total_documents'] > 0 else 0
    
    return stats


def print_batch_summary(results: list[dict]):
    """
    Print a human-readable summary of batch detection results.
    """
    stats = analyze_batch_results(results)
    
    print("\n" + "=" * 80)
    print("BATCH DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total Documents Processed: {stats['total_documents']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print()
    print(f"📋 Summary Cards (Physician Review): {stats['summary_card_count']} ({stats['summary_card_percentage']:.1f}%)")
    print(f"📌 Task Only (Staff Action): {stats['task_only_count']} ({100-stats['summary_card_percentage']:.1f}%)")
    print(f"📠 Fax Covers Detected: {stats['fax_cover_count']}")
    print()
    print("Document Type Distribution:")
    for doc_type, count in sorted(stats['doc_type_distribution'].items(), key=lambda x: x[1], reverse=True):
        summary_count = stats['summary_card_by_type'].get(doc_type, 0)
        print(f"  {doc_type}: {count} total ({summary_count} get Summary Cards)")
    print("=" * 80)