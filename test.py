# document_detector_multi_report.py
"""
Enhanced multi-report document detector that handles documents with multiple reports.
Uses document summarizer output as the primary signal for context-aware detection.

Key Features:
- Detects ALL main reports in a document, not just the first one
- Summary-driven: Uses summarizer output to understand document context
- Filters out supporting documents, references, and attachments
- Context-aware title handling
- Multi-report safe detection
"""
import logging
from typing import List
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
# 1. Define the output schema for MULTIPLE reports
# --------------------------

class SingleReportDetection(BaseModel):
    """Detection result for a single report within the document"""
    doc_type: str = Field(
        description="The document type (RFA, PR2, DFR, QME, IMAGING, CONSULT, UR, etc.), "
                    "OR if none match, use the actual title from the document."
    )
    title: str = Field(
        description="The actual title or heading of this report as it appears in the document. "
                    "Must be specific and extracted from the document content, not generic."
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0 for this specific report detection",
        ge=0.0, le=1.0
    )
    is_main_report: bool = Field(
        description="True if this is a main report discussed in the document. "
                    "False if it's a supporting document, attachment, or reference."
    )
    reasoning: str = Field(
        description="Why this was identified as a main report or supporting document."
    )
    is_valid_for_summary_card: bool = Field(
        description="True if this report requires physician clinical interpretation/decision-making.",
        default=False
    )
    summary_card_reasoning: str = Field(
        description="Explanation for why this report does or does not need a Summary Card.",
        default=""
    )


class MultiReportDetectionResult(BaseModel):
    """Complete detection result supporting multiple reports"""
    reports: List[SingleReportDetection] = Field(
        description="List of all main reports detected in the document. "
                    "Should only include actual main reports, not references or attachments.",
        default_factory=list
    )
    document_summary: str = Field(
        description="Brief summary of what the overall document contains and how many main reports are present.",
        default=""
    )
    total_main_reports: int = Field(
        description="Total count of main reports detected (excluding supporting documents)",
        default=0
    )
    has_multiple_reports: bool = Field(
        description="True if document contains more than one main report",
        default=False
    )
    overall_confidence: float = Field(
        description="Overall confidence in the detection results",
        ge=0.0, le=1.0,
        default=0.0
    )


parser = PydanticOutputParser(pydantic_object=MultiReportDetectionResult)

# --------------------------
# 2. Enhanced System Prompt for Multi-Report Detection
# --------------------------

MULTI_REPORT_SYSTEM_PROMPT = """
You are an expert medical document classifier specializing in multi-report document analysis.

**YOUR PRIMARY TASK:**
Detect ALL main reports present in the document based on the document summarizer's output.
Return EVERY main report, not just the first one you find.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**CRITICAL: USE DOCUMENT SUMMARY AS PRIMARY SIGNAL**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The document summarizer output provides the CONTEXT and OVERVIEW of what's in the document.
- Trust the summary to understand what reports are ACTUALLY discussed
- Use it to distinguish main reports from references/attachments
- Don't just match on keywords - understand the narrative

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**MAIN REPORTS vs SUPPORTING DOCUMENTS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**✅ MAIN REPORTS (is_main_report=true):**
- Reports that are FULLY PRESENT with detailed content in the document
- Reports that are the PRIMARY SUBJECT of the document
- Reports that are DISCUSSED IN DETAIL in the summary
- Reports where findings, recommendations, or conclusions are provided

Examples:
- "The document contains an MRI report showing disc herniation at L4-L5..."
- "This is a QME report by Dr. Smith evaluating the patient's lumbar spine..."
- "The PR2 report documents the patient's progress over 6 weeks..."
- "Two reports are present: an MRI and a consultation note from orthopedics..."

**❌ SUPPORTING DOCUMENTS (is_main_report=false - DO NOT INCLUDE):**
- Documents that are only MENTIONED or REFERENCED
- Attachments listed but not fully included
- Prior reports cited for historical context
- Documents requested but not yet received
- Factual mentions without detailed content

Examples:
- "...requesting MRI report from January 2024"
- "Attached: prior PR2 from June 2023 (for reference)"
- "Please provide the QME report when available"
- "Patient mentioned having an X-ray done last month"
- "Reference: DFR dated 03/15/2024"

**THE KEY QUESTION:** Is this report FULLY PRESENT and DISCUSSED IN DETAIL, 
or just MENTIONED/REFERENCED?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**DOCUMENT TYPE CATEGORIES**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

Procedure Reports:
- Surgery Reports
- Endoscopy / Colonoscopy Reports
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**DETECTION STRATEGY**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **Read the summary carefully** - Understand the full context
2. **Identify all reports mentioned** - Look for multiple reports
3. **Classify each as main or supporting** - Use the summary narrative
4. **Extract actual titles** - Get specific names/headings from the text
5. **Validate against context** - Does the title match what's actually in the document?
6. **Return all main reports** - Don't stop at the first one

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**TITLE EXTRACTION RULES**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Extract actual titles from the document:**
- Look for report headers, headings, or stated titles
- Use specific names: "MRI Report - Lumbar Spine" not just "MRI"
- Include relevant context: "QME Report by Dr. Smith" not just "QME"
- Be specific about body parts/procedures: "CT Cervical Spine" not just "CT"

**Validate titles against summary:**
- If summary says "MRI shows..." there should be an MRI report
- If summary mentions "Dr. X's consultation", look for that consultation
- If summary discusses findings from multiple reports, include all of them

**Never use:**
- Generic placeholders: "Report 1", "Document A"
- Fax cover information: "Fax from Dr. Smith"
- Administrative headers: "Medical Records Request"
- Vague descriptions: "Some kind of medical report"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**SUMMARY CARD ELIGIBILITY (per report)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Set `is_valid_for_summary_card = true` for EACH report if it:
1. ✅ Contains clinical findings or medical opinions
2. ✅ Requires physician interpretation
3. ✅ Could affect treatment decisions or work status
4. ✅ Staff cannot safely act without physician review

**Reports that typically GET Summary Cards:**
- MRI/CT/X-ray with diagnostic findings
- QME/AME/IME reports with medical opinions
- Lab/Pathology results
- Surgery reports
- Specialist consultations with recommendations
- FCE evaluations
- Psychological evaluations

**Reports that typically DON'T get Summary Cards:**
- RFAs (authorization requests - administrative)
- UR decisions (already reviewed)
- Attorney letters (legal correspondence)
- Scheduling notifications
- Pharmacy/medication lists
- Billing documents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**EXAMPLES**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Example 1: Single Report**
Summary: "MRI of lumbar spine showing L4-L5 disc herniation with nerve root compression"

Expected output:
- reports: [
    {
      doc_type: "MRI",
      title: "MRI Report - Lumbar Spine",
      is_main_report: true,
      is_valid_for_summary_card: true
    }
  ]
- total_main_reports: 1
- has_multiple_reports: false

**Example 2: Multiple Reports**
Summary: "Document contains an MRI report showing cervical stenosis and a consultation note from orthopedic surgeon recommending surgical intervention"

Expected output:
- reports: [
    {
      doc_type: "MRI",
      title: "MRI Report - Cervical Spine",
      is_main_report: true,
      is_valid_for_summary_card: true
    },
    {
      doc_type: "CONSULT",
      title: "Orthopedic Consultation",
      is_main_report: true,
      is_valid_for_summary_card: true
    }
  ]
- total_main_reports: 2
- has_multiple_reports: true

**Example 3: Main Report + Reference (Don't include reference)**
Summary: "QME report by Dr. Smith finding permanent disability. References prior MRI from June 2023."

Expected output:
- reports: [
    {
      doc_type: "QME",
      title: "QME Report by Dr. Smith",
      is_main_report: true,
      is_valid_for_summary_card: true,
      reasoning: "Main QME report with detailed evaluation and findings"
    }
  ]
- total_main_reports: 1
- has_multiple_reports: false

Note: Prior MRI is only referenced, not included as a main report

**Example 4: RFA with Supporting Documents (Only RFA is main)**
Summary: "Request for authorization for lumbar surgery. Attached MRI and consultation reports support the request."

Expected output:
- reports: [
    {
      doc_type: "RFA",
      title: "Request for Authorization - Lumbar Surgery",
      is_main_report: true,
      is_valid_for_summary_card: false,
      reasoning: "RFA is the main document; supporting docs are attachments"
    }
  ]
- total_main_reports: 1
- has_multiple_reports: false

Note: Attached MRI/consultation are supporting documents, not main reports in this context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**CRITICAL REMINDERS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ ALWAYS analyze the FULL summary before making decisions
2. ✅ DETECT ALL main reports, not just the first one
3. ✅ USE CONTEXT to distinguish main reports from references
4. ✅ EXTRACT SPECIFIC TITLES from the document text
5. ✅ VALIDATE titles against the summary narrative
6. ❌ NEVER include references, attachments, or mentions as main reports
7. ❌ NEVER stop after finding one report if multiple exist
8. ❌ NEVER use generic or fax cover titles

**Your goal: Return a complete, accurate list of ALL main reports in the document.**
"""

# --------------------------
# 3. Enhanced User Prompt Template
# --------------------------

USER_PROMPT_TEMPLATE = """
**DOCUMENT SUMMARY (Primary Signal):**
{summary}

**DOCUMENT TEXT (For Title Extraction & Validation):**
{text}

**YOUR TASK:**
1. Analyze the summary to understand what main reports are present
2. Detect ALL main reports discussed in detail (not just references)
3. Extract specific titles from the document text
4. For each main report, determine if it needs a Summary Card
5. Return complete detection results

{format_instructions}
"""

# --------------------------
# 4. Helper Functions for Text Processing
# --------------------------

def preprocess_text_for_detection(text: str, max_length: int = 8000) -> str:
    """
    Clean and prepare text for detection while preserving important content.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Truncate if too long, but try to keep both beginning and end
    if len(text) > max_length:
        mid_point = max_length // 2
        text = text[:mid_point] + "\n...[content truncated]...\n" + text[-mid_point:]
    
    return text


def extract_fax_indicators(text: str) -> bool:
    """
    Quick check to see if text appears to be primarily fax cover content.
    """
    if not text:
        return False
    
    fax_phrases = [
        "please find attached",
        "enclosed please find",
        "fax transmission",
        "pages including cover",
        "to: from: re: date:",
        "fax: pages:"
    ]
    
    text_lower = text.lower()[:500]  # Check first 500 chars
    fax_count = sum(1 for phrase in fax_phrases if phrase in text_lower)
    
    return fax_count >= 2  # If 2+ fax indicators in first 500 chars, likely fax cover


# --------------------------
# 5. Main Multi-Report Detection Function
# --------------------------

def detect_document_types_multi(
    summarizer_output: str = None,
    raw_text: str = None,
    use_fallback: bool = True
) -> dict:
    """
    Enhanced document type detection supporting multiple reports.
    
    Args:
        summarizer_output: Summary text from document summarizer (PRIMARY SIGNAL)
        raw_text: Raw extracted text from document (for title extraction)
        use_fallback: Whether to use raw_text as fallback if summary analysis is uncertain
    
    Returns:
        Dictionary containing:
        - reports: List of detected main reports
        - document_summary: Overall document description
        - total_main_reports: Count of main reports
        - has_multiple_reports: Boolean flag
        - overall_confidence: Overall detection confidence
        - source: Which input was used primarily
    """
    
    logger.info("=" * 80)
    logger.info("MULTI-REPORT DOCUMENT TYPE DETECTION")
    logger.info("=" * 80)
    
    try:
        # Initialize LLM
        model = AzureChatOpenAI(
                azure_endpoint=CONFIG.get("azure_openai_endpoint"),
                api_key=CONFIG.get("azure_openai_api_key"),
                deployment_name=CONFIG.get("azure_openai_deployment"),
                api_version=CONFIG.get("azure_openai_api_version"),
                temperature=0.0,
                timeout=60
        )

        
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", MULTI_REPORT_SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE)
        ])
        
        # Prepare inputs
        primary_text = summarizer_output or raw_text or ""
        processed_raw_text = preprocess_text_for_detection(raw_text) if raw_text else ""
        
        if not primary_text:
            raise ValueError("Either summarizer_output or raw_text must be provided")
        
        logger.info(f"Primary text length: {len(primary_text)} chars")
        logger.info(f"Raw text length: {len(processed_raw_text)} chars")
        logger.info(f"Using summarizer output: {bool(summarizer_output)}")
        
        # Check for fax cover
        is_likely_fax = extract_fax_indicators(primary_text)
        if is_likely_fax:
            logger.warning("⚠️ Primary text appears to be fax cover - will prioritize processed text")
        
        # Prepare chain input
        chain_input = {
            "summary": primary_text[:3000],  # Use more of summary for context
            "text": processed_raw_text[:5000],  # Use raw text for title extraction
            "format_instructions": parser.get_format_instructions()
        }
        
        logger.info("→ Analyzing document for multiple reports...")
        
        # Invoke LLM
        messages = prompt.format_prompt(**chain_input).to_messages()
        response = model.invoke(messages)
        
        # Parse response
        result = parser.parse(response.content)
        result_dict = result.model_dump()
        result_dict["source"] = "summarizer" if summarizer_output else "raw_text"
        
        # Log results
        logger.info("=" * 80)
        logger.info("DETECTION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total main reports detected: {result_dict['total_main_reports']}")
        logger.info(f"Multiple reports: {result_dict['has_multiple_reports']}")
        logger.info(f"Overall confidence: {result_dict['overall_confidence']:.2f}")
        logger.info(f"Document summary: {result_dict['document_summary'][:150]}...")
        
        logger.info("\nDetected Reports:")
        for idx, report in enumerate(result_dict['reports'], 1):
            logger.info(f"\n  Report #{idx}:")
            logger.info(f"    Type: {report['doc_type']}")
            logger.info(f"    Title: {report['title']}")
            logger.info(f"    Confidence: {report['confidence']:.2f}")
            logger.info(f"    Is Main Report: {report['is_main_report']}")
            logger.info(f"    Summary Card: {report['is_valid_for_summary_card']}")
            logger.info(f"    Reasoning: {report['reasoning'][:100]}...")
        
        # Filter to only main reports
        main_reports = [r for r in result_dict['reports'] if r['is_main_report']]
        result_dict['reports'] = main_reports
        result_dict['total_main_reports'] = len(main_reports)
        result_dict['has_multiple_reports'] = len(main_reports) > 1
        
        # Validation: If no main reports found and confidence is low, try fallback
        if (not main_reports or result_dict['overall_confidence'] < 0.6) and use_fallback and processed_raw_text:
            logger.info("→ Low confidence or no reports - trying fallback with raw text...")
            
            chain_input_fallback = {
                "summary": processed_raw_text[:3000],
                "text": processed_raw_text[:5000],
                "format_instructions": parser.get_format_instructions()
            }
            
            messages_fallback = prompt.format_prompt(**chain_input_fallback).to_messages()
            response_fallback = model.invoke(messages_fallback)
            result_fallback = parser.parse(response_fallback.content)
            result_fallback_dict = result_fallback.model_dump()
            
            # Use fallback if it has higher confidence or more reports
            if (result_fallback_dict['overall_confidence'] > result_dict['overall_confidence'] or
                result_fallback_dict['total_main_reports'] > result_dict['total_main_reports']):
                logger.info("✓ Using fallback result (better confidence or more reports)")
                result_dict = result_fallback_dict
                result_dict["source"] = "raw_text_fallback"
                
                # Filter again
                main_reports = [r for r in result_dict['reports'] if r['is_main_report']]
                result_dict['reports'] = main_reports
                result_dict['total_main_reports'] = len(main_reports)
                result_dict['has_multiple_reports'] = len(main_reports) > 1
        
        logger.info("=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"✅ Total Main Reports: {result_dict['total_main_reports']}")
        logger.info(f"📊 Overall Confidence: {result_dict['overall_confidence']:.2f}")
        logger.info(f"📄 Source: {result_dict['source']}")
        
        if result_dict['reports']:
            logger.info("\nFinal Report List:")
            for idx, report in enumerate(result_dict['reports'], 1):
                logger.info(f"  {idx}. {report['doc_type']} - {report['title']}")
                logger.info(f"     Summary Card: {'YES' if report['is_valid_for_summary_card'] else 'NO'}")
        else:
            logger.warning("⚠️ No main reports detected!")
        
        logger.info("=" * 80)
        
        return result_dict
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR IN MULTI-REPORT DETECTION")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        logger.error("=" * 80)
        raise


# --------------------------
# 6. Batch Processing
# --------------------------

def batch_detect_multi_reports(documents: list[dict]) -> list[dict]:
    """
    Process multiple documents for multi-report detection.
    
    Args:
        documents: List of dicts with 'summarizer_output' and/or 'raw_text' keys
    
    Returns:
        List of detection results with 'doc_id' if provided
    """
    results = []
    
    for idx, doc in enumerate(documents):
        doc_id = doc.get('doc_id', f'doc_{idx}')
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing document {doc_id} ({idx+1}/{len(documents)})")
        logger.info(f"{'='*80}")
        
        try:
            result = detect_document_types_multi(
                summarizer_output=doc.get('summarizer_output'),
                raw_text=doc.get('raw_text')
            )
            result['doc_id'] = doc_id
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")
            results.append({
                'doc_id': doc_id,
                'reports': [],
                'document_summary': f'Error: {str(e)}',
                'total_main_reports': 0,
                'has_multiple_reports': False,
                'overall_confidence': 0.0,
                'source': 'error'
            })
    
    return results


# --------------------------
# 7. Analytics & Reporting
# --------------------------

def analyze_multi_report_batch(results: list[dict]) -> dict:
    """
    Analyze batch results for multi-report detection.
    
    Returns statistics on:
    - Total documents vs total reports
    - Single vs multi-report documents
    - Report type distribution
    - Summary card requirements
    """
    stats = {
        'total_documents': len(results),
        'total_reports_detected': sum(r['total_main_reports'] for r in results),
        'multi_report_documents': sum(1 for r in results if r['has_multiple_reports']),
        'single_report_documents': sum(1 for r in results if r['total_main_reports'] == 1),
        'no_reports_detected': sum(1 for r in results if r['total_main_reports'] == 0),
        'avg_reports_per_document': 0,
        'avg_confidence': sum(r['overall_confidence'] for r in results) / len(results) if results else 0,
        'report_type_distribution': {},
        'summary_card_count': 0,
        'reports_by_doc_count': {}
    }
    
    if stats['total_documents'] > 0:
        stats['avg_reports_per_document'] = stats['total_reports_detected'] / stats['total_documents']
    
    # Count report types and summary cards
    for result in results:
        doc_report_count = result['total_main_reports']
        stats['reports_by_doc_count'][doc_report_count] = stats['reports_by_doc_count'].get(doc_report_count, 0) + 1
        
        for report in result.get('reports', []):
            doc_type = report.get('doc_type', 'UNKNOWN')
            stats['report_type_distribution'][doc_type] = stats['report_type_distribution'].get(doc_type, 0) + 1
            
            if report.get('is_valid_for_summary_card', False):
                stats['summary_card_count'] += 1
    
    return stats


def print_multi_report_summary(results: list[dict]):
    """
    Print human-readable summary of multi-report detection results.
    """
    stats = analyze_multi_report_batch(results)
    
    print("\n" + "=" * 80)
    print("MULTI-REPORT BATCH DETECTION SUMMARY")
    print("=" * 80)
    print(f"📄 Total Documents Processed: {stats['total_documents']}")
    print(f"📋 Total Reports Detected: {stats['total_reports_detected']}")
    print(f"📊 Average Reports per Document: {stats['avg_reports_per_document']:.2f}")
    print(f"🎯 Average Confidence: {stats['avg_confidence']:.2f}")
    print()
    print("Document Distribution:")
    print(f"  📄 Single Report Documents: {stats['single_report_documents']}")
    print(f"  📑 Multi-Report Documents: {stats['multi_report_documents']}")
    print(f"  ⚠️  No Reports Detected: {stats['no_reports_detected']}")
    print()
    print("Reports per Document Breakdown:")
    for count in sorted(stats['reports_by_doc_count'].keys()):
        doc_count = stats['reports_by_doc_count'][count]
        print(f"  {count} report(s): {doc_count} document(s)")
    print()
    print("Report Type Distribution:")
    for doc_type, count in sorted(stats['report_type_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_type}: {count}")
    print()
    print(f"🎯 Summary Cards Required: {stats['summary_card_count']} / {stats['total_reports_detected']} reports")
    print("=" * 80)


# --------------------------
# 8. Backward Compatibility Function
# --------------------------

def detect_document_type_legacy(
    summarizer_output: str = None,
    raw_text: str = None
) -> dict:
    """
    Legacy function that returns single document type (for backward compatibility).
    Uses the first/highest confidence report from multi-report detection.
    
    Returns format matching the original detect_document_type function.
    """
    multi_result = detect_document_types_multi(summarizer_output, raw_text)
    
    if not multi_result['reports']:
        # No reports found
        return {
            'doc_type': 'UNKNOWN',
            'confidence': multi_result['overall_confidence'],
            'reasoning': 'No main reports detected in document',
            'source': multi_result['source'],
            'is_fax_cover': False,
            'is_valid_for_summary_card': False,
            'summary_card_reasoning': 'No reports found'
        }
    
    # Get the first report (or highest confidence if you prefer)
    primary_report = multi_result['reports'][0]
    
    # If multiple reports, note it in the reasoning
    if multi_result['has_multiple_reports']:
        primary_report['reasoning'] += f" (Note: Document contains {multi_result['total_main_reports']} reports)"
    
    return {
        'doc_type': primary_report['doc_type'],
        'confidence': primary_report['confidence'],
        'reasoning': primary_report['reasoning'],
        'source': multi_result['source'],
        'is_fax_cover': False,
        'is_valid_for_summary_card': primary_report['is_valid_for_summary_card'],
        'summary_card_reasoning': primary_report['summary_card_reasoning']
    }


# --------------------------
# Backward Compatibility Alias
# --------------------------

def detect_document_type(
    summarizer_output: str = None,
    raw_text: str = None
) -> dict:
    """
    Backward compatible alias for detect_document_type_legacy.
    
    DEPRECATED: Use detect_document_types_multi for new code to handle multiple reports.
    
    This function maintains the original API contract:
    - Returns a single document type result
    - Uses legacy single-report behavior
    
    For multi-report handling, use detect_document_types_multi directly.
    """
    return detect_document_type_legacy(summarizer_output, raw_text)


# --------------------------
# Multi-Report Processing Helper
# --------------------------

def process_multi_report_detection(
    summarizer_output: str = None,
    raw_text: str = None
) -> dict:
    """
    Comprehensive multi-report detection that returns both:
    - Legacy single-report result (for backward compatibility)
    - Full multi-report result (for new multi-report handling)
    
    Returns:
        Dictionary with:
        - 'primary': Legacy single-report result (first/best report)
        - 'all_reports': Full multi-report detection result
        - 'has_multiple_reports': Boolean flag for easy checking
        - 'total_reports': Count of main reports detected
    """
    try:
        # Get full multi-report detection
        multi_result = detect_document_types_multi(summarizer_output, raw_text)
        
        # Build legacy-compatible primary result
        if multi_result['reports']:
            primary_report = multi_result['reports'][0]
            primary_result = {
                'doc_type': primary_report['doc_type'],
                'confidence': primary_report['confidence'],
                'reasoning': primary_report['reasoning'],
                'source': multi_result['source'],
                'is_fax_cover': False,
                'is_valid_for_summary_card': primary_report['is_valid_for_summary_card'],
                'summary_card_reasoning': primary_report['summary_card_reasoning'],
                'title': primary_report.get('title', primary_report['doc_type'])
            }
        else:
            primary_result = {
                'doc_type': 'UNKNOWN',
                'confidence': multi_result['overall_confidence'],
                'reasoning': 'No main reports detected in document',
                'source': multi_result['source'],
                'is_fax_cover': False,
                'is_valid_for_summary_card': False,
                'summary_card_reasoning': 'No reports found',
                'title': 'Unknown Document'
            }
        
        return {
            'primary': primary_result,
            'all_reports': multi_result,
            'has_multiple_reports': multi_result['has_multiple_reports'],
            'total_reports': multi_result['total_main_reports'],
            'reports': multi_result['reports']
        }
        
    except Exception as e:
        logger.error(f"Error in process_multi_report_detection: {e}")
        # Return safe fallback
        return {
            'primary': {
                'doc_type': 'UNKNOWN',
                'confidence': 0.0,
                'reasoning': f'Detection failed: {str(e)}',
                'source': 'error',
                'is_fax_cover': False,
                'is_valid_for_summary_card': True,  # Default to True for safety
                'summary_card_reasoning': 'Detection failed - defaulting to requiring summary card',
                'title': 'Unknown Document'
            },
            'all_reports': {
                'reports': [],
                'document_summary': f'Error: {str(e)}',
                'total_main_reports': 0,
                'has_multiple_reports': False,
                'overall_confidence': 0.0,
                'source': 'error'
            },
            'has_multiple_reports': False,
            'total_reports': 0,
            'reports': []
        }


# --------------------------
# 9. Usage Example
# --------------------------

# if __name__ == "__main__":
#     # Example usage
#     sample_summary = """
#     This document contains two main reports:
#     1. MRI Report of Lumbar Spine showing L4-L5 disc herniation with nerve root compression
#     2. Orthopedic Consultation by Dr. Smith recommending surgical intervention
#     The patient has chronic low back pain and radiculopathy.
#     Reference is made to prior X-rays from June 2023.
#     """
    
#     sample_text = """
#     MRI REPORT - LUMBAR SPINE
#     Date: January 15, 2026
#     [MRI findings...]
    
#     ORTHOPEDIC CONSULTATION
#     Dr. John Smith, MD
#     [Consultation notes...]
    
#     Referenced: X-ray Lumbar Spine 06/12/2023
#     """
    
#     # Multi-report detection
#     result = detect_document_types_multi(
#         summarizer_output=sample_summary,
#         raw_text=sample_text
#     )
    
#     print(f"\nDetected {result['total_main_reports']} report(s):")
#     for report in result['reports']:
#         print(f"  - {report['doc_type']}: {report['title']}")
    
#     # Legacy single-report mode
#     legacy_result = detect_document_type_legacy(
#         summarizer_output=sample_summary,
#         raw_text=sample_text
#     )
    
#     print(f"\nLegacy mode (single report): {legacy_result['doc_type']}")