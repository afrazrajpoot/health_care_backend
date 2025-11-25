# document_detector_simple.py
"""
Minimal context-aware document type detector.
Uses GPT-4o + LangChain OutputParser (Pydantic).
No regex, no embeddings, no rule bias.
Simplified: Single LLM call to classify or extract title as doc_type.
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
        description="The main document type. "
                    "If matches a standard type, use that exact name. "
                    "Otherwise, use the extracted main title/heading from the document."
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Short reason why this document type was chosen, including if it's a standard type or extracted title."
    )
    is_standard_type: bool = Field(
        description="True if this is a known standard document type, False if it's an extracted title."
    )

parser = PydanticOutputParser(pydantic_object=DocumentTypeOut)

# Standard document types for reference (passed to prompt)
standard_types = [
    # Imaging Reports
    "MRI", "CT", "X-ray", "Ultrasound", "EMG", "Mammogram", "PET Scan", "Bone Scan",
    "DEXA Scan", "Fluoroscopy", "Angiogram",
    
    # Laboratory & Diagnostics
    "Labs", "Pathology", "Biopsy", "Genetic Testing", "Toxicology", "Allergy Testing",
    
    # Progress Reports & Evaluations
    "PR-2", "PR-4", "DFR", "Consult", "Progress Note", "Office Visit", "Clinic Note", "Telemedicine",
    
    # Medical Evaluations
    "QME", "AME", "IME", "IMR", "FCE", "Peer Review", "Independent Review",
    
    # Authorization & Utilization
    "RFA", "UR", "Authorization", "Peer-to-Peer", "Treatment Authorization", "Procedure Authorization",
    
    # Therapy & Treatment Notes
    "Physical Therapy", "Occupational Therapy", "Chiropractic", "Acupuncture", "Massage Therapy", "Pain Management",
    
    # Clinical Notes - Nursing and additional types
    "Nursing", "Nursing Note", "Vital Signs", "Medication Administration",
    
    # Surgical Documents
    "Surgery Report", "Operative Note", "Anesthesia Report", "Pre-Op", "Post-Op", "Discharge",
    
    # Hospital Documents
    "Admission Note", "Hospital Course", "ER Report", "Emergency Room", "Hospital Progress",
    
    # Specialty Reports
    "Cardiology", "Neurology", "Orthopedics", "Psychiatry", "Psychology", "Psychotherapy", 
    "Behavioral Health", "Rheumatology", "Endocrinology", "Gastroenterology", "Pulmonology",
    
    # Diagnostic Studies
    "Sleep Study", "EKG", "ECG", "Holter Monitor", "Echocardiogram", "Stress Test", 
    "Pulmonary Function", "Nerve Conduction",
    
    # Medications & Pharmacy
    "Med Refill", "Prescription", "Pharmacy", "Medication List", "Prior Authorization",
    
    # Administrative & Correspondence
    "Adjuster", "Attorney", "NCM", "Signature Request", "Referral", "Correspondence", 
    "Appeal", "Denial Letter", "Approval Letter",
    
    # Work Status & Disability
    "Work Status", "Work Restrictions", "Return to Work", "Disability", "Claim Form",
    
    # Legal Documents
    "Deposition", "Interrogatory", "Subpoena", "Affidavit",
    
    # Employer & Vocational
    "Employer Report", "Vocational Rehab", "Job Analysis", "Work Capacity",
    
    # Generic Fallbacks
    "Clinical Note", "Medical Report", "Administrative", "Unknown"
]

# --------------------------
# 2. Build the LLM + prompt (single call for classification or title extraction)
# --------------------------

SYSTEM_PROMPT = """
You are a professional document classification model for medical and workers' compensation reports.
Your job is to determine the primary document type by understanding the FULL CONTEXT and PURPOSE of the document.

**CRITICAL INSTRUCTIONS:**

1. **NEVER return "Fax", "Facsimile", "Facsimile Report", or any fax-related term as the doc_type.**
   - Fax headers are transmission metadata, NOT the document type.
   - Always look past fax headers to find the actual medical/legal document content.

2. **Context-First Approach:**
   - Read and understand the ENTIRE document context and purpose
   - What is this document actually about? What is its primary function?
   - Who created it and why? (medical report, therapy note, authorization, etc.)

3. **Classification Priority:**
   a) FIRST: Analyze the document's content, purpose, and context
   b) THEN: Check if it matches ANY standard type from the list below → use EXACT standard name, is_standard_type=True
   c) IF NO standard match: Extract the most meaningful title/heading that describes the document's purpose
      - Look for prominent headings, report titles, or document labels
      - If multiple titles exist, choose the one that best describes the PRIMARY purpose
      - DO NOT use generic terms like "Report" alone - be specific (e.g., "Physical Therapy Evaluation")
   d) Set is_standard_type=False for extracted titles

4. **Title Extraction Guidelines:**
   - Ignore: fax headers, cover sheets, transmission data, page numbers, dates in isolation
   - Focus on: main headings, report types, evaluation names, procedure descriptions
   - Be specific: "Lumbar Spine MRI" not just "MRI" if that's the actual title
   - Context matters: If content is about PT but title says "Progress Report", consider which is more accurate

5. **Confidence Scoring:**
   - High (0.8-1.0): Clear standard type match OR obvious primary title
   - Medium (0.5-0.79): Extracted title but requires interpretation
   - Low (0.0-0.49): Ambiguous or unclear document

Standard types (use exact matches):
{standard_list}

Remember: Understand the document's PURPOSE first, classify second. Never return fax-related terms.
"""

HUMAN_PROMPT = """
Analyze the following document text to determine its type.

**Step 1:** Read the entire text and understand what this document is actually about (its purpose and content).
**Step 2:** Determine if it matches a standard type. If yes, use that exact name.
**Step 3:** If no standard match, extract the most appropriate title that describes the document's primary purpose.
**Step 4:** NEVER use "Fax" or "Facsimile Report" - these are transmission methods, not document types.

Return strict JSON using this schema:
{format_instructions}

TEXT:
{text}
"""

# --------------------------
# 3. Simplified detector function
# --------------------------

def detect_document_type(text: str) -> DocumentTypeOut:
    """
    Detects document type: Returns standard type if matched, else extracted title as doc_type.
    Single LLM call for simplicity.
    """
    logger.info("Starting document type detection")
    
    try:
        model = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )

        # Use first 4000 characters for detection (adjust if needed)
        detection_text = text[:4000]
        
        # Build prompt with standard types list
        standard_list_str = ", ".join(standard_types)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT.strip().format(standard_list=standard_list_str)),
            ("human", HUMAN_PROMPT.strip())
        ]).partial(format_instructions=parser.get_format_instructions())

        messages = prompt.format_prompt(text=detection_text).to_messages()
        response = model.invoke(messages)
        result = parser.parse(response.content)

        logger.info(f'📝 Document type: {result.doc_type}')
        logger.info(f'📊 Confidence: {result.confidence}')
        logger.info(f'💡 Reasoning: {result.reasoning}')
        logger.info(f'🏷️ Standard type: {result.is_standard_type}')

        return result
        
    except Exception as e:
        logger.error(f"ERROR during document type detection: {e}")
        # Simple fallback: "Unknown"
        return DocumentTypeOut(
            doc_type="Unknown",
            confidence=0.0,
            reasoning=f"Detection failed: {str(e)}",
            is_standard_type=False
        )