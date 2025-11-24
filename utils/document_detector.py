# models/data_models.py
from pydantic import BaseModel, Field
from typing import Literal

# Convert enum to BaseModel with Literal types
class DocumentType(BaseModel):
    """All supported document types - 70+ document types"""
    
    # Using Literal for type safety while maintaining flexibility
    doc_type: Literal[
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
        "Clinical Note", "Medical Report", "Administrative", "Unknown",
        
        # Allow any string for custom types
        str
    ] = Field(description="Document type - can be predefined or custom")

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
Your job is to decide the primary type of document from raw extracted text.

**CRITICAL INSTRUCTIONS:**
1. FIRST, check if the document matches ANY standard type from the list below. If it does, set doc_type to that EXACT standard name, is_standard_type=True, and high confidence.
2. If NO match to ANY standard type, extract the ACTUAL main title/heading from the document (look at the first prominent heading or line) and set doc_type to that title, is_standard_type=False, and moderate confidence.
3. NEVER use "OTHER", "UNKNOWN", or generic terms if a title can be extracted.
4. Be specific and context-aware. Reasoning should explain the choice briefly.

Standard types (exact matches only):
{standard_list}

If unsure, prioritize extracting a clear title over guessing a standard type.
"""

HUMAN_PROMPT = """
Determine the most appropriate document type based on the provided text.
If standard match: use it. Else: extract and use the main title as doc_type.

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

        # Use first 1000 characters for detection (adjust if needed)
        detection_text = text[:3000]
        
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