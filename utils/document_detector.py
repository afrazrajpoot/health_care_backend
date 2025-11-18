# document_detector_simple.py
"""
Minimal context-aware document type detector.
Uses GPT-4o + LangChain OutputParser (Pydantic).
No regex, no embeddings, no rule bias.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from config.settings import CONFIG

# --------------------------
# 1. Define the output schema
# --------------------------

class DocumentTypeOut(BaseModel):
    doc_type: str = Field(
        description="The main document type inferred from the text. "
                    "Must be one of: RFA, PR2, DFR, QME, IMAGING, CONSULT, UR, OTHER."
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Short reason why this document type was chosen."
    )

parser = PydanticOutputParser(pydantic_object=DocumentTypeOut)

# --------------------------
# 2. Build the LLM + prompt
# --------------------------

SYSTEM_PROMPT = """
You are a professional document classification model for medical and workers' compensation reports.
Your job is to decide the primary type of document from raw extracted text.

You will handle report types such as:
- RFA (Request for Authorization)
- PR2 (Progress Report)
- PR4 (Permanent/Stationary Report)
- DFR (Doctor’s First Report)
- QME (Qualified Medical Evaluation)
- AME (Agreed Medical Evaluation)
- IME (Independent Medical Evaluation)
- IMR (Independent Medical Review)
- UR (Utilization Review)
- CONSULT (Consultation or Office Visit)
- IMAGING (Generic Imaging Report)
- MRI (Magnetic Resonance Imaging)
- CT (Computed Tomography)
- X-ray (X-ray Imaging)
- Ultrasound (Ultrasound Imaging)
- EMG (Electromyography / Nerve Study)
- Progress Reports (PR2, PR4)
- Progress Notes
- Consult Reports
- Radiology Reports
- Surgery Reports
- PT/OT/Chiro/Acupuncture Notes
- Peer Reviews
- UR / IMR Decisions
- Medication / Pharmacy Documents
- Nurse Case Manager Notes
- Attorney Letters
- DFR / PR-2 / PR-4 Forms
- Treatment Plans
- Lab Reports
- Pathology Reports
- Cardiology Reports
- EMG/NCS Reports
- FCE (Functional Capacity Evaluation)
- Work Status Reports
- Return-to-Work / Restriction Notes
- Surgery Pre-Op / Post-Op Notes
- Anesthesia Reports
- Pain Management Notes
- Psychological / Psychiatric Reports
- Emergency Department Reports
- Discharge Summaries
- Admission Summaries
- Hospital Course Documents
- Nursing Notes
- Disability / Claim Forms
- Pharmacy Logs
- Legal Correspondence
- Employer Incident Reports
- Job Requirements Reports
- Medication Administration Records
- Telemedicine Notes
- Endoscopy / Colonoscopy Reports
- Biopsy Reports
- Genetic Testing Reports
- Sleep Study Reports
- Appeal / Denial Letters
- ICD/CPT Billing Summaries

Guidelines:
- Consider the **context**, not just keywords.
- If a form is only *mentioned* (e.g., "Attach the Doctor’s First Report"), do NOT classify as that type.
- The **title or heading appearing first** is usually the main document, but use full context to confirm.
- Be concise and objective. Never invent facts.
"""

HUMAN_PROMPT = """
Determine the most appropriate document type based on the provided text.

Return strict JSON using this schema:
{format_instructions}

TEXT:
{text}
"""

# --------------------------
# 3. Detector function
# --------------------------

def detect_document_type(text: str) -> dict:
    """
    Detects the most suitable document type using GPT-4o with context reasoning.
    Returns dict: {"doc_type": "...", "confidence": ..., "reasoning": "..."}
    """
    model = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )


    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.strip()),
        ("human", HUMAN_PROMPT.strip())
    ]).partial(format_instructions=parser.get_format_instructions())

    chain_input = {"text": text[:8000]}  # cap text length for efficiency

    messages = prompt.format_prompt(**chain_input).to_messages()
    response = model.invoke(messages)  # Use .invoke() instead of calling directly
    result = parser.parse(response.content)
    result = result.model_dump()

    print(f'📝 Document type: {result["doc_type"]}')
    print(f'📝 Confidence: {result["confidence"]}')
    print(f'📝 Reasoning: {result["reasoning"]}')

    return result
