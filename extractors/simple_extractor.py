"""
Generic extractor for simple document types (RFA, UR, Auth, Admin letters, etc.)
"""
import re
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class SimpleExtractor:
    """Generic extractor for simpler document types"""

    # 🧠 Universal clarity add-on for all prompt instructions
    CLEAR_EXTRACTION_GUIDE = (
        "Ensure all extracted information is explicit and actionable. That helps medical professionals quickly understand key details."
        "When listing body parts, diagnoses, or findings, name each explicitly. "
        "Avoid vague terms like '+1 more', 'etc.', or incomplete fragments. "
        "Always produce full, meaningful, concise phrases. "
        "Do not skip important context, but keep output brief and factual. "
        "Ensure wording is clear, complete, and human-readable."
    )

    TEMPLATES = {
        "RFA": {
            "fields": ["date", "service_requested", "body_part"],
            "format": "RFA {date} = {service_requested} {body_part}",
            "prompt": (
                "Extract: date (MM/DD/YY), service_requested (e.g., 'PT 6v', 'MRI'), "
                "and body_part (list clearly; e.g., 'R shoulder, L knee'). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "UR": {
            "fields": ["date", "service_denied", "reason"],
            "format": "UR {date} = {service_denied} denied; {reason}",
            "prompt": (
                "Extract: date, service_denied (what was denied), and reason (brief, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Authorization": {
            "fields": ["date", "service_approved", "body_part"],
            "format": "Auth {date} = {service_approved} {body_part} approved",
            "prompt": (
                "Extract: date (MM/DD/YY), service_approved (e.g., 'MRI', 'PT 6v'), and body_part. "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "DFR": {
            "fields": ["date", "doi", "diagnosis", "plan"],
            "format": "DFR {date} = DOI {doi}; {diagnosis}; {plan}",
            "prompt": (
                "Extract: report date, DOI (date of injury), primary diagnosis, and initial plan (max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "PR-4": {
            "fields": ["date", "mmi_status", "future_care"],
            "format": "PR-4 {date} = {mmi_status}; {future_care}",
            "prompt": (
                "Extract: date (MM/DD/YY), mmi_status ('MMI' or 'ongoing'), and future_care (max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Adjuster": {
            "fields": ["date", "request"],
            "format": "Adjuster {date} = {request}",
            "prompt": (
                "Extract: date (MM/DD/YY), and request (what is being requested, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Attorney": {
            "fields": ["date", "side", "topic"],
            "format": "{side} Attorney {date} = {topic}",
            "prompt": (
                "Extract: date (MM/DD/YY), side ('Applicant' or 'Defense'), and topic (main subject, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "NCM": {
            "fields": ["date", "topic"],
            "format": "NCM {date} = {topic}",
            "prompt": (
                "Extract: date (MM/DD/YY), and topic (main update or request, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Signature Request": {
            "fields": ["date", "form_type"],
            "format": "Signature req {date} = {form_type}",
            "prompt": (
                "Extract: date (MM/DD/YY), and form_type (what form requires signature, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Referral": {
            "fields": ["date", "specialty", "reason"],
            "format": "Referral {date} = {specialty} eval for {reason}",
            "prompt": (
                "Extract: date (MM/DD/YY), specialty (where patient is referred), and reason (brief, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Discharge": {
            "fields": ["date", "diagnosis", "plan"],
            "format": "Discharge {date} = {diagnosis}; {plan}",
            "prompt": (
                "Extract: date (MM/DD/YY), primary diagnosis, and discharge plan (max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Med Refill": {
            "fields": ["date", "medication"],
            "format": "Med refill {date} = {medication}",
            "prompt": (
                "Extract: date (MM/DD/YY), and medication (include name + dose, e.g., 'Ibuprofen 800mg'). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
        "Labs": {
            "fields": ["date", "key_abnormal"],
            "format": "Lab {date} = {key_abnormal}",
            "prompt": (
                "Extract: date (MM/DD/YY), and key_abnormal (most critical abnormal value, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
        },
    }

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Generic extraction for simple document types"""
        template = self.TEMPLATES.get(doc_type)
        if not template:
            logger.warning(f"⚠️ No template for {doc_type}")
            return ExtractionResult(
                document_type=doc_type,
                document_date=fallback_date,
                summary_line=f"{doc_type} {fallback_date} = manual review required",
                raw_data={},
            )

        # Build field list for JSON
        field_json = ", ".join([f'"{field}": "value"' for field in template["fields"]])

        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant, that helps doctors and medical professionals by extracting actual actionable and useful information from medical documents. Extract information from this {doc_type} document.

INSTRUCTION: {instruction}

RULES:
- Extract ONLY explicitly stated information
- Use MM/DD/YY date format
- If field not found, return empty string
- Follow word limits strictly
- Avoid vague or placeholder phrases (e.g., '+1 more', 'etc.')
- Ensure the summary is clear, concise, and fully meaningful

Document text:
{text}

Return JSON:
{{{field_json}}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "instruction", "field_json"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke(
                {
                    "text": text[:5000],
                    "doc_type": doc_type,
                    "instruction": template["prompt"],
                    "field_json": field_json,
                }
            )

            # Use fallback date if not extracted
            if "date" in result and not result["date"]:
                result["date"] = fallback_date

            # Build summary from format string
            summary = template["format"]
            for field, value in result.items():
                if value:
                    summary = summary.replace(f"{{{field}}}", value)

            # Clean up placeholders and spacing
            summary = re.sub(r"\{[^}]+\}\s*", "", summary)
            summary = re.sub(r"\s+", " ", summary).strip()

            return ExtractionResult(
                document_type=doc_type,
                document_date=result.get("date", fallback_date),
                summary_line=summary,
                raw_data=result,
            )

        except Exception as e:
            logger.error(f"❌ {doc_type} extraction failed: {e}")
            return ExtractionResult(
                document_type=doc_type,
                document_date=fallback_date,
                summary_line=f"{doc_type} {fallback_date} = extraction failed",
                raw_data={},
            )
