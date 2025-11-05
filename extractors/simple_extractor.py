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
    
    TEMPLATES = {
        "RFA": {
            "fields": ["date", "service_requested", "body_part"],
            "format": "RFA {date} = {service_requested} {body_part}",
            "prompt": "Extract: date (MM/DD/YY), service_requested (e.g., 'PT 6v', 'MRI'), body_part"
        },
        "UR": {
            "fields": ["date", "service_denied", "reason"],
            "format": "UR {date} = {service_denied} denied; {reason}",
            "prompt": "Extract: date, service_denied, reason (brief, max 6 words)"
        },
        "Authorization": {
            "fields": ["date", "service_approved", "body_part"],
            "format": "Auth {date} = {service_approved} {body_part} approved",
            "prompt": "Extract: date, service_approved, body_part"
        },
        "DFR": {
            "fields": ["date", "doi", "diagnosis", "plan"],
            "format": "DFR {date} = DOI {doi}; {diagnosis}; {plan}",
            "prompt": "Extract: report date, DOI (date of injury), primary diagnosis, initial plan (max 6 words)"
        },
        "PR-4": {
            "fields": ["date", "mmi_status", "future_care"],
            "format": "PR-4 {date} = {mmi_status}; {future_care}",
            "prompt": "Extract: date, mmi_status ('MMI' or 'ongoing'), future_care (max 6 words)"
        },
        "Adjuster": {
            "fields": ["date", "request"],
            "format": "Adjuster {date} = {request}",
            "prompt": "Extract: date, request (what is being requested, max 8 words)"
        },
        "Attorney": {
            "fields": ["date", "side", "topic"],
            "format": "{side} Attorney {date} = {topic}",
            "prompt": "Extract: date, side ('Applicant' or 'Defense'), topic (main subject, max 8 words)"
        },
        "NCM": {
            "fields": ["date", "topic"],
            "format": "NCM {date} = {topic}",
            "prompt": "Extract: date, topic (main update/request, max 8 words)"
        },
        "Signature Request": {
            "fields": ["date", "form_type"],
            "format": "Signature req {date} = {form_type}",
            "prompt": "Extract: date, form_type (what needs signature, max 5 words)"
        },
        "Referral": {
            "fields": ["date", "specialty", "reason"],
            "format": "Referral {date} = {specialty} eval for {reason}",
            "prompt": "Extract: date, specialty (where referring to), reason (brief, max 5 words)"
        },
        "Discharge": {
            "fields": ["date", "diagnosis", "plan"],
            "format": "Discharge {date} = {diagnosis}; {plan}",
            "prompt": "Extract: date, primary diagnosis, discharge plan (max 6 words)"
        },
        "Med Refill": {
            "fields": ["date", "medication"],
            "format": "Med refill {date} = {medication}",
            "prompt": "Extract: date, medication (name + dose, e.g., 'Ibuprofen 800mg')"
        },
        "Labs": {
            "fields": ["date", "key_abnormal"],
            "format": "Lab {date} = {key_abnormal}",
            "prompt": "Extract: date, key_abnormal (most critical abnormal value, max 5 words)"
        }
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
                raw_data={}
            )
        
        # Build field list for JSON
        field_json = ", ".join([f'"{field}": "value"' for field in template["fields"]])
        
        prompt = PromptTemplate(
            template="""
Extract information from this {doc_type} document.

INSTRUCTION: {instruction}

RULES:
- Extract ONLY explicitly stated information
- Use MM/DD/YY date format
- If field not found, return empty string
- Follow word limits strictly

Document text:
{text}

Return JSON:
{{{field_json}}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "instruction", "field_json"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:5000],
                "doc_type": doc_type,
                "instruction": template["prompt"],
                "field_json": field_json
            })
            
            # Use fallback date if not extracted
            if "date" in result and not result["date"]:
                result["date"] = fallback_date
            
            # Build summary from format string
            summary = template["format"]
            for field, value in result.items():
                if value:
                    summary = summary.replace(f"{{{field}}}", value)
            
            # Remove empty placeholders
            summary = re.sub(r'\{[^}]+\}\s*', '', summary)
            summary = re.sub(r'\s+', ' ', summary).strip()
            
            return ExtractionResult(
                document_type=doc_type,
                document_date=result.get("date", fallback_date),
                summary_line=summary,
                raw_data=result
            )
            
        except Exception as e:
            logger.error(f"❌ {doc_type} extraction failed: {e}")
            return ExtractionResult(
                document_type=doc_type,
                document_date=fallback_date,
                summary_line=f"{doc_type} {fallback_date} = extraction failed",
                raw_data={}
            )
