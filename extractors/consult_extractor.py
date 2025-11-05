"""
Specialist Consult extractor
"""
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ConsultExtractor:
    """Specialized extractor for Specialist Consult Reports"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
    
    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract Consult report"""
        prompt = PromptTemplate(
            template="""
Extract key information from this specialist consultation report.

RULES:
1. Physician: MUST have Dr./MD/DO title
2. Specialty: "Ortho", "Neuro", "Pain", "PM&R", etc. (short form)
3. Plan: primary recommendation (max 8 words)

Document text:
{text}

Extract:
- consult_date: Date of consultation (MM/DD/YY or {fallback_date})
- physician_name: Consulting physician (with title)
- specialty: Medical specialty (short form)
- recommendations: Primary plan/recommendation (max 8 words)

Return JSON:
{{
  "consult_date": "date",
  "physician_name": "name",
  "specialty": "specialty",
  "recommendations": "plan"
}}

{format_instructions}
""",
            input_variables=["text", "fallback_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:5000],
                "fallback_date": fallback_date
            })
            
            date = result.get("consult_date", fallback_date)
            physician = result.get("physician_name", "")
            specialty = result.get("specialty", "")
            recommendations = result.get("recommendations", "")
            
            # Build summary: Dr [Name] [Specialty] Consult [date] = [recommendations]
            summary_parts = []
            if physician:
                summary_parts.append(physician)
            if specialty:
                summary_parts.append(specialty)
            summary_parts.append(f"Consult {date} = {recommendations}")
            
            summary = " ".join(summary_parts)
            
            return ExtractionResult(
                document_type="Consult",
                document_date=date,
                summary_line=summary,
                examiner_name=physician,
                specialty=specialty,
                raw_data=result
            )
            
        except Exception as e:
            logger.error(f"❌ Consult extraction failed: {e}")
            return ExtractionResult(
                document_type="Consult",
                document_date=fallback_date,
                summary_line=f"Consult {fallback_date} = extraction failed",
                raw_data={}
            )
