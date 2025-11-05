"""
Imaging reports extractor (MRI, CT, X-ray, Ultrasound, EMG)
"""
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ImagingExtractor:
    """Specialized extractor for MRI/CT/X-ray/Ultrasound/EMG"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
    
    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract imaging report"""
        prompt = PromptTemplate(
            template="""
Extract key information from this {doc_type} report.

RULES:
1. Extract ONLY the PRIMARY finding (not all findings)
2. Body part: use abbreviations (R/L shoulder, C4-6, etc.)
3. Date: MM/DD/YY format
4. Finding: max 6 words, focus on diagnosis/pathology

Document text:
{text}

Extract:
- study_date: Date of imaging (MM/DD/YY or {fallback_date})
- body_part: Anatomical area studied
- primary_finding: Primary diagnosis/finding (max 6 words)

Return JSON:
{{
  "study_date": "date",
  "body_part": "part",
  "primary_finding": "finding"
}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "fallback_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:5000],
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            
            date = result.get("study_date", fallback_date)
            body_part = result.get("body_part", "")
            finding = result.get("primary_finding", "")
            
            summary = f"{doc_type} {body_part} {date} = {finding}"
            
            return ExtractionResult(
                document_type=doc_type,
                document_date=date,
                summary_line=summary,
                body_parts=[body_part] if body_part else [],
                raw_data=result
            )
            
        except Exception as e:
            logger.error(f"❌ Imaging extraction failed: {e}")
            return ExtractionResult(
                document_type=doc_type,
                document_date=fallback_date,
                summary_line=f"{doc_type} {fallback_date} = extraction failed",
                raw_data={}
            )
