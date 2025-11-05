"""
PR-2 Progress Report extractor
"""
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class PR2Extractor:
    """Specialized extractor for PR-2 Progress Reports"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
    
    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract PR-2 report"""
        prompt = PromptTemplate(
            template="""
Extract key information from this PR-2 Progress Report.

RULES:
1. Physician name: MUST have Dr./MD/DO title (ignore signatures)
2. Status: "improved", "unchanged", "worsened" (one word)
3. Plan: immediate next action (max 8 words)

Document text:
{text}

Extract:
- report_date: Date of report (MM/DD/YY or {fallback_date})
- physician_name: Primary treating physician (with title)
- body_part: Primary body part being treated
- current_status: Current status (1-2 words)
- next_plan: Next treatment action (max 8 words)

Return JSON:
{{
  "report_date": "date",
  "physician_name": "name",
  "body_part": "part",
  "current_status": "status",
  "next_plan": "plan"
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
            
            date = result.get("report_date", fallback_date)
            physician = result.get("physician_name", "")
            body_part = result.get("body_part", "")
            status = result.get("current_status", "")
            plan = result.get("next_plan", "")
            
            # Build summary: Dr [Name] PR-2 [date] [body part] = [status]; [plan]
            summary_parts = []
            if physician:
                summary_parts.append(physician)
            summary_parts.append(f"PR-2 {date}")
            if body_part:
                summary_parts.append(body_part)
            summary_parts.append("=")
            if status:
                summary_parts.append(f"{status};")
            if plan:
                summary_parts.append(plan)
            
            summary = " ".join(summary_parts)
            
            return ExtractionResult(
                document_type="PR-2",
                document_date=date,
                summary_line=summary,
                examiner_name=physician,
                body_parts=[body_part] if body_part else [],
                raw_data=result
            )
            
        except Exception as e:
            logger.error(f"❌ PR-2 extraction failed: {e}")
            return ExtractionResult(
                document_type="PR-2",
                document_date=fallback_date,
                summary_line=f"PR-2 {fallback_date} = extraction failed",
                raw_data={}
            )
