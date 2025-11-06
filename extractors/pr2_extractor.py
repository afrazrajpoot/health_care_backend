"""
PR-2 Progress Report extractor (v2.1 – updated for clarity & completeness)
"""
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class PR2Extractor:
    """Specialized extractor for PR-2 Progress Reports with enhanced clarity."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract PR-2 report data and produce concise summary card."""
        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant, that helps doctors and medical professionals by extracting actual actionable and useful information from medical documents. You are extracting concise structured data from a PR-2 (Progress Report).

EXTRACTION RULES (UPDATED):
1. Physician name MUST include title (Dr./MD/DO). Ignore electronic signatures.
2. Status: one word or short phrase (e.g., "improved", "unchanged", "worsened", "stable", or "uncertain").
3. Body part: primary area treated in this report (e.g., "R shoulder", "L knee").
4. Plan: clear next treatment step(s). Include follow-ups, referrals, or testing.
5. Treatment recommendations: include any new orders, procedures, or specific medications.
6. Work status: include phrases like "TTD", "modified duty", "return to full duty", or similar.
7. If any “?” is used (e.g., "? MMI" or "? improvement"), replace it with a brief clarification (e.g., "uncertain, pending further evaluation").
8. Output must be short, factual, and readable.

Document text:
{text}

Extract these fields:
- report_date: Date of report (MM/DD/YY or {fallback_date})
- physician_name: Treating physician with title
- body_part: Primary area addressed
- current_status: Patient’s current clinical status (max 5-10 words)
- treatment_recommendations: New or continued treatments, including medications (max 10 words)
- work_status: Work ability or restriction (max 16 words)
- next_plan: Next step / follow-up (max 16 words)

Return JSON:
{{
  "report_date": "MM/DD/YY or {fallback_date}",
  "physician_name": "Dr. Full Name or empty",
  "body_part": "Primary part or empty",
  "current_status": "Status term or empty",
  "treatment_recommendations": "Treatments or meds or empty",
  "work_status": "Work status or empty",
  "next_plan": "Follow-up plan or empty"
}}

{format_instructions}
""",
            input_variables=["text", "fallback_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({"text": text[:5000], "fallback_date": fallback_date})

            date = result.get("report_date", fallback_date)
            physician = result.get("physician_name", "").strip()
            body_part = result.get("body_part", "").strip()
            status = result.get("current_status", "").strip()
            treatment = result.get("treatment_recommendations", "").strip()
            work_status = result.get("work_status", "").strip()
            plan = result.get("next_plan", "").strip()

            # --- Build concise but meaningful summary ---
            # Example:
            # 09/12/25: PR-2 (Dr. Smith, Ortho) for R shoulder = improved; continue PT; modified duty; f/u 2 weeks
            summary_parts = []
            summary_parts.append(f"{date}: PR-2")
            if physician:
                last_name = physician.replace("Dr.", "").replace("MD", "").replace("DO", "").strip().split()[-1]
                summary_parts.append(f"(Dr {last_name})")
            if body_part:
                summary_parts.append(f"for {body_part}")
            summary_parts.append("=")

            # Status and plan components
            key_items = []
            if status:
                key_items.append(status)
            if treatment:
                key_items.append(treatment)
            if work_status:
                key_items.append(work_status)
            if plan:
                key_items.append(plan)

            summary_parts.append("; ".join(key_items))
            summary = " ".join(summary_parts)

            # Limit summary to 70 words for readability
            words = summary.split()
            if len(words) > 70:
                summary = " ".join(words[:70]) + "..."

            return ExtractionResult(
                document_type="PR-2",
                document_date=date,
                summary_line=summary,
                examiner_name=physician,
                body_parts=[body_part] if body_part else [],
                raw_data=result,
            )

        except Exception as e:
            logger.error(f"❌ PR-2 extraction failed: {e}")
            return ExtractionResult(
                document_type="PR-2",
                document_date=fallback_date,
                summary_line=f"PR-2 {fallback_date} = extraction failed",
                raw_data={},
            )
