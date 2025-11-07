"""
PR-2 Progress Report extractor (v3.2 – Simplified to match ImagingExtractor structure)
"""
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class PR2Extractor:
    """Specialized extractor for PR-2 Progress Reports with deep medical analysis."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract PR-2 report data with comprehensive medical analysis."""
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        return initial_result

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data with deep medical analysis"""
        system_template = """
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

            Extract these fields:
            - report_date: Date of report (MM/DD/YY or {fallback_date})
            - physician_name: Treating physician with title (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.))
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
            """
        human_template = """
            You are analyzing this PR-2 Progress Report for structured extraction.

            Document text:
            {text}

            Fallback date: {fallback_date}

            Use the system extraction rules and formatting to identify and extract only the following fields:
            - report_date
            - physician_name (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.))
            - body_part
            - current_status
            - treatment_recommendations
            - work_status
            - next_plan

            Format the JSON output in this format:
            {{
            "report_date": "MM/DD/YY or {fallback_date}",
            "physician_name": "Dr. Full Name (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.)) or empty",
            "body_part": "Primary part or empty",
            "current_status": "Status term or empty",
            "treatment_recommendations": "Treatments or meds or empty",
            "work_status": "Work status or empty",
            "next_plan": "Follow-up plan or empty",
            "formatted_summary": "[Dr. Name] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]"

            }}

            {format_instructions}

            Return valid JSON only.
            """

        try:
            # Create system message prompt template
            system_prompt = SystemMessagePromptTemplate.from_template(
                system_template, input_variables=[]
            )

            # Create human message prompt template with partial format_instructions
            human_prompt_template = PromptTemplate(
                template=human_template,
                input_variables=["text", "fallback_date"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            human_prompt = HumanMessagePromptTemplate(prompt=human_prompt_template)

            # Build chat prompt
            chat_prompt = ChatPromptTemplate.from_messages([
                system_prompt,
                human_prompt
            ])

            # Execute extraction
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:8000],
                "fallback_date": fallback_date
            })

            return result

        except Exception as e:
            logger.error(f"❌ PR-2 raw extraction failed: {e}")
            return {}

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Stage 2: Build initial result with validation and summary"""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_pr2_summary(cleaned, doc_type, fallback_date)
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        cleaned = {}
        
        # Date validation
        date = result.get("report_date", "").strip()
        cleaned["report_date"] = date if date and date != "empty" else fallback_date

        # Physician validation
        physician = result.get("physician_name", "").strip()
        if physician and physician != "empty":
            # Ensure it contains physician indicators
            physician_upper = physician.upper()
            has_title = any(indicator in physician_upper for indicator in ['DR.', 'MD', 'DO', 'M.D.', 'D.O.', ', MD', ', DO'])
            
            if has_title:
                cleaned["physician_name"] = physician
            else:
                logger.warning(f"Physician name lacks professional title: {physician}")
                cleaned["physician_name"] = ""
        else:
            cleaned["physician_name"] = ""

        # Clean other fields
        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        current_status = result.get("current_status", "").strip()
        cleaned["current_status"] = current_status if current_status and current_status != "empty" else ""

        treatment_recommendations = result.get("treatment_recommendations", "").strip()
        cleaned["treatment_recommendations"] = treatment_recommendations if treatment_recommendations and treatment_recommendations != "empty" else ""

        work_status = result.get("work_status", "").strip()
        cleaned["work_status"] = work_status if work_status and work_status != "empty" else ""

        next_plan = result.get("next_plan", "").strip()
        cleaned["next_plan"] = next_plan if next_plan and next_plan != "empty" else ""

        critical_flags = result.get("critical_flags", "").strip()
        cleaned["critical_flags"] = critical_flags if critical_flags and critical_flags != "empty" else ""

        clinical_trajectory = result.get("clinical_trajectory", "").strip()
        cleaned["clinical_trajectory"] = clinical_trajectory if clinical_trajectory and clinical_trajectory != "empty" else ""

        return cleaned

    def _build_pr2_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build concise, human-readable summary line for PR-2 report"""
        date = data.get("report_date", fallback_date)
        physician = data.get("physician_name", "")
        body_part = data.get("body_part", "")
        trajectory = data.get("clinical_trajectory", "")
        flags = data.get("critical_flags", "")

        # Build summary parts
        summary_parts = [f"{date}: {doc_type}"]
        
        # Add physician if available
        if physician:
            last_name = (
                physician.replace("Dr.", "")
                .replace("MD", "")
                .replace("DO", "")
                .replace("M.D.", "")
                .replace("D.O.", "")
                .strip()
                .split()[-1]
            )
            if last_name:
                summary_parts.append(f"(Dr {last_name})")
        
        # Add body part if available
        if body_part:
            summary_parts.append(f"- {body_part}")
        
        # Add trajectory if available (shortened)
        if trajectory:
            trajectory_short = trajectory.split('.')[0][:50]
            summary_parts.append(f"| {trajectory_short}")
        
        # Add critical flags if present
        if flags:
            flags_short = flags.split('.')[0][:40]
            summary_parts.append(f"⚠️ {flags_short}")

        summary = " ".join(summary_parts)
        
        # Ensure summary is readable length
        if len(summary) > 150:
            summary = summary[:147] + "..."

        logger.info(f"✅ PR-2 Summary: {summary}")
        return summary