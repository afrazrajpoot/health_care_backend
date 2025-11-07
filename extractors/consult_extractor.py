"""
Specialist Consult extractor (v2.2 – refactored to match ImagingExtractor layout)
"""
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ConsultExtractor:
    """Specialized extractor for Specialist Consultation Reports with improved clarity and completeness."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract specialist consult report and generate concise summary."""
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        return initial_result

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data using system and human templates"""
        system_template = """
You are an AI Medical Assistant, that helps doctors and medical professionals by extracting actual actionable and useful information from medical documents. You are extracting structured clinical information from a specialist consultation report
to generate a concise, readable summary for a medical timeline card.

━━━ STAGE : DOCTOR/Physician name EXTRACTION (CRITICAL VALIDATION) ━━━
CONSULTING DOCTOR/Physician name EXTRACTION GUIDELINES:
- MUST have explicit title: "Dr.", "MD", "DO", "M.D.", "D.O." (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.))
- Look in: signatures, consultations, specialist mentions
- Extract FULL NAME with title (e.g., "Dr. Jane Smith")
- IF name found WITHOUT title → ADD to verification_notes: "Doctor name lacks title: [name]"
- Do NOT extract patient names, admin names, signature without context
- If no consultant → "Not specified"

EXTRACTION RULES:
1. Physician name: MUST include title (Dr./MD/DO) (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.)). Ignore electronic signatures.
2. Specialty: Convert to short form if possible ("Orthopedic Surgery" → "Ortho", "Neurology" → "Neuro", "Pain Management" → "Pain").
3. Body part: Extract if specified (use short form: "R shoulder", "L knee", etc.).
4. Findings: Summarize key diagnostic impression (max 16 words, e.g., "partial rotator cuff tear", "lumbar disc bulge").
5. Recommendations: Include specific next steps or treatment plan (max 16 words, including referrals or follow-ups).
6. Treatment recommendations: Include medications, therapies, or procedures explicitly recommended.
7. Work status/restrictions: Include if mentioned (e.g., "modified duty", "TTD", "full duty").
8. If any “?” uncertainty exists (e.g., "? impingement"), replace with “possible [finding]”.
9. Maintain concise, human-readable phrasing — suitable for compact UI display.

Extract these fields:
- consult_date: Date of consultation (MM/DD/YY or {fallback_date})
- physician_name: Consulting physician (with title) (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.))
- specialty: Medical specialty (short form)
- body_part: Body part(s) evaluated (abbreviated)
- findings: Primary impression or diagnosis (max 16 words)
- recommendations: Overall treatment or follow-up plan (max 16 words)
- treatment_recommendations: Specific treatments or meds (max 20 words)
- work_status: Work ability or restrictions (max 16 words)

Return JSON:
{{
  "consult_date": "MM/DD/YY or {fallback_date}",
  "physician_name": "Dr. Full Name or empty",
  "specialty": "short form or empty",
  "body_part": "abbreviated or empty",
  "findings": "key finding or empty",
  "recommendations": "primary plan or empty",
  "treatment_recommendations": "treatments/meds or empty",
  "work_status": "work status or empty"
}}
"""

        human_template = """
You are analyzing this specialist consultation report for structured extraction.

Document text:
{text}

Fallback date: {fallback_date}

Follow the extraction and validation rules from the system prompt above.
Return only valid JSON with the fields:
- consult_date
- physician_name
- specialty
- body_part
- findings
- recommendations
- treatment_recommendations
- work_status

{format_instructions}

Return JSON only.
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

            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke(
                {"text": text[:5000], "fallback_date": fallback_date}
            )
            return result
        except Exception as e:
            logger.error(f"❌ Raw extraction failed: {e}")
            return {}

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Stage 2: Build initial result with validation and summary"""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_consult_summary(cleaned, doc_type, fallback_date)
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("consult_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name"),
            specialty=cleaned.get("specialty"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        cleaned = {}
        date = result.get("consult_date", "").strip()
        cleaned["consult_date"] = date if date and date != "empty" else fallback_date

        # Validate physician name has title
        physician = result.get("physician_name", "").strip()
        if physician and not any(title in physician for title in ["Dr.", "MD", "DO", "M.D.", "D.O."]):
            physician = ""
            logger.warning("Physician name lacked required title; cleared.")
        cleaned["physician_name"] = physician

        specialty = result.get("specialty", "").strip()
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""

        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        findings = result.get("findings", "").strip()
        cleaned["findings"] = findings if findings and findings != "empty" else ""

        recommendations = result.get("recommendations", "").strip()
        cleaned["recommendations"] = recommendations if recommendations and recommendations != "empty" else ""

        treatment_recommendations = result.get("treatment_recommendations", "").strip()
        cleaned["treatment_recommendations"] = treatment_recommendations if treatment_recommendations and treatment_recommendations != "empty" else ""

        work_status = result.get("work_status", "").strip()
        cleaned["work_status"] = work_status if work_status and work_status != "empty" else ""

        return cleaned

    def _build_consult_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build concise, human-readable summary line"""
        date = data.get("consult_date", fallback_date)
        physician = data.get("physician_name", "")
        specialty = data.get("specialty", "")
        body_part = data.get("body_part", "")
        findings = data.get("findings", "")
        recommendations = data.get("recommendations", "")
        treatment = data.get("treatment_recommendations", "")
        work_status = data.get("work_status", "")

        # Build concise summary
        # Example: 20/04/25: Consult (Dr Patel, Ortho) for R shoulder = partial tear; PT + NSAIDs; modified duty
        summary_parts = []
        summary_parts.append(f"{date}: Consult")

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
                if specialty:
                    summary_parts.append(f"(Dr {last_name}, {specialty})")
                else:
                    summary_parts.append(f"(Dr {last_name})")

        if body_part:
            summary_parts.append(f"for {body_part}")

        summary_parts.append("=")

        key_phrases = []
        if findings:
            key_phrases.append(findings)
        if treatment:
            key_phrases.append(treatment)
        if recommendations:
            key_phrases.append(recommendations)
        if work_status:
            key_phrases.append(work_status)

        summary_parts.append("; ".join(key_phrases))
        summary = " ".join(summary_parts)

        # Limit to ~70 words for UI brevity
        words = summary.split()
        if len(words) > 70:
            summary = " ".join(words[:70]) + "..."

        return summary