"""
Specialist Consult extractor (v2.1 – enhanced for clarity and consistency)
"""
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
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
        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant, that helps doctors and medical professionals by extracting actual actionable and useful information from medical documents. You are extracting structured clinical information from a specialist consultation report
to generate a concise, readable summary for a medical timeline card.

EXTRACTION RULES:
1. Physician name: MUST include title (Dr./MD/DO). Ignore electronic signatures.
2. Specialty: Convert to short form if possible ("Orthopedic Surgery" → "Ortho", "Neurology" → "Neuro", "Pain Management" → "Pain").
3. Body part: Extract if specified (use short form: "R shoulder", "L knee", etc.).
4. Findings: Summarize key diagnostic impression (max 16 words, e.g., "partial rotator cuff tear", "lumbar disc bulge").
5. Recommendations: Include specific next steps or treatment plan (max 16 words, including referrals or follow-ups).
6. Treatment recommendations: Include medications, therapies, or procedures explicitly recommended.
7. Work status/restrictions: Include if mentioned (e.g., "modified duty", "TTD", "full duty").
8. If any “?” uncertainty exists (e.g., "? impingement"), replace with “possible [finding]”.
9. Maintain concise, human-readable phrasing — suitable for compact UI display.

Document text:
{text}

Extract these fields:
- consult_date: Date of consultation (MM/DD/YY or {fallback_date})
- physician_name: Consulting physician (with title)
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

{format_instructions}
""",
            input_variables=["text", "fallback_date"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke(
                {"text": text[:5000], "fallback_date": fallback_date}
            )

            date = result.get("consult_date", fallback_date).strip()
            physician = result.get("physician_name", "").strip()
            specialty = result.get("specialty", "").strip()
            body_part = result.get("body_part", "").strip()
            findings = result.get("findings", "").strip()
            recommendations = result.get("recommendations", "").strip()
            treatment = result.get("treatment_recommendations", "").strip()
            work_status = result.get("work_status", "").strip()

            # --- Build concise summary ---
            # Example: 20/04/25: Consult (Dr Patel, Ortho) for R shoulder = partial tear; PT + NSAIDs; modified duty
            summary_parts = []
            summary_parts.append(f"{date}: Consult")

            if physician:
                last_name = (
                    physician.replace("Dr.", "")
                    .replace("MD", "")
                    .replace("DO", "")
                    .strip()
                    .split()[-1]
                )
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

            return ExtractionResult(
                document_type="Consult",
                document_date=date,
                summary_line=summary,
                examiner_name=physician,
                specialty=specialty,
                body_parts=[body_part] if body_part else [],
                raw_data=result,
            )

        except Exception as e:
            logger.error(f"❌ Consult extraction failed: {e}")
            return ExtractionResult(
                document_type="Consult",
                document_date=fallback_date,
                summary_line=f"Consult {fallback_date} = extraction failed",
                raw_data={},
            )
