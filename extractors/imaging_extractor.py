"""
Imaging reports extractor (MRI, CT, X-ray, Ultrasound, EMG)
v2.1 – enhanced clarity and precision for concise summaries
"""
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ImagingExtractor:
    """Specialized extractor for MRI/CT/X-ray/Ultrasound/EMG reports with improved contextual clarity."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract imaging report and generate concise, clinically meaningful summary."""
        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant, that helps doctors and medical professionals by extracting actual actionable and useful information from medical documents. You are extracting key clinical data from an imaging report ({doc_type}) to create a concise, clear summary line.

EXTRACTION RULES:
1. Focus ONLY on the primary diagnostic finding (most clinically significant).
2. If multiple findings exist, select the one with highest diagnostic importance.
3. If normal study → output "normal study" or "no acute findings".
4. If uncertain or possible finding (marked with “?”), rewrite as “possible [finding]”.
5. Body part: concise format (e.g., "R shoulder", "L knee", "C4-6", "L-spine").
6. Date: MM/DD/YY format.
7. For MRI/CT, indicate if with or without contrast when explicitly stated.
8. Finding: brief but complete (max 16 words) — avoid general terms like "abnormal" alone.
9. Do not include technical details (e.g., sequences, imaging parameters).
10. The summary should be easily readable on a compact card.

Document text:
{text}

Extract these fields:
- study_date: Imaging date (MM/DD/YY or {fallback_date})
- body_part: Anatomical area studied (abbreviated form)
- contrast_used: "with contrast", "without contrast", or empty if not mentioned
- primary_finding: Most important diagnostic finding (max 16 words)
- impression_status: "normal", "abnormal", "post-op", or "inconclusive" if applicable

Return JSON:
{{
  "study_date": "MM/DD/YY or {fallback_date}",
  "body_part": "abbreviated part or empty",
  "contrast_used": "contrast detail or empty",
  "primary_finding": "main finding (max 16 words)",
  "impression_status": "normal/abnormal/post-op/inconclusive"
}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "fallback_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke(
                {"text": text[:5000], "doc_type": doc_type, "fallback_date": fallback_date}
            )

            date = result.get("study_date", fallback_date).strip()
            body_part = result.get("body_part", "").strip()
            finding = result.get("primary_finding", "").strip()
            contrast = result.get("contrast_used", "").strip()
            status = result.get("impression_status", "").strip()

            # --- Build concise, human-readable summary ---
            # Example:
            # MRI R shoulder 09/12/25 = partial rotator cuff tear
            # CT L ankle 07/18/25 = post-op changes, no acute findings
            summary_parts = []

            # Imaging type and body part
            summary_parts.append(f"{doc_type}")
            if body_part:
                summary_parts.append(body_part)
            summary_parts.append(date)

            # Build findings segment
            findings_list = []
            if finding:
                findings_list.append(finding)
            elif status.lower() == "normal":
                findings_list.append("normal study")
            elif status.lower() == "post-op":
                findings_list.append("post-op changes")
            elif status.lower() == "inconclusive":
                findings_list.append("inconclusive findings")

            # Add contrast info if useful
            if contrast and contrast.lower() in ["with contrast", "without contrast"]:
                findings_list.append(f"({contrast})")

            # Join all parts
            finding_str = ", ".join(findings_list) if findings_list else "no significant abnormality"
            summary = f"{' '.join(summary_parts)} = {finding_str}".strip()

            # Limit to ~70 words for visual card brevity
            words = summary.split()
            if len(words) > 70:
                summary = " ".join(words[:70]) + "..."

            return ExtractionResult(
                document_type=doc_type,
                document_date=date,
                summary_line=summary,
                body_parts=[body_part] if body_part else [],
                raw_data=result,
            )

        except Exception as e:
            logger.error(f"❌ Imaging extraction failed: {e}")
            return ExtractionResult(
                document_type=doc_type,
                document_date=fallback_date,
                summary_line=f"{doc_type} {fallback_date} = extraction failed",
                raw_data={},
            )
