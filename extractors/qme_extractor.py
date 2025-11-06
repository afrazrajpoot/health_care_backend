"""
QME/AME/IME specialized extractor with LLM chaining
"""
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier

logger = logging.getLogger("document_ai")


class QMEExtractorChained:
    """
    Enhanced QME extractor with multi-stage LLM chaining:
    Stage 1: Extract raw data
    Stage 2: Build summary
    Stage 3: Verify and correct
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract with verification chain"""
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        final_result = self.verifier.verify_and_fix(initial_result)
        return final_result

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data using system and user prompts"""
        
        system_prompt = """You are a specialized medical document analyst that extracts key information from Qualified Medical Evaluator (QME/AME/IME) reports.

CRITICAL RULES FOR EXAMINER IDENTIFICATION:
1. ONLY extract the consulting/examining physician who:
   - Actually performed the evaluation/examination
   - Authored/wrote the report
   - Signed the report as the primary examiner
2. IGNORE all other names mentioned in the report:
   - Referring doctors
   - Treating physicians
   - Other consultants
   - Medical staff (nurses, assistants)
   - Administrative personnel
3. Look for explicit signatures, "Respectfully submitted by", "Examined by", or similar concluding phrases
4. The examiner name MUST include appropriate title (Dr./MD/DO) and be the full name

DOCUMENT ANALYSIS APPROACH:
- Focus on factual, clinical information actually stated in the report
- Extract explicit functional restrictions, not generic "modified duty"
- Note diagnostic uncertainties marked with "?" with brief context
- Be concise and use standard medical abbreviations
- Output structured JSON according to the specified schema"""

        user_prompt = """
ANALYZE THIS QME/AME REPORT:

DOCUMENT TEXT:
{text}

EXTRACTION INSTRUCTIONS:
Extract the following fields with precision:

- document_date: Date of exam/report (MM/DD/YY format, or use {fallback_date} if not found)
- examiner_name: ONLY the consulting/examining physician who wrote and signed this report (must include Dr./MD/DO title) - USE FULL NAME, DO NOT ABBREVIATE
- specialty: Physician specialty (e.g., Orthopedic Surgery, Neurology, Pain Management)
- body_parts_evaluated: All body parts evaluated (e.g., ["R shoulder", "L knee"])
- diagnoses_confirmed: Key confirmed diagnoses (up to 3 most important)
- MMI_status: MMI reached/deferred/pending/ongoing
- impairment_summary: WPI or impairment rating if stated
- causation_opinion: Apportionment or causation percentages if applicable
- treatment_recommendations: Specific treatments or procedures recommended (max 12 words)
- medication_recommendations: Specific medications or drug names recommended (max 10 words)
- work_restrictions: Explicit functional restrictions (max 12 words)
- future_medical_recommendations: Follow-up or future care recommendations (max 12 words)

WORK RESTRICTIONS RULE:
- DO NOT summarize as "modified work" or "restricted duty"
- ALWAYS extract the actual functional restrictions
- If restrictions not explicit, infer from context (e.g., "avoid overhead lifting" for shoulder pain)

UNCERTAINTY HANDLING:
- For "?" notations, add brief context: "? rotator cuff tear" → "? rotator cuff tear (diagnosis uncertain)"

Return valid JSON:
{{
  "document_date": "MM/DD/YY or {fallback_date}",
  "examiner_name": "Dr. Full Name or empty",
  "specialty": "Specialty or empty",
  "body_parts_evaluated": ["part1", "part2", ...],
  "diagnoses_confirmed": ["diagnosis1", "diagnosis2", ...],
  "MMI_status": "status phrase or empty",
  "impairment_summary": "WPI or impairment note or empty",
  "causation_opinion": "apportionment or empty",
  "treatment_recommendations": "procedures or therapies or empty",
  "medication_recommendations": "medications or empty",
  "work_restrictions": "explicit restrictions or empty",
  "future_medical_recommendations": "ongoing/follow-up plan or empty"
}}

{format_instructions}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:8000], 
                "fallback_date": fallback_date,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            logger.error(f"❌ Raw extraction failed: {e}")
            return {}

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_qme_summary(cleaned, doc_type, fallback_date)
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("document_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("examiner_name"),
            specialty=cleaned.get("specialty"),
            body_parts=cleaned.get("body_parts_evaluated", []),
            raw_data=cleaned,
        )

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        cleaned = {}
        date = result.get("document_date", "").strip()
        cleaned["document_date"] = date if date and date != "empty" else fallback_date

        # Enhanced examiner name validation
        examiner = result.get("examiner_name", "").strip()
        if examiner and any(t in examiner for t in ["Dr.", "MD", "DO"]):
            # Additional check to ensure it's not just a mentioned doctor
            if not any(term in examiner.lower() for term in ["refer", "treat", "primary", "family", "emergency"]):
                cleaned["examiner_name"] = examiner
            else:
                cleaned["examiner_name"] = ""
        else:
            cleaned["examiner_name"] = ""

        specialty = result.get("specialty", "").strip()
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""

        body_parts = result.get("body_parts_evaluated", [])
        cleaned["body_parts_evaluated"] = [bp.strip() for bp in body_parts if bp and bp != "empty"]

        diagnoses = result.get("diagnoses_confirmed", [])
        cleaned["diagnoses_confirmed"] = [dx.strip() for dx in diagnoses if dx and dx != "empty"]

        string_fields = [
            "causation_opinion",
            "impairment_summary",
            "MMI_status",
            "work_status",
            "work_restrictions",
            "future_medical_recommendations",
            "treatment_recommendations",
            "follow_up_instructions",
            "attorney_or_adjuster_notes",
        ]
        for f in string_fields:
            v = result.get(f, "").strip()
            cleaned[f] = v if v and v.lower() not in ["", "empty", "none", "n/a", "not mentioned"] else ""

        return cleaned

    def _build_qme_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        date = data.get("document_date", fallback_date)
        examiner = data.get("examiner_name", "")
        specialty = data.get("specialty", "")
        body_parts = data.get("body_parts_evaluated", [])
        mmi = data.get("MMI_status", "")
        impairment = data.get("impairment_summary", "")
        causation = data.get("causation_opinion", "")
        work_status = data.get("work_status", "")
        restrictions = data.get("work_restrictions", "")
        future_med = data.get("future_medical_recommendations", "")
        treatment = data.get("treatment_recommendations", "")

        parts = [f"{date}: {doc_type}"]

        if examiner:
            # Use full examiner name without abbreviating
            specialty_abbr = self._abbreviate_specialty(specialty) if specialty else 'QME'
            parts.append(f"({examiner}, {specialty_abbr})")

        if body_parts:
            body_str = ", ".join(body_parts[:3])
            parts.append(f"for {body_str}")

        findings = []
        if mmi:
            findings.append(mmi)
        if impairment:
            findings.append(impairment)
        if work_status:
            findings.append(work_status)

        if findings:
            parts.append(f"= {'; '.join(findings)}")

        recommendations = []
        if treatment:
            recommendations.append(treatment)
        if future_med:
            recommendations.append(future_med)

        if recommendations:
            parts.append(f"→ {'; '.join(recommendations)}")

        if restrictions:
            parts.append(f"; {restrictions}")
        elif causation:
            parts.append(f"; {causation}")

        summary = " ".join(parts)
        words = summary.split()
        if len(words) > 70:
            summary = " ".join(words[:70]) + "..."

        return summary

    def _abbreviate_specialty(self, specialty: str) -> str:
        abbreviations = {
            "Orthopedic Surgery": "Ortho",
            "Orthopedics": "Ortho",
            "Neurology": "Neuro",
            "Pain Management": "Pain",
            "Psychiatry": "Psych",
            "Psychology": "Psych",
            "Physical Medicine & Rehabilitation": "PM&R",
            "Physical Medicine and Rehabilitation": "PM&R",
            "Internal Medicine": "IM",
            "Occupational Medicine": "Occ Med",
        }
        return abbreviations.get(specialty, specialty[:10])