"""
QME/AME/IME specialized extractor with LLM chaining
"""
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
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
        # Stage 1: Extract raw data
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        
        # Stage 2: Build initial summary
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 3: Verify and correct format
        final_result = self.verifier.verify_and_fix(initial_result)
        
        return final_result
    
    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data"""
        # [KEEP EXACT PROMPT AND LOGIC FROM YOUR FILE - file:27]
        prompt = PromptTemplate(
            template="""
You are extracting key information from a {doc_type} report into structured fields following the DocLatch™ QME schema.

EXTRACTION RULES (STRICT):
1. Extract ONLY explicitly stated information - NO inference or assumptions
2. Focus on CONCLUSIONS, OPINIONS, and RECOMMENDATIONS sections - NOT history/narrative
3. For doctor names: MUST have title (Dr., MD, DO) - ignore signatures without titles
4. For dates: use MM/DD/YY format
5. For body parts: normalize to standard abbreviations (R/L shoulder, R/L knee, C4-6, L4-5, etc.)
6. For diagnoses: list ONLY accepted/confirmed diagnoses (max 3 most important)
7. For causation/apportionment: extract ONLY if explicitly stated with percentages
8. For recommendations: extract ONLY future treatment/care (max 10 words per field)
9. For restrictions: extract ONLY specific functional limitations (max 8 words)
10. If ANY field not found in document, return empty string for that field
11. Do NOT extract from patient history - focus on examiner's CONCLUSIONS

Document text (focusing on conclusions and recommendations):
{text}

Extract these fields with precision:
- document_date: Date of QME exam/report (MM/DD/YY format, or use {fallback_date} if not found)
- examiner_name: Full name of QME physician (MUST include Dr./MD/DO title, e.g., "Dr. Kevin Calhoun")
- specialty: QME specialty in standard form ("Orthopedic Surgery", "Neurology", "Pain Management", "Psychiatry", "Physical Medicine & Rehabilitation", etc.)
- body_parts_evaluated: ALL body parts explicitly addressed in evaluation (list with normalized abbreviations, e.g., ["R shoulder", "L knee", "cervical spine"])
- diagnoses_confirmed: Final accepted or newly confirmed diagnoses ONLY (list, max 3 most important, e.g., ["Partial rotator cuff tear", "Meniscal tear", "Lumbar strain"])
- causation_opinion: Work-related vs. non-industrial opinion with apportionment percentages if stated (e.g., "80% industrial, 20% non-industrial" or "100% industrial" or empty if not discussed)
- impairment_summary: WPI percentage or impairment rating if present (max 6 words, e.g., "8% WPI" or "WPI deferred pending treatment" or empty)
- MMI_status: Maximum Medical Improvement status - use EXACT phrase: "MMI reached" OR "MMI deferred" OR "MMI pending" OR "Ongoing treatment" (if not mentioned, return empty)
- work_restrictions: Specific functional work limitations if listed (max 10 words, e.g., "No overhead lifting >10 lb; no repetitive bending" or empty)
- future_medical_recommendations: Primary ongoing care, referrals, or future medical needs (max 12 words, e.g., "Continue PT; pain management follow-up; PRN ortho consult" or empty)
- treatment_recommendations: Immediate or new treatment orders (max 10 words, e.g., "Resume PT 6 visits; consider ESI if no improvement" or empty)
- follow_up_instructions: Next steps or follow-up interval if specified (max 8 words, e.g., "Re-eval in 3 months" or "Follow-up QME in 1 year" or empty)
- attorney_or_adjuster_notes: Any references to panel requests, disputes, or legal communications (max 10 words or empty, e.g., "Panel requested by applicant attorney")

Return JSON with ALL fields (use empty string "" for any field not found):
{{
  "document_date": "MM/DD/YY or {fallback_date}",
  "examiner_name": "Dr. Full Name or empty",
  "specialty": "Full specialty name or empty",
  "body_parts_evaluated": ["part1", "part2"] or [],
  "diagnoses_confirmed": ["diagnosis1", "diagnosis2"] or [],
  "causation_opinion": "percentage breakdown or empty",
  "impairment_summary": "WPI or impairment note or empty",
  "MMI_status": "exact status phrase or empty",
  "work_restrictions": "restrictions or empty",
  "future_medical_recommendations": "ongoing care or empty",
  "treatment_recommendations": "immediate treatments or empty",
  "follow_up_instructions": "follow-up plan or empty",
  "attorney_or_adjuster_notes": "legal notes or empty"
}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "fallback_date"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:8000],
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            return result
        except Exception as e:
            logger.error(f"❌ Raw extraction failed: {e}")
            return {}
    
    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Stage 2: Build initial result from raw data"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_qme_summary(cleaned, doc_type, fallback_date)
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("document_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("examiner_name"),
            specialty=cleaned.get("specialty"),
            body_parts=cleaned.get("body_parts_evaluated", []),
            raw_data=cleaned
        )
    
    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate extracted data and clean empty/invalid values"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        cleaned = {}
        
        # Date validation
        date = result.get("document_date", "").strip()
        cleaned["document_date"] = date if date and date != "empty" else fallback_date
        
        # Examiner name validation (must have title)
        examiner = result.get("examiner_name", "").strip()
        if examiner and examiner != "empty":
            if any(title in examiner for title in ["Dr.", "Dr ", "MD", "DO", "M.D.", "D.O."]):
                cleaned["examiner_name"] = examiner
            else:
                cleaned["examiner_name"] = ""
        else:
            cleaned["examiner_name"] = ""
        
        # Specialty
        specialty = result.get("specialty", "").strip()
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""
        
        # Body parts (list)
        body_parts = result.get("body_parts_evaluated", [])
        if isinstance(body_parts, list):
            cleaned["body_parts_evaluated"] = [bp.strip() for bp in body_parts if bp and bp != "empty"]
        else:
            cleaned["body_parts_evaluated"] = []
        
        # Diagnoses (list)
        diagnoses = result.get("diagnoses_confirmed", [])
        if isinstance(diagnoses, list):
            cleaned["diagnoses_confirmed"] = [dx.strip() for dx in diagnoses if dx and dx != "empty"]
        else:
            cleaned["diagnoses_confirmed"] = []
        
        # String fields
        string_fields = [
            "causation_opinion", "impairment_summary", "MMI_status", "work_restrictions",
            "future_medical_recommendations", "treatment_recommendations",
            "follow_up_instructions", "attorney_or_adjuster_notes"
        ]
        
        for field in string_fields:
            value = result.get(field, "").strip()
            if value and value.lower() not in ["empty", "", "none", "n/a", "not mentioned", "not stated", "not found"]:
                cleaned[field] = value
            else:
                cleaned[field] = ""
        
        return cleaned
    
    def _build_qme_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build summary"""
        # [KEEP EXACT LOGIC FROM YOUR FILE - file:27]
        date = data.get("document_date", fallback_date)
        examiner = data.get("examiner_name", "")
        specialty = data.get("specialty", "")
        body_parts = data.get("body_parts_evaluated", [])
        mmi = data.get("MMI_status", "")
        impairment = data.get("impairment_summary", "")
        causation = data.get("causation_opinion", "")
        restrictions = data.get("work_restrictions", "")
        future_med = data.get("future_medical_recommendations", "")
        treatment = data.get("treatment_recommendations", "")
        
        parts = []
        parts.append(f"{date}: {doc_type}")
        
        if examiner:
            name_parts = examiner.replace("Dr.", "").replace("MD", "").replace("DO", "").strip().split()
            last_name = name_parts[-1] if name_parts else ""
            if last_name:
                if specialty:
                    specialty_short = self._abbreviate_specialty(specialty)
                    parts.append(f"(Dr {last_name}, {specialty_short})")
                else:
                    parts.append(f"(Dr {last_name})")
        
        if body_parts:
            body_str = " + ".join(body_parts[:2])
            if len(body_parts) > 2:
                body_str += f" + {len(body_parts) - 2} more"
            parts.append(f"for {body_str}")
        
        findings = []
        if mmi:
            findings.append(mmi)
        if impairment:
            findings.append(impairment)
        
        if findings:
            parts.append(f"= {'; '.join(findings)}")
        
        recommendations = []
        if treatment:
            recommendations.append(treatment)
        if future_med:
            recommendations.append(future_med)
        
        if recommendations:
            rec_str = "; ".join(recommendations)
            parts.append(f"→ {rec_str}")
        
        additional = []
        if causation:
            additional.append(causation)
        elif restrictions:
            additional.append(restrictions)
        
        if additional:
            parts.append(f"; {additional[0]}")
        
        summary = " ".join(parts)
        
        # Enforce word limit
        words = summary.split()
        if len(words) > 35:
            summary = " ".join(words[:35]) + "..."
        
        return summary
    
    def _abbreviate_specialty(self, specialty: str) -> str:
        """Convert specialty to short form"""
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
            "Occupational Medicine": "Occ Med"
        }
        return abbreviations.get(specialty, specialty[:10])
