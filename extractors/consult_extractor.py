"""
Specialist Consult extractor with DoctorDetector integration.
Consistent doctor detection flow and labeled summary format.
"""
import logging
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class ConsultExtractorChained:
    """
    Enhanced Consult extractor with DoctorDetector integration:
    - Stage 1: Extract structured clinical data (NO doctor extraction)
    - Stage 2: Doctor detection via DoctorDetector (zone-aware)
    - Stage 3: Build initial result
    - Stage 4: Verify and correct
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        logger.info("✅ ConsultExtractorChained initialized with DoctorDetector")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract with DoctorDetector integration.
        
        Args:
            text: Layout-preserved text from Document AI
            doc_type: Document type (CONSULT)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            raw_text: Original flat text (for backward compatibility)
        """
        # Stage 1: Extract clinical data (NO doctor extraction in prompt)
        raw_result = self._extract_clinical_data(text, doc_type, fallback_date)
        
        # Stage 2: Doctor detection via DoctorDetector (zone-aware)
        consulting_physician = self._detect_consultant(text, page_zones)
        raw_result["physician_name"] = consulting_physician
        
        # Stage 3: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        verified_result = self.verifier.verify_and_fix(initial_result)
        
        return verified_result

    def _extract_clinical_data(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Stage 1: Extract clinical data ONLY (NO doctor extraction).
        Doctor extraction is handled by DoctorDetector.
        """
        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant extracting structured clinical data from a specialist consultation report.

EXTRACTION RULES:
1. Focus on CLINICAL DATA ONLY (specialty, findings, recommendations, treatment).
2. DO NOT extract consultant/doctor names - this is handled separately.
3. Specialty: Convert to short form (e.g., "Orthopedic Surgery" → "Ortho", "Neurology" → "Neuro", "Pain Management" → "Pain").
4. Body part: Use abbreviated form (e.g., "R shoulder", "L knee").
5. Findings: Summarize key diagnostic impression (max 16 words).
6. Recommendations: Include specific next steps or treatment plan (max 16 words).
7. Treatment recommendations: Medications, therapies, or procedures (max 16 words).
8. Work status: Include if mentioned (e.g., "modified duty", "TTD", "full duty").
9. If "?" uncertainty exists, replace with "possible [finding]".
10. Output must be concise and readable.

Document text:
{text}

Extract these fields (do NOT extract physician_name):
- consult_date: Date of consultation (MM/DD/YY format, or use {fallback_date} if not found)
- specialty: Medical specialty (short form)
- body_part: Body part(s) evaluated (abbreviated)
- findings: Primary impression or diagnosis (max 16 words)
- recommendations: Treatment plan or follow-up (max 16 words)
- treatment_recommendations: Specific treatments or medications (max 16 words)
- work_status: Work ability or restrictions (max 16 words)

CRITICAL REASONING RULES - VERIFY BEFORE RETURNING:
1. ONLY extract POSITIVE/ACTIONABLE findings. DO NOT extract negative statements.
   ✗ BAD: "No significant pathology", "No recommendations", "Cleared for full duty"
   ✓ GOOD: "AC joint arthritis", "ESI recommended", "Modified duty - no lifting >20 lbs"
   
2. If a field has NO meaningful positive data, return empty string "" - DO NOT return negative phrases.
   
3. For findings: ONLY return if there's an actual diagnosis or significant clinical finding.
   ✗ BAD: "Normal exam", "No acute findings"
   ✓ GOOD: "Rotator cuff tendinopathy", "L5-S1 radiculopathy"
   
4. For work_status: ONLY return if there are actual restrictions or specific status.
   ✗ BAD: "Full duty", "No restrictions"
   ✓ GOOD: "TTD", "Modified duty - sedentary only"

5. REASONING CHECK: Before returning each field, ask yourself:
   - "Is this information ACTIONABLE for the treating physician?"
   - "Does this tell me what TO DO or what IS present?"
   - If answer is NO → return empty string for that field

Return JSON:
{{
  "consult_date": "MM/DD/YY or {fallback_date}",
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
            result = chain.invoke({
                "text": text[:8000],
                "fallback_date": fallback_date,
            })
            logger.info("✅ Stage 1: Clinical data extraction complete")
            return result
        except Exception as e:
            logger.error(f"❌ Consult clinical data extraction failed: {e}")
            return {}

    def _detect_consultant(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """
        Stage 2: Detect consulting specialist using DoctorDetector (zone-aware).
        
        Args:
            text: Full document text
            page_zones: Page zones for zone-aware detection
        
        Returns:
            Consultant name with title, or empty string
        """
        logger.info("🔍 Stage 2: Running DoctorDetector for consultant (zone-aware)...")
        
        if page_zones:
            logger.info(f"✅ Consult extractor received page_zones with {len(page_zones)} pages: {list(page_zones.keys())}")
        else:
            logger.warning("⚠️ Consult extractor did NOT receive page_zones")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(
                f"✅ Consultant detected: {detection_result['doctor_name']} "
                f"(confidence: {detection_result['confidence']}, "
                f"source: {detection_result['source']})"
            )
            return detection_result["doctor_name"]
        else:
            logger.warning(
                f"⚠️ No valid consultant found: {detection_result['validation_notes']}"
            )
            return ""

    def _build_initial_result(
        self,
        raw_data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> ExtractionResult:
        """Stage 3: Build initial result with validation and summary"""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_consult_summary(cleaned, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("consult_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name", ""),
            specialty=cleaned.get("specialty"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )
        
        logger.info(f"✅ Stage 3: Initial result built (consultant: {result.examiner_name})")
        return result

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        cleaned = {}
        
        # Date validation
        date = result.get("consult_date", "").strip()
        cleaned["consult_date"] = date if date and date != "empty" else fallback_date

        # Physician (from DoctorDetector - already validated)
        physician = result.get("physician_name", "").strip()
        cleaned["physician_name"] = physician

        # Specialty validation
        specialty = result.get("specialty", "").strip()
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""

        # Body part validation
        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        # String fields validation
        string_fields = [
            "findings",
            "recommendations",
            "treatment_recommendations",
            "work_status",
        ]
        for f in string_fields:
            v = result.get(f, "").strip()
            cleaned[f] = v if v and v.lower() not in [
                "", "empty", "none", "n/a", "not mentioned", "not specified"
            ] else ""

        return cleaned

    def _build_consult_summary(
        self,
        data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Build concise, physician-friendly summary for Consult report.
        Uses explicit field labels for clarity (matching PR2 format).
        Enforces 35-65 word range.
        """
        date = data.get("consult_date", fallback_date)
        physician = data.get("physician_name", "")
        specialty = data.get("specialty", "")
        body_part = data.get("body_part", "")
        findings = data.get("findings", "")
        recommendations = data.get("recommendations", "")
        treatment = data.get("treatment_recommendations", "")
        work_status = data.get("work_status", "")

        # Doctor section (only if valid)
        doctor_section = ""
        if physician:
            if specialty:
                doctor_section = f" by {physician} ({specialty})"
            else:
                doctor_section = f" by {physician}"

        # Compose enhanced summary prompt with explicit labeling rules
        summary_prompt = f"""You are an expert medical summarizer creating a concise Specialist Consult summary for physicians.

REQUIRED FORMAT WITH LABELS:
 Consult {doctor_section} for [Body part] : [DATE] | Findings → [Diagnosis/impression] | Treatment → [Meds/procedures] | Recommendations → [Follow-up] | Work status → [Status]

CRITICAL RULES:
1. ALWAYS prefix field values with labels for clarity:
   - "Findings →" before findings
   - "Treatment →" before treatment_recommendations
   - "Recommendations →" before recommendations
   - "Work status →" before work_status (if present)
2. Use "→" arrows to separate labels from values
3. Use "|" pipes to separate major sections
4. Omit entire sections (including labels) if field is empty
5. Keep concise (35-65 words total)
6. Do NOT write standalone ambiguous terms - always include the label

EXAMPLES:
Good: "09/26/25: Consult by Dr. Smith (Ortho) for R shoulder | Findings → rotator cuff tear | Treatment → PT, NSAIDs | Recommendations → Re-eval in 4 weeks | Work status → modified duty"
Good: "10/05/25: Consult for L knee | Findings → possible meniscal tear | Treatment → MRI ordered | Recommendations → Follow-up post-MRI"
Bad: "09/26/25: Consult for shoulder = Tear; Modified duty" (missing labels!)

EXTRACTED FIELDS:
- consult_date: {date}
- physician_name: {physician}
- specialty: {specialty}
- body_part: {body_part}
- findings: {findings}
- treatment_recommendations: {treatment}
- recommendations: {recommendations}
- work_status: {work_status}

Generate the summary now (35-65 words, with field labels):"""

        try:
            summary_chain = PromptTemplate(
                template="""{summary_prompt}""",
                input_variables=["summary_prompt"]
            ) | self.llm
            
            response = summary_chain.invoke({"summary_prompt": summary_prompt})
            summary = response.content if hasattr(response, 'content') else str(response)
            summary = summary.strip()
            
            # Validate that labels are present
            required_labels = []
            if findings:
                required_labels.append("Findings →")
            if treatment or recommendations:
                required_labels.append("→")  # At least one arrow should be present
            
            missing_labels = [label for label in required_labels if label not in summary]
            if missing_labels:
                logger.warning(f"⚠️ Summary missing required labels: {missing_labels}. Rebuilding...")
                summary = self._build_manual_summary(date, doc_type, doctor_section, body_part, findings, treatment, recommendations, work_status)
            
            logger.info(f"✅ Consult Summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Consult summary LLM failed: {e}")
            return self._build_manual_summary(date, doc_type, doctor_section, body_part, findings, treatment, recommendations, work_status)

    def _build_manual_summary(
        self,
        date: str,
        doc_type: str,
        doctor_section: str,
        body_part: str,
        findings: str,
        treatment: str,
        recommendations: str,
        work_status: str
    ) -> str:
        """
        Manual summary construction with explicit field labels.
        Fallback when LLM fails or produces unlabeled output.
        """
        parts = [f"{date}: Consult"]
        
        if doctor_section:
            parts.append(doctor_section)
        
        if body_part:
            parts.append(f"for {body_part}")
        
        # Findings with label
        sections = []
        if findings:
            sections.append(f"Findings → {findings}")
        if treatment:
            sections.append(f"Treatment → {treatment}")
        if recommendations:
            sections.append(f"Recommendations → {recommendations}")
        if work_status:
            sections.append(f"Work status → {work_status}")
        
        if sections:
            parts.append(f"| {' | '.join(sections)}")
        
        summary = " ".join(parts)
        logger.info(f"✅ Manual Consult summary built: {summary}")
        return summary
