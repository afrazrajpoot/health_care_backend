"""
QME/AME/IME specialized extractor with LLM chaining and full doctor name extraction
"""
import logging
import re
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class QMEExtractorChained:
    """
    Enhanced QME extractor with multi-stage LLM chaining and full doctor name extraction:
    Stage 1: Extract raw data
    Stage 2: Detect examiner via DoctorDetector (zone-aware)
    Stage 3: Build summary
    Stage 4: Verify and correct
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )
        logger.info("✅ QMEExtractorChained initialized with DoctorDetector")

    def extract(
        self, 
        text: str, 
        doc_type: str, 
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract with DoctorDetector integration and verification chain.
        
        Args:
            text: Layout-preserved text from Document AI
            doc_type: Document type (QME/AME/IME)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            raw_text: Original flat text (for backward compatibility)
        """
        # Debug: Check if page_zones is provided
        if page_zones:
            logger.info(f"✅ QME extractor received page_zones with {len(page_zones)} pages: {list(page_zones.keys())}")
        else:
            logger.warning("⚠️ QME extractor did NOT receive page_zones")
        
        # Stage 1: Extract clinical data (NO doctor extraction in prompt)
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        
        # Stage 2: Detect examiner via DoctorDetector (zone-aware)
        examiner_name = self._detect_examiner(text, page_zones)
        raw_result["examiner_name"] = examiner_name
        
        # Stage 3: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        final_result = self.verifier.verify_and_fix(initial_result)
        return final_result

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data using concise QME summary schema"""
        prompt = PromptTemplate(
            template="""
You are summarizing a Qualified Medical Evaluator (QME/AME) report into a concise,
structured, actual/actionable summary line for a physician dashboard and treatment timeline.

CRITICAL PHYSICIAN IDENTIFICATION RULES:

1. PRIMARY PHYSICIAN (QME/AME Examiner):
   - MUST have medical credentials: Dr., MD, M.D., DO, D.O., MBBS
   - Look for: "QME:", "AME:", "Examiner:", "Evaluating Physician:"
   - Check signatures: "Electronically signed by:", "Dictated by:", "Signed by:"
   - EXTRACT FULL NAME: Always extract the complete physician name with title
     Examples:
       ✓ "Dr. John Michael Smith" → "Dr. John Michael Smith"
       ✓ "Sarah Jennifer Johnson, MD" → "Sarah Jennifer Johnson, MD"
       ✓ "Robert K. Chen, D.O." → "Robert K. Chen, D.O."

2. REFERRAL DOCTOR FALLBACK (ONLY if no primary physician found):
   - Use referral doctor ONLY when no QME/AME examiner is identified
   - Look for: "Referred by:", "Referral from:", "PCP:", "Primary Care:"
   - Must have medical credentials
   - EXTRACT FULL NAME: Always extract complete name with title
     Examples: ✓ "Referred by: Dr. James William Wilson" → "Dr. James William Wilson"

3. REJECT NON-DOCTOR SIGNATORIES:
   - ✗ Reject names without medical credentials (e.g., "Syed Akbar")
   - ✗ Reject administrators, coordinators, case managers
   - ✗ Reject any name without proper medical credentials

4. FULL NAME EXTRACTION REQUIREMENTS:
   - ALWAYS extract complete name including first, middle (if present), and last name
   - PRESERVE medical titles: "Dr.", "MD", "DO", "M.D.", "D.O."
   - DO NOT shorten names: "Dr. J. Smith" → try to find full name elsewhere in document
   - If only shortened name available, use as-is but note in verification

5. PRIORITY ORDER:
   1. QME/AME Examiner (with credentials and full name)
   2. Referral Doctor (with credentials and full name, ONLY if no examiner)
   3. "Not specified" (if no qualified doctors found)

INPUT:
Full QME/AME report text.

INSTRUCTIONS:
1. Extract the following key elements:
   - Date of exam/report
   - QME/AME physician FULL NAME with credentials (complete name with title)
   - Body parts evaluated
   - Key diagnoses confirmed
   - MMI/WPI status
   - Apportionment (industrial vs non-industrial)
   - Specific treatment recommendations (PT, injections, surgery, consults)
   - Specific medication recommendations (use drug names when present)
   - Work status AND specific functional restrictions
     Examples:
       * "No lifting >10 lbs"
       * "Limit overhead reaching"
       * "Sedentary work only"
       * "No repetitive bending/twisting"
       * "No pushing/pulling >5 lbs"
   - Future medical and follow-up recommendations

2. Output a structured summary schema — concise and factual, optimized for clinical timeline parsing.

3. WORK RESTRICTIONS RULE:
   - DO NOT summarize work status as “modified work” or “restricted duty.”
   - ALWAYS extract and list the actual functional restrictions.
   - If restrictions are not explicitly stated, infer the most accurate
     functional description from context (e.g., for chronic shoulder pain:
     “avoid overhead lifting/reaching; lifting ≤10 lbs to waist height”).

4. HANDLING QUESTION MARKS (“?”):
   - If the report uses “?” to show uncertainty, clarify what is uncertain
     using a short parenthetical (≤8 words).
     Examples:
       "? rotator cuff tear" → "? rotator cuff tear (diagnosis uncertain)"
       "? repeat MRI" → "? repeat MRI (pending rationale)"

5. PHYSICIAN EXTRACTION PRIORITY:
   - First: QME/AME examiner with credentials and FULL NAME
   - Fallback: Referral doctor with credentials and FULL NAME (ONLY if no examiner)
   - Last: "Not specified" (if no qualified doctors)

6. FULL NAME EXAMPLES:
   ✓ ACCEPT: "Dr. Michael Jonathan Brown"
   ✓ ACCEPT: "Sarah Elizabeth Martinez, MD"
   ✓ ACCEPT: "Robert K. Chen, D.O."
   ✓ ACCEPT: "Dr. Jennifer Marie O'Malley"
   ✗ REJECT: "Dr. Smith" (too short - find full name)
   ✗ REJECT: "J. Johnson, MD" (initial only - find full name)
   ✗ REJECT: "Syed Akbar" (no medical title)

7. Tone:
   - Neutral, clinical, concise.
   - Use standard medical abbreviations.
   - Avoid speculation or redundant phrasing.

Document text:
{text}

Extract these fields with precision:
- document_date: Date of exam/report (MM/DD/YY format, or use {fallback_date} if not found)
- examiner_name: FULL NAME of QME/AME physician (must include Dr./MD/DO and complete name) - minimum first and last name with title
- referral_physician: FULL NAME of referral doctor if explicitly mentioned (must have credentials and complete name)
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

CRITICAL REASONING RULES - VERIFY BEFORE RETURNING:
1. ONLY extract POSITIVE/ACTIONABLE findings. DO NOT extract negative statements.
   ✗ BAD: "No additional treatment indicated", "No future medical care indicated", "Injury fully resolved"
   ✓ GOOD: "Continue PT 2x/week", "ESI recommended", "Follow-up in 6 weeks"
   
2. If a field has NO meaningful positive data, return empty string "" - DO NOT return negative phrases.
   
3. For MMI_status: ONLY return if MMI is explicitly stated with a positive finding
   ✗ BAD: "reached" (too vague)
   ✓ GOOD: "MMI reached", "MMI deferred pending MRI", "Not at MMI"
   
4. For impairment_summary: ONLY return if there's an actual impairment percentage > 0%
   ✗ BAD: "0% Whole Person Impairment"
   ✓ GOOD: "15% WPI", "8% lower extremity impairment"
   
5. For causation_opinion: ONLY return if there's meaningful apportionment data
   ✗ BAD: "no apportionment indicated", "0% industrial"
   ✓ GOOD: "60% industrial, 40% non-industrial", "100% industrial"

6. REASONING CHECK: Before returning each field, ask yourself:
   - "Is this information ACTIONABLE for the treating physician?"
   - "Does this tell me what TO DO or what IS present?"
   - If answer is NO → return empty string for that field

Return JSON:
{{
  "document_date": "MM/DD/YY or {fallback_date}",
  "examiner_name": "Full Name with Title (e.g., 'Dr. John Michael Smith', 'Sarah Jennifer Johnson, MD') or empty if no qualified doctor",
  "referral_physician": "Full Name with Title (e.g., 'Dr. James William Wilson') or empty",
  "specialty": "Specialty or empty",
  "body_parts_evaluated": ["part1", "part2", ...] or [],
  "diagnoses_confirmed": ["diagnosis1", "diagnosis2", ...] or [],
  "MMI_status": "status phrase or empty",
  "impairment_summary": "WPI or impairment note or empty",
  "causation_opinion": "apportionment or empty",
  "treatment_recommendations": "procedures or therapies or empty",
  "medication_recommendations": "medications or empty",
  "work_restrictions": "explicit restrictions or empty",
  "future_medical_recommendations": "ongoing/follow-up plan or empty"
}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "fallback_date"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke(
                {"text": text[:8000], "doc_type": doc_type, "fallback_date": fallback_date}
            )
            return result
        except Exception as e:
            logger.error(f"❌ Raw extraction failed: {e}")
            return {}

    def _detect_examiner(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """
        Stage 2: Detect QME/AME examiner using DoctorDetector (zone-aware).
        
        Args:
            text: Full document text
            page_zones: Page zones for zone-aware detection
        
        Returns:
            Examiner name with title, or empty string
        """
        logger.info("🔍 Stage 2: Running DoctorDetector for QME/AME examiner (zone-aware)...")
        
        # Debug: Check if page_zones is provided
        if page_zones:
            logger.info(f"✅ QME extractor received page_zones with {len(page_zones)} pages: {list(page_zones.keys())}")
        else:
            logger.warning("⚠️ QME extractor did NOT receive page_zones")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(
                f"✅ Examiner detected: {detection_result['doctor_name']} "
                f"(confidence: {detection_result['confidence']}, "
                f"source: {detection_result['source']})"
            )
            examiner_name = detection_result["doctor_name"]
            logger.info(f"🎯 QME extractor returning examiner: '{examiner_name}'")
            return examiner_name
        else:
            logger.warning(
                f"⚠️ No valid examiner found: {detection_result['validation_notes']}"
            )
            logger.info("🎯 QME extractor returning empty examiner name")
            return ""

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        
        # Apply referral doctor fallback logic
        final_examiner = self._apply_referral_fallback(cleaned)
        cleaned["final_examiner"] = final_examiner
        
        summary_line = self._build_qme_summary(cleaned, doc_type, fallback_date)
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("document_date", fallback_date),
            summary_line=summary_line,
            examiner_name=final_examiner,
            specialty=cleaned.get("specialty"),
            body_parts=cleaned.get("body_parts_evaluated", []),
            raw_data=cleaned,
        )

    def _apply_referral_fallback(self, data: Dict) -> str:
        """
        Apply referral doctor fallback logic:
        - Primary: QME/AME examiner with credentials
        - Fallback: Referral physician with credentials (ONLY if no examiner)
        - Final: "Not specified"
        """
        examiner_md = data.get("examiner_name", "")
        referral_md = data.get("referral_physician", "")
        
        # Primary: Use QME/AME examiner if qualified
        if examiner_md and examiner_md != "":
            logger.info(f"✅ Using QME/AME examiner: {examiner_md}")
            return examiner_md
        
        # Fallback: Use referral doctor ONLY if no examiner found
        if referral_md and referral_md != "":
            logger.info(f"🔄 Using referral doctor as fallback: {referral_md}")
            return referral_md
        
        # No qualified doctors found
        logger.info("❌ No qualified doctors found")
        return ""

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        cleaned = {}
        date = result.get("document_date", "").strip()
        cleaned["document_date"] = date if date and date != "empty" else fallback_date

        # Physician validation with strict credential checking and name validation
        examiner = result.get("examiner_name", "").strip()
        referral_physician = result.get("referral_physician", "").strip()
        
        cleaned["examiner_name"] = self._validate_physician_full_name(examiner)
        cleaned["referral_physician"] = self._validate_physician_full_name(referral_physician)

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
        
        # Negative phrases that should be filtered out (non-actionable information)
        negative_phrases = [
            "no additional treatment", "no future medical", "no treatment",
            "no recommendations", "injury fully resolved", "injury resolved",
            "no restrictions", "no limitations", "no follow-up",
            "no apportionment", "0% industrial", "0% wpi", "0% whole person",
            "not indicated", "not recommended", "not needed", "not required",
            "no significant", "unremarkable", "within normal limits",
            "no impairment", "no disability", "no work restrictions"
        ]
        
        for f in string_fields:
            v = result.get(f, "").strip()
            v_lower = v.lower()
            
            # Check if empty or placeholder
            if not v or v_lower in ["", "empty", "none", "n/a", "not mentioned", "reached"]:
                cleaned[f] = ""
                continue
            
            # Check if contains negative/non-actionable phrases
            is_negative = any(neg_phrase in v_lower for neg_phrase in negative_phrases)
            if is_negative:
                logger.debug(f"⚠️ Filtering out negative phrase in {f}: {v}")
                cleaned[f] = ""
                continue
            
            # Keep only meaningful, actionable information
            cleaned[f] = v

        return cleaned

    def _validate_physician_full_name(self, name: str) -> str:
        """Efficiently validate physician name has proper medical credentials and full name."""
        if not name or name.lower() in ["not specified", "not found", "none", "n/a", ""]:
            return ""
        
        name_lower = name.lower()
        
        # Fast rejection using set membership (O(1) lookup)
        reject_terms = {
            "admin", "administrator", "case manager", "coordinator", "manager",
            "therapist", "technician", "assistant", "technologist",
            "staff", "authority", "personnel", "clerk", "secretary",
            "signed by", "dictated by", "transcribed by"
        }
        
        # Check if any reject term is in the name
        if any(term in name_lower for term in reject_terms):
            return ""
        
        # Use pre-compiled regex for efficiency - MUST have medical credentials
        if not self.medical_credential_pattern.search(name_lower):
            return ""
        
        # Validate name has sufficient length for full name (at least 2 words + title)
        words = name.split()
        if len(words) < 2:
            logger.warning(f"⚠️ Name too short for full name: {name}")
            return ""
        
        # Check if name appears to be a full name (not just initials or single names)
        # Look for patterns like "Dr. First Last" or "First Last, MD"
        has_proper_name_structure = (
            (len(words) >= 3 and any(title in words[0].lower() for title in ["dr", "dr."])) or  # "Dr. First Last"
            (len(words) >= 2 and any(title in words[-1].lower() for title in ["md", "do", "m.d.", "d.o."]))  # "First Last, MD"
        )
        
        if not has_proper_name_structure:
            logger.warning(f"⚠️ Name doesn't appear to be full name: {name}")
            # Still return if it has credentials and at least 2 words
            if len(words) >= 2 and self.medical_credential_pattern.search(name_lower):
                logger.info(f"✅ Accepting name with credentials: {name}")
                return name
            return ""
        
        return name

    def _build_qme_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        date = data.get("document_date", fallback_date)
        examiner = data.get("final_examiner", "")
        referral_physician = data.get("referral_physician", "")
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
            # Use full examiner name in summary (no abbreviation)
            specialty_abbrev = self._abbreviate_specialty(specialty) if specialty else 'QME'
            if examiner == referral_physician and referral_physician != "":
                parts.append(f"(Referral: {examiner}, {specialty_abbrev})")
            else:
                parts.append(f"({examiner}, {specialty_abbrev})")

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