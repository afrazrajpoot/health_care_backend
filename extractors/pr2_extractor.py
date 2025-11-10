"""
PR-2 Progress Report extractor with DoctorDetector integration.
All doctor extraction logic delegated to DoctorDetector for consistency.
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


class PR2ExtractorChained:
    """
    Enhanced PR-2 extractor with DoctorDetector integration:
    - Stage 1: Extract structured clinical data (NO doctor extraction)
    - Stage 2: Build summary
    - Stage 3: Verify and correct
    - Stage 4: Doctor detection via DoctorDetector (zone-aware)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        self.doctor_detector = DoctorDetector(llm)
        logger.info("✅ PR2ExtractorChained initialized with DoctorDetector")

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
            doc_type: Document type (PR-2)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            raw_text: Original flat text (for backward compatibility)
        """
        # Stage 1: Extract clinical data (NO doctor extraction in prompt)
        raw_result = self._extract_clinical_data(text, doc_type, fallback_date)
        
        # Stage 2: Doctor detection via DoctorDetector (zone-aware)
        physician_name = self._detect_physician(text, page_zones)
        raw_result["physician_name"] = physician_name
        
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
You are an AI Medical Assistant extracting structured clinical data from a PR-2 (Progress Report).

EXTRACTION RULES:
1. Focus on CLINICAL DATA ONLY (status, body part, treatment, work status, plan).
2. DO NOT extract physician/doctor names - this is handled separately.
3. Status: one word or short phrase (e.g., "improved", "unchanged", "worsened", "stable").
4. Body part: primary area treated (e.g., "R shoulder", "L knee").
5. Plan: clear next treatment step(s). Include follow-ups, referrals, or testing.
6. Treatment recommendations: new orders, procedures, or specific medications.
7. Work status: include phrases like "TTD", "modified duty", "return to full duty", or similar.
8. If any "?" is used (e.g., "? MMI"), replace with a brief clarification (e.g., "uncertain, pending evaluation").
9. Output must be short, factual, and readable.

Document text:
{text}

Extract these fields:
- report_date: Date of report (MM/DD/YY format, or use {fallback_date} if not found)
- body_part: Primary area addressed (e.g., "R shoulder", "L knee")
- current_status: Patient's current clinical status (max 10 words)
- treatment_recommendations: New or continued treatments, including medications (max 12 words)
- work_status: Work ability or restriction (max 16 words)
- next_plan: Next step / follow-up (max 16 words)

CRITICAL REASONING RULES - VERIFY BEFORE RETURNING:

1. EXTRACT ALL KEY FINDINGS - both positive and negative clinical findings:
   ✓ GOOD (Positive): "Continue PT 2x/week", "Add gabapentin 300mg", "Follow-up in 2 weeks"
   ✓ GOOD (Negative): "No treatment changes - stable", "Resolved - no further treatment", "Full duty - no restrictions"
   ✗ BAD (Placeholders): "Not mentioned", "Not provided", "N/A"
   
2. If a field has NO actual information (truly not mentioned in document), return empty string "".
   If field has actual clinical finding (even negative), include it with context.
   
3. For current_status: Extract actual status assessment:
   ✓ GOOD: "Improved", "Unchanged", "Worsened", "Stable", "Resolved", "Plateaued"
   ✗ BAD: "Not specified" (use empty string instead)
   
4. For treatment_recommendations: Extract actual treatment plan:
   ✓ GOOD: "Continue PT 2x/week", "Start NSAIDs", "ESI scheduled", "No changes - continue current meds"
   ✓ GOOD: "No further treatment needed - condition resolved"
   ✗ BAD: "No new treatments" (too vague - specify: "Continue current treatment" or "No treatment changes")
   
5. For work_status: Extract actual work status/restrictions:
   ✓ GOOD: "TTD", "Modified duty - no lifting >10 lbs", "Full duty - no restrictions", "Return to work 01/15/25"
   ✓ GOOD: "Restrictions lifted - cleared for full duty"
   ✗ BAD: "Not mentioned" (use empty string instead)
   
6. For next_plan: Extract actual follow-up plan:
   ✓ GOOD: "Follow-up in 2 weeks", "Re-eval in 1 month", "Discharge to home exercise program"
   ✓ GOOD: "No follow-up needed - patient discharged", "PRN follow-up if symptoms worsen"
   ✗ BAD: "No follow-up" without context (specify: "No follow-up needed - condition stable")

7. REASONING CHECK: Before returning each field, ask yourself:
   - "Does this contain actual information from the PR-2 report (whether positive or negative)?"
   - "Would a treating physician or case manager find this clinically useful?"
   - If YES → include it (include negative findings like "stable", "no restrictions", "resolved")
   - If NO (placeholder like "not mentioned") → return empty string

Return JSON:
{{
  "report_date": "MM/DD/YY or {fallback_date}",
  "body_part": "Primary part or empty",
  "current_status": "Status term or empty",
  "treatment_recommendations": "Treatments or meds or empty",
  "work_status": "Work status or empty",
  "next_plan": "Follow-up plan or empty"
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
            logger.error(f"❌ PR-2 clinical data extraction failed: {e}")
            return {}

    def _detect_physician(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """
        Stage 2: Detect physician using DoctorDetector (zone-aware).
        
        Args:
            text: Full document text
            page_zones: Page zones for zone-aware detection
        
        Returns:
            Physician name with title, or empty string
        """
        logger.info("🔍 Stage 2: Running DoctorDetector (zone-aware)...")
        
        # Debug: Check if page_zones is provided
        if page_zones:
            logger.info(f"✅ PR2 extractor received page_zones with {len(page_zones)} pages: {list(page_zones.keys())}")
        else:
            logger.warning("⚠️ PR2 extractor did NOT receive page_zones")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(
                f"✅ Physician detected: {detection_result['doctor_name']} "
                f"(confidence: {detection_result['confidence']}, "
                f"source: {detection_result['source']})"
            )
            physician_name = detection_result["doctor_name"]
            logger.info(f"🎯 PR2 extractor returning physician: '{physician_name}'")
            return physician_name
        else:
            logger.warning(
                f"⚠️  No valid physician found: {detection_result['validation_notes']}"
            )
            logger.info("🎯 PR2 extractor returning empty physician name")
            return ""

    def _build_initial_result(
        self,
        raw_data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> ExtractionResult:
        """Stage 3: Build initial result with validation and summary"""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_pr2_summary(cleaned, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name", ""),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )
        
        logger.info(f"✅ Stage 3: Initial result built (physician: {result.examiner_name})")
        return result

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        cleaned = {}
        
        # Date validation
        date = result.get("report_date", "").strip()
        cleaned["report_date"] = date if date and date != "empty" else fallback_date

        # Physician (from DoctorDetector - already validated)
        physician = result.get("physician_name", "").strip()
        cleaned["physician_name"] = physician

        # Body part validation
        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        # String fields validation
        string_fields = [
            "current_status",
            "treatment_recommendations",
            "work_status",
            "next_plan",
        ]
        
        # Negative phrases that should be filtered out (non-actionable information)
        negative_phrases = [
            "no treatment", "no changes", "no new", "no follow-up",
            "fully resolved", "resolved", "no restrictions",
            "released to full duty", "full duty", "unrestricted",
            "not indicated", "not recommended", "not needed",
            "no significant", "within normal limits", "stable with no changes"
        ]
        
        for f in string_fields:
            v = result.get(f, "").strip()
            v_lower = v.lower()
            
            # Check if empty or placeholder
            if not v or v_lower in [
                "", "empty", "none", "n/a", "not mentioned", "not specified"
            ]:
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

    def _build_pr2_summary(
    self,
    data: Dict,
    doc_type: str,
    fallback_date: str
) -> str:
        """
        Build concise, physician-friendly summary for PR-2 report.
        Uses explicit field labels for clarity (e.g., "Status → improved", "Work status → TTD").
        Enforces 35-65 word range.
        """
        date = data.get("report_date", fallback_date)
        physician = data.get("physician_name", "")
        body_part = data.get("body_part", "")
        status = data.get("current_status", "")
        treatment = data.get("treatment_recommendations", "")
        work_status = data.get("work_status", "")
        next_plan = data.get("next_plan", "")

        # Doctor section (only if valid)
        doctor_section = ""
        if physician:
            doctor_section = f" {physician}"

        # Compose enhanced summary prompt with explicit labeling rules
        summary_prompt = f"""You are an expert medical summarizer creating a concise PR-2 Progress Report summary for physicians.

    REQUIRED FORMAT WITH LABELS:
    PR-2{doctor_section} for [Body part] [DATE]: | Clinical status → [Status] | Work status → [Work status] | Treatment → [Meds/procedures] | Plan → [Follow-up/next steps]

    CRITICAL RULES:
    1. ALWAYS prefix field values with labels for clarity:
    - "Clinical status →" before current_status
    - "Work status →" before work_status  
    - "Treatment →" before treatment_recommendations
    - "Plan →" before next_plan
    2. Use "→" arrows to separate labels from values
    3. Use "|" pipes to separate major sections
    4. Omit entire sections (including labels) if field is empty
    5. Keep concise (35-65 words total)
    6. Do NOT write standalone ambiguous terms like "Uncertain" - always include the label

    EXAMPLES:
    Good: "09/26/25: PR-2 by Dr. Smith for R shoulder | Clinical status → improved | Work status → TTD | Treatment → Continue PT, add NSAIDs | Plan → Re-eval in 2 weeks"
    Good: "10/05/25: PR-2 for L knee | Clinical status → stable | Work status → modified duty, no lifting >20 lbs | Plan → Follow-up in 4 weeks"
    Bad: "09/26/25: PR-2 for Head = Improved; Uncertain → Continue meds" (missing labels!)

    EXTRACTED FIELDS:
    - report_date: {date}
    - physician_name: {physician}
    - body_part: {body_part}
    - current_status: {status}
    - treatment_recommendations: {treatment}
    - work_status: {work_status}
    - next_plan: {next_plan}

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
            if status:
                required_labels.append("Clinical status →")
            if work_status:
                required_labels.append("Work status →")
            if treatment or next_plan:
                required_labels.append("→")  # At least one arrow should be present
            
            missing_labels = [label for label in required_labels if label not in summary]
            if missing_labels:
                logger.warning(f"⚠️ Summary missing required labels: {missing_labels}. Rebuilding...")
                # Fallback to manual construction
                summary = self._build_manual_summary(date, doc_type, doctor_section, body_part, status, treatment, work_status, next_plan)
            
            logger.info(f"✅ PR-2 Summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ PR-2 summary LLM failed: {e}")
            # Fallback: manual construction with labels
            return self._build_manual_summary(date, doc_type, doctor_section, body_part, status, treatment, work_status, next_plan)

    def _build_manual_summary(
        self,
        date: str,
        doc_type: str,
        doctor_section: str,
        body_part: str,
        status: str,
        treatment: str,
        work_status: str,
        next_plan: str
    ) -> str:
        """
        Manual summary construction with explicit field labels.
        Fallback when LLM fails or produces unlabeled output.
        """
        parts = [f"{date}: {doc_type}"]
        
        if doctor_section:
            parts.append(doctor_section)
        
        if body_part:
            parts.append(f"for {body_part}")
        
        # Clinical findings with labels
        findings = []
        if status:
            findings.append(f"Clinical status → {status}")
        if work_status:
            findings.append(f"Work status → {work_status}")
        
        if findings:
            parts.append(f"| {' | '.join(findings)}")
        
        # Treatment and plan with labels
        actions = []
        if treatment:
            actions.append(f"Treatment → {treatment}")
        if next_plan:
            actions.append(f"Plan → {next_plan}")
        
        if actions:
            parts.append(f"| {' | '.join(actions)}")
        
        summary = " ".join(parts)
        logger.info(f"✅ Manual PR-2 summary built: {summary}")
        return summary