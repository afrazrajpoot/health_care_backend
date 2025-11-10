"""
Imaging reports extractor (MRI, CT, X-ray, Ultrasound, EMG)
Enhanced with DoctorDetector integration for consistent doctor extraction.
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


class ImagingExtractorChained:
    """
    Enhanced Imaging extractor with DoctorDetector integration:
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
        logger.info("✅ ImagingExtractorChained initialized with DoctorDetector")

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
            doc_type: Document type (MRI/CT/X-ray/Ultrasound/EMG)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            raw_text: Original flat text (for backward compatibility)
        """
        logger.info(f"🔍 Starting extraction for {doc_type} report")
        
        # Stage 1: Extract clinical/imaging data (NO doctor extraction in prompt)
        raw_result = self._extract_clinical_data(text, doc_type, fallback_date)
        
        # Stage 2: Doctor detection via DoctorDetector (zone-aware)
        radiologist_name = self._detect_radiologist(text, page_zones)
        raw_result["consulting_doctor"] = radiologist_name
        
        # Stage 3: Build initial result
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Stage 4: Verify and fix
        verified_result = self.verifier.verify_and_fix(initial_result)
        
        logger.info(f"✅ Extraction completed for {doc_type}")
        return verified_result

    def _extract_clinical_data(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Stage 1: Extract imaging/clinical data ONLY (NO doctor extraction).
        Doctor extraction is handled by DoctorDetector.
        """
        logger.info(f"🎯 Stage 1: Clinical data extraction for {doc_type}")
        
        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant extracting structured data from imaging reports.

EXTRACTION RULES:
1. Focus on IMAGING DATA ONLY (findings, impressions, body part, contrast).
2. DO NOT extract radiologist/doctor names - this is handled separately.
3. Primary finding: Most clinically significant finding (max 16 words).
4. If normal study → "normal study" or "no acute findings".
5. If uncertain finding (marked with "?") → rewrite as "possible [finding]".
6. Body part: concise format (e.g., "R shoulder", "L knee", "C4-6", "L-spine").
7. Date: MM/DD/YY format.
8. Contrast: Indicate if "with contrast" or "without contrast" when explicitly stated.
9. Avoid technical details (sequences, imaging parameters).
10. Impression status: "normal", "abnormal", "post-op", or "inconclusive".

Document text:
{text}

Extract these fields (do NOT extract consulting_doctor):
- study_date: Imaging date (MM/DD/YY format, or use {fallback_date} if not found)
- document_type: Type of report (MRI, CT, X-ray, Ultrasound, EMG)
- body_part: Anatomical area studied (abbreviated form)
- contrast_used: "with contrast", "without contrast", or empty
- primary_finding: Most important diagnostic finding (max 16 words)
- impression_status: "normal", "abnormal", "post-op", or "inconclusive"

CRITICAL REASONING RULES - VERIFY BEFORE RETURNING:
1. ONLY extract meaningful findings. DO NOT extract "normal" or "unremarkable" as primary_finding.
   ✗ BAD: "No acute findings", "Unremarkable study", "Within normal limits"
   ✓ GOOD: "Grade 2 AC joint separation", "L4-5 disc herniation", "Rotator cuff tear"
   
2. If the study is completely normal with no pathology, set primary_finding to empty string "".
   
3. For impression_status: Use "normal" ONLY if study is truly normal. If abnormalities exist, use "abnormal".
   
4. REASONING CHECK: Before returning primary_finding, ask yourself:
   - "Is this finding ACTIONABLE or clinically significant?"
   - "Would this change treatment or diagnosis?"
   - If answer is NO → return empty string for primary_finding

Return JSON:
{{
  "study_date": "MM/DD/YY or {fallback_date}",
  "document_type": "e.g., MRI, CT, X-ray",
  "body_part": "abbreviated part or empty",
  "contrast_used": "contrast detail or empty",
  "primary_finding": "main diagnostic finding (max 16 words)",
  "impression_status": "normal/abnormal/post-op/inconclusive"
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
            logger.info(f"  - study_date: {result.get('study_date', 'Not found')}")
            logger.info(f"  - body_part: {result.get('body_part', 'Not found')}")
            logger.info(f"  - primary_finding: {result.get('primary_finding', 'Not found')}")
            return result
        except Exception as e:
            logger.error(f"❌ Imaging clinical data extraction failed: {e}")
            return {}

    def _detect_radiologist(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """
        Stage 2: Detect radiologist/interpreting physician using DoctorDetector (zone-aware).
        
        Args:
            text: Full document text
            page_zones: Page zones for zone-aware detection
        
        Returns:
            Radiologist name with title, or empty string
        """
        logger.info("🔍 Stage 2: Running DoctorDetector for radiologist (zone-aware)...")
        
        if page_zones:
            logger.info(f"✅ Imaging extractor received page_zones with {len(page_zones)} pages: {list(page_zones.keys())}")
        else:
            logger.warning("⚠️ Imaging extractor did NOT receive page_zones")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(
                f"✅ Radiologist detected: {detection_result['doctor_name']} "
                f"(confidence: {detection_result['confidence']}, "
                f"source: {detection_result['source']})"
            )
            return detection_result["doctor_name"]
        else:
            logger.warning(
                f"⚠️ No valid radiologist found: {detection_result['validation_notes']}"
            )
            return ""

    def _build_initial_result(
        self,
        raw_data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> ExtractionResult:
        """Stage 3: Build initial result with validation and summary"""
        logger.info("🎯 Stage 3: Building initial result with validation")
        
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_imaging_summary(cleaned, doc_type, fallback_date)
        
        result = ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("study_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("consulting_doctor", ""),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )
        
        logger.info(f"✅ Stage 3: Initial result built (radiologist: {result.examiner_name})")
        return result

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        logger.info("🔧 Validating and cleaning extracted data")
        cleaned = {}
        
        # Date validation
        date = result.get("study_date", "").strip()
        cleaned["study_date"] = date if date and date != "empty" else fallback_date
        logger.info(f"  📅 Date cleaned: {cleaned['study_date']}")

        # Radiologist (from DoctorDetector - already validated)
        radiologist = result.get("consulting_doctor", "").strip()
        cleaned["consulting_doctor"] = radiologist
        logger.info(f"  👨‍⚕️ Radiologist: {radiologist if radiologist else 'None'}")

        # Body part validation
        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""
        logger.info(f"  🦴 Body Part: {cleaned['body_part']}")

        # Primary finding validation
        primary_finding = result.get("primary_finding", "").strip()
        cleaned["primary_finding"] = primary_finding if primary_finding and primary_finding != "empty" else ""
        logger.info(f"  🔍 Primary Finding: {cleaned['primary_finding']}")

        # Contrast validation
        contrast = result.get("contrast_used", "").strip()
        cleaned["contrast_used"] = contrast if contrast and contrast != "empty" else ""
        logger.info(f"  💉 Contrast: {cleaned['contrast_used']}")

        # Impression status validation
        status = result.get("impression_status", "").strip()
        cleaned["impression_status"] = status if status and status != "empty" else ""
        logger.info(f"  📊 Status: {cleaned['impression_status']}")

        return cleaned

    def _build_imaging_summary(
        self,
        data: Dict,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Build concise, physician-friendly summary for imaging report.
        Uses explicit field labels for clarity (matching PR2 format).
        Enforces 35-65 word range.
        """
        logger.info("🎯 Building imaging summary")
        
        physician = data.get("consulting_doctor", "").strip()
        body_part = data.get("body_part", "")
        finding = data.get("primary_finding", "")
        date = data.get("study_date", fallback_date)
        contrast = data.get("contrast_used", "")
        status = data.get("impression_status", "")

        # Doctor section (only if valid)
        doctor_section = ""
        if physician:
            last_name = self._extract_physician_last_name(physician)
            if last_name:
                doctor_section = f" - {last_name}"

        # Compose enhanced summary prompt with explicit labeling rules
        summary_prompt = f"""You are an expert medical summarizer creating a concise imaging report summary for physicians.

    REQUIRED FORMAT WITH LABELS:
    [DATE]: {doc_type}{doctor_section} for [Body part] | Impression → [Status] | Findings → [Primary finding]

    CRITICAL RULES:
    1. ALWAYS prefix field values with labels for clarity:
    - "Impression →" before impression_status (normal/abnormal/post-op/inconclusive)
    - "Findings →" before primary_finding
    2. If contrast info exists, include in body part: "[Body part] (with/without contrast)"
    3. Use "→" arrows to separate labels from values
    4. Use "|" pipes to separate major sections
    5. Omit entire sections (including labels) if field is empty
    6. Keep concise (35-65 words total)
    7. Do NOT write standalone ambiguous terms - always include the label

    EXAMPLES:
    Good: "09/26/25: MRI by Dr. Smith for R shoulder (without contrast) | Impression → abnormal | Findings → full-thickness rotator cuff tear; mild AC joint arthritis"
    Good: "10/05/25: CT for L knee | Impression → normal | Findings → no acute fracture or effusion"
    Good: "11/10/25: X-ray for C-spine | Impression → post-op | Findings → hardware in place, no complications"
    Bad: "09/26/25: MRI for shoulder = Abnormal; Tear" (missing labels!)

    EXTRACTED FIELDS:
    - study_date: {date}
    - consulting_doctor: {physician}
    - body_part: {body_part}
    - contrast_used: {contrast}
    - primary_finding: {finding}
    - impression_status: {status}

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
                required_labels.append("Impression →")
            if finding:
                required_labels.append("Findings →")
            
            missing_labels = [label for label in required_labels if label not in summary]
            if missing_labels:
                logger.warning(f"⚠️ Summary missing required labels: {missing_labels}. Rebuilding...")
                summary = self._build_manual_imaging_summary(date, doc_type, doctor_section, body_part, contrast, status, finding)
            
            logger.info(f"✅ Imaging Summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Imaging summary LLM failed: {e}")
            return self._build_manual_imaging_summary(date, doc_type, doctor_section, body_part, contrast, status, finding)

    def _build_manual_imaging_summary(
        self,
        date: str,
        doc_type: str,
        doctor_section: str,
        body_part: str,
        contrast: str,
        status: str,
        finding: str
    ) -> str:
        """
        Manual summary construction with explicit field labels.
        Fallback when LLM fails or produces unlabeled output.
        """
        parts = [f"{date}: {doc_type}"]
        
        if doctor_section:
            parts.append(doctor_section)
        
        if body_part:
            body_str = body_part
            if contrast:
                body_str += f" ({contrast})"
            parts.append(f"for {body_str}")
        
        # Impression and findings with labels
        sections = []
        if status:
            sections.append(f"Impression → {status}")
        if finding:
            sections.append(f"Findings → {finding}")
        
        if sections:
            parts.append(f"| {' | '.join(sections)}")
        
        summary = " ".join(parts)
        logger.info(f"✅ Manual imaging summary built: {summary}")
        return summary

    def _extract_physician_last_name(self, physician_name: str) -> str:
        """Extract last name from physician name string"""
        if not physician_name:
            return ""

        # Remove common titles and suffixes
        clean_name = (
            physician_name
            .replace("Dr.", "")
            .replace("MD", "")
            .replace("DO", "")
            .replace("M.D.", "")
            .replace("D.O.", "")
            .replace("MBBS", "")
            .replace("MBChB", "")
            .replace(",", "")
            .strip()
        )

        # Get the last word as last name
        parts = clean_name.split()
        if parts:
            last_name = parts[-1]
            logger.info(f"  🔍 Extracted last name: '{last_name}' from '{physician_name}'")
            return last_name
        return ""
