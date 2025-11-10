"""
Generic extractor for simple document types (RFA, UR, Auth, Admin letters, etc.)
Enhanced with optional DoctorDetector integration and labeled summary format.
"""
import re
import logging
from typing import Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.doctor_detector import DoctorDetector

logger = logging.getLogger("document_ai")


class SimpleExtractor:
    """Generic extractor for simpler document types with optional doctor detection"""

    # Universal clarity guide
    CLEAR_EXTRACTION_GUIDE = (
        "Ensure all extracted information is explicit and actionable. "
        "When listing body parts, diagnoses, or findings, name each explicitly. "
        "Avoid vague terms like '+1 more', 'etc.', or incomplete fragments. "
        "Always produce full, meaningful, concise phrases (30-60 words for summaries). "
        "Do NOT extract physician/doctor names - this is handled separately. "
        "\n\nCRITICAL REASONING RULES:\n"
        "1. ONLY extract POSITIVE/ACTIONABLE findings. DO NOT extract negative statements.\n"
        "   ✗ BAD: 'No treatment needed', 'Not approved', 'No restrictions'\n"
        "   ✓ GOOD: 'PT 6 visits approved', 'MRI denied - insufficient medical necessity'\n"
        "2. If a field has NO meaningful positive data, return empty string '' - DO NOT return negative phrases.\n"
        "3. For denials/UR: Include the reason if it provides actionable information.\n"
        "4. REASONING CHECK: Before returning each field, ask yourself:\n"
        "   - 'Is this information ACTIONABLE for the treating physician?'\n"
        "   - 'Does this tell me what TO DO or what IS happening?'\n"
        "   - If answer is NO → return empty string for that field"
    )

    TEMPLATES = {
        "RFA": {
            "fields": ["date", "service_requested", "body_part"],
            "format": "[DATE]: RFA{doctor_section} | Service → {service_requested} | Body part → {body_part}",
            "prompt": (
                "Extract: date (MM/DD/YY), service_requested (e.g., 'PT 6 visits', 'MRI L-spine'), "
                "and body_part (list clearly; e.g., 'R shoulder', 'L knee'). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,  # NEW: flag to include doctor detection
        },
        "UR": {
            "fields": ["date", "service_denied", "reason"],
            "format": "[DATE]: UR Decision{doctor_section} | Service denied → {service_denied} | Reason → {reason}",
            "prompt": (
                "Extract: date (MM/DD/YY), service_denied (what was denied, e.g., 'MRI', 'PT'), "
                "and reason (brief rationale, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,
        },
        "Authorization": {
            "fields": ["date", "service_approved", "body_part"],
            "format": "[DATE]: Authorization{doctor_section} | Service approved → {service_approved} | Body part → {body_part}",
            "prompt": (
                "Extract: date (MM/DD/YY), service_approved (e.g., 'MRI', 'PT 6 visits'), "
                "and body_part (e.g., 'R shoulder'). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,
        },
        "DFR": {
            "fields": ["date", "doi", "diagnosis", "plan"],
            "format": "[DATE]: DFR{doctor_section} | DOI → {doi} | Diagnosis → {diagnosis} | Plan → {plan}",
            "prompt": (
                "Extract: report date (MM/DD/YY), DOI (date of injury in MM/DD/YY), "
                "primary diagnosis, and initial treatment plan (max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,
        },
        "PR-4": {
            "fields": ["date", "mmi_status", "future_care"],
            "format": "[DATE]: PR-4{doctor_section} | MMI Status → {mmi_status} | Future care → {future_care}",
            "prompt": (
                "Extract: date (MM/DD/YY), mmi_status ('MMI reached', 'Ongoing treatment', or 'Deferred'), "
                "and future_care (future medical needs, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,
        },
        "Adjuster": {
            "fields": ["date", "request"],
            "format": "[DATE]: Adjuster letter | Request → {request}",
            "prompt": (
                "Extract: date (MM/DD/YY), and request (what is being requested, max 20 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": False,
        },
        "Attorney": {
            "fields": ["date", "side", "topic"],
            "format": "[DATE]: {side} Attorney | Topic → {topic}",
            "prompt": (
                "Extract: date (MM/DD/YY), side ('Applicant' or 'Defense'), "
                "and topic (main subject, max 20 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": False,
        },
        "NCM": {
            "fields": ["date", "topic"],
            "format": "[DATE]: Nurse Case Manager | Topic → {topic}",
            "prompt": (
                "Extract: date (MM/DD/YY), and topic (main update or request, max 20 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": False,
        },
        "Signature Request": {
            "fields": ["date", "form_type"],
            "format": "[DATE]: Signature Request | Form → {form_type}",
            "prompt": (
                "Extract: date (MM/DD/YY), and form_type (what form requires signature, max 20 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": False,
        },
        "Referral": {
            "fields": ["date", "specialty", "reason"],
            "format": "[DATE]: Referral | Specialty → {specialty} | Reason → {reason}",
            "prompt": (
                "Extract: date (MM/DD/YY), specialty (where patient is referred, e.g., 'Orthopedics'), "
                "and reason (brief, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,
        },
        "Discharge": {
            "fields": ["date", "diagnosis", "plan"],
            "format": "[DATE]: Discharge{doctor_section} | Diagnosis → {diagnosis} | Plan → {plan}",
            "prompt": (
                "Extract: date (MM/DD/YY), primary discharge diagnosis, "
                "and discharge plan (max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": True,
        },
        "Med Refill": {
            "fields": ["date", "medication"],
            "format": "[DATE]: Med Refill | Medication → {medication}",
            "prompt": (
                "Extract: date (MM/DD/YY), and medication (include name + dose, e.g., 'Ibuprofen 800mg'). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": False,
        },
        "Labs": {
            "fields": ["date", "key_abnormal"],
            "format": "[DATE]: Lab Results | Key abnormal → {key_abnormal}",
            "prompt": (
                "Extract: date (MM/DD/YY), and key_abnormal (most critical abnormal value, max 16 words). "
                + CLEAR_EXTRACTION_GUIDE
            ),
            "include_doctor": False,
        },
    }

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.doctor_detector = DoctorDetector(llm)
        logger.info("✅ SimpleExtractor initialized with DoctorDetector")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None
    ) -> ExtractionResult:
        """
        Generic extraction for simple document types with optional doctor detection.
        
        Args:
            text: Layout-preserved text from Document AI
            doc_type: Document type (RFA, UR, Authorization, etc.)
            fallback_date: Fallback date if not found
            page_zones: Per-page zone extraction (optional, for doctor detection)
            raw_text: Original flat text (for backward compatibility)
        """
        template = self.TEMPLATES.get(doc_type)
        if not template:
            logger.warning(f"⚠️ No template for {doc_type}")
            return ExtractionResult(
                document_type=doc_type,
                document_date=fallback_date,
                summary_line=f"{doc_type} {fallback_date} = manual review required",
                raw_data={},
            )

        # Stage 1: Extract clinical/document data (NO doctor extraction)
        raw_result = self._extract_data(text, doc_type, template, fallback_date)
        
        # Stage 2: Optional doctor detection (for clinical documents)
        physician_name = ""
        if template.get("include_doctor", False):
            logger.info(f"🔍 {doc_type} requires doctor detection (include_doctor=True)")
            physician_name = self._detect_physician(text, page_zones, doc_type)
            raw_result["physician_name"] = physician_name
            logger.info(f"🎯 {doc_type} physician_name set to: '{physician_name}'")
        else:
            logger.info(f"ℹ️ {doc_type} does not require doctor detection (include_doctor=False)")
        
        # Stage 3: Build summary with labels
        summary_line = self._build_summary(raw_result, template, doc_type, fallback_date, physician_name)
        logger.info(f"📝 {doc_type} summary built: {summary_line}")
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=raw_result.get("date", fallback_date),
            summary_line=summary_line,
            examiner_name=physician_name if physician_name else None,
            raw_data=raw_result,
        )

    def _extract_data(self, text: str, doc_type: str, template: Dict, fallback_date: str) -> Dict:
        """Extract document-specific data (NO doctor extraction)"""
        field_json = ", ".join([f'"{field}": "value"' for field in template["fields"]])
        
        prompt = PromptTemplate(
            template="""
You are an AI Medical Assistant extracting structured data from a {doc_type} document.

INSTRUCTION: {instruction}

RULES:
- Extract ONLY explicitly stated information
- DO NOT extract physician/doctor names - handled separately
- Use MM/DD/YY date format
- If field not found, return empty string
- Follow word limits strictly
- Avoid vague phrases (e.g., '+1 more', 'etc.')
- Ensure output is clear and meaningful

Document text:
{text}

Return JSON:
{{{field_json}}}

{format_instructions}
""",
            input_variables=["text", "doc_type", "instruction", "field_json"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        try:
            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:6000],
                "doc_type": doc_type,
                "instruction": template["prompt"],
                "field_json": field_json,
            })
            
            # Use fallback date if not extracted
            if "date" in result and not result["date"]:
                result["date"] = fallback_date
            
            logger.info(f"✅ {doc_type} data extraction complete")
            return result
        except Exception as e:
            logger.error(f"❌ {doc_type} extraction failed: {e}")
            return {"date": fallback_date}

    def _detect_physician(
        self,
        text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]],
        doc_type: str
    ) -> str:
        """Optional doctor detection for clinical documents"""
        logger.info(f"🔍 Running DoctorDetector for {doc_type} (zone-aware)...")
        
        # Debug: Check if page_zones is provided
        if page_zones:
            logger.info(f"✅ {doc_type} extractor received page_zones with {len(page_zones)} pages: {list(page_zones.keys())}")
        else:
            logger.warning(f"⚠️ {doc_type} extractor did NOT receive page_zones")
        
        detection_result = self.doctor_detector.detect_doctor(
            text=text,
            page_zones=page_zones
        )
        
        if detection_result["doctor_name"]:
            logger.info(
                f"✅ Physician detected: {detection_result['doctor_name']} "
                f"(confidence: {detection_result['confidence']}, source: {detection_result['source']})"
            )
            physician_name = detection_result["doctor_name"]
            logger.info(f"🎯 {doc_type} extractor returning physician: '{physician_name}'")
            return physician_name
        else:
            logger.warning(f"⚠️ No valid physician found for {doc_type}: {detection_result['validation_notes']}")
            logger.info(f"🎯 {doc_type} extractor returning empty physician name")
            return ""

    def _build_summary(
        self,
        data: Dict,
        template: Dict,
        doc_type: str,
        fallback_date: str,
        physician_name: str
    ) -> str:
        """Build labeled summary from template with explicit field markers"""
        date = data.get("date", fallback_date)
        
        # Build doctor section if included
        doctor_section = ""
        if physician_name:
            doctor_section = f" - {physician_name}"
            logger.info(f"🩺 Building doctor_section for {doc_type}: '{doctor_section}'")
        else:
            logger.info(f"ℹ️ No physician name for {doc_type}, doctor_section will be empty")
        
        # Start with format template
        summary = template["format"]
        logger.info(f"📋 Template format: {summary}")
        
        # Replace date
        summary = summary.replace("[DATE]", date)
        
        # Replace doctor section
        summary = summary.replace("{doctor_section}", doctor_section)
        logger.info(f"📝 After doctor_section replacement: {summary}")
        
        # Replace all field values
        for field, value in data.items():
            if field != "date" and field != "physician_name" and value:
                summary = summary.replace(f"{{{field}}}", str(value))
        
        # Clean up empty placeholders and extra spacing
        summary = re.sub(r"\{[^}]+\}\s*", "", summary)  # Remove unfilled placeholders
        summary = re.sub(r"\|\s*\|", "|", summary)  # Remove double pipes
        summary = re.sub(r"\s+", " ", summary)  # Normalize spacing
        summary = summary.strip()
        
        # Remove trailing separators
        summary = re.sub(r"\s*[|→]\s*$", "", summary)
        
        logger.info(f"✅ {doc_type} Summary: {summary}")
        return summary
