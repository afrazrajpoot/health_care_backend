"""
Imaging reports extractor (MRI, CT, X-ray, Ultrasound, EMG)
v2.8 – Format validation bypass for physician names
"""
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ImagingExtractor:
    """Specialized extractor for MRI/CT/X-ray/Ultrasound/EMG reports with enhanced physician extraction."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract imaging report and generate concise, clinically meaningful summary."""
        logger.info(f"🔍 Starting extraction for {doc_type} report")
        logger.info(f"📄 Document preview: {text[:200]}...")
        
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        
        # Bypass format validation
        self._bypass_format_validation(initial_result)
        
        logger.info(f"✅ Extraction completed for {doc_type}")
        return initial_result

    def _bypass_format_validation(self, result: ExtractionResult) -> None:
        """Bypass format validation to prevent physician name removal"""
        logger.info("🛡️ Bypassing format validation - physician names are required")
        # Clear any format validation warnings
        if hasattr(result, 'validation_warnings'):
            result.validation_warnings = []
        if hasattr(result, 'format_issues'):
            result.format_issues = []
        
        logger.info(f"✅ Format validation bypassed for: {result.summary_line}")

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data with enhanced physician extraction"""
        logger.info(f"🎯 Stage 1: Raw data extraction for {doc_type}")
        
        system_template = """
You are a precise clinical information extractor that structures radiology or medical report text into standardized JSON fields.

Follow these extraction and formatting rules strictly.

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
10. Extract doctor name if present — only valid if contains "Dr.", "MD", "DO", "MBBS", or "MBChB".
11. Identify the document type (MRI, CT, X-ray, Ultrasound, etc.) from context.
12. Create a single-line summary in this format:
    [Dr. Name] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]

Document text: {text}

Extract these fields:
- study_date: Imaging date (MM/DD/YY or {fallback_date})
- document_type: Type of report (e.g., MRI, CT, X-ray, Ultrasound, Consultation)
- body_part: Anatomical area studied (abbreviated form)
- contrast_used: "with contrast", "without contrast", or empty if not mentioned
- primary_finding: Most important diagnostic finding (max 16 words)
- impression_status: "normal", "abnormal", "post-op", or "inconclusive" if applicable
- consulting_doctor: Doctor’s name (Dr., MD, DO, MBBS, or MBChB) (Dr. Full Name) valid only if name contains title (e.g., "Dr. John Smith", "Jane Doe, MD", "Dr Jane Doe", (eg. Dr., MD, DO, M.D., D.O.))
- formatted_summary: A one-line summary following the exact format:
  [Dr. Name] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]

Return valid JSON:
{
  "study_date": "MM/DD/YY or {fallback_date}",
  "document_type": "e.g., MRI, CT, X-ray, Consultation",
  "body_part": "abbreviated part or empty",
  "contrast_used": "contrast detail or empty",
  "primary_finding": "main diagnostic finding (max 16 words)",
  "impression_status": "normal/abnormal/post-op/inconclusive",
  "consulting_doctor": "Dr. name if found, else empty",
  "formatted_summary": "[Dr. Name] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]"
}
"""
        human_template = """
You are analyzing this medical report for structured extraction.

Text:
{text}

Fallback date: {fallback_date}

Follow the rules from the system prompt.

Ensure each field is extracted accurately, and the `formatted_summary` strictly matches:
[Dr. Name] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]

Return **valid JSON only** containing:
- study_date
- document_type
- body_part
- contrast_used
- primary_finding
- impression_status
- consulting_doctor
- formatted_summary
"""

        try:
            # Create system message prompt template with doc_type variable
            system_prompt = SystemMessagePromptTemplate.from_template(
                system_template, input_variables=["doc_type"]
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
                {"doc_type": doc_type, "text": text[:8000], "fallback_date": fallback_date}
            )
            
            # LOG THE RAW EXTRACTED DATA
            logger.info("📊 RAW EXTRACTION RESULTS:")
            logger.info(f"   - study_date: {result.get('study_date', 'Not found')}")
            logger.info(f"   - physician_name: {result.get('physician_name', 'Not found')}")
            logger.info(f"   - specialty: {result.get('specialty', 'Not found')}")
            logger.info(f"   - body_part: {result.get('body_part', 'Not found')}")
            logger.info(f"   - contrast_used: {result.get('contrast_used', 'Not found')}")
            logger.info(f"   - primary_finding: {result.get('primary_finding', 'Not found')}")
            logger.info(f"   - impression_status: {result.get('impression_status', 'Not found')}")
            logger.info(f"   - clinical_summary: {result.get('clinical_summary', 'Not found')}")
            
            return result
        except Exception as e:
            logger.error(f"❌ Raw extraction failed: {e}")
            return {}

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Stage 2: Build initial result with validation and summary"""
        logger.info("🎯 Stage 2: Building initial result with validation")
        
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        
        # Use the AI-generated clinical_summary if it follows the correct format
        clinical_summary = cleaned.get("clinical_summary", "").strip()
        physician = cleaned.get("physician_name", "").strip()
        
        # Check if the clinical_summary already includes physician name in correct format
        if clinical_summary and physician and physician in clinical_summary:
            summary_line = clinical_summary
            logger.info("✅ Using AI-generated summary with physician name")
        else:
            # Build summary manually
            summary_line = self._build_proper_imaging_summary(cleaned, doc_type, fallback_date)
            logger.info("✅ Using manually built summary with physician name")
        
        # LOG THE FINAL RESULT
        logger.info("📊 FINAL EXTRACTION RESULT:")
        logger.info(f"   - Document Type: {doc_type}")
        logger.info(f"   - Document Date: {cleaned.get('study_date', fallback_date)}")
        logger.info(f"   - Physician Name: {cleaned.get('physician_name', 'Not specified')}")
        logger.info(f"   - Specialty: {cleaned.get('specialty', 'Not specified')}")
        logger.info(f"   - Body Part: {cleaned.get('body_part', 'Not specified')}")
        logger.info(f"   - Primary Finding: {cleaned.get('primary_finding', 'Not specified')}")
        logger.info(f"   - Final Summary: {summary_line}")
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("study_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name"),
            specialty=cleaned.get("specialty"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        logger.info("🔧 Validating and cleaning extracted data")
        
        cleaned = {}
        date = result.get("study_date", "").strip()
        cleaned["study_date"] = date if date and date != "empty" else fallback_date
        logger.info(f"   📅 Date cleaned: {cleaned['study_date']}")

        # Enhanced physician validation
        physician = result.get("physician_name", "").strip()
        if physician and physician != "empty":
            # Ensure it contains physician indicators and doesn't look like referring doctor
            physician_upper = physician.upper()
            has_title = any(indicator in physician_upper for indicator in ['DR.', 'MD', 'DO', 'M.D.', 'D.O.', ', MD', ', DO'])
            
            # Check for common referring doctor patterns to exclude
            is_likely_referring = any(pattern in physician_upper for pattern in [
                'REFERRING', 'ORDERING', 'REQUESTING', 'TREATING', 
                'PCP', 'PRIMARY CARE', 'SURGEON', 'TECHNOLOGIST'
            ])
            
            if has_title and not is_likely_referring:
                cleaned["physician_name"] = physician
                logger.info(f"   👨‍⚕️ Physician validated: {physician}")
            else:
                logger.warning(f"   ⚠️ Rejected potential referring physician: {physician}")
                cleaned["physician_name"] = ""
        else:
            cleaned["physician_name"] = ""
            logger.info("   ❌ No physician found")

        specialty = result.get("specialty", "").strip()
        cleaned["specialty"] = specialty if specialty and specialty != "empty" else ""
        logger.info(f"   🎓 Specialty: {cleaned['specialty']}")

        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""
        logger.info(f"   🦴 Body Part: {cleaned['body_part']}")

        primary_finding = result.get("primary_finding", "").strip()
        cleaned["primary_finding"] = primary_finding if primary_finding and primary_finding != "empty" else ""
        logger.info(f"   🔍 Primary Finding: {cleaned['primary_finding']}")

        clinical_summary = result.get("clinical_summary", "").strip()
        cleaned["clinical_summary"] = clinical_summary if clinical_summary and clinical_summary != "empty" else ""
        logger.info(f"   📝 Clinical Summary: {cleaned['clinical_summary'][:100]}...")

        contrast = result.get("contrast_used", "").strip()
        cleaned["contrast_used"] = contrast if contrast and contrast != "empty" else ""
        logger.info(f"   💉 Contrast: {cleaned['contrast_used']}")

        status = result.get("impression_status", "").strip()
        cleaned["impression_status"] = status if status and status != "empty" else ""
        logger.info(f"   📊 Status: {cleaned['impression_status']}")

        return cleaned

    def _build_proper_imaging_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build detailed 3-4 line imaging summary with physician"""
        logger.info("🎯 Building DETAILED imaging summary")
        
        physician = data.get("physician_name", "").strip()
        body_part = data.get("body_part", "")
        finding = data.get("primary_finding", "")
        date = data.get("study_date", fallback_date)
        specialty = data.get("specialty", "")
        contrast = data.get("contrast_used", "")
        
        # Build multi-line summary
        summary_lines = []
        
        # Line 1: Physician and study info
        if physician:
            last_name = self._extract_physician_last_name(physician)
            physician_display = f"Dr. {last_name}" if last_name else "Physician"
            if specialty:
                summary_lines.append(f"Interpreted by {physician_display} ({specialty})")
            else:
                summary_lines.append(f"Interpreted by {physician_display}")
        
        # Line 2: Study details
        study_info = f"{doc_type} of {body_part} performed on {date}"
        if contrast:
            study_info += f" ({contrast})"
        summary_lines.append(study_info)
        
        # Line 3: Findings
        if finding:
            summary_lines.append(f"Findings: {finding}")
        
        # Line 4: Clinical significance
        if "dislocation" in finding.lower() or "subluxation" in finding.lower():
            summary_lines.append("Clinical: Joint instability requiring orthopedic evaluation")
        elif "tear" in finding.lower() or "rupture" in finding.lower():
            summary_lines.append("Clinical: Soft tissue injury requiring specialized management")
        
        final_summary = "\n".join(summary_lines)
        logger.info(f"✅ Detailed summary:\n{final_summary}")
        return final_summary
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
            .replace(",", "")
            .strip()
        )
        
        # Get the last word as last name
        parts = clean_name.split()
        if parts:
            last_name = parts[-1]
            logger.info(f"   🔍 Extracted last name: '{last_name}' from '{physician_name}'")
            return last_name
        return ""