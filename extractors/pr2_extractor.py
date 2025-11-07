"""
PR-2 Progress Report extractor (v3.2 – Simplified to match ImagingExtractor structure)
"""
import logging
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class PR2Extractor:
    """Specialized extractor for PR-2 Progress Reports with deep medical analysis."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract PR-2 report data with comprehensive medical analysis."""
        raw_result = self._extract_raw_data(text, doc_type, fallback_date)
        initial_result = self._build_initial_result(raw_result, doc_type, fallback_date)
        return initial_result

    def _extract_raw_data(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Stage 1: Extract raw structured data with deep medical analysis"""
        system_template = """
You are an expert Medical AI Assistant specializing in workers' compensation case analysis. Your role is to help physicians by performing deep, thorough analysis of PR-2 (Progress Report) documents.

━━━ PHYSICIAN EXTRACTION (CRITICAL) ━━━
CONSULTING/TREATING PHYSICIAN EXTRACTION RULES:
- MUST extract the PRIMARY TREATING PHYSICIAN who authored and signed this PR-2 report
- Look for these explicit indicators:
  * "Electronically signed by:", "Signed by:", "Attestation:"
  * Signature blocks at report end
  * "Provider:", "Attending Physician:", "Treating Physician:"
  * "I, [Dr. Name], have examined..."
- IGNORE ALL OTHER DOCTORS AND STAFF NAMES including:
  * Consulting physicians mentioned in body
  * Referring physicians
  * Specialists for future referrals
  * Physicians from prior treatment history
  * Any doctor mentioned as "referred to Dr. X"
- Extract FULL NAME with title (e.g., "Dr. Jane Smith", "John Doe, MD")
- If no clear treating physician found → leave physician_name EMPTY
- PRIMARY FOCUS: Signature blocks, "Dictated by:", "Report by:", "Electronically signed by:"
- MUST have title: "Dr.", "MD", "DO", "M.D.", "D.O."
- IGNORE all other doctors mentioned in content (referrals, consults, PCPs)
- If name found WITHOUT title → ADD to verification_notes: "Consulting doctor lacks title: [name]"
- If no clear author/signer → "Not specified"
CORE COMPETENCIES:
- Extract ALL medically relevant information from the document
- Identify clinical trends, improvements, or deterioration
- Recognize treatment efficacy and response patterns
- Flag critical information requiring physician attention
- Understand medical terminology, abbreviations, and context
- Synthesize complex medical data into actionable insights

ANALYSIS APPROACH:
1. Read the ENTIRE document carefully, not just surface-level information
2. Identify explicit AND implicit clinical information
3. Connect related findings across different sections
4. Note any discrepancies, concerns, or red flags
5. Consider the workers' compensation context (MMI, work restrictions, causality)
6. Extract specific details: medications with dosages, therapy frequencies, objective measurements
7. Capture the clinical narrative and trajectory of care

OUTPUT REQUIREMENTS:
- Be precise and specific (not vague or generic)
- Include quantifiable data when available (ROM degrees, pain scales, visit frequencies)
- Preserve medical accuracy and terminology
- Prioritize actionable information for case management
- Flag any inconsistencies or concerns
"""

        human_template = """
Analyze this PR-2 Progress Report and extract comprehensive medical information.

DOCUMENT TEXT:
{text}

EXTRACTION FIELDS (extract ALL relevant information for each):

1. **report_date**: Date of this report (format: MM/DD/YY, or use {fallback_date} if not found)

2. **physician_name**: The PRIMARY TREATING PHYSICIAN who authored this PR-2 report (not consulting or referring physicians). Look for:
   - Electronic signature at the END of the document (most reliable)
   - "Electronically signed by:" or "Signed by:"
   - Attestation sections at bottom
   - "Provider:" or "Attending Physician:" in header
   - Signature blocks with dates matching the report date
   
   CRITICAL RULES:
   - IGNORE consulting physicians, specialists mentioned in the body
   - IGNORE referring physicians or "referred to Dr. X"
   - IGNORE physicians mentioned in treatment history
   - PRIORITIZE the signature block at the document END
   - Only return the physician who actually wrote and signed THIS report
   - DO NOT use placeholder names like "[Name]" or "REDACTED"
   - If the actual name is redacted/masked, return empty string ""

3. **body_part**: Primary anatomical area(s) being treated. Include laterality (R/L) and be specific (e.g., "Right shoulder rotator cuff", "Lumbar spine L4-L5", "Left knee medial meniscus").

4. **current_status**: Detailed clinical status including:
   - Subjective complaints and pain levels
   - Objective findings (ROM, strength, special tests)
   - Functional limitations
   - Comparison to previous visits (improved/unchanged/worsened)
   - Any complications or new issues
   
5. **treatment_recommendations**: Comprehensive treatment plan including:
   - Current medications (names, dosages, frequencies)
   - Physical therapy (type, frequency, duration)
   - Injections or procedures performed or planned
   - Diagnostic tests ordered
   - Modalities (ice, heat, TENS, etc.)
   - Home exercise program details
   - Any treatment modifications or discontinuations

6. **work_status**: Complete work capacity assessment:
   - Current work status (TTD, modified duty, full duty, etc.)
   - Specific restrictions (lifting limits, positioning, repetitive activities)
   - Duration of restrictions
   - Anticipated changes to work status
   - Return to work timeline if applicable

7. **next_plan**: Future care plan including:
   - Follow-up appointment timing and purpose
   - Anticipated next steps in treatment
   - Specialist referrals or consultations
   - MMI considerations or timeline
   - Any pending decisions or evaluations
   - Goals for next visit

8. **critical_flags**: Any concerning findings requiring attention:
   - Non-compliance with treatment
   - Unexpected deterioration
   - Complications or adverse reactions
   - Inconsistencies in presentation
   - Medicolegal concerns
   - Red flag symptoms

9. **clinical_trajectory**: Overall progress assessment:
   - Direction of clinical course (improving/plateauing/declining)
   - Response to current treatment
   - Likelihood of achieving MMI
   - Estimated timeline to recovery or MMI

Return ONLY valid JSON with this exact structure:
{{
  "report_date": "MM/DD/YY",
  "physician_name": "Dr. Full Name or empty",
  "body_part": "Specific anatomical area with laterality",
  "current_status": "Detailed clinical status with objective and subjective findings",
  "treatment_recommendations": "Complete treatment plan with specifics",
  "work_status": "Full work capacity assessment with restrictions",
  "next_plan": "Detailed follow-up and future care plan",
  "critical_flags": "Any concerns or red flags, or empty string if none",
  "clinical_trajectory": "Overall progress and prognosis assessment"
}}

IMPORTANT: 
- Extract actual information from the document, not generic placeholders
- Include specific numbers, dates, and measurements when available
- If information is truly not present, use empty string ""
- Be thorough but concise - focus on clinical relevance
- Maintain medical accuracy and terminology

{format_instructions}
"""

        try:
            # Create system message prompt template
            system_prompt = SystemMessagePromptTemplate.from_template(
                system_template, input_variables=[]
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

            # Execute extraction
            chain = chat_prompt | self.llm | self.parser
            result = chain.invoke({
                "text": text[:8000],
                "fallback_date": fallback_date
            })

            return result

        except Exception as e:
            logger.error(f"❌ PR-2 raw extraction failed: {e}")
            return {}

    def _build_initial_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Stage 2: Build initial result with validation and summary"""
        cleaned = self._validate_and_clean(raw_data, fallback_date)
        summary_line = self._build_pr2_summary(cleaned, doc_type, fallback_date)
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("report_date", fallback_date),
            summary_line=summary_line,
            examiner_name=cleaned.get("physician_name"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )

    def _validate_and_clean(self, result: Dict, fallback_date: str) -> Dict:
        """Validate and clean extracted data"""
        cleaned = {}
        
        # Date validation
        date = result.get("report_date", "").strip()
        cleaned["report_date"] = date if date and date != "empty" else fallback_date

        # Physician validation
        physician = result.get("physician_name", "").strip()
        if physician and physician != "empty":
            # Ensure it contains physician indicators
            physician_upper = physician.upper()
            has_title = any(indicator in physician_upper for indicator in ['DR.', 'MD', 'DO', 'M.D.', 'D.O.', ', MD', ', DO'])
            
            if has_title:
                cleaned["physician_name"] = physician
            else:
                logger.warning(f"Physician name lacks professional title: {physician}")
                cleaned["physician_name"] = ""
        else:
            cleaned["physician_name"] = ""

        # Clean other fields
        body_part = result.get("body_part", "").strip()
        cleaned["body_part"] = body_part if body_part and body_part != "empty" else ""

        current_status = result.get("current_status", "").strip()
        cleaned["current_status"] = current_status if current_status and current_status != "empty" else ""

        treatment_recommendations = result.get("treatment_recommendations", "").strip()
        cleaned["treatment_recommendations"] = treatment_recommendations if treatment_recommendations and treatment_recommendations != "empty" else ""

        work_status = result.get("work_status", "").strip()
        cleaned["work_status"] = work_status if work_status and work_status != "empty" else ""

        next_plan = result.get("next_plan", "").strip()
        cleaned["next_plan"] = next_plan if next_plan and next_plan != "empty" else ""

        critical_flags = result.get("critical_flags", "").strip()
        cleaned["critical_flags"] = critical_flags if critical_flags and critical_flags != "empty" else ""

        clinical_trajectory = result.get("clinical_trajectory", "").strip()
        cleaned["clinical_trajectory"] = clinical_trajectory if clinical_trajectory and clinical_trajectory != "empty" else ""

        return cleaned

    def _build_pr2_summary(self, data: Dict, doc_type: str, fallback_date: str) -> str:
        """Build concise, human-readable summary line for PR-2 report"""
        date = data.get("report_date", fallback_date)
        physician = data.get("physician_name", "")
        body_part = data.get("body_part", "")
        trajectory = data.get("clinical_trajectory", "")
        flags = data.get("critical_flags", "")

        # Build summary parts
        summary_parts = [f"{date}: {doc_type}"]
        
        # Add physician if available
        if physician:
            last_name = (
                physician.replace("Dr.", "")
                .replace("MD", "")
                .replace("DO", "")
                .replace("M.D.", "")
                .replace("D.O.", "")
                .strip()
                .split()[-1]
            )
            if last_name:
                summary_parts.append(f"(Dr {last_name})")
        
        # Add body part if available
        if body_part:
            summary_parts.append(f"- {body_part}")
        
        # Add trajectory if available (shortened)
        if trajectory:
            trajectory_short = trajectory.split('.')[0][:50]
            summary_parts.append(f"| {trajectory_short}")
        
        # Add critical flags if present
        if flags:
            flags_short = flags.split('.')[0][:40]
            summary_parts.append(f"⚠️ {flags_short}")

        summary = " ".join(summary_parts)
        
        # Ensure summary is readable length
        if len(summary) > 150:
            summary = summary[:147] + "..."

        logger.info(f"✅ PR-2 Summary: {summary}")
        return summary