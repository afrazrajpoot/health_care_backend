"""
PR-2 Progress Report Extractor - Enhanced physician recognition with referral doctor fallback
"""
import logging
import re
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class PR2Extractor:
    """Enhanced extractor for PR-2 Progress Reports with referral doctor fallback."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract PR-2 report with proper physician identification."""
        raw_result = self._extract_medical_content(text, doc_type, fallback_date)
        result = self._build_result(raw_result, doc_type, fallback_date)
        return result

    def _extract_medical_content(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Extract progress updates and identify physicians efficiently."""
        
        system_template = """You are a medical data extraction specialist analyzing PR-2 Progress Reports.

CRITICAL PHYSICIAN IDENTIFICATION PROTOCOL:

1. PRIMARY PHYSICIAN (Treating/Signing):
   - MUST have medical credentials: Dr., MD, M.D., DO, D.O., MBBS
   - Look for: "Treating physician:", "Provider:", "Examined by:", "Attending:"
   - Check signatures: "Electronically signed by:", "Dictated by:", "Signed by:"

2. REFERRAL DOCTOR FALLBACK (ONLY if no primary physician found):
   - Use referral doctor ONLY when no treating/signing physician is identified
   - Look for: "Referred by:", "Referral from:", "PCP:", "Primary Care:"
   - Must have medical credentials
   - Examples: ✓ Referred by: Dr. James Wilson

3. REJECT NON-DOCTOR SIGNATORIES:
   - ✗ Reject names without medical credentials (e.g., "Syed Akbar")
   - ✗ Reject administrators, case managers, coordinators
   - ✗ Reject therapists, technicians, assistants
   - ✗ Reject any name without proper medical credentials

4. PRIORITY ORDER:
   1. Treating Physician (with credentials)
   2. Signing Physician (with credentials)  
   3. Referral Doctor (with credentials, ONLY if no treating/signing physician)
   4. "Not specified" (if no qualified doctors found)

VALID PHYSICIAN FORMATS:
   ✓ "Dr. Robert Chen"
   ✓ "Lisa Martinez, MD"
   ✓ "Dr. David Kumar, DO"
   ✓ "Emily Thompson M.D."

IMMEDIATE REJECTION:
   ✗ Names without medical credentials
   ✗ "Case Manager", "Coordinator", "Administrator"
   ✗ "Therapist", "Technician", "Assistant"
   ✗ Isolated names: "John Doe", "Syed Akbar"

PR-2 SPECIFIC EXTRACTION:
Focus on documenting patient progress and status changes:
- Current clinical status and functional improvements/declines
- Treatment modifications and medication adjustments
- Work capacity and restrictions
- Therapy progress and outcomes
- Pain levels and symptom changes
- Follow-up plans and duration of restrictions

REQUIRED OUTPUT:
- report_date: Date in MM/DD/YY format
- treating_physician: Primary doctor documenting progress (MUST have credentials)
- signing_physician: Doctor who signed/authorized (MUST have credentials)
- referral_physician: Referral doctor if explicitly mentioned
- clinical_progress: Current status and changes since last visit (1-2 sentences)
- treatment_changes: New treatments, medications, or therapy modifications (1 sentence)
- work_capacity: Current work status, restrictions, or return-to-work timeline (1 sentence)
- recommendations: Next steps, follow-up, or ongoing care plan (1 sentence)

Return clean JSON without explanations or markdown."""

        human_template = """Extract information from this PR-2 Progress Report:

{text}

PHYSICIAN EXTRACTION PRIORITY:
1. First: Treating/Signing physician with credentials (Dr./MD/DO/MBBS)
2. Fallback: Referral doctor with credentials (ONLY if no treating/signing physician)
3. Last: "Not specified" (if no qualified doctors)

JSON format:
{format_instructions}"""
        
        try:
            system_prompt = SystemMessagePromptTemplate.from_template(system_template)
            
            human_prompt_template = PromptTemplate(
                template=human_template,
                input_variables=["text", "doc_type"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            human_prompt = HumanMessagePromptTemplate(prompt=human_prompt_template)

            chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            chain = chat_prompt | self.llm | self.parser
            
            result = chain.invoke({
                "text": text,
                "doc_type": doc_type
            })
            
            logger.info("📊 PR-2 Extraction Results:")
            logger.info(f"   - Treating Physician: {result.get('treating_physician', 'Not found')}")
            logger.info(f"   - Signing Physician: {result.get('signing_physician', 'Not found')}")
            logger.info(f"   - Referral Physician: {result.get('referral_physician', 'Not found')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ PR-2 extraction failed: {e}")
            return {}

    def _build_result(self, raw_data: Dict, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Build final result with cleaned data and referral doctor fallback."""
        
        cleaned = self._clean_extracted_data(raw_data, fallback_date)
        
        # Apply referral doctor fallback logic
        final_physician = self._apply_referral_fallback(cleaned)
        cleaned["final_physician"] = final_physician
        
        summary = self._build_targeted_summary(cleaned, doc_type)
        
        return ExtractionResult(
            document_type=doc_type,
            document_date=cleaned.get("report_date", fallback_date),
            summary_line=summary,
            examiner_name=final_physician,
            body_parts=[],
            raw_data=cleaned,
        )

    def _apply_referral_fallback(self, data: Dict) -> str:
        """
        Apply referral doctor fallback logic:
        - Primary: Treating physician with credentials
        - Secondary: Signing physician with credentials  
        - Fallback: Referral physician with credentials (ONLY if no treating/signing)
        - Final: "Not specified"
        """
        treating_md = data.get("treating_physician", "")
        signing_md = data.get("signing_physician", "")
        referral_md = data.get("referral_physician", "")
        
        # Primary: Use treating physician if qualified
        if treating_md and treating_md != "Not specified":
            logger.info(f"✅ Using treating physician: {treating_md}")
            return treating_md
        
        # Secondary: Use signing physician if qualified
        if signing_md and signing_md != "Not specified":
            logger.info(f"✅ Using signing physician: {signing_md}")
            return signing_md
        
        # Fallback: Use referral doctor ONLY if no treating/signing found
        if referral_md and referral_md != "Not specified":
            logger.info(f"🔄 Using referral doctor as fallback: {referral_md}")
            return referral_md
        
        # No qualified doctors found
        logger.info("❌ No qualified doctors found")
        return "Not specified"

    def _clean_extracted_data(self, result: Dict, fallback_date: str) -> Dict:
        """Clean and validate extracted data."""
        
        cleaned = {}
        
        # Date
        date = result.get("report_date", "").strip()
        cleaned["report_date"] = date if date and date.lower() not in ["empty", ""] else fallback_date

        # Physician validation
        treating_physician = result.get("treating_physician", "").strip()
        signing_physician = result.get("signing_physician", "").strip()
        referral_physician = result.get("referral_physician", "").strip()
        
        cleaned["treating_physician"] = self._validate_physician_name(treating_physician)
        cleaned["signing_physician"] = self._validate_physician_name(signing_physician)
        cleaned["referral_physician"] = self._validate_physician_name(referral_physician)

        # Medical content
        cleaned["clinical_progress"] = result.get("clinical_progress", "").strip()
        cleaned["treatment_changes"] = result.get("treatment_changes", "").strip()
        cleaned["work_capacity"] = result.get("work_capacity", "").strip()
        cleaned["recommendations"] = result.get("recommendations", "").strip()

        return cleaned

    def _validate_physician_name(self, name: str) -> str:
        """Efficiently validate physician name has proper medical credentials."""
        if not name or name.lower() in ["not specified", "not found", "none", "n/a", ""]:
            return "Not specified"
        
        name_lower = name.lower()
        
        # Fast rejection using set membership (O(1) lookup)
        reject_terms = {
            "admin", "administrator", "case manager", "coordinator", "manager",
            "therapist", "physical therapist", "pt", "ot", "technician",
            "assistant", "technologist", "staff", "authority", "personnel",
            "clerk", "signed by", "dictated by", "transcribed by"
        }
        
        # Check if any reject term is in the name
        if any(term in name_lower for term in reject_terms):
            return "Not specified"
        
        # Use pre-compiled regex for efficiency - MUST have medical credentials
        if self.medical_credential_pattern.search(name_lower):
            return name
        
        # Reject names without credentials (including "Syed Akbar" type names)
        return "Not specified"

    def _build_targeted_summary(self, data: Dict, doc_type: str) -> str:
        """Build precise 50-60 word summary of progress report."""
        
        physician = data.get("final_physician", "")
        treating_physician = data.get("treating_physician", "")
        signing_physician = data.get("signing_physician", "")
        referral_physician = data.get("referral_physician", "")
        clinical_progress = data.get("clinical_progress", "")
        treatment_changes = data.get("treatment_changes", "")
        work_capacity = data.get("work_capacity", "")
        recommendations = data.get("recommendations", "")
        
        # Build summary components
        parts = []
        
        # 1. Physician header (8-15 words)
        physician_valid = physician and physician != "Not specified"
        
        if physician_valid:
            # Add context based on physician role
            if physician == referral_physician and physician != treating_physician:
                parts.append(f"Referral {physician} progress report")
            elif treating_physician and signing_physician and treating_physician != signing_physician:
                parts.append(f"{treating_physician} progress report, signed by {signing_physician}")
            else:
                parts.append(f"{physician} progress report")
        else:
            parts.append("PR-2 progress report")
        
        # 2. Prioritize medical content by importance
        content_segments = []
        
        # Priority 1: Clinical progress (most important)
        if clinical_progress:
            content_segments.append(clinical_progress)
        
        # Priority 2: Work capacity (critical for PR-2)
        if work_capacity:
            content_segments.append(work_capacity)
        
        # Priority 3: Treatment changes
        if treatment_changes:
            content_segments.append(treatment_changes)
        
        # Priority 4: Recommendations
        if recommendations:
            content_segments.append(recommendations)
        
        # Combine medical content
        full_content = " ".join(content_segments)
        
        if full_content:
            # Calculate available words
            header = " ".join(parts)
            current_words = len(header.split())
            target_words = 55  # Target middle of 50-60 range
            words_for_content = target_words - current_words
            
            if words_for_content > 10:  # Ensure meaningful content
                content_words = full_content.split()
                
                if len(content_words) > words_for_content:
                    # Smart truncation
                    truncated = " ".join(content_words[:words_for_content])
                    
                    # Try to end at sentence boundary
                    if '.' in truncated:
                        sentences = truncated.split('.')
                        # Keep complete sentences only
                        complete_sentences = '. '.join(sentences[:-1])
                        if complete_sentences:
                            truncated = complete_sentences + '.'
                        else:
                            truncated = sentences[0] + '.'
                    else:
                        # End at comma or add ellipsis
                        if ',' in truncated[-30:]:
                            last_comma = truncated.rfind(',')
                            truncated = truncated[:last_comma] + '.'
                        else:
                            truncated += '...'
                    
                    parts.append(truncated)
                else:
                    parts.append(full_content)
        
        # Combine all parts
        if len(parts) > 1:
            # Use colon for separation between header and content
            summary = parts[0] + ": " + parts[1]
        else:
            summary = parts[0]
        
        # Ensure proper ending
        if not summary.endswith(('.', '...')):
            summary += '.'
        
        # Log word count for monitoring
        word_count = len(summary.split())
        logger.info(f"📝 Summary word count: {word_count}")
        
        return summary