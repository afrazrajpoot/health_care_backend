"""
Imaging Reports Extractor (MRI, CT, X-ray, Ultrasound, EMG)
Ultra-efficient physician recognition with referral doctor fallback
"""
import logging
import re
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ImagingExtractor:
    """Extractor for imaging reports with referral doctor fallback."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract key medical information with proper doctor identification."""
        logger.info(f"🔍 Extracting {doc_type} report")
        
        raw_result = self._extract_medical_content(text, doc_type, fallback_date)
        result = self._build_result(raw_result, doc_type, fallback_date)
        
        logger.info(f"✅ Extraction completed")
        return result

    def _extract_medical_content(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Extract medical findings and identify physicians efficiently."""
        
        system_template = """You are a medical AI trained to extract structured data from radiology reports. Extract only factual information present in the document.

CRITICAL PHYSICIAN IDENTIFICATION RULES:

1. PRIMARY CONSULTING DOCTOR (Interpreting Radiologist):
   - MUST have medical credentials: Dr., MD, M.D., DO, D.O., MBBS
   - Look for: "Interpreted by:", "Read by:", "Radiologist:", "Dictated by:"
   - Examples: ✓ Dr. Sarah Johnson, ✓ Michael Chen, MD

2. REFERRAL DOCTOR FALLBACK (ONLY if no primary consulting doctor found):
   - Use referral doctor ONLY when no interpreting radiologist is identified
   - Look for: "Referred by:", "Referral from:", "PCP:", "Primary Care:"
   - Must have medical credentials
   - Examples: ✓ Referred by: Dr. James Wilson

3. REJECT NON-DOCTOR SIGNATORIES:
   - ✗ Reject "Syed Akbar" (no credentials) even if they signed
   - ✗ Reject administrators, technicians, coordinators
   - ✗ Reject any name without proper medical credentials

4. PRIORITY ORDER:
   1. Interpreting Radiologist (with credentials)
   2. Referral Doctor (with credentials, ONLY if no radiologist)
   3. "Not specified" (if no qualified doctors found)

MEDICAL CONTENT EXTRACTION:
Extract concise, clinically relevant information:
• study_date: Report date (MM/DD/YY format)
• body_part: Anatomical region(s) scanned
• findings: Key abnormalities detected (2-3 sentences)
• impression: Radiologist's conclusion/diagnosis (1-2 sentences)
• interpreting_physician: Qualified doctor following priority rules above
• referral_physician: Referral doctor if explicitly mentioned

Guidelines:
- Be precise and concise
- Use medical terminology
- Prioritize significant findings over normal anatomy

Output only valid JSON with no explanations."""

        human_template = """Extract structured data from this {doc_type} imaging report:

{text}

PHYSICIAN EXTRACTION PRIORITY:
1. First: Interpreting radiologist with credentials (Dr./MD/DO/MBBS)
2. Fallback: Referral doctor with credentials (ONLY if no radiologist found)
3. Last: "Not specified" (if no qualified doctors)

Return JSON:
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
            
            logger.info("📊 Extraction Results:")
            logger.info(f"   - Physician: {result.get('interpreting_physician', 'Not found')}")
            logger.info(f"   - Referral: {result.get('referral_physician', 'Not found')}")
            logger.info(f"   - Body Part: {result.get('body_part', 'Not found')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
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
            document_date=cleaned.get("study_date", fallback_date),
            summary_line=summary,
            examiner_name=final_physician,
            specialty=cleaned.get("specialty", "Radiology"),
            body_parts=[cleaned.get("body_part")] if cleaned.get("body_part") else [],
            raw_data=cleaned,
        )

    def _apply_referral_fallback(self, data: Dict) -> str:
        """
        Apply referral doctor fallback logic:
        - Primary: Interpreting physician with credentials
        - Fallback: Referral physician with credentials (ONLY if no primary)
        - Final: "Not specified"
        """
        interpreting_md = data.get("interpreting_physician", "")
        referral_md = data.get("referral_physician", "")
        
        # Primary: Use interpreting physician if qualified
        if interpreting_md and interpreting_md != "Not specified":
            logger.info(f"✅ Using primary interpreting physician: {interpreting_md}")
            return interpreting_md
        
        # Fallback: Use referral doctor ONLY if no primary found
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
        date = result.get("study_date", "").strip()
        cleaned["study_date"] = date if date and date.lower() not in ["empty", ""] else fallback_date

        # Physician validation
        interpreting_physician = result.get("interpreting_physician", "").strip()
        referral_physician = result.get("referral_physician", "").strip()
        
        cleaned["interpreting_physician"] = self._validate_physician_name(interpreting_physician)
        cleaned["referral_physician"] = self._validate_physician_name(referral_physician)

        # Medical content
        cleaned["body_part"] = result.get("body_part", "").strip()
        cleaned["findings"] = result.get("findings", "").strip()
        cleaned["impression"] = result.get("impression", "").strip()

        return cleaned

    def _validate_physician_name(self, name: str) -> str:
        """Efficiently validate physician name has proper medical credentials."""
        if not name or name.lower() in ["not specified", "not found", "none", "n/a", ""]:
            return "Not specified"
        
        name_lower = name.lower()
        
        # Fast rejection using set membership (O(1) lookup)
        reject_terms = {
            "admin", "administrator", "technician", "technologist", "tech",
            "assistant", "coordinator", "manager", "staff", "authority",
            "personnel", "clerk", "secretary", "radiologic technologist"
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
        """Build optimized 50-60 word summary with smart content selection."""
        
        physician = data.get("final_physician", "")
        body_part = data.get("body_part", "")
        findings = data.get("findings", "")
        impression = data.get("impression", "")
        
        # Prioritize impression over findings (impression is more clinically relevant)
        primary_content = impression if impression else findings
        
        # Build header (physician + study info)
        if physician and physician != "Not specified":
            # Add context if it's a referral doctor
            if physician == data.get("referral_physician") and physician != data.get("interpreting_physician"):
                header = f"Referral {physician} {doc_type}"
            else:
                header = f"{physician} {doc_type}"
        else:
            header = doc_type
        
        if body_part:
            header += f" of {body_part}"
        
        # Calculate target distribution
        header_words = len(header.split())
        target_total = 55  # Middle of 50-60 range
        available_for_content = target_total - header_words
        
        # Build final summary
        if primary_content and available_for_content > 15:
            content_words = primary_content.split()
            
            if len(content_words) <= available_for_content:
                # Content fits completely
                summary = f"{header}: {primary_content}"
            else:
                # Smart truncation needed
                truncated = self._smart_truncate(
                    primary_content, 
                    available_for_content
                )
                summary = f"{header}: {truncated}"
        else:
            summary = header
        
        # Ensure proper punctuation
        if summary and not summary.endswith(('.', '!', '?', '...')):
            summary += '.'
        
        # Log final word count
        word_count = len(summary.split())
        logger.info(f"📝 Summary: {word_count} words")
        
        return summary

    def _smart_truncate(self, text: str, max_words: int) -> str:
        """Intelligently truncate text at natural boundaries."""
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        # Get target substring
        truncated = " ".join(words[:max_words])
        
        # Try to end at sentence boundary
        if '.' in truncated:
            sentences = [s.strip() for s in truncated.split('.') if s.strip()]
            if sentences:
                # Keep complete sentences that fit
                result = ""
                for sentence in sentences:
                    test = result + sentence + ". "
                    if len(test.split()) <= max_words:
                        result = test
                    else:
                        break
                if result:
                    return result.strip()
        
        # Try to end at semicolon or comma
        for delimiter in [';', ',']:
            if delimiter in truncated[-30:]:  # Look in last 30 chars
                last_delim = truncated.rfind(delimiter)
                if last_delim > len(truncated) * 0.7:  # At least 70% through
                    return truncated[:last_delim] + '.'
        
        # Last resort: add ellipsis
        return truncated + '...'