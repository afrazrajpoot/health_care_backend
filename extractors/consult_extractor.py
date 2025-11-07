"""
Specialist Consult Extractor - Enhanced physician recognition with referral doctor fallback
"""
import logging
import re
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult

logger = logging.getLogger("document_ai")


class ConsultExtractor:
    """Enhanced extractor for Specialist Consultation Reports with referral doctor fallback."""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        # Pre-compile regex for efficiency
        self.medical_credential_pattern = re.compile(
            r'\b(dr\.?|doctor|m\.?d\.?|d\.?o\.?|mbbs|m\.?b\.?b\.?s\.?)\b',
            re.IGNORECASE
        )

    def extract(self, text: str, doc_type: str, fallback_date: str) -> ExtractionResult:
        """Extract consultation report with proper physician identification."""
        raw_result = self._extract_medical_content(text, doc_type, fallback_date)
        result = self._build_result(raw_result, doc_type, fallback_date)
        return result

    def _extract_medical_content(self, text: str, doc_type: str, fallback_date: str) -> Dict:
        """Extract medical changes and identify physicians efficiently."""
        
        system_template = """You are a medical data extraction specialist analyzing consultation reports.

CRITICAL PHYSICIAN IDENTIFICATION PROTOCOL:

1. PRIMARY PHYSICIAN (Consulting/Signing):
   - MUST have medical credentials: Dr., MD, M.D., DO, D.O., MBBS
   - Look for: "Consulting physician:", "Examined by:", "Provider:", "Attending:"
   - Check signatures: "Electronically signed by:", "Dictated by:", "Signed by:"

2. REFERRAL DOCTOR FALLBACK (ONLY if no primary physician found):
   - Use referral doctor ONLY when no consulting/signing physician is identified
   - Look for: "Referred by:", "Referral from:", "PCP:", "Primary Care:"
   - Must have medical credentials
   - Examples: ✓ Referred by: Dr. James Wilson

3. REJECT NON-DOCTOR SIGNATORIES:
   - ✗ Reject names without medical credentials (e.g., "Syed Akbar")
   - ✗ Reject administrators, technicians, coordinators
   - ✗ Reject any name without proper medical credentials

4. PRIORITY ORDER:
   1. Consulting Physician (with credentials)
   2. Signing Physician (with credentials)  
   3. Referral Doctor (with credentials, ONLY if no consulting/signing physician)
   4. "Not specified" (if no qualified doctors found)

VALID PHYSICIAN FORMATS:
   ✓ "Dr. Sarah Johnson"
   ✓ "Michael Brown, MD"
   ✓ "Dr. Ahmed Khan, MBBS"
   ✓ "Jennifer Lee M.D."

IMMEDIATE REJECTION:
   ✗ Names without medical credentials
   ✗ "Administrator", "Manager", "Coordinator"
   ✗ "Technician", "Assistant", "Staff"
   ✗ Isolated names: "John Smith", "Syed Akbar"

MEDICAL CONTENT EXTRACTION:
- Focus on NEW findings, diagnoses, and clinical changes
- Extract primary impression/assessment
- Identify key recommendations and treatment modifications
- Summarize in 2-3 concise sentences

REQUIRED OUTPUT:
- consult_date: Date in MM/DD/YY format
- consulting_physician: Doctor who examined patient (MUST have credentials)
- signing_physician: Doctor who signed report (MUST have credentials)
- referral_physician: Referral doctor if explicitly mentioned
- primary_assessment: Chief complaint and clinical impression (1-2 sentences)
- key_findings: Significant clinical findings (1-2 sentences)
- recommendations: Treatment plan and follow-up (1-2 sentences)

Return clean JSON without explanations or markdown."""

        human_template = """Extract information from this {doc_type} report:

{text}

PHYSICIAN EXTRACTION PRIORITY:
1. First: Consulting/Signing physician with credentials (Dr./MD/DO/MBBS)
2. Fallback: Referral doctor with credentials (ONLY if no consulting/signing physician)
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
            
            logger.info("📊 Consult Extraction Results:")
            logger.info(f"   - Consulting Physician: {result.get('consulting_physician', 'Not found')}")
            logger.info(f"   - Signing Physician: {result.get('signing_physician', 'Not found')}")
            logger.info(f"   - Referral Physician: {result.get('referral_physician', 'Not found')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Consult extraction failed: {e}")
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
            document_date=cleaned.get("consult_date", fallback_date),
            summary_line=summary,
            examiner_name=final_physician,
            specialty="",
            body_parts=[],
            raw_data=cleaned,
        )

    def _apply_referral_fallback(self, data: Dict) -> str:
        """
        Apply referral doctor fallback logic:
        - Primary: Consulting physician with credentials
        - Secondary: Signing physician with credentials  
        - Fallback: Referral physician with credentials (ONLY if no consulting/signing)
        - Final: "Not specified"
        """
        consulting_md = data.get("consulting_physician", "")
        signing_md = data.get("signing_physician", "")
        referral_md = data.get("referral_physician", "")
        
        # Primary: Use consulting physician if qualified
        if consulting_md and consulting_md != "Not specified":
            logger.info(f"✅ Using consulting physician: {consulting_md}")
            return consulting_md
        
        # Secondary: Use signing physician if qualified
        if signing_md and signing_md != "Not specified":
            logger.info(f"✅ Using signing physician: {signing_md}")
            return signing_md
        
        # Fallback: Use referral doctor ONLY if no consulting/signing found
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
        date = result.get("consult_date", "").strip()
        cleaned["consult_date"] = date if date and date.lower() not in ["empty", ""] else fallback_date

        # Physician validation
        consulting_physician = result.get("consulting_physician", "").strip()
        signing_physician = result.get("signing_physician", "").strip()
        referral_physician = result.get("referral_physician", "").strip()
        
        cleaned["consulting_physician"] = self._validate_physician_name(consulting_physician)
        cleaned["signing_physician"] = self._validate_physician_name(signing_physician)
        cleaned["referral_physician"] = self._validate_physician_name(referral_physician)

        # Medical content
        cleaned["primary_assessment"] = result.get("primary_assessment", "").strip()
        cleaned["key_findings"] = result.get("key_findings", "").strip()
        cleaned["recommendations"] = result.get("recommendations", "").strip()

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
            "personnel", "clerk", "secretary", "signed by", "dictated by"
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
        """Build precise 50-60 word summary of consultation."""
        
        physician = data.get("final_physician", "")
        consulting_physician = data.get("consulting_physician", "")
        signing_physician = data.get("signing_physician", "")
        referral_physician = data.get("referral_physician", "")
        primary_assessment = data.get("primary_assessment", "")
        key_findings = data.get("key_findings", "")
        recommendations = data.get("recommendations", "")
        
        # Build summary components
        parts = []
        
        # 1. Physician header (8-15 words)
        physician_valid = physician and physician != "Not specified"
        
        if physician_valid:
            # Add context based on physician role
            if physician == referral_physician and physician != consulting_physician:
                parts.append(f"Referral {physician} consultation")
            elif consulting_physician and signing_physician and consulting_physician != signing_physician:
                parts.append(f"{consulting_physician} consultation, signed by {signing_physician}")
            else:
                parts.append(f"{physician} consultation")
        else:
            parts.append("Consultation report")
        
        # 2. Combine medical content for remaining words
        medical_content = []
        
        if primary_assessment:
            medical_content.append(primary_assessment)
        
        if key_findings:
            medical_content.append(key_findings)
        
        if recommendations:
            medical_content.append(recommendations)
        
        # Join medical content
        full_medical = " ".join(medical_content)
        
        if full_medical:
            # Calculate available words
            header = " ".join(parts)
            current_words = len(header.split())
            target_words = 55  # Target middle of 50-60 range
            words_for_content = target_words - current_words
            
            if words_for_content > 10:  # Ensure meaningful content
                content_words = full_medical.split()
                
                if len(content_words) > words_for_content:
                    # Smart truncation
                    truncated = " ".join(content_words[:words_for_content])
                    
                    # Try to end at sentence boundary
                    if '.' in truncated:
                        sentences = truncated.split('.')
                        truncated = '. '.join(sentences[:-1]) + '.'
                    else:
                        # End at comma or natural break
                        if ',' in truncated[-20:]:
                            last_comma = truncated.rfind(',')
                            truncated = truncated[:last_comma] + '.'
                        else:
                            truncated += '...'
                    
                    parts.append(truncated)
                else:
                    parts.append(full_medical)
        
        # Combine all parts
        if len(parts) > 1:
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