"""
OPTIMIZED Enhanced Report Analyzer with Layout-Awareness and Zone Detection
Now leverages Document AI layout preservation for better doctor detection
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, TypedDict
from datetime import datetime, timedelta
import re, json
import logging

# Import our modular document detector (kept for hybrid approach)
from utils.document_detector import detect_document_type
from utils.doctor_detector import DoctorDetector
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

from models.data_models import DocumentAnalysis, VerificationResult, BriefSummary

# ============================================================================
# OPTIMIZED ENHANCED REPORT ANALYZER WITH LAYOUT AWARENESS
# ============================================================================

class EnhancedReportAnalyzer:
    """
    OPTIMIZED Enhanced analyzer with:
    - Single-pass extraction
    - System/User prompts
    - Layout-aware doctor detection
    - Zone-based signature prioritization
    """

    def __init__(self):
        """Initialize with LLM and all components"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=CONFIG.get("azure_openai_endpoint"),
            api_key=CONFIG.get("azure_openai_api_key"),
            deployment_name=CONFIG.get("azure_openai_deployment"),
            api_version=CONFIG.get("azure_openai_api_version"),
            temperature=0.0,
            timeout=120
        )

        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        self.brief_summary_parser = JsonOutputParser(pydantic_object=BriefSummary)
        
    # No longer need DocumentTypeDetector instance; use detect_document_type function
        
        # NEW: Initialize layout-aware doctor detector
        self.doctor_detector = DoctorDetector(self.llm)
        logger.info("✅ OPTIMIZED EnhancedReportAnalyzer initialized (Layout-aware)")

    def detect_document_type(self, document_text: str) -> str:
        """
        Updated: Uses detect_document_type function from utils.document_detector.
        Returns the final_doc_type string from the detection result.
        """
        try:
            logger.info("🔍 Fast document type detection...")
            detection_result = detect_document_type(document_text)
            doc_type = detection_result.get("doc_type", "medical_document")
            logger.info(f"✅ Document type detected: {doc_type}")
            return doc_type
        except Exception as e:
            logger.error(f"❌ Document type detection failed: {str(e)}")
            return "medical_document"

    def validate_doctor_name(self, name: str, document_text: str = None) -> Tuple[bool, str]:
        """
        Uses doctor_detector to validate and correct doctor names. Falls back to 'Not specified' if invalid or not found.
        """
        try:
            if not name or name == "Not specified":
                return True, name
            # Use doctor_detector to validate/correct
            if hasattr(self, "doctor_detector"):
                detection_result = self.doctor_detector.detect_doctor(document_text or name)
                detected_name = detection_result.get("doctor_name", "")
                confidence_str = detection_result.get("confidence", "none")
                # Map string confidence to numeric for thresholding
                confidence_map = {"high": 1.0, "medium": 0.7, "low": 0.4, "none": 0.0}
                confidence = confidence_map.get(confidence_str, 0.0)
                if detected_name and confidence >= 0.5:
                    return True, detected_name
                else:
                    logger.warning(f"⚠️ Doctor name not confidently detected: {name}")
                    return False, "Not specified"
            else:
                logger.warning("⚠️ doctor_detector not initialized; fallback to Not specified")
                return False, "Not specified"
        except Exception as e:
            logger.error(f"❌ Doctor name validation failed: {e}")
            return False, "Not specified"

    def extract_signature_context(self, page_zones: Optional[Dict[int, Dict[str, str]]]) -> str:
        """
        NEW: Extract signature/footer zones for enhanced doctor detection
        """
        if not page_zones:
            return ""
        
        signature_contexts = []
        for page_num, zones in page_zones.items():
            # Prioritize signature zone, fallback to footer
            if zones.get("signature"):
                signature_contexts.append(f"PAGE {page_num} SIGNATURE:\n{zones['signature']}")
            elif zones.get("footer"):
                signature_contexts.append(f"PAGE {page_num} FOOTER:\n{zones['footer']}")
        
        return "\n\n".join(signature_contexts)

    def create_optimized_extraction_prompt(
        self, 
        detected_doc_type: str,
        has_signature_context: bool = False
    ) -> ChatPromptTemplate:
        """
        OPTIMIZED: System/User prompt structure with layout awareness.
        Critical: Consulting Doctor = Report author/signer ONLY
        """
        
        # Additional context for signature-aware extraction
        signature_guidance = ""
        if has_signature_context:
            signature_guidance = """
━━━ SIGNATURE ZONE CONTEXT PROVIDED ━━━

You have been provided with extracted SIGNATURE/FOOTER zones from the document.
These zones are the MOST RELIABLE source for consulting doctor names.

PRIORITY ORDER for doctor detection:
1. FIRST: Check signature zones (highest confidence)
2. SECOND: Check header/document metadata
3. LAST: Check body text (lowest confidence, verify not referral)

When extracting consulting_doctor:
- Prioritize names from signature zones
- Must include title (Dr., MD, DO, etc.)
- Ignore "Dictated by", "Transcribed by", "CC:", "Referred by"
- If found in signature → high confidence
- If found elsewhere → verify it's the report author, not a referral

"""

        # SYSTEM PROMPT - Role and Instructions
        system_template = f"""
You are an expert medical document analyzer with BUILT-IN quality assurance.

DETECTED DOCUMENT TYPE: {{detected_doc_type}}
(Use this as guidance for extraction priorities and context)

DOCUMENT TEXT:
{{document_text}}

{signature_guidance}

{{signature_context}}

CURRENT DATE: {{current_date}}

═══════════════════════════════════════════════════════════════════════
COMPREHENSIVE ANALYSIS INSTRUCTIONS (ALL IN ONE PASS)
═══════════════════════════════════════════════════════════════════════

PERFORM ALL STAGES SIMULTANEOUSLY:

━━━ STAGE 1: DOCUMENT STRUCTURE ANALYSIS ━━━

- Document type: {{detected_doc_type}} (validate and refine if needed)
- Prioritize extraction based on type:
  * QME/AME/IME → Focus on: impairment, apportionment, MMI, work restrictions
  * Imaging → Focus on: findings, impressions, body part, abnormalities
  * Progress Report → Focus on: status change, treatment updates, next steps
  * Consult → Focus on: recommendations, specialist opinions
- Identify workers comp indicators: "WRKCMP", "Work Comp", "industrial injury", claim references
- Assess urgency: Pain scales (7+/10=urgent), severe symptoms, critical findings
- Document sections: Demographics, Subjective, Objective, Assessment, Plan, Findings, Impressions

━━━ STAGE 2: PATIENT & CLAIM EXTRACTION ━━━

CRITICAL CLAIM NUMBER RULE:
- Scan for keywords: "CLAIM", "Claim #", "Claim Number", "CL#" within 20 chars of a number
- If keyword + number nearby → extract claim number
- If NO keyword → "Not specified"
- Examples:
  ✅ "Claim #12345" → "12345"
  ✅ "Claim Number: 98765" → "98765"
  ❌ "Patient ID 12345" (no "claim" keyword) → "Not specified"

Extract:
- patient_name: Full name from demographics. If not found → "Not specified"
- claim_number: Use strict rule above
- dob: "DOB:", "Date of Birth:", birth patterns. Format YYYY-MM-DD. If not found → "Not specified"
- doi: Date of injury from history/workers comp context. Format YYYY-MM-DD. If not found → "Not specified"

━━━ STAGE 3: CLINICAL EXTRACTION ━━━

- status: Based on urgency (normal/elevated/urgent/critical). If unclear → "Not specified"
- diagnosis: Primary condition + 2-3 key findings (5-10 words). If not found → "Not specified"
- key_concern: Main issue in 2-3 words. If not found → "Not specified"

BODY PART ANALYSIS (HANDLE MULTIPLE):
- body_part: Primary body part (e.g., "lumbar spine", "right knee"). If not found → "Not specified"
- body_parts_analysis: If MULTIPLE body parts mentioned, create SEPARATE analysis for EACH:
  * body_part: Specific part
  * diagnosis: Diagnosis for this part
  * key_concern: Key concern for this part
  * clinical_summary: Important findings (2-3 sentences)
  * treatment_plan: Treatments/therapies for this part (2-3 sentences)
  * extracted_recommendation: Recommendations for this part
  * adls_affected: ADLs affected by this part
  * work_restrictions: Restrictions for this part
- adls_affected: Limited activities in 2-3 words. If not found → "Not specified"
- work_restrictions: Work limitations in 2-3 words. If not found → "Not specified"

━━━ STAGE 4: KEYWORD EXTRACTION (STRICT RULES) ━━━

EXTRACTED RECOMMENDATION:
- STRICTLY scan for keywords: 'recommend', 'recommendation', 'plan', 'follow-up', 'therapy', 'PT', 'medication', 'surgery', 'consult', 'referral'
- ONLY if keywords present → extract comma-separated keywords (NOT sentences)
- Example: "PT twice weekly, follow-up 4 weeks, surgical consult"
- If NO keywords → "Not specified"

EXTRACTED DECISION:
- Scan Assessment/Plan/Impressions for: "Decision:", "Plan:", "Judgment:", "Proceed with"
- Extract comma-separated medical terms only
- If not found → "Not specified"

EXTRACTED UR DECISION:
- Scan for: 'Utilization Review', 'UR', 'prior authorization', 'PA', 'approved', 'denied'
- ONLY if keywords present → extract decision (comma-separated)
- If denial → extract ur_denial_reason (1-2 sentences explaining why)
- If NO keywords → ur_decision="Not specified", ur_denial_reason=None

━━━ STAGE 5: DOCTOR & INTERPRETING PHYSICIAN EXTRACTION ━━━

CONSULTING DOCTOR (HIGHEST PRIORITY - USE SIGNATURE ZONES):
- **FIRST check signature zones if provided above**
- MUST have explicit title: "Dr.", "MD", "DO", "M.D.", "D.O.", "MBBS", or "MBChB"
- Look in order: (1) Signature zones, (2) Header/metadata, (3) Body text
- Extract FULL NAME with title (e.g., "Dr. Jane Smith")
- IGNORE: "Dictated by", "Transcribed by", "CC:", administrative names
- Do NOT extract patient names, referral doctors in body text
- If no consultant → ""

REFERRAL DOCTOR:
- MUST have explicit title: "Dr.", "MD", "DO", "M.D.", "D.O.", "MBBS", or "MBChB"
- Look for: "Referred to", "Referral to", "Referred by", "PCP", "Primary Care"
- Extract FULL NAME with title
- If no referral → ""

━━━ STAGE 6: FORMATTED SUMMARY LINE ━━━

Follow these concise clinical summary rules:

EXTRACTION RULES:
1. Focus ONLY on the primary diagnostic finding (most clinically significant).
2. If multiple findings exist, select the one with highest diagnostic importance.
3. If normal study → output "normal study" or "no acute findings".
4. If uncertain or possible finding (marked with "?"), rewrite as "possible [finding]".
5. Body part: concise format (e.g., "R shoulder", "L knee", "C4-6", "L-spine").
6. Date: MM/DD/YY format.
7. For MRI/CT, indicate if with or without contrast when explicitly stated.
8. Finding: brief but complete (max 16 words).
9. Do not include technical details (e.g., sequences, imaging parameters).
10. The summary should be easily readable on a compact card.

Generate ONE line summary using this format:
[Dr. Name if available else no title] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]

━━━ STAGE 7: AI OUTCOME & TASK ANALYSIS ━━━

AI OUTCOME:
- Based on diagnosis, recommendations, decisions → generate outcome prediction (comma-separated keywords)
- Tie to evidence in document
- Example: "full recovery 6 weeks, monitor pain, low risk"

TASK NEED ANALYSIS:
- Set is_task_needed=TRUE if:
  * PENDING actions not completed
  * FUTURE appointments need scheduling
  * NEW recommendations need implementation
  * AUTHORIZATIONS needed
  * REFERRALS to process
- Set is_task_needed=FALSE if:
  * Everything COMPLETED/DONE
  * No pending actions
  * Purely historical/descriptive

━━━ STAGE 8: DATE REASONING ━━━

- Extract ALL dates, convert to YYYY-MM-DD
- Classify as DOB (birth context), DOI (injury), RD (report/signature/end)
- Use document flow: early dates likely DOB/DOI, late dates likely RD
- Provide reasoning, contexts, confidence scores (0.0-1.0)

━━━ STAGE 9: BUILT-IN SELF-VERIFICATION ━━━

extraction_confidence (0.0-1.0):
- 1.0: All critical fields extracted with certainty
- 0.8: Most fields extracted, minor ambiguity
- 0.6: Several fields uncertain
- 0.4: Many fields missing
- 0.2: Minimal extraction

═══════════════════════════════════════════════════════════════════════
CRITICAL OUTPUT RULES
═══════════════════════════════════════════════════════════════════════

- Extract ONLY from document text. If not found → "Not specified"
- Doctor names MUST include titles. Flag in verification_notes if missing
- Keywords: comma-separated terms, NO full sentences
- Dates: YYYY-MM-DD format only
- Output valid JSON matching schema
- Include extraction_confidence, verified=true, verification_notes

{{format_instructions}}
"""

        # USER PROMPT - Document and Specific Instructions
        user_template = """
You are analyzing a clinical or radiology document for structured extraction and validation.

DOCUMENT TEXT:
{document_text}

{signature_context}

CURRENT DATE: {current_date}

Use the system instructions to extract and structure all key fields.

Ensure the `formatted_summary` line strictly follows this format:
[Dr. Name if available else no title] [Document Type] [Body Part] [Date] = [Primary Finding] → [Impression] [Recommendations]-[Medication]-[Follow-up]-[Future Treatment]-[Comments]

Return **valid JSON only** following the schema in the system template.
"""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    # reasoning_agent.py (key sections updated)

    def extract_document_data_with_reasoning(
        self,
        document_text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,  # NEW parameter
        raw_text: Optional[str] = None
    ) -> DocumentAnalysis:
        """
        OPTIMIZED: Single-pass extraction with zone-awareness.
        
        Args:
            document_text: Layout-preserved text from Document AI
            page_zones: Per-page zone extraction {page_num: {header, body, footer, signature}}
            raw_text: Original flat text (for backward compatibility)
        """
        try:
            logger.info("🚀 Starting OPTIMIZED extraction (Zone-aware)...")
            
            # Debug: Check if page_zones is provided
            if page_zones:
                logger.info(f"📦 page_zones received with {len(page_zones)} pages: {list(page_zones.keys())}")
            else:
                logger.warning("⚠️ No page_zones received in extract_document_data_with_reasoning")
            
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Fast pattern-based detection
            logger.info("📄 Stage 1: Fast document type detection")
            detected_doc_type = self.detect_document_type(document_text)

            # Extract signature context for prompts
            signature_context = ""
            has_signature_context = False
            if page_zones:
                signature_context = self.extract_signature_context(page_zones)
                has_signature_context = bool(signature_context)
                if has_signature_context:
                    logger.info(f"✅ Signature context extracted ({len(signature_context)} chars)")

            # Single comprehensive extraction with System/User prompts
            logger.info("🔍 Stage 2: Comprehensive extraction with zone context")
            prompt = self.create_optimized_extraction_prompt(
                detected_doc_type,
                has_signature_context
            )

            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "document_text": document_text,
                "signature_context": signature_context if has_signature_context else "",
                "current_date": current_date,
                "detected_doc_type": detected_doc_type,
                "format_instructions": self.parser.get_format_instructions()
            })

            analysis = DocumentAnalysis(**result)

            # **ENHANCED: Zone-aware doctor validation**
            logger.info("✅ Stage 3: Zone-aware doctor validation")
            
            # Validate consulting doctor with zone data
            if analysis.consulting_doctor and analysis.consulting_doctor != "Not specified":
                # Run enhanced detection with zones
                logger.info("🔍 Running zone-aware doctor detection...")
                logger.info(f"🔍 page_zones before detector call: {page_zones is not None}, keys: {list(page_zones.keys()) if page_zones else 'None'}")
                detection_result = self.doctor_detector.detect_doctor(
                    text=document_text,
                    page_zones=page_zones  # NEW: pass zones
                )
                
                if detection_result["doctor_name"]:
                    # Use detector result if confidence is high/medium
                    if detection_result["confidence"] in ["high", "medium"]:
                        analysis.consulting_doctor = detection_result["doctor_name"]
                        analysis.verification_notes.append(
                            f"Zone-aware detection: {detection_result['doctor_name']} "
                            f"(confidence: {detection_result['confidence']}, "
                            f"source: {detection_result['source']})"
                        )
                        logger.info(f"✅ Consulting doctor: {detection_result['doctor_name']} "
                                f"from {detection_result['source']}")
                else:
                    # No valid doctor found
                    analysis.consulting_doctor = "Not specified"
                    analysis.verification_notes.append(
                        "No valid main/treating doctor detected with required title (Dr./MD/DO)"
                    )
                    logger.warning("⚠️ No valid consulting doctor found")

            # Validate referral doctor (NOTE: we do NOT extract referrals in main detector)
            if analysis.referral_doctor and analysis.referral_doctor != "Not specified":
                # Just validate it has a title
                if not self._has_medical_title(analysis.referral_doctor):
                    analysis.referral_doctor = "Not specified"
                    analysis.verification_notes.append("Referral doctor rejected: no title")

            # Set verified flag
            analysis.verified = True

            # Log results
            logger.info(f"🎉 OPTIMIZED extraction complete:")
            logger.info(f"   - Patient: {analysis.patient_name}")
            logger.info(f"   - Document Type: {analysis.document_type}")
            logger.info(f"   - CONSULTING DOCTOR: {analysis.consulting_doctor}")
            logger.info(f"   - Referral Doctor: {analysis.referral_doctor}")
            logger.info(f"   - Confidence: {analysis.extraction_confidence:.2f}")

            return analysis

        except Exception as e:
            logger.error(f"❌ Optimized extraction failed: {str(e)}")
            return self.create_fallback_analysis()

    def _has_medical_title(self, name: str) -> bool:
        """Helper to check if name has medical title."""
        if not name:
            return False
        t = name.upper()
        return any(title in t for title in ["DR.", " MD", " DO", "M.D.", "D.O.", "MBBS", "MBCHB"])

    def generate_brief_summary(self, document_text: str) -> str:
        """
        Generate brief summary (unchanged).
        """
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Use System/User prompts for summary too
            system_prompt = SystemMessagePromptTemplate.from_template(
                "You are a medical summarization expert. Generate concise 1-2 sentence professional summaries."
            )

            human_prompt = HumanMessagePromptTemplate.from_template(
                """Generate a concise 1-2 sentence professional summary of this medical document.

DOCUMENT TEXT:
{document_text}

CURRENT DATE: {current_date}

Focus: Patient condition, key findings, recommendations. Use clinical language.

{format_instructions}"""
            )

            summary_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({
                "document_text": document_text,
                "current_date": current_date,
                "format_instructions": self.brief_summary_parser.get_format_instructions()
            })

            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated summary: {brief_summary}")
            return brief_summary

        except Exception as e:
            logger.error(f"❌ Summary generation failed: {str(e)}")
            return "Brief summary unavailable"

    def create_fallback_analysis(self) -> DocumentAnalysis:
        """Create fallback analysis when extraction fails"""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return DocumentAnalysis(
            patient_name="Not specified",
            claim_number="Not specified",
            dob=timestamp,
            doi=timestamp,
            status="Not specified",
            rd=timestamp,
            body_part="Not specified",
            body_parts_analysis=[],
            diagnosis="Not specified",
            key_concern="Not specified",
            extracted_recommendation="Not specified",
            extracted_decision="Not specified",
            ur_decision="Not specified",
            ur_denial_reason=None,
            adls_affected="Not specified",
            work_restrictions="Not specified",
            consulting_doctor="Not specified",
            referral_doctor="Not specified",
            ai_outcome="Insufficient data; full evaluation needed",
            document_type="medical_document",
            summary_points=["Not specified"],
            date_reasoning=None,
            is_task_needed=False,
            extraction_confidence=0.0,
            verified=False,
            verification_notes=["Fallback analysis - extraction failed"]
        )

    def get_date_reasoning(self, document_text: str) -> Dict[str, Any]:
        """Get date reasoning results"""
        try:
            full_analysis = self.extract_document_data_with_reasoning(document_text)
            return {
                "date_reasoning": full_analysis.date_reasoning.dict() if full_analysis.date_reasoning else {}
            }
        except Exception as e:
            logger.error(f"❌ Date reasoning failed: {str(e)}")
            return {"date_reasoning": {}}
