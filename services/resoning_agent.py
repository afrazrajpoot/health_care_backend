"""
OPTIMIZED Enhanced Report Analyzer with Mode-Aware Extraction (WC/GM)
Extracts and structures data differently based on workers comp vs general medicine mode
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
from config.settings import CONFIG

logger = logging.getLogger("document_ai")

from models.data_models import DocumentAnalysis, VerificationResult, BriefSummary

# ============================================================================
# MODE-AWARE ENHANCED REPORT ANALYZER (WC/GM)
# ============================================================================

class EnhancedReportAnalyzer:
    """
    MODE-AWARE Enhanced analyzer with:
    - WC (Workers Comp) vs GM (General Medicine) specific extraction
    - Mode-appropriate field mapping
    - Different clinical focus based on mode
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
        
        logger.info("✅ MODE-AWARE EnhancedReportAnalyzer initialized (WC/GM support)")

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

    def extract_signature_context(self, page_zones: Optional[Dict[int, Dict[str, str]]]) -> str:
        """
        Extract signature/footer zones for enhanced doctor detection
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

    def create_mode_aware_extraction_prompt(
        self, 
        detected_doc_type: str,
        mode: str,
        has_signature_context: bool = False
    ) -> ChatPromptTemplate:
        """
        FIXED: Complete mode-aware prompt with all variables properly handled
        """
        
        # Generate ALL mode-specific content upfront
        mode_content = self._generate_complete_mode_content(mode)
        
        # Signature guidance - generate this as a string, not a template variable
        signature_guidance_text = ""
        if has_signature_context:
            signature_guidance_text = """
    ━━━ SIGNATURE ZONE CONTEXT PROVIDED ━━━

    You have been provided with extracted SIGNATURE/FOOTER zones from the document.
    These zones are the MOST RELIABLE source for consulting doctor names.

    PRIORITY ORDER for doctor detection:
    1. FIRST: Check signature zones (highest confidence)
    2. SECOND: Check header/document metadata
    3. LAST: Check body text (lowest confidence, verify not referral)
    """

        # SIMPLE SYSTEM TEMPLATE - Include ALL variables that will be passed
        system_template = """You are an expert medical document analyzer with MODE-AWARE extraction capabilities.

    DETECTED DOCUMENT TYPE: {detected_doc_type}
    EXTRACTION MODE: {mode_display}

    {mode_guidance}

    DOCUMENT TEXT:
    {document_text}

    {signature_guidance}

    {signature_context}

    CURRENT DATE: {current_date}

    ═══════════════════════════════════════════════════════════════════════
    MODE-AWARE COMPREHENSIVE ANALYSIS INSTRUCTIONS
    ═══════════════════════════════════════════════════════════════════════

    PERFORM ALL STAGES SIMULTANEOUSLY WITH {mode_upper} FOCUS:

    {mode_specific_instructions}

    {format_instructions}"""

        # SIMPLE USER TEMPLATE
        user_template = """You are analyzing a medical document for {mode_upper} mode extraction.

    DOCUMENT TEXT:
    {document_text}

    {signature_context}

    CURRENT DATE: {current_date}

    Use the system instructions to extract and structure all key fields with {mode_upper} focus.

    Ensure the `formatted_summary` line uses the appropriate format for {mode_upper} mode.

    Return **valid JSON only** following the schema in the system template."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
    def _generate_complete_mode_content(self, mode: str) -> dict:
        """Generate ALL mode-specific content as a dictionary"""
        if mode == "wc":
            return {
                "mode_display": "WC (Workers Compensation)",
                "mode_upper": "WC",
                "mode_guidance": """
    WORKERS COMPENSATION FOCUS:
    - Priority: Work-related injuries, body parts, claim numbers
    - Key elements: Date of injury, body parts, work restrictions, MMI status
    - Legal/administrative: Claim numbers, utilization review, permanent impairment
    - Outcome focus: Return to work, work capacity, disability ratings
    """,
                "mode_specific_instructions": self._get_wc_specific_instructions()
            }
        else:
            return {
                "mode_display": "GM (General Medicine)", 
                "mode_upper": "GM",
                "mode_guidance": """
    GENERAL MEDICINE FOCUS:
    - Priority: Medical conditions, symptoms, medications, quality of life
    - Key elements: Chronic conditions, medications, lifestyle factors, comorbidities
    - Clinical focus: Disease management, treatment adherence, preventive care
    - Outcome focus: Symptom control, functional status, quality of life
    """,
                "mode_specific_instructions": self._get_gm_specific_instructions()
            }

    def _get_wc_specific_instructions(self) -> str:
        """Get Workers Comp specific instructions - return as plain string"""
        return """
    ━━━ STAGE 1: WORKERS COMP DOCUMENT ANALYSIS ━━━
    - Focus on: Work-related injuries, body parts, claim numbers, work restrictions
    - Extract: claim_number, doi (date of injury), body_part, work_restrictions
    - WC-specific fields: injury_type, work_relatedness, mmi_status, return_to_work_plan

    ━━━ STAGE 2: WORKERS COMP CLINICAL EXTRACTION ━━━
    - body_part: Primary body part injured
    - body_parts_analysis: Multiple body parts with WC-specific fields
    - ADLS & WORK IMPACT: work_impact, physical_demands, work_capacity

    ━━━ STAGE 3: WORKERS COMP KEYWORD EXTRACTION ━━━
    - EXTRACTED UR DECISION: Scan for 'Utilization Review', 'UR', 'prior authorization'
    - WORKERS COMP SUMMARY FORMAT: [Body Part] [Work Restrictions] [MMI Status]

    ━━━ CONSULTING PHYSICIAN/AUTHOR DETECTION ━━━
    - Identify the author who signed the report as the "consulting_doctor" name (e.g., from signature block, "Dictated by:", or closing statement).
    - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
    - Extract specialty only if explicitly stated near the signature.
    - If no clear signer is found, leave empty.
    """

    def _get_gm_specific_instructions(self) -> str:
        """Get General Medicine specific instructions - return as plain string"""
        return """
    ━━━ STAGE 1: GENERAL MEDICINE DOCUMENT ANALYSIS ━━━
    - Focus on: Medical conditions, symptoms, medications, quality of life
    - Extract: patient_id, onset_date, condition, medications
    - GM-specific fields: condition_severity, symptoms, chronic_condition, comorbidities

    ━━━ STAGE 2: GENERAL MEDICINE CLINICAL EXTRACTION ━━━
    - condition: Primary medical condition
    - conditions_analysis: Multiple conditions with GM-specific fields
    - FUNCTIONAL IMPACT: daily_living_impact, symptom_impact, quality_of_life

    ━━━ STAGE 3: GENERAL MEDICINE KEYWORD EXTRACTION ━━━
    - EXTRACTED AUTHORIZATION DECISION: Scan for 'prior auth', 'authorization'
    - GENERAL MEDICINE SUMMARY FORMAT: [Condition] [Medications] [Follow-up]

    ━━━ CONSULTING PHYSICIAN/AUTHOR DETECTION ━━━
    - Identify the author who signed the report as the "consulting_doctor" name (e.g., from signature block, "Dictated by:", or closing statement).
    - It is NOT mandatory that this author is a qualified doctor; extract the name as explicitly signed, regardless of credentials.
    - Extract specialty only if explicitly stated near the signature.
    - If no clear signer is found, leave empty.
    """
    def extract_document_data_with_reasoning(
        self,
        document_text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None,
        mode: Optional[str] = "wc"
    ) -> DocumentAnalysis:
        """
        FIXED: MODE-AWARE extraction with ALL template variables passed
        """
        try:
            logger.info(f"🚀 Starting MODE-AWARE extraction (Mode: {mode})...")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            detected_doc_type = self.detect_document_type(document_text)

            # Extract signature context
            signature_context = ""
            if page_zones:
                signature_context = self.extract_signature_context(page_zones)

            # Generate ALL mode content upfront
            mode_content = self._generate_complete_mode_content(mode)
            
            # Signature guidance text (not a template variable - pre-rendered)
            signature_guidance_text = ""
            if signature_context:
                signature_guidance_text = """━━━ SIGNATURE ZONE CONTEXT PROVIDED ━━━

    You have been provided with extracted SIGNATURE/FOOTER zones from the document.
    These zones are the MOST RELIABLE source for consulting doctor names.

    PRIORITY ORDER for doctor detection:
    1. FIRST: Check signature zones (highest confidence)
    2. SECOND: Check header/document metadata  
    3. LAST: Check body text (lowest confidence, verify not referral)"""

            # Mode-aware comprehensive extraction
            logger.info(f"🔍 Stage 2: {mode.upper()} mode extraction")
            prompt = self.create_mode_aware_extraction_prompt(
                detected_doc_type,
                mode,
                bool(signature_context)
            )

            chain = prompt | self.llm | self.parser
            
            # ✅ FIXED: Pass ALL required template variables
            invocation_data = {
                "document_text": document_text,
                "signature_context": signature_context,
                "current_date": current_date,
                "detected_doc_type": detected_doc_type,
                "format_instructions": self.parser.get_format_instructions(),
                # Mode-specific variables
                "mode_upper": mode_content["mode_upper"],
                "mode_display": mode_content["mode_display"], 
                "mode_guidance": mode_content["mode_guidance"],
                "mode_specific_instructions": mode_content["mode_specific_instructions"],
                "signature_guidance": signature_guidance_text
            }
            
            logger.info(f"📋 Passing template variables: {list(invocation_data.keys())}")
            result = chain.invoke(invocation_data)

            analysis = DocumentAnalysis(**result)
            
            # Set verified flag
            analysis.verified = True

            # Log results
            logger.info(f"🎉 {mode.upper()} MODE extraction complete:")
            logger.info(f"   - Patient: {analysis.patient_name}")
            logger.info(f"   - Document Type: {analysis.document_type}")
            logger.info(f"   - Mode: {mode}")
            logger.info(f"   - CONSULTING DOCTOR: {analysis.consulting_doctor}")
            logger.info(f"   - Confidence: {analysis.extraction_confidence:.2f}")

            return analysis

        except Exception as e:
            logger.error(f"❌ {mode.upper()} mode extraction failed: {str(e)}")
            return self.create_fallback_analysis(mode)
    def generate_brief_summary(self, document_text: str, mode: str = "wc") -> str:
        """
        Generate a brief summary of the medical document.
        """
        logger.info(f"🔍 Generating summary for {mode.upper()} mode...")
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Mode-specific summary guidance
            mode_focus = "workers compensation injury" if mode == "wc" else "general medical condition"
            
            system_prompt = SystemMessagePromptTemplate.from_template(
                f"You are a medical summarization expert. Generate concise 1-2 sentence professional summaries with {mode_focus} focus."
            )

            human_prompt = HumanMessagePromptTemplate.from_template(
                f"""Analyze the following medical document for {mode.upper()} mode. Identify the report type and extract all critical, actionable findings specific to {mode_focus}.

Generate a highly detailed, professional summary that immediately provides the physician with the most crucial {mode_focus} information.

DOCUMENT TEXT:
{{document_text}}

CURRENT DATE: {{current_date}}

Focus: Patient condition, key findings, recommendations. Use clinical language appropriate for {mode.upper()} mode.

{{format_instructions}}"""
            )

            summary_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({
                "document_text": document_text,
                "current_date": current_date,
                "format_instructions": self.brief_summary_parser.get_format_instructions()
            })

            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated {mode.upper()} summary: {brief_summary}")
            return brief_summary

        except Exception as e:
            logger.error(f"❌ {mode.upper()} summary generation failed: {str(e)}")
            return f"{mode_upper()} brief summary unavailable"

    def create_fallback_analysis(self, mode: str = "wc") -> DocumentAnalysis:
        """Create mode-aware fallback analysis when extraction fails"""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        # Mode-specific fallback fields
        if mode == "wc":
            key_concern = "Work-related injury evaluation needed"
            ai_outcome = "Insufficient data; full workers comp evaluation needed"
        else:
            key_concern = "Medical condition evaluation needed" 
            ai_outcome = "Insufficient data; full medical evaluation needed"
            
        return DocumentAnalysis(
            patient_name="Not specified",
            claim_number="Not specified",
            dob=timestamp,
            doi=timestamp if mode == "wc" else "Not specified",
            status="Not specified",
            rd=timestamp,
            body_part="Not specified",
            body_parts_analysis=[],
            diagnosis="Not specified",
            key_concern=key_concern,
            extracted_recommendation="Not specified",
            extracted_decision="Not specified",
            ur_decision="Not specified",
            ur_denial_reason=None,
            adls_affected="Not specified",
            work_restrictions="Not specified",
            consulting_doctor="Not specified",
            referral_doctor="Not specified",
            ai_outcome=ai_outcome,
            document_type="medical_document",
            summary_points=["Not specified"],
            date_reasoning=None,
            is_task_needed=False,
            extraction_confidence=0.0,
            verified=False,
            verification_notes=[f"Fallback analysis - {mode.upper()} extraction failed"]
        )

    def get_date_reasoning(self, document_text: str, mode: str = "wc") -> Dict[str, Any]:
        """Get mode-aware date reasoning results"""
        try:
            full_analysis = self.extract_document_data_with_reasoning(document_text, mode=mode)
            return {
                "date_reasoning": full_analysis.date_reasoning.dict() if full_analysis.date_reasoning else {}
            }
        except Exception as e:
            logger.error(f"❌ {mode.upper()} date reasoning failed: {str(e)}")
            return {"date_reasoning": {}}