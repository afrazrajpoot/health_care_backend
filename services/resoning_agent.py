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

    ⚠️ REPORT DATE INSTRUCTION:
    - Identify the ACTUAL report/examination/document date from the text above
    - DO NOT use current/today's date - only use dates explicitly mentioned in the document
    - IMPORTANT: US date format is MM/DD/YYYY. Example: 11/25/2025 means November 25, 2025 (NOT day 11 of month 25)
    - Convert to YYYY-MM-DD format. Example: 11/25/2025 → 2025-11-25
    - If no report date found, use "00/00/0000" as placeholder

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
date
    {signature_context}

    ⚠️ REPORT DATE: Extract the actual report/examination date from the document above. DO NOT use current date.
    - US date format: MM/DD/YYYY. Example: 11/25/2025 = November 25, 2025
    - Output in YYYY-MM-DD format. Example: 11/25/2025 → 2025-11-25

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
    - Extract: Patient/Patient name, claim/claim_number, doi (date of injury), body_part, work_restrictions
    - WC-specific fields: injury_type, work_relatedness, mmi_status, return_to_work_plan
    
    ━━━ CLAIM NUMBER EXTRACTION PATTERNS ━━━
    CRITICAL: Scan the ENTIRE document mainly (header, footer, cc: lines, letterhead) for claim numbers.
    
    Common claim number patterns (case-insensitive):
    - "[Claim #XXXXXXXXX]" or "[Claim #XXXXX-XXX]"
    - "Claim Number: XXXXXXXXX" or "Claim #: XXXXXXXXX"
    - "Claim: XXXXXXXXX" or "Claim #XXXXXXXXX"
    - "WC Claim: XXXXXXXXX" or "Workers Comp Claim: XXXXXXXXX"
    - "Policy/Claim: XXXXXXXXX"
    - In "cc:" lines: "Broadspire [Claim #XXXXXXXXX]"
    - In subject lines or reference fields: "Claim #XXXXXXX"
    
    Extract the FULL claim number including any suffixes (e.g., -001, -002).
    If multiple claim numbers found, use the one that appears most prominently or in the header/footer.

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
    
    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract ALL physician/doctor names mentioned ANYWHERE in the document into the "all_doctors" list.
    - Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, QME, AME, IME, radiologist, etc.
    - Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
    - Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
    - Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
    - If no doctors found, leave list empty [].
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
    - But it should be a valid person name, not a business or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.
    - Extract specialty only if explicitly stated near the signature.
    - If no clear signer is found, leave empty.
    
    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract ALL physician/doctor names mentioned ANYWHERE in the document into the "all_doctors" list.
    - Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
    - Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
    - Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
    - Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
    - If no doctors found, leave list empty [].
    """
    
    def extract_document_data_with_reasoning(
        self,
        document_text: str,
        page_zones: Optional[Dict[str, Dict[str, str]]] = None,
        raw_text: Optional[str] = None,
        mode: Optional[str] = "wc",
        pre_extracted_patient_details: Optional[Dict[str, Any]] = None
    ) -> DocumentAnalysis:
        """
        MODE-AWARE extraction with pre-extracted patient details support.
        
        Args:
            document_text: The document text to analyze
            page_zones: Optional page zones for signature extraction
            raw_text: Optional raw text
            mode: "wc" for Workers Comp or "gm" for General Medicine
            pre_extracted_patient_details: Patient details pre-extracted from Document AI
                Keys: patient_name, dob, doi, claim_number, date_of_report
        """
        try:
            logger.info(f"🚀 Starting MODE-AWARE extraction (Mode: {mode})...")
            
            # Log pre-extracted patient details if provided
            if pre_extracted_patient_details:
                logger.info("📋 Pre-extracted patient details received from Document AI:")
                for key, value in pre_extracted_patient_details.items():
                    if value:
                        logger.info(f"   - {key}: {value}")
            # print(document_text,'document text')
            # NOTE: Report date extraction is handled by LLM from document context
            # OPTIMIZATION: Removed redundant separate detection call
            detected_doc_type = "Unknown - Please Identify from text"

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
            
            # ✅ Override LLM results with pre-extracted patient details if available
            # Pre-extracted values from Document AI (AI + Layout Parser) are more reliable
            if pre_extracted_patient_details:
                logger.info("🔄 Overriding LLM results with pre-extracted patient details...")
                
                # Helper function to check if a value is invalid/placeholder
                def is_invalid_value(value):
                    if not value:
                        return True
                    value_str = str(value).lower().strip()
                    # Check for common placeholder patterns
                    return (
                        value_str in ["not specified", "unknown", "n/a", "", "none"] or
                        value_str.startswith("00/00/") or  # Catches 00/00/0000, 00/00/00, etc.
                        value_str == "0000-00-00"
                    )
                
                # Override patient_name if pre-extracted and LLM returned "Not specified" or empty
                pre_patient_name = pre_extracted_patient_details.get("patient_name")
                if pre_patient_name and is_invalid_value(analysis.patient_name):
                    logger.info(f"   📝 patient_name: '{analysis.patient_name}' → '{pre_patient_name}'")
                    analysis.patient_name = pre_patient_name
                
                # Override dob if pre-extracted and LLM returned empty or placeholder
                pre_dob = pre_extracted_patient_details.get("dob")
                if pre_dob and is_invalid_value(analysis.dob):
                    logger.info(f"   📝 dob: '{analysis.dob}' → '{pre_dob}'")
                    analysis.dob = pre_dob
                
                # Override doi if pre-extracted and LLM returned empty or placeholder
                pre_doi = pre_extracted_patient_details.get("doi")
                if pre_doi and is_invalid_value(analysis.doi):
                    logger.info(f"   📝 doi: '{analysis.doi}' → '{pre_doi}'")
                    analysis.doi = pre_doi
                
                # Override claim_number if pre-extracted and LLM returned empty
                pre_claim = pre_extracted_patient_details.get("claim_number")
                if pre_claim and is_invalid_value(analysis.claim_number):
                    logger.info(f"   📝 claim_number: '{analysis.claim_number}' → '{pre_claim}'")
                    analysis.claim_number = pre_claim
                
                # Override rd (report date) if pre-extracted and LLM returned empty or placeholder
                pre_rd = pre_extracted_patient_details.get("date_of_report")
                if pre_rd and is_invalid_value(analysis.rd):
                    logger.info(f"   📝 rd (report date): '{analysis.rd}' → '{pre_rd}'")
                    analysis.rd = pre_rd
                
                # Override consulting_doctor (author) if pre-extracted and LLM returned empty
                pre_author = pre_extracted_patient_details.get("author")
                if pre_author and (
                    not analysis.consulting_doctor or 
                    str(analysis.consulting_doctor).lower() in ["not specified", "unknown", "n/a", ""]
                ):
                    logger.info(f"   📝 consulting_doctor: '{analysis.consulting_doctor}' → '{pre_author}'")
                    analysis.consulting_doctor = pre_author
            
            # Set verified flag
            analysis.verified = True

            # Log results
            logger.info(f"🎉 {mode.upper()} MODE extraction complete:")
            logger.info(f"   - Patient: {analysis.patient_name}")
            logger.info(f"   - Document Type: {analysis.document_type}")
            logger.info(f"   - Mode: {mode}")
            logger.info(f"   - REPORT DATE (rd): {analysis.rd}")  # Log the extracted report date
            logger.info(f"   - DOB: {analysis.dob}")
            logger.info(f"   - DOI: {analysis.doi}")
            logger.info(f"   - Author: {analysis.consulting_doctor}")
            logger.info(f"   - ALL DOCTORS: {analysis.all_doctors if analysis.all_doctors else '[]'}")
            logger.info(f"   - Confidence: {analysis.extraction_confidence:.2f}")

            return analysis

        except Exception as e:
            logger.error(f"❌ {mode.upper()} mode extraction failed: {str(e)}")
            return self.create_fallback_analysis(mode)
        
    def generate_brief_summary(self, summarizer_output: str, mode: str = "wc") -> str:
        """
        Generate a brief summary of the medical document with detailed clinical specifics.
        """
        logger.info(f"🔍 Generating summary for {mode.upper()} mode...")
        try:
            # NOTE: Report date extraction is handled by LLM from document context

            # Mode-specific summary guidance
            mode_focus = "workers compensation injury" if mode == "wc" else "general medical condition"
            
            system_prompt = SystemMessagePromptTemplate.from_template(
                f"""You are a medical summarization expert. Generate concise 2-3 sentence professional summaries with {mode_focus} focus.
                
    CRITICAL RULES:
    - Extract ONLY information explicitly stated in the document
    - Include specific names: medications (with dosages if mentioned), procedures, therapies, diagnoses
    - Use exact medical terminology from the document
    - Never infer, assume, or add information not present
    - If specific details are absent, do not fabricate them"""
            )

            human_prompt = HumanMessagePromptTemplate.from_template(
                f"""Analyze the following medical document for {mode.upper()} mode and create a detailed yet concise summary.

    DOCUMENT TEXT:
    {{summarizer_output}}

    ⚠️ REPORT DATE: Extract the actual report/examination date from the document above. DO NOT use current/today's date. If not found, use "00/00/0000".
    - US date format: MM/DD/YYYY. Example: 11/25/2025 = November 25, 2025
    - Output in YYYY-MM-DD format. Example: 11/25/2025 → 2025-11-25

    SUMMARY REQUIREMENTS:
    1. Include the specific diagnosis/condition (use exact terms from document)
    2. List key findings or test results (with specific values if mentioned)
    3. Mention specific treatments:
    - Medication names (include dosages if stated: e.g., "ibuprofen 600mg")
    - Therapy types (e.g., "physical therapy", "cognitive behavioral therapy")
    - Procedures performed or recommended (e.g., "MRI of lumbar spine", "arthroscopic surgery")
    4. Include actionable recommendations (e.g., "follow-up in 2 weeks", "continue current regimen")
    5. For {mode_focus}: emphasize work-relatedness, restrictions, or return-to-work status if mentioned

    OUTPUT FORMAT:
    - 2-3 clear, information-dense sentences
    - Clinical terminology appropriate for physician review
    - ONLY use information explicitly present in the document
    - If a detail category (medication/therapy/procedure) is not mentioned, omit it entirely

    {{format_instructions}}"""
            )

            summary_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            chain = summary_prompt | self.llm | self.brief_summary_parser
            result = chain.invoke({
                "summarizer_output": summarizer_output,
                "format_instructions": self.brief_summary_parser.get_format_instructions()
            })

            brief_summary = result.get('brief_summary', 'Not specified')
            logger.info(f"✅ Generated {mode.upper()} summary: {brief_summary}")
            return brief_summary

        except Exception as e:
            logger.error(f"❌ {mode.upper()} summary generation failed: {str(e)}")
            return f"{mode.upper()} brief summary unavailable"

    def create_fallback_analysis(self, mode: str = "wc") -> DocumentAnalysis:
        """Create mode-aware fallback analysis when extraction fails"""
        # Use placeholder date instead of current date (we don't know the actual report date)
        fallback_date = "00/00/0000"
        
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
            dob=fallback_date,
            doi=fallback_date if mode == "wc" else "Not specified",
            status="Not specified",
            rd=fallback_date,
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
            all_doctors=[],
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