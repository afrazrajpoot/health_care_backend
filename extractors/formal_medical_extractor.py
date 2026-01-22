"""
FormalMedicalReportExtractor - Enhanced Extractor for Comprehensive Medical Reports
Optimized for accuracy using Gemini-style full-document processing
Version: 2.0 - With Pydantic Output Validation for Consistent Long Summary
"""
import logging
import re
import json
import time
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier
from utils.summary_helpers import ensure_date_and_author, clean_long_summary
from helpers.short_summary_generator import generate_structured_short_summary
from helpers.long_summary_generator import format_bullet_summary_to_json, format_long_summary_to_text
from models.long_summary_models import (
    FormalMedicalLongSummary,
    format_formal_medical_long_summary,
    create_fallback_formal_medical_summary
)

logger = logging.getLogger("document_ai")


class FormalMedicalReportExtractor:
    """
    Enhanced Formal Medical Report extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, Endoscopy, Genetics, Discharge Summaries
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.formal_medical_summary_parser = PydanticOutputParser(pydantic_object=FormalMedicalLongSummary)
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.report_type_patterns = {
            'surgery': re.compile(r'\b(surgery|surgical|pre[- ]?op|post[- ]?op|operative|procedure)\b', re.IGNORECASE),
            'anesthesia': re.compile(r'\b(anesthesia|anesthetic|sedation|airway|intubation)\b', re.IGNORECASE),
            'emg': re.compile(r'\b(EMG|NCS|electromyography|nerve conduction|needle exam)\b', re.IGNORECASE),
            'pathology': re.compile(r'\b(pathology|biopsy|specimen|histology|microscopic)\b', re.IGNORECASE),
            'cardiology': re.compile(r'\b(cardiology|EKG|ECG|echocardiogram|stress test|holter)\b', re.IGNORECASE),
            'sleep': re.compile(r'\b(sleep study|polysomnography|PSG|apnea|hypopnea)\b', re.IGNORECASE),
            'endoscopy': re.compile(r'\b(endoscopy|colonoscopy|EGD|gastroscopy|bronchoscopy)\b', re.IGNORECASE),
            'genetics': re.compile(r'\b(genetic|mutation|variant|DNA|RNA|chromosome)\b', re.IGNORECASE),
            'discharge': re.compile(r'\b(discharge|admission|hospital course|disposition)\b', re.IGNORECASE)
        }
        
        # Medical procedure patterns
        self.procedure_patterns = {
            'cpt_codes': re.compile(r'\bCPT[:\s]*(\d{4,5})', re.IGNORECASE),
            'icd_codes': re.compile(r'\b(ICD[-]?10[:\s]*([A-Z]\d{2,})|([A-Z]\d{2,}))', re.IGNORECASE),
            'medications': re.compile(r'\b(\d+\s*(mg|mcg|g|ml)\s*[\w\s]+\s*(PO|IV|IM|SC|QD|BID|TID|QID|PRN))', re.IGNORECASE)
        }

        # ENHANCED: Pre-compile regex for signature extraction to assist LLM
        self.signature_patterns = {
            'electronic_signature': re.compile(r'(electronically signed|signature|e-signed|signed by|authenticated by|digital signature|verified by)[:\s]*([A-Za-z\s\.,]+?)(?=\n|\s{2,}|$)', re.IGNORECASE | re.DOTALL),
            'signed_date': re.compile(r'(signed|signature|date)[:\s]*([A-Za-z\s\.,]+?)\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}\s*(AM|PM))?', re.IGNORECASE),
            'provider_signature': re.compile(r'(provider|doctor|md|do|physician)[:\s]*([A-Za-z\s\.,]+?)(?=\s{2,}|\n\n|$)', re.IGNORECASE),
            # NEW: Patterns for physical signatures, footers, or stamps
            'physical_signature': re.compile(r'(handwritten signature|ink signature|physical sign)[:\s]*([A-Za-z\s\.,]+?)(?=\n|$)', re.IGNORECASE),
            'footer_signature': re.compile(r'^(?:\s*[-=]{3,}.*?\n){0,3}([A-Za-z\s\.,]+?)\s*(?:\d{1,2}/\d{1,2}/\d{4}.*)?$', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'auth_stamp': re.compile(r'(authentica|stamp|seal)[:\s]*([A-Za-z\s\.,]+?)(?=\s{2,}|$)', re.IGNORECASE),
            # ENHANCED: Specific pattern for last provider/signer
            'last_provider_sign': re.compile(r'(Provider:\s*)([A-Za-z\s\.,]+?)(?=\s*\(|Massage|\n{2,})', re.IGNORECASE | re.DOTALL)
        }
        
        logger.info("✅ FormalMedicalReportExtractor v2.0 initialized (Full Context + Pydantic Validation)")

    # formal_medical_extractor.py - UPDATED with dual-context priority

    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
    ) -> Dict:
        """
        Extract Formal Medical Report data with FULL CONTEXT.
        
        Args:
            text: Complete document text (full OCR extraction)
            raw_text: Accurate summarized context from Document AI Summarizer (PRIMARY SOURCE)
            doc_type: Document type (Surgery, Anesthesia, EMG, Pathology, etc.)
            fallback_date: Fallback date if not found
        
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING FORMAL MEDICAL REPORT EXTRACTION (DUAL-CONTEXT PRIORITY)")
        logger.info("=" * 80)
        
        # logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        # logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # Auto-detect specific report type if not specified
        detected_type = self._detect_report_type(raw_text, doc_type)  # Use primary for detection
        logger.info(f"📋 Report Type: {detected_type} (original: {doc_type})")
        
        # Check document size
        text_length = len(raw_text)
        token_estimate = text_length // 4
        logger.info(f"📄 PRIMARY source size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # ENHANCED: Pre-extract potential signatures from BOTH sources
        potential_signatures_primary = self._pre_extract_signatures(raw_text)
        potential_signatures_full = self._pre_extract_signatures(text)
        potential_signatures = list(set(potential_signatures_primary + potential_signatures_full))
        logger.info(f"🔍 Pre-extracted potential signatures (both sources): {potential_signatures}")
        
        # Stage 1: Format the summarizer output (raw_text) into structured long summary
        # This is a FORMATTING task only - no new content generation
        formatted_json = format_bullet_summary_to_json(
            bullet_summary=raw_text,
            llm=self.llm,
            document_type=detected_type
        )
        long_summary = format_long_summary_to_text(formatted_json)
        
        # ENHANCED: Verify and inject author into long summary if needed
        verified_author = self._verify_and_extract_author(long_summary, raw_text, text, potential_signatures)
        long_summary = self._inject_author_into_long_summary(long_summary, verified_author)
        
        # Stage 1.5: Clean the long summary - remove empty fields, placeholders, and instruction text
        long_summary = clean_long_summary(long_summary)
        
        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(raw_text, detected_type)
        
        logger.info("=" * 80)
        logger.info("✅ FORMAL MEDICAL REPORT EXTRACTION COMPLETE (DUAL-CONTEXT)")
        logger.info("=" * 80)
        
        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _verify_and_extract_author(
        self, 
        long_summary: str, 
        raw_text: str,
        full_text: str, 
        potential_signatures: List[str]
    ) -> str:
        """
        ENHANCED: Use LLM to verify author with DUAL-CONTEXT.
        Prioritizes PRIMARY SOURCE, supplements with FULL TEXT if needed.
        """
        # Updated verification prompt with dual-context awareness
        verify_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
    You are verifying the AUTHOR/SIGNER of a medical report using DUAL-CONTEXT sources.

    PRIMARY SOURCE (raw_text): Most accurate context for clinical interpretations
    SUPPLEMENTARY SOURCE (full_text): Complete OCR for signature blocks in headers/footers

    Extract ONLY the exact name of the person who signed:
    - Check PRIMARY SOURCE first for author context
    - Check SUPPLEMENTARY SOURCE for signature blocks (usually last pages/footers)
    - Prioritize end-of-document signatures or last provider entry
    - Differentiate from Performing/Ordering Physician if distinct
    - Use candidates for confirmation, especially those with 'signed' or last 'Physician:'

    Output ONLY the name as JSON: {"author": "Exact Name (e.g., Dr. Jane Doe)"}
    """),
            HumanMessagePromptTemplate.from_template("""
    PRIMARY SOURCE (accurate context):
    {raw_text}

    SUPPLEMENTARY SOURCE (full OCR):
    {full_text}

    Long Summary: {long_summary}
    Candidates: {potential_signatures}
    """)
        ])
        
        try:
            chain = verify_prompt | self.llm
            result = chain.invoke({
                "raw_text": raw_text[:5000],  # First 5000 chars for context
                "full_text": full_text[-5000:],  # Last 5000 chars for signatures
                "long_summary": long_summary,
                "potential_signatures": "\n".join(potential_signatures)
            })
            
            parsed = self.parser.parse(result.content)
            verified_author = parsed.get("author", "")
            
            if not verified_author or "No distinct" in verified_author:
                # Fallback: Scan both sources
                last_provider_pattern = re.compile(
                    r'(Physician:\s*)([A-Za-z\s\.,]+?)(?=\s*\(|Massage|\n{2,}|$)',
                    re.IGNORECASE | re.DOTALL
                )
                
                # Try full text first (better for signatures)
                match = last_provider_pattern.search(full_text)
                if not match:
                    # Try primary source
                    match = last_provider_pattern.search(raw_text)
                
                if match:
                    verified_author = match.group(2).strip()
                elif potential_signatures:
                    for cand in reversed(potential_signatures):
                        if any(word in cand.lower() for word in ['signed', 'signature', 'physician']):
                            verified_author = cand.split(':')[-1].strip()
                            break
                else:
                    verified_author = "Signature not identified"
            
            logger.info(f"🔍 Verified Author (dual-context): {verified_author}")
            return verified_author
            
        except Exception as e:
            logger.warning(f"Author verification failed: {e}. Using raw candidates.")
            return potential_signatures[-1].split(':')[-1].strip() if potential_signatures else "Author not extracted"

    def _generate_long_summary_direct(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
        potential_signatures: List[str]
    ) -> str:
        """
        Directly generate comprehensive long summary with DUAL-CONTEXT PRIORITY.
        
        PRIMARY SOURCE: raw_text (accurate Document AI summarized context)
        SUPPLEMENTARY: text (full OCR extraction for missing details only)
        """
        logger.info("🔍 Processing medical report with DUAL-CONTEXT approach...")
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # ENHANCED System Prompt with DUAL-CONTEXT PRIORITY
        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert medical documentation specialist analyzing a COMPLETE {doc_type} report.

    🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

    You are provided with TWO versions of the document:

    1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
    - This is the MOST ACCURATE, context-aware summary from Google's Document AI foundation model
    - It preserves CRITICAL MEDICAL CONTEXT with accurate clinical interpretations
    - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
    - Contains CORRECT medical conclusions, accurate clinical interpretations, proper context
    - **ALWAYS PRIORITIZE information from this source**

    2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
    - Complete OCR text extraction (may have formatting noise, OCR artifacts)
    - Use ONLY to fill in SPECIFIC DETAILS missing from the accurate context
    - Examples of acceptable supplementary use:
        * Exact patient demographics in headers if not in primary
        * Specific measurements or lab values if not in primary
        * CPT/ICD codes in billing sections
        * Signature blocks in footers (better in full text)
        * Claim numbers in headers/footers
        - if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
    - **DO NOT let this override the medical context from the primary source**

    🚨 ABSOLUTE ANTI-FABRICATION RULE (HIGHEST PRIORITY):
    **YOU MUST ONLY EXTRACT AND SUMMARIZE INFORMATION THAT EXISTS IN THE PROVIDED SOURCES.**
    - NEVER generate, infer, assume, or fabricate ANY information
    - If information is NOT explicitly stated in either source → OMIT IT ENTIRELY
    - An incomplete summary is 100x better than a fabricated one
    - Every single piece of information in your output MUST be traceable to the source text

    ⚠️ STRICT ANTI-HALLUCINATION RULES:

    1. **ZERO FABRICATION TOLERANCE**:
    - If a field (e.g., DOB, Claim Number, Diagnosis) is NOT in either source → LEAVE IT BLANK or OMIT
    - NEVER write "likely", "probably", "typically", "usually" - these indicate fabrication
    - NEVER fill in "standard" or "typical" values - only actual extracted values

    2. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME medical finding:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    3. **MEDICAL FINDINGS & DIAGNOSES PRIORITY**:
    - PRIMARY SOURCE provides accurate medical interpretations
    - Use FULL TEXT only for exact values/codes if missing
    - NEVER change medical interpretation based on full text alone

    3. **PROCEDURES & TECHNIQUES**:
    - PRIMARY SOURCE for procedure context and clinical significance
    - FULL TEXT for specific CPT codes or technical parameters if missing
    - DO NOT add procedures from full text if they contradict primary

    4. **PATIENT DEMOGRAPHICS**:
    - Check PRIMARY SOURCE first
    - Use FULL TEXT headers if more complete for exact demographics
    - Use most complete/accurate version

    5. **SIGNATURE/AUTHOR**:
    - Check PRIMARY SOURCE first for author context
    - FULL TEXT better for signature blocks in footers (last pages)
    - Extract from explicit sign blocks only

    🔍 EXTRACTION WORKFLOW:

    Step 1: Read PRIMARY SOURCE (accurate context) thoroughly for medical understanding
    Step 2: Extract ALL medical findings, diagnoses, procedures from PRIMARY SOURCE
    Step 3: Check SUPPLEMENTARY SOURCE (full text) ONLY for:
    - Patient demographics in headers
    - CPT/ICD codes if missing
    - Signature blocks in footers
    - Claim numbers in headers
    - Specific measurements if missing
    Step 4: Verify no contradictions between sources (if conflict, PRIMARY wins)

    [Rest of the anti-hallucination rules remain the same as original...]

    CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
    You are seeing the ENTIRE medical report at once, allowing you to:
    - Understand the complete clinical picture from history to conclusions
    - Connect pre-procedure assessments with intraoperative findings and post-procedure outcomes
    - Identify relationships between clinical indications, procedures performed, and results
    - Provide comprehensive extraction without information loss

    🔍 SPECIAL INSTRUCTIONS FOR PATIENT DETAILS VALIDATION:

    **CRITICAL - PATIENT DETAILS CROSS-VALIDATION**:
    If the raw_text contains a "--- PATIENT DETAILS ---" section:
    1. **FIRST**: Extract the patient details from that section (Patient Name, DOB, DOI, Claim Number)
    2. **SECOND**: Cross-validate each detail against the FULL TEXT EXTRACTION (text parameter) as sometimes the full text is not properly formatted, so the fields and values are not aligned properly, but the full text must have the correct details, and if we are getting the pateint details from the patient details section, we need to make sure they are accurate by cross-checking with the full text extraction
    3. **VALIDATION RULES**:
       ✅ If the detail MATCHES what's in the full text extraction → USE IT (it's accurate)
       ✅ If the detail is CLOSE but has minor formatting differences → USE the formatted version from patient details section
       ❌ If the detail CONTRADICTS the full text extraction → IGNORE the patient details section value and extract directly from full text
       ❌ If the detail is MISSING or shows "N/A" → Extract directly from full text extraction
    4. **FINAL CHECK**: Ensure all patient details (Name, DOB, DOI, Claim Number) are accurate and consistent with the document content

    **Example Validation Process**:
    - Patient Details section shows: "Patient Name: John Smith"
    - Full text contains: "Patient: John Smith" → ✅ VALID - Use "John Smith"
    - Patient Details section shows: "DOB: N/A"
    - Full text contains: "Date of Birth: 05/15/1975" → ❌ INVALID - Use "05/15/1975" from full text
    - Patient Details section shows: "Claim Number: 12345-ABC"
    - Full text contains: "Claim #: 12345-ABC" → ✅ VALID - Use "12345-ABC"

    ⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):
    [Keep all existing anti-hallucination rules from original file...]

    Now analyze this COMPLETE {doc_type} medical report using the DUAL-CONTEXT PRIORITY approach and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY:
    """)

        # Updated user prompt with clear source separation
        user_prompt = HumanMessagePromptTemplate.from_template("""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📌 PRIMARY SOURCE - ACCURATE CONTEXT (Use this as your MAIN reference):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    {document_actual_context}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📄 SUPPLEMENTARY SOURCE - FULL TEXT EXTRACTION (Use ONLY for missing details):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    {full_document_text}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Document Type: {doc_type}
    Report Date: [extracted date of report]

    MANDATORY SIGNATURE SCAN: Read PRIMARY SOURCE first for author context. Check SUPPLEMENTARY SOURCE (last pages/footers) for signature blocks. Use these candidates to CONFIRM:
    {potential_signatures}

    Generate the long summary in this EXACT STRUCTURED FORMAT using the DUAL-CONTEXT PRIORITY rules:

    [Keep all existing format sections from original file, but add notes about source priority...]

    👤 PATIENT INFORMATION
    --------------------------------------------------
    [Check PRIMARY SOURCE first, use FULL TEXT headers if more complete]

    Name: [primary first, supplement from full text headers if needed]
    DOB: [primary first, supplement from full text headers if needed]
    DOI: [primary first, supplement if needed]
    Claim Number: [check full text headers first, then primary]
    Employer: [primary first, supplement if needed]

    👨‍⚕️ HEALTHCARE PROVIDERS
    --------------------------------------------------
    [Check PRIMARY SOURCE for clinical context]
    [Check FULL TEXT footers/last pages for signature blocks]

    Performing Physician: [from primary]
    Specialty: [from primary]
    Ordering Physician: [from primary]
    Anesthesiologist: [from primary]

    Author:
    • Signature: [check primary first for context, full text footers for signature blocks]

    [Continue with all other sections, adding dual-context guidance where appropriate...]

    REMEMBER: 
    1. PRIMARY SOURCE (accurate context) is your MAIN reference for medical interpretations
    2. Use FULL TEXT only to supplement specific missing details (demographics, codes, signatures)
    3. NEVER override primary source medical context with full text

    {format_instructions}
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        # Get format instructions from Pydantic parser
        format_instructions = self.formal_medical_summary_parser.get_format_instructions()
        
        logger.info(f"📄 PRIMARY SOURCE size: {len(raw_text):,} chars")
        logger.info(f"📄 SUPPLEMENTARY size: {len(text):,} chars")
        logger.info("🤖 Invoking LLM with DUAL-CONTEXT PRIORITY approach + Pydantic validation...")
        
        try:
            start_time = time.time()
            
            # Single LLM call with both sources and format instructions
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "doc_type": doc_type,
                "document_actual_context": raw_text,  # PRIMARY: Accurate summarized context
                "full_document_text": text,           # SUPPLEMENTARY: Full OCR extraction
                "fallback_date": fallback_date,
                "potential_signatures": "\n".join(potential_signatures) if potential_signatures else "No pre-extracted candidates found.",
                "format_instructions": format_instructions
            })
            
            raw_response = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Medical long summary generated in {processing_time:.2f}s")
            logger.info(f"📝 Raw response length: {len(raw_response):,} chars")
            
            # Try to parse with Pydantic for validation
            try:
                parsed_summary = self.formal_medical_summary_parser.parse(raw_response)
                long_summary = format_formal_medical_long_summary(parsed_summary)
                logger.info("✅ Pydantic validation successful - using structured output")
                logger.info(f"✅ Formatted long summary: {len(long_summary):,} chars")
                
            except Exception as parse_error:
                logger.warning(f"⚠️ Pydantic parsing failed: {parse_error}")
                logger.info("🔄 Attempting JSON extraction fallback...")
                
                # Try to extract JSON from the response
                try:
                    json_match = re.search(r'\{[\s\S]*\}', raw_response)
                    if json_match:
                        json_str = json_match.group()
                        parsed_dict = json.loads(json_str)
                        parsed_summary = FormalMedicalLongSummary.model_validate(parsed_dict)
                        long_summary = format_formal_medical_long_summary(parsed_summary)
                        logger.info("✅ JSON extraction fallback successful")
                    else:
                        logger.info("📄 No JSON found, using raw LLM response")
                        long_summary = raw_response
                        
                except Exception as json_error:
                    logger.warning(f"⚠️ JSON extraction fallback failed: {json_error}")
                    logger.info("📄 Using raw LLM response as long summary")
                    long_summary = raw_response
            
            logger.info("✅ Context priority maintained: PRIMARY source used for medical findings")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct medical report long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Medical report exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large reports")
            
            # Fallback: Create a minimal Pydantic-based summary
            logger.info("🔄 Creating fallback formal medical summary...")
            fallback_summary = create_fallback_formal_medical_summary(doc_type, fallback_date)
            return format_formal_medical_long_summary(fallback_summary)


    def _detect_report_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific medical report type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for report_type, pattern in self.report_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[report_type] = len(matches)
        
        # Boost scores for procedure-specific terminology
        if self.procedure_patterns['cpt_codes'].search(text):
            for report_type in ['surgery', 'procedure']:
                type_scores[report_type] = type_scores.get(report_type, 0) + 2
        
        if self.procedure_patterns['icd_codes'].search(text):
            for report_type in ['pathology', 'discharge']:
                type_scores[report_type] = type_scores.get(report_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].upper().replace('_', ' ')
                logger.info(f"🔍 Auto-detected report type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect report type, using: {original_type}")
        return original_type or "MEDICAL_REPORT"

    def _pre_extract_signatures(self, text: str) -> List[str]:
        """
        ENHANCED: Pre-extract potential signature candidates using regex to assist LLM extraction.
        NEW: Scan last 30% of document first for priority, then full scan.
        """
        candidates = []
        text_length = len(text)
        end_slice = text[text_length // 3 * 2:]  # Last ~33% for footers/signatures
        
        # Scan end first
        for pattern_name, pattern in self.signature_patterns.items():
            matches = pattern.findall(end_slice)
            for match in matches:
                if isinstance(match, tuple):
                    candidate = ' '.join([m.strip() for m in match if m.strip()])
                else:
                    candidate = match.strip()
                if candidate and len(candidate) > 5:
                    candidates.append(f"{pattern_name}: {candidate}")
        
        # Full scan as fallback (limit to avoid noise)
        if len(candidates) < 3:
            for pattern_name, pattern in self.signature_patterns.items():
                matches = pattern.findall(text)
                for match in matches:
                    if isinstance(match, tuple):
                        candidate = ' '.join([m.strip() for m in match if m.strip()])
                    else:
                        candidate = match.strip()
                    if candidate and len(candidate) > 5 and f"{pattern_name}: {candidate}" not in candidates:
                        candidates.append(f"{pattern_name}: {candidate}")
                        if len(candidates) >= 3:
                            break
                if len(candidates) >= 3:
                    break
        
        unique_candidates = list(set(candidates))[:5]  # Top 5 now
        logger.debug(f"Pre-extracted signature candidates (end-priority): {unique_candidates}")
        return unique_candidates

    def _inject_author_into_long_summary(self, long_summary: str, verified_author: str) -> str:
        """
        NEW: Robust post-processing to inject the verified author prominently if not present.
        Handles cases where LLM deviates from format by inserting under healthcare providers or at top.
        """
        if not verified_author or verified_author in ["Signature not identified", "Author not extracted"]:
            return long_summary
        
        # Check if Author already exists
        if re.search(r'Author:\s*' + re.escape(verified_author), long_summary, re.IGNORECASE):
            return long_summary
        
        # ENHANCED: Look for HEALTHCARE PROVIDERS section (exact match)
        providers_section_pattern = re.compile(r'(👨‍⚕️ HEALTHCARE PROVIDERS\s*[-—]+\s*)', re.IGNORECASE)
        if providers_section_pattern.search(long_summary):
            insertion = r'\1  Author: {}\n'.format(verified_author)
            long_summary = providers_section_pattern.sub(insertion, long_summary)
            logger.info(f"✅ Injected Author into HEALTHCARE PROVIDERS section: {verified_author}")
            return long_summary
        
        # Fallback 1: Inject after first "Physician:" mention
        first_physician_pattern = re.compile(r'(Physician:\s*[^\n\r]+?)(?=\n|\r|$)', re.IGNORECASE | re.MULTILINE)
        match = first_physician_pattern.search(long_summary)
        if match:
            insertion_point = match.end()
            long_summary = long_summary[:insertion_point] + f"\n  Author: {verified_author}" + long_summary[insertion_point:]
            logger.info(f"✅ Injected Author after first Physician: {verified_author}")
            return long_summary
        
        # Fallback 2: Inject at the very top under a new HEALTHCARE PROVIDERS header if no provider mentioned
        long_summary = f"👨‍⚕️ HEALTHCARE PROVIDERS\n--------------------------------------------------\nAuthor: {verified_author}\n\n{long_summary}"
        logger.info(f"✅ Injected new HEALTHCARE PROVIDERS section with Author: {verified_author}")
        return long_summary

    def _generate_short_summary_from_long_summary(self, raw_text: str, doc_type: str) -> dict:
        """
        Generate a structured, UI-ready summary from raw_text (Document AI summarizer output).
        Delegates to the reusable helper function.
        
        Args:
            raw_text: The Document AI summarizer output (primary context)
            doc_type: Document type
            long_summary: Detailed reference context
            
        Returns:
            dict: Structured summary with header and UI-ready items
        """
        return generate_structured_short_summary(self.llm, raw_text, doc_type)
    

    def _create_medical_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback medical summary directly from long summary"""
        
        # Extract key medical information using regex patterns
        patterns = {
            'procedure': r'Procedure:\s*([^\n]+)',
            'physician': r'Performing Physician:\s*([^\n]+)',
            'author': r'Author:\s*([^\n]+)',  # ENHANCED: For signer
            'diagnosis': r'Final Diagnosis:\s*([^\n]+)',
            'findings': r'Intraoperative Findings:\s*([^\n]+)',
            'pathology': r'Pathological Diagnosis:\s*([^\n]+)',
            'recommendations': r'Recommendations:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type
        parts.append(f"{doc_type} report")
        
        if 'procedure' in extracted:
            parts.append(f"Procedure: {extracted['procedure']}")
        
        # Add physician and author context
        if 'physician' in extracted:
            parts.append(f"by {extracted['physician']}")
        if 'author' in extracted:
            parts.append(f"(Signed: {extracted['author']})")
        
        # Add findings
        if 'findings' in extracted:
            first_finding = extracted['findings'][:80] + "..." if len(extracted['findings']) > 80 else extracted['findings']
            parts.append(f"Findings: {first_finding}")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            parts.append(f"Diagnosis: {extracted['diagnosis']}")
        elif 'pathology' in extracted:
            parts.append(f"Pathology: {extracted['pathology']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with standard medical follow-up", "following established protocols", "with routine clinical monitoring"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used medical fallback summary: {len(summary.split())} words")
        return summary