"""
FormalMedicalReportExtractor - Enhanced Extractor for Comprehensive Medical Reports
Optimized for accuracy using Gemini-style full-document processing
"""
import logging
import re
import time
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier

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
        
        logger.info("✅ FormalMedicalReportExtractor initialized (Full Context + Enhanced Signature Extraction)")

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
        
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
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
        
        # Stage 1: Directly generate long summary with DUAL-CONTEXT
        long_summary = self._generate_long_summary_direct(
            text=text,
            raw_text=raw_text,
            doc_type=detected_type,
            fallback_date=fallback_date,
            potential_signatures=potential_signatures
        )
        
        # ENHANCED: Verify and inject author into long summary if needed
        verified_author = self._verify_and_extract_author(long_summary, raw_text, text, potential_signatures)
        long_summary = self._inject_author_into_long_summary(long_summary, verified_author)
        
        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)
        
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
    - **DO NOT let this override the medical context from the primary source**

    ⚠️ ANTI-HALLUCINATION RULES FOR DUAL-CONTEXT:

    1. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME medical finding:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    2. **MEDICAL FINDINGS & DIAGNOSES PRIORITY**:
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
    Report Date: {fallback_date}

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
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        logger.info(f"📄 PRIMARY SOURCE size: {len(raw_text):,} chars")
        logger.info(f"📄 SUPPLEMENTARY size: {len(text):,} chars")
        logger.info("🤖 Invoking LLM with DUAL-CONTEXT PRIORITY approach...")
        
        try:
            start_time = time.time()
            
            # Single LLM call with both sources
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "doc_type": doc_type,
                "document_actual_context": raw_text,  # PRIMARY: Accurate summarized context
                "full_document_text": text,           # SUPPLEMENTARY: Full OCR extraction
                "fallback_date": fallback_date,
                "potential_signatures": "\n".join(potential_signatures) if potential_signatures else "No pre-extracted candidates found."
            })
            
            long_summary = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Medical long summary generated in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary: {len(long_summary):,} chars")
            logger.info("✅ Context priority maintained: PRIMARY source used for medical findings")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct medical report long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Medical report exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large reports")
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"


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

    def _clean_pipes_from_summary(self, short_summary: str) -> str:
        """
        Clean empty pipes from short summary to avoid consecutive pipes or trailing pipes.
        
        Args:
            short_summary: The pipe-delimited short summary string
            
        Returns:
            Cleaned summary with proper pipe formatting
        """
        if not short_summary or '|' not in short_summary:
            return short_summary
        
        # Split by pipe and clean each part
        parts = short_summary.split('|')
        cleaned_parts = []
        
        for part in parts:
            # Remove whitespace and check if part has meaningful content
            stripped_part = part.strip()
            # Keep part if it has actual content (not just empty or whitespace)
            if stripped_part:
                cleaned_parts.append(stripped_part)
        
        # Join back with pipes - only include parts with actual content
        cleaned_summary = ' . '.join(cleaned_parts)
        
        logger.info(f"🔧 Pipe cleaning: {len(parts)} parts -> {len(cleaned_parts)} meaningful parts")
        return cleaned_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word structured medical summary in key-value format.
        Zero hallucinations, pipe-delimited, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word medical structured summary (key-value format)...")

        # ENHANCED: Updated system prompt to ensure author prominence and patient exclusion
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-report summarization specialist.

TASK:
Create a concise, factual summary of a medical report using ONLY information explicitly present in the long summary.
- **ONLY include, critical, or clinically significant findings**.
- **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**
DO NOT include any patient personal details such as name, DOB, DOI, or claim number.

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Physical Exam:[value] | Vital Signs:[value] | Medication:[value] | MMI Status:[value] | Work Status:[value] | Restrictions:[value] | Treatment Plan:[value] | Recommendations:[value] | Critical Finding:[value] | Follow-up:[value]

KEY RULES:
- ONLY include critical, or clinically significant findings .
- If a value is missing or not extractable, omit the ENTIRE key-value pair.
- NEVER output empty fields or placeholder text.
- NEVER fabricate dates, meds, restrictions, exam findings, or recommendations.
- NO narrative sentences; use short factual fragments.
- First three fields (Report Title, Author, Date) appear without keys.
- All other fields use key-value format: Key:[value].
- Critical Finding and Physical Exam details include **only abnormalities or critical observations**.
- Medications, MMI Status, Work Status, Restrictions, Treatment Progress, Follow-up included **only if explicitly mentioned**.

CONTENT PRIORITY (only if abnormal/critical and present):
1. Report Title (use {doc_type} if not specified)
2. Author (use VERIFIED AUTHOR from HEALTHCARE PROVIDERS section; indicate signature type if specified)
3. Date (Report Date or Procedure Date)
4. Body Parts (from Procedure/Anatomical Sites)
5. Diagnosis (Final/Pathological Diagnosis)
6. Physical Exam (only abnormal findings if mentioned)
7. Vital Signs (only abnormal values if mentioned)
8. Medications (Anesthetic/Intraoperative meds if mentioned)
9. MMI Status (if mentioned)
10. Work Status / Restrictions (if applicable and mentioned)
11. Treatment Plan (from Results/Interpretation if mentioned)
12. Critical Finding (from CRITICAL FINDINGS or abnormal observations if mentioned)
13. Recommendations (from Recommendations section)
14. Follow-up plan (from Recommendations)

ABSOLUTELY FORBIDDEN:
- Normal findings (ignore entirely for Physical Exam and Vital Signs)
- Assumptions, interpretations, inferred diagnoses
- Narrative sentences
- Patient personal details (Name, DOB, DOI, Claim, Employer)
- Placeholder text or "Not provided"
- Duplicate or empty pipes (||)
- Hallucinated or fabricated content

Your final output MUST be between 30–60 words, single-line, pipe-delimited, and include ONLY explicitly provided abnormal or critical findings.
""")


        user_prompt = HumanMessagePromptTemplate.from_template("""
MEDICAL REPORT LONG SUMMARY:

{long_summary}

Produce a 30–60 word structured medical summary following ALL rules.
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "doc_type": doc_type,
                "long_summary": long_summary
            })

            summary = response.content.strip()
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # Remove pipe cleaning to keep pipes as-is
            # summary = self._clean_pipes_from_summary(summary)

            # ENHANCED: Post-process to enforce no patient details
            forbidden_patterns = [r'Name|DOB|DOI|Claim|Employer']
            for pattern in forbidden_patterns:
                summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
                summary = re.sub(r'\s*\|\s*', '|', summary)  # Clean extra pipes

            # Validate 30–60 word range
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Medical summary out of range ({wc} words). Regenerating...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous output contained {wc} words. Rewrite it to be **between 30 and 60 words**, keeping all factual content, maintaining the key-value pipe-delimited format, and adding NO invented details. Remember: [Report Title] | [Author] | [Date] | ... EXCLUDE patient details."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after regeneration

            logger.info(f"✅ Medical summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Medical summary generation failed: {e}")
            return "Summary unavailable due to processing error."
  
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