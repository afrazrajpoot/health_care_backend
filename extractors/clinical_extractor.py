"""
ClinicalNoteExtractor - Enhanced Extractor for Clinical Progress Notes and Therapy Reports
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
Version: 1.3 - Strict Anti-Hallucination for Signatures
"""
import logging
import re
import time
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from utils.summary_helpers import ensure_date_and_author
from models.data_models import ExtractionResult
from utils.extraction_verifier import ExtractionVerifier

logger = logging.getLogger("document_ai")


class ClinicalNoteExtractor:
    """
    Enhanced Clinical Note extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different clinical specialties
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Progress Notes, PT/OT/Chiro/Acupuncture, Pain Management, Psychiatry, Nursing Notes
    - Direct LLM generation for long summary (removes intermediate extraction)
    - Extracts patient details and signature author for long summary (strict extraction from sign block only)
    - Short summary focuses only on critical findings and abnormal actions (no patient details), includes author if explicitly signed
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.note_type_patterns = {
            'progress_note': re.compile(r'\b(progress note|follow[- ]?up|office visit|clinic note)\b', re.IGNORECASE),
            'physical_therapy': re.compile(r'\b(physical therapy|PT|therapeutic exercise|range of motion|ROM)\b', re.IGNORECASE),
            'occupational_therapy': re.compile(r'\b(occupational therapy|OT|ADL|activities of daily living|functional capacity)\b', re.IGNORECASE),
            'chiropractic': re.compile(r'\b(chiropractic|chiropractor|adjustment|manipulation|subluxation)\b', re.IGNORECASE),
            'acupuncture': re.compile(r'\b(acupuncture|needle|meridian|qi|energy flow)\b', re.IGNORECASE),
            'pain_management': re.compile(r'\b(pain management|pain clinic|chronic pain|pain scale|analgesic)\b', re.IGNORECASE),
            'psychiatry': re.compile(r'\b(psychiatry|psychiatric|mental status|affect|mood|psychotropic)\b', re.IGNORECASE),
            'psychology': re.compile(r'\b(psychology|psychological|therapy session|counseling|behavioral)\b', re.IGNORECASE),
            'nursing': re.compile(r'\b(nursing note|nurse visit|vital signs|nursing assessment|medication administration)\b', re.IGNORECASE)
        }
        
        # Clinical assessment patterns
        self.assessment_patterns = {
            'pain_scale': re.compile(r'\b(pain|discomfort)\s*(scale|level|score)?\s*[:\-]?\s*(\d+/10|\d+\s*out\s*of\s*10)', re.IGNORECASE),
            'rom_measurements': re.compile(r'\b(ROM|range of motion)\s*[:\-]?\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'functional_status': re.compile(r'\b(able to|unable to|independent|assist|assistance|with help)\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'treatment_codes': re.compile(r'\b(CPT[:\s]*(\d{4,5})|(9716[01234]|9753[05]|9775[05]))', re.IGNORECASE)
        }
        
        # Patterns for patient details
        self.patient_patterns = {
            'name': re.compile(r'\b(patient name|name|mr\.?\s*mrs\.?\s*ms\.?\s*)\s*[:\-]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE),
            'dob': re.compile(r'\b(dob|date of birth|birthdate)\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', re.IGNORECASE)
        }
        
        # Enhanced patterns for signature (more comprehensive, distinguish physical/electronic)
        self.signature_patterns = {
            'physical_author': re.compile(r'(?i)(?:handwritten|wet|physical signature|ink signature)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:,?\s*(?:MD|DO|PA|NP|RN|PT|OT|DPT|DC|PhD|etc\.?))?)', re.DOTALL),
            'electronic_author': re.compile(r'(?i)(?:electronically signed|e-signature|digital signature|/s/|typed signature)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:,?\s*(?:MD|DO|PA|NP|RN|PT|OT|DPT|DC|PhD|etc\.?))?)', re.DOTALL),
            'sign_block': re.compile(r'(?i)(signature|sign off|attestation|certification|approval)\s*(?:section|block)?[:\-]?\s*(.*?)(?=\n{2,}|\Z)', re.DOTALL)
        }
        
        logger.info("✅ ClinicalNoteExtractor v1.3 initialized (Full Context + Strict Anti-Hallucination for Signatures)")

    # clinical_extractor.py - UPDATED with dual-context priority

    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
    ) -> Dict:
        """
        Extract Clinical Note data with FULL CONTEXT.
        
        Args:
            text: Complete document text (full OCR extraction)
            raw_text: Accurate summarized context from Document AI Summarizer (PRIMARY SOURCE)
            doc_type: Document type (Progress Note, PT, OT, Chiro, Acupuncture, Pain Management, Psychiatry, Nursing)
            fallback_date: Fallback date if not found
        
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("🏥 STARTING CLINICAL NOTE EXTRACTION (DUAL-CONTEXT PRIORITY)")
        logger.info("=" * 80)
        
        # Auto-detect specific note type if not specified
        detected_type = self._detect_note_type(text, doc_type)
        logger.info(f"📋 Clinical Note Type: {detected_type} (original: {doc_type})")
        
        # Check document size
        # logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        # logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        text_length = len(raw_text)
        token_estimate = text_length // 4
        logger.info(f"📄 PRIMARY source size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Directly generate long summary with DUAL-CONTEXT (raw_text PRIMARY + text SUPPLEMENTARY)
        long_summary = self._generate_long_summary_direct(
            text=text,
            raw_text=raw_text,
            doc_type=detected_type,
            fallback_date=fallback_date
        )
        
        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)
        
        logger.info("=" * 80)
        logger.info("✅ CLINICAL NOTE EXTRACTION COMPLETE (DUAL-CONTEXT)")
        logger.info("=" * 80)
        
        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _generate_long_summary_direct(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Directly generate comprehensive long summary with DUAL-CONTEXT PRIORITY.
        
        PRIMARY SOURCE: raw_text (accurate Document AI summarized context)
        SUPPLEMENTARY: text (full OCR extraction for missing details only)
        """
        logger.info("🔍 Processing clinical note with DUAL-CONTEXT approach...")
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # Enhanced System Prompt with DUAL-CONTEXT PRIORITY
        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert clinical documentation specialist analyzing a COMPLETE {doc_type}.

    🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

    You are provided with TWO versions of the document:

    1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
    - This is the MOST ACCURATE, context-aware summary from Google's Document AI foundation model
    - It preserves CRITICAL CLINICAL CONTEXT with accurate interpretations
    - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
    - Contains CORRECT clinical assessments, accurate treatment context, proper interpretations
    - **ALWAYS PRIORITIZE information from this source**

    2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
    - Complete OCR text extraction (may have formatting noise, OCR artifacts)
    - Use ONLY to fill in SPECIFIC DETAILS missing from the accurate context
    - Examples of acceptable supplementary use:
        * Exact ROM measurements (specific degrees) if not in primary
        * Specific CPT codes or procedure codes
        * Exact medication dosages if not in primary
        * Patient demographics in headers if not in primary
        * Exact pain scores if not in primary
    - **DO NOT let this override the clinical context from the primary source**

    ⚠️ ANTI-HALLUCINATION RULES FOR DUAL-CONTEXT:

    1. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME clinical finding:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    2. **CLINICAL ASSESSMENT PRIORITY**:
    - PRIMARY SOURCE provides accurate clinical interpretations and assessments
    - Use FULL TEXT only for specific measurements if missing
    - NEVER change clinical interpretation based on full text alone

    3. **TREATMENT & MODALITIES**:
    - PRIMARY SOURCE contains accurate treatment context
    - Use FULL TEXT for specific parameters (durations, frequencies, CPT codes) if missing
    - DO NOT add treatments from full text if they contradict primary source

    4. **OBJECTIVE FINDINGS**:
    - PRIMARY SOURCE for clinically significant findings and context
    - Use FULL TEXT only for specific measurements (ROM degrees, strength grades) if missing
    - DO NOT add normal findings from full text if primary focuses on abnormalities

    5. **PATIENT DEMOGRAPHICS**:
    - Check both sources for patient name, DOB, ID
    - PRIMARY SOURCE preferred, but FULL TEXT headers often better for exact demographics
    - Use most complete/accurate version

    6. **SIGNATURE/AUTHOR**:
    - Check PRIMARY SOURCE first for signing provider
    - If not clear, scan FULL TEXT signature blocks (usually last pages)
    - Extract ONLY from explicit sign blocks with signing language

    🔍 EXTRACTION WORKFLOW:

    Step 1: Read PRIMARY SOURCE (accurate context) thoroughly for clinical understanding
    Step 2: Extract ALL clinical findings, assessments, treatments from PRIMARY SOURCE
    Step 3: Check SUPPLEMENTARY SOURCE (full text) ONLY for:
    - Specific measurements missing from primary (ROM degrees, pain scores)
    - Patient demographics in headers
    - CPT codes or procedure codes
    - Additional details not in primary
    Step 4: Verify no contradictions between sources (if conflict, PRIMARY wins)

    ⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY - ABSOLUTE FOR SIGNATURES) (donot include in output, for LLM use only):

    1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
    - If NOT explicitly mentioned in PRIMARY SOURCE, check SUPPLEMENTARY
    - If still not found, return EMPTY string "" or empty list []
    - DO NOT infer, assume, or extrapolate clinical information

    2. **SUBJECTIVE COMPLAINTS - PATIENT'S EXACT WORDS**
    - Extract from PRIMARY SOURCE for accurate context
    - Use FULL TEXT only for exact patient quotes if more complete
    - DO NOT interpret or rephrase

    3. **OBJECTIVE FINDINGS - MEASURED VALUES ONLY**
    - PRIMARY SOURCE for clinical significance
    - Supplement with exact measurements from FULL TEXT if missing
    - DO NOT calculate or estimate ranges

    4. **TREATMENTS & MODALITIES - SPECIFIC DETAILS ONLY**
    - PRIMARY SOURCE for treatment context
    - FULL TEXT for specific parameters if missing
    - DO NOT add typical treatment protocols

    5. **ASSESSMENT & PLAN - CLINICIAN'S EXACT WORDING**
    - PRIMARY SOURCE for clinical reasoning
    - FULL TEXT for exact wording only if more specific
    - DO NOT interpret clinical reasoning

    6. **PATIENT DETAILS - FROM BOTH SOURCES**:
    - Check PRIMARY SOURCE first
    - Use FULL TEXT headers/demographics sections if more complete
    - Use exact formatting from clearest source

    7. **SIGNATURE AUTHOR - STRICTLY FROM SIGN BLOCK ONLY (PHYSICAL/ELECTRONIC) - NO HALLUCINATIONS**
    - Check PRIMARY SOURCE first for author
    - If not clear, scan FULL TEXT signature blocks (last pages)
    - MUST have explicit signing language
    - OMIT if no explicit sign block found

    EXTRACTION FOCUS - 10 CRITICAL CLINICAL NOTE CATEGORIES:

    I. NOTE IDENTITY & ENCOUNTER CONTEXT
    II. PATIENT INFORMATION
    III. SUBJECTIVE FINDINGS (PATIENT'S PERSPECTIVE)
    IV. OBJECTIVE EXAMINATION FINDINGS (CLINICIAN'S OBSERVATIONS)
    V. TREATMENT PROVIDED (SESSION-SPECIFIC)
    VI. ASSESSMENT & CLINICAL IMPRESSION
    VII. TREATMENT PLAN & GOALS
    VIII. WORK STATUS & FUNCTIONAL CAPACITY
    IX. SIGNATURE & AUTHOR (STRICT PHYSICAL/ELECTRONIC - NO ASSUMPTIONS)
    X. OUTCOME MEASURES & PROGRESS TRACKING

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

    ⚠️ FINAL REMINDER (donot include in output, for LLM use only):
    - PRIMARY SOURCE is your MAIN reference for clinical context
    - Use FULL TEXT only for specific missing details (measurements, codes, exact demographics)
    - NEVER override primary source clinical interpretations with full text
    - SIGNATURE: ONLY from explicit sign block; OMIT if no explicit signing phrase

    Now analyze this COMPLETE {doc_type} using the DUAL-CONTEXT PRIORITY approach and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
    """)

        # Updated User Prompt with clear source separation
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

    Generate the long summary in this EXACT STRUCTURED FORMAT using the DUAL-CONTEXT PRIORITY rules:

    👤 PATIENT INFORMATION
    --------------------------------------------------
    [Check PRIMARY SOURCE first, use FULL TEXT headers if more complete]

    Name: [from primary, supplement from full text headers if needed]
    DOB: [from primary, supplement from full text headers if needed]
    Other Details: [claim number - check full text headers first, then primary]

    📋 CLINICAL ENCOUNTER OVERVIEW
    --------------------------------------------------
    Note Type: {doc_type}
    Visit Date: [from primary source, supplement if needed]
    Visit Type: [from primary]
    Duration: [from primary]
    Facility: [from primary]

    👨‍⚕️ PROVIDER INFORMATION
    --------------------------------------------------
    Treating Provider: [from primary source]
    Credentials: [from primary]
    Specialty: [from primary]

    ━━━ CLAIM NUMBER EXTRACTION ━━━
    - if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
    - Check FULL TEXT headers/footers FIRST for exact claim numbers
    - Then check PRIMARY SOURCE if full text unclear
    - Scan for patterns: "[Claim #XXXXXXXXX]", "Claim Number:", "WC Claim:"

    All Doctors Involved:
    • [extract from BOTH sources, deduplicate, prefer primary source format]

    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract from BOTH sources (primary + supplementary)
    - Deduplicate: If same doctor in both, use PRIMARY SOURCE format
    - Include: treating physician, referring doctor, etc.

    🗣️ SUBJECTIVE FINDINGS
    --------------------------------------------------
    [FROM PRIMARY SOURCE for accurate clinical context]
    [Supplement with exact patient quotes from FULL TEXT if more complete]

    Chief Complaint: [primary source]
    Pain: [primary source, supplement exact scores from full text]

    Functional Limitations:
    • [from primary source, list up to 5, exact wording]

    🔍 OBJECTIVE EXAMINATION
    --------------------------------------------------
    [FROM PRIMARY SOURCE for clinically significant findings]
    [Supplement with exact measurements from FULL TEXT if missing]

    Range of Motion:
    • [primary for clinical significance, full text for exact degrees if missing]

    Manual Muscle Testing:
    • [primary for findings, full text for exact grades if missing]

    Special Tests:
    • [from primary source, list up to 3, with results]

    💆 TREATMENT PROVIDED
    --------------------------------------------------
    [FROM PRIMARY SOURCE for treatment context]
    [Supplement with specific parameters from FULL TEXT if missing]

    Treatment Techniques:
    • [primary source, supplement CPT codes from full text if missing]

    Therapeutic Exercises:
    • [from primary source, list up to 5]

    Modalities Used:
    • [primary source, supplement parameters from full text if missing]

    🏥 CLINICAL ASSESSMENT
    --------------------------------------------------
    [ALL FROM PRIMARY SOURCE for accurate clinical reasoning]

    Assessment: [primary source]
    Progress: [primary source]
    Clinical Impression: [primary source]
    Prognosis: [primary source]

    🎯 TREATMENT PLAN
    --------------------------------------------------
    [FROM PRIMARY SOURCE for treatment planning context]

    Short-term Goals:
    • [from primary source, list up to 3]

    Home Exercise Program:
    • [from primary source, list up to 3]

    Frequency/Duration: [primary source]
    Next Appointment: [primary source]

    💼 WORK STATUS
    --------------------------------------------------
    Current Status: [primary source]

    Work Restrictions:
    • [from primary source, list up to 5, exact wording]

    Functional Capacity: [primary source]

    📊 OUTCOME MEASURES
    --------------------------------------------------
    Pain Scale: [primary source, supplement exact number from full text if missing]

    Functional Scores:
    • [from primary source, list up to 3]

    ✍️ SIGNATURE & AUTHOR
    --------------------------------------------------
    [Check PRIMARY SOURCE first, then FULL TEXT signature blocks if unclear]

    Author:
    hint: check primary source first, then full text signature block (last pages) if unclear
    • Signature: [extracted name/title if physical or electronic signature present; otherwise omit]

    🚨 CRITICAL CLINICAL FINDINGS
    --------------------------------------------------
    • [from PRIMARY SOURCE - list up to 8 most significant items]

    REMEMBER: 
    1. PRIMARY SOURCE (accurate context) is your MAIN reference for clinical interpretations
    2. Use FULL TEXT only to supplement specific missing details (measurements, codes, exact demographics)
    3. NEVER override primary source clinical context with full text
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
                "fallback_date": fallback_date
            })
            
            long_summary = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Clinical long summary generated in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary: {len(long_summary):,} chars")
            logger.info("✅ Context priority maintained: PRIMARY source used for clinical findings")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct clinical note long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Clinical note exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large notes")
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"

    def _detect_note_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific clinical note type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for note_type, pattern in self.note_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[note_type] = len(matches)
        
        # Boost scores for treatment-specific terminology
        if self.assessment_patterns['treatment_codes'].search(text):
            for note_type in ['physical_therapy', 'occupational_therapy', 'chiropractic']:
                type_scores[note_type] = type_scores.get(note_type, 0) + 2
        
        if self.assessment_patterns['pain_scale'].search(text):
            for note_type in ['pain_management', 'progress_note']:
                type_scores[note_type] = type_scores.get(note_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].replace('_', ' ').title()
                logger.info(f"🔍 Auto-detected note type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect note type, using: {original_type}")
        return original_type or "Clinical Note"

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
        cleaned_summary = ' | '.join(cleaned_parts)
        
        logger.info(f"🔧 Pipe cleaning: {len(parts)} parts -> {len(cleaned_parts)} meaningful parts")
        return cleaned_summary
    
    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word clinical note summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        Focuses ONLY on critical findings and abnormal actions (no patient details), includes author ONLY if explicitly signed with type.
        Starts with Report Title first.
        """

        logger.info("🎯 Generating 30–60 word clinical structured summary (critical findings & abnormal actions only, strict author with type, Report Title first)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a clinical documentation specialist.

TASK:
Create a concise, factual clinical summary using ONLY information explicitly present in the long summary.
- **ONLY include, critical, or clinically significant findings**.
- **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Report Title] | Author:[value] | Date:[value] | Diagnosis:[value] | Physical Exam:[value] | Vital Signs:[value] | Critical Findings:[value] | Actions:[value] | Medications:[value] | Work Status:[value] | Recommendation:[value] | Follow-up:[value]

KEY RULES (VERY IMPORTANT):
- **ONLY include, critical, or clinically significant findings**.
- **If a value is missing or cannot be extracted, omit the ENTIRE key-value pair.**
- **Do NOT output empty fields, empty values, or placeholder text.**
 - For Author never use "Dr." with it
- **Do NOT output double pipes (`||`).**
- NO hallucination. NO invented findings, meds, or follow-up.
- MUST be **one single line**, pipe-delimited, no line breaks.

FIELD DEFINITIONS:
- Report Title: Always include this field first → `: {doc_type}`
- Author: Only if explicitly signed with signature type
  ("Signature: Smith" → "Author: Smith")  
  If there is no signature text, omit the entire field.
- Critical Findings: Significant clinical findings, worsening conditions, flagged issues.
- Abnormal Actions: Any abnormal interventions, adverse reactions, therapy responses, compliance issues.
- Medications: ONLY meds explicitly mentioned; skip if none.
- Physical Exam: ONLY abnormal exam findings (reduced ROM, weakness, swelling, tenderness, abnormal vitals).
- Follow-up: ONLY if explicitly stated (return date, re-eval, imaging follow-up, specialist referral).

CONTENT PRIORITY (ONLY IF ABNORMAL AND PRESENT):
1. Report Title: Always included
2. Author (only explicit signed)
3. Critical Findings
4. objective findings
5. actions or responses
6. Medications
7. Physical Exam ( abnormal only if present)
8. Follow-up instructions
9. Vital Signs (abnormal only if present)
10. Recommendations (if given)
11. Work Status (if given)

ABSOLUTELY FORBIDDEN:
- Normal findings for physical exam and vital signs (ignore these fields if all normal)
- Patient details (name, DOB, demographics)
- Using provider names as author without explicit signature
- For Author never use "Dr." with it
- Assumptions, interpretations, or inferred issues
- Narrative writing
- Placeholder text (e.g., “None”, “Not provided”)
- Duplicate pipes or empty fields

Your final output MUST be between 30–60 words, follow the exact pipe-delimited key-value style, contain ONLY factual abnormal findings explicitly in the summary, and omit any fields not extracted.
""")


        user_prompt = HumanMessagePromptTemplate.from_template("""
    CLINICAL LONG SUMMARY:

    {long_summary}

    Now produce a 30–60 word structured clinical summary following ALL rules, starting with Report Title, including author ONLY if explicitly signed with type and signing language, focusing only on critical findings and abnormal actions.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
                "doc_type": doc_type,
                "long_summary": long_summary
            })
            summary = response.content.strip()

            # Normalize whitespace only - no pipe cleaning
            summary = re.sub(r"\s+", " ", summary).strip()
            # Programmatically add missing Date or Author if LLM missed them
            summary = ensure_date_and_author(summary, long_summary)

            # Validate word count
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Clinical summary out of range ({wc} words). Attempting auto-fix...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior summary contained {wc} words. Rewrite it to be between 30 and 60 words. "
                        "DO NOT add fabricated details or patient info. Preserve all factual elements including author ONLY if explicit with type and signing language, Report Title first. "
                        "Maintain key-value pipe-delimited format: Report Title:[value] | Author:[value] | Critical Findings:[value] | Abnormal Actions:[value] (skip missing keys)"
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # No pipe cleaning after auto-fix
                # Programmatically add missing Date or Author if LLM missed them
                summary = ensure_date_and_author(summary, long_summary)

            logger.info(f"✅ Clinical summary generated: {len(summary.split())} words")
            return self._clean_pipes_from_summary(summary)

        except Exception as e:
            logger.error(f"❌ Clinical summary generation failed: {e}")
            return "Summary unavailable due to processing error."

    def _create_clinical_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback clinical summary directly from long summary"""
        
        # Extract key clinical information using regex patterns
        patterns = {
            'provider': r'Treating Provider:\s*([^\n]+)',
            'pain': r'Pain Scale:\s*([^\n]+)',
            'complaint': r'Chief Complaint:\s*([^\n]+)',
            'rom': r'Range of Motion:(.*?)(?:\n\n|\n[A-Z]|$)',
            'treatment': r'Treatment Techniques:(.*?)(?:\n\n|\n[A-Z]|$)',
            'assessment': r'Assessment:\s*([^\n]+)',
            'plan': r'Frequency/Duration:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and date
        parts.append(f"{doc_type} visit")
        
        # Add provider context
        if 'provider' in extracted:
            parts.append(f"by {extracted['provider']}")
        
        # Add chief complaint and pain
        if 'complaint' in extracted:
            complaint = extracted['complaint'][:60] + "..." if len(extracted['complaint']) > 60 else extracted['complaint']
            parts.append(f"for {complaint}")
        
        if 'pain' in extracted:
            parts.append(f"Pain: {extracted['pain']}")
        
        # Add key findings
        if 'rom' in extracted:
            first_rom = extracted['rom'].split('\n')[0].replace('•', '').strip()[:50]
            parts.append(f"Findings: {first_rom}")
        
        # Add treatment
        if 'treatment' in extracted:
            first_treatment = extracted['treatment'].split('\n')[0].replace('•', '').strip()[:50]
            parts.append(f"Treatment: {first_treatment}")
        
        # Add assessment and plan
        if 'assessment' in extracted:
            assessment = extracted['assessment'][:60] + "..." if len(extracted['assessment']) > 60 else extracted['assessment']
            parts.append(f"Assessment: {assessment}")
        
        if 'plan' in extracted:
            parts.append(f"Plan: {extracted['plan']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with continued clinical management", "following treatment protocols", "with ongoing progress monitoring"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used clinical fallback summary: {len(summary.split())} words")
        return summary