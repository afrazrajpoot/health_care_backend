"""
Specialist Consult Enhanced Extractor - Full Context
Optimized for accuracy using Gemini-style full-document processing
8 CRITICAL FIELDS FOCUSED (NO HALLUCINATION, ONLY EXPLICIT INFORMATION)
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
from utils.summary_helpers import ensure_date_and_author

logger = logging.getLogger("document_ai")


class ConsultExtractorChained:
    """
    Enhanced Consult extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning for specialist recommendation evaluation
    - 8-FIELD FOCUSED extraction for zero hallucination
    - Optimized for Workers' Compensation consultation analysis
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)

    # consult_extractor.py - UPDATED with dual-context priority

    def extract(self, text: str, raw_text: str, doc_type: str, fallback_date: str) -> Dict:
        """
        Extract specialist consult report with full context processing.
        
        Args:
            text: Complete document text (full OCR extraction)
            raw_text: Accurate summarized context from Document AI Summarizer (PRIMARY SOURCE)
            doc_type: Document type
            fallback_date: Fallback date if not found
        
        Returns dictionary with long_summary and short_summary like QME extractor.
        """
        logger.info("=" * 80)
        logger.info("👨‍⚕️ STARTING CONSULT EXTRACTION (DUAL-CONTEXT PRIORITY)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Directly generate long summary with DUAL-CONTEXT (raw_text PRIMARY + text SUPPLEMENTARY)
            long_summary = self._generate_long_summary_direct(
                text=text,
                raw_text=raw_text,
                doc_type=doc_type,
                fallback_date=fallback_date
            )
            
            # Step 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context consultation extraction completed in {elapsed_time:.2f}s")
            logger.info("=" * 80)
            logger.info("✅ CONSULT EXTRACTION COMPLETE (DUAL-CONTEXT)")
            logger.info("=" * 80)
            
            # Return dictionary with both summaries like QME extractor
            return {
                "long_summary": long_summary,
                "short_summary": short_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            # Return fallback result structure
            return {
                "long_summary": f"Consultation extraction failed: {str(e)}",
                "short_summary": "Consultation summary not available"
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
        logger.info("🔍 Processing consultation report with DUAL-CONTEXT approach...")
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # Updated System Prompt with DUAL-CONTEXT PRIORITY
        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert Workers' Compensation consultation report specialist analyzing a COMPLETE Specialist Consultation Report.

    🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

    You are provided with TWO versions of the document:

    1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
    - This is the MOST ACCURATE, context-aware summary from Google's Document AI foundation model
    - It preserves CRITICAL CONSULTATION CONTEXT with accurate clinical interpretations
    - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
    - Contains CORRECT diagnostic conclusions, accurate specialist assessments, proper treatment justifications
    - **ALWAYS PRIORITIZE information from this source**

    2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
    - Complete OCR text extraction (may have formatting noise, OCR artifacts)
    - Use ONLY to fill in SPECIFIC DETAILS missing from the accurate context
    - Examples of acceptable supplementary use:
        * Exact medication dosages if not in primary source
        * Specific claim numbers or identifiers in headers/footers
        * Additional doctor names mentioned
        * Precise ROM measurements or lab values
        * Exact work restriction wording if more specific in full text
    - **DO NOT let this override the clinical context from the primary source**

    ⚠️ ANTI-HALLUCINATION RULES FOR DUAL-CONTEXT:

    1. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME clinical finding:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    2. **DIAGNOSIS PRIORITY**:
    - PRIMARY SOURCE provides accurate diagnostic context
    - Use FULL TEXT only to add ICD-10 codes or body part specifics if missing
    - NEVER change diagnosis interpretation based on full text alone

    3. **RECOMMENDATIONS PRIORITY** (MOST CRITICAL FOR AUTHORIZATION):
    - PRIMARY SOURCE contains accurate, contextually appropriate recommendations
    - Use FULL TEXT only for specific procedure details (CPT codes, exact injection locations) if missing
    - DO NOT add recommendations from full text if they contradict primary source

    4. **WORK RESTRICTIONS**:
    - PRIMARY SOURCE provides accurate work capacity assessment
    - Use FULL TEXT for exact restriction wording only if more specific
    - NEVER add restrictions from full text not present in primary source

    5. **OBJECTIVE FINDINGS**:
    - PRIMARY SOURCE contains clinically significant findings
    - Use FULL TEXT only for specific measurements (ROM degrees, strength scores) if missing
    - DO NOT add normal findings from full text if primary source focuses on abnormalities

    6. **CLAIM NUMBERS & IDENTIFIERS**:
    - if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
    - These are often in headers/footers (better in FULL TEXT)
    - Check FULL TEXT first for exact claim numbers
    - Use PRIMARY SOURCE if full text is unclear

    🔍 EXTRACTION WORKFLOW:

    Step 1: Read PRIMARY SOURCE (accurate context) thoroughly for clinical understanding
    Step 2: Extract ALL consultation findings, diagnoses, recommendations from PRIMARY SOURCE
    Step 3: Check SUPPLEMENTARY SOURCE (full text) ONLY for:
    - Specific details missing from primary (exact dosages, measurements, codes)
    - Administrative info (claim numbers, exact dates in headers)
    - Additional doctor names
    Step 4: Verify no contradictions between sources (if conflict, PRIMARY wins)

    ⚠️ CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY) (donot include in output, for LLM use only):

    1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
    - If NOT explicitly mentioned in PRIMARY SOURCE, check SUPPLEMENTARY
    - If still not found, return EMPTY string "" or empty list []
    - DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps

    2. **DIAGNOSES - INCLUDE ALL WITH EXACT TERMINOLOGY**
    - Extract PRIMARY diagnosis from PRIMARY SOURCE
    - Supplement with ICD-10 codes from FULL TEXT if not in primary
    - Include ALL SECONDARY diagnoses from PRIMARY SOURCE

    3. **WORK RESTRICTIONS - EXACT WORDING ONLY**
    - Use PRIMARY SOURCE for restriction context
    - Use FULL TEXT for exact wording only if more specific
    - If "no overhead work" stated in primary, extract "no overhead work" (NOT "no overhead reaching")

    4. **RECOMMENDATIONS - ZERO TOLERANCE FOR ASSUMPTIONS**
    - Extract ONLY recommendations explicitly stated in PRIMARY SOURCE
    - Supplement with procedure details from FULL TEXT only if missing
    - DO NOT extract treatments "considered but not recommended"

    5. **EMPTY FIELDS ARE BETTER THAN GUESSED FIELDS**
    - Leave fields empty if information not found in PRIMARY SOURCE
    - DO NOT use "Not mentioned", "Not stated", "Unknown" - just return ""

    6. **CONSULTING PHYSICIAN/AUTHOR DETECTION**:
    - Check PRIMARY SOURCE first for author/signing physician
    - If not clear, scan FULL TEXT signature blocks (usually last pages)
    - Extract name as explicitly signed, regardless of credentials

    7. **ALL DOCTORS EXTRACTION**:
    - Extract from BOTH sources (primary + supplementary)
    - Deduplicate: If same doctor in both, use primary source version
    - Include all physicians with credentials

    CONSULTATION EXTRACTION FOCUS - 8 CRITICAL FIELDS FOR CLAIMS ADMINISTRATION:

    FIELD 1: HEADER & CONTEXT - Report Identity & Authority
    FIELD 2: CHIEF COMPLAINT - Patient's Primary Issue
    FIELD 3: DIAGNOSIS & ASSESSMENT - Medical Conclusion (HIGHEST PRIORITY)
    FIELD 4: HISTORY OF PRESENT ILLNESS - Symptoms & Severity Context
    FIELD 5: PRIOR TREATMENT & EFFICACY - Failure of Conservative Care
    FIELD 6: OBJECTIVE FINDINGS - Verifiable Evidence (Exam & Imaging)
    FIELD 7: PLAN - RECOMMENDED TREATMENTS (MOST CRITICAL FOR AUTHORIZATION)
    FIELD 8: WORK STATUS & IMPAIRMENT - Legal/Administrative Status

    Now analyze this COMPLETE specialist consultation report using the DUAL-CONTEXT PRIORITY approach and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
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

    📋 CONSULTATION OVERVIEW
    --------------------------------------------------
    Document Type: {doc_type}
    Consultation Date: [extracted or {fallback_date}]
    Consulting Physician: [from primary source, check full text signature if unclear]
    Specialty: [from primary source]
    Referring Physician: [from primary source]

    👤 PATIENT INFORMATION
    --------------------------------------------------
    Name: [from primary source, supplement if needed]
    Date of Birth: [from primary source, supplement if needed]
    Date of Injury: [from primary source]
    Claim Number: [from full text headers/footers first]

    All Doctors Involved:
    • [extract from BOTH sources, deduplicate, prefer primary source format]

    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract ALL physician/doctor names from BOTH sources
    - Deduplicate: If same doctor appears in both, use PRIMARY SOURCE format
    - Include: consulting doctor, referring doctor, ordering physician, treating physician, etc.
    - Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor)
    - Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles
    - Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO")
    - If no doctors found, leave list empty []

    🎯 CHIEF COMPLAINT
    --------------------------------------------------
    Primary Complaint: [from primary source]
    Location: [from primary source]
    Duration: [from primary source]
    Radiation Pattern: [from primary source]

    Author:
    hint: check primary source first, then full text signature block (last pages) if unclear
    • Signature: [extracted name/title; should not be business name or generic title]

    🏥 DIAGNOSIS & ASSESSMENT
    --------------------------------------------------
    Primary Diagnosis: [from PRIMARY SOURCE - accurate diagnostic context]
    - ICD-10: [supplement from full text if not in primary]
    - Certainty: [from primary source]

    Secondary Diagnoses:
    • [from PRIMARY SOURCE - list up to 5]

    Causation: [from primary source interpretation]

    🔬 CLINICAL HISTORY & SYMPTOMS
    --------------------------------------------------
    [All from PRIMARY SOURCE for accurate clinical context]

    Pain Quality: [from primary]
    Pain Location: [from primary]
    Radiation: [from primary]

    Aggravating Factors:
    • [from primary source, list up to 5, exact wording]

    Alleviating Factors:
    • [from primary source, list up to 5, exact wording]

    💊 PRIOR TREATMENT & EFFICACY
    --------------------------------------------------
    Prior Treatments Received:
    • [from PRIMARY SOURCE for treatment efficacy context, supplement durations from full text if missing]

    Level of Relief:
    • [from primary source - accurate efficacy assessment]

    Treatment Failure Statement: [from primary source]

    📊 OBJECTIVE FINDINGS
    --------------------------------------------------
    Physical Examination:
    • [FROM PRIMARY SOURCE for clinically significant findings]
    • [Supplement with specific measurements from full text if missing: ROM degrees, strength scores]

    Imaging Review:
    • [FROM PRIMARY SOURCE for imaging interpretation]
    • [Supplement with specific measurements from full text if missing]

    🎯 TREATMENT RECOMMENDATIONS (⚠️ MOST CRITICAL - PRIMARY SOURCE PRIORITY)
    --------------------------------------------------
    Injections Requested:
    • [FROM PRIMARY SOURCE - accurate treatment justification]
    • [Supplement with exact injection sites/CPT codes from full text if missing]

    Procedures Requested:
    • [FROM PRIMARY SOURCE - list up to 5 with reasons]

    Surgery Recommended:
    • [FROM PRIMARY SOURCE - list up to 3 with urgency]

    Diagnostics Ordered:
    • [FROM PRIMARY SOURCE - list up to 5 with reasons]

    Medication Changes:
    • [FROM PRIMARY SOURCE - supplement exact dosages from full text if missing]

    Therapy Recommendations:
    • [FROM PRIMARY SOURCE - supplement frequency details from full text if missing]

    💼 WORK STATUS & IMPAIRMENT
    --------------------------------------------------
    Current Work Status: [from PRIMARY SOURCE]

    Work Restrictions:
    • [FROM PRIMARY SOURCE for restriction context]
    • [Use full text for exact wording only if more specific]

    Restriction Duration: [from primary source]
    Return to Work Plan: [from primary source]

    🚨 CRITICAL FINDINGS
    --------------------------------------------------
    • [list up to 8 most actionable items from PRIMARY SOURCE]

    REMEMBER: 
    1. PRIMARY SOURCE (accurate context) is your MAIN reference for clinical interpretations
    2. Use FULL TEXT only to supplement specific missing details (measurements, codes, exact dates)
    3. NEVER override primary source clinical context with full text
    """)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            user_prompt
        ])
        
        logger.info(f"📄 PRIMARY SOURCE size: {len(raw_text):,} chars")
        logger.info(f"📄 SUPPLEMENTARY size: {len(text):,} chars")
        logger.info("🤖 Invoking LLM with DUAL-CONTEXT PRIORITY approach...")
        
        # Invoke LLM with both sources
        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "document_actual_context": raw_text,  # PRIMARY: Accurate summarized context
                "full_document_text": text,           # SUPPLEMENTARY: Full OCR extraction
                "fallback_date": fallback_date,
                "doc_type": doc_type
            })
            
            long_summary = result.content.strip()
            
            logger.info("✅ Generated consultation long summary with DUAL-CONTEXT PRIORITY")
            logger.info("✅ Context priority maintained: PRIMARY source used for clinical findings")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct LLM generation failed: {str(e)}")
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"


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
    
    def _generate_short_summary_from_long_summary(self, long_summary: str) -> str:
        """
        Generate a precise 30–60 word consultation summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word consultation structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-legal consultation specialist.

TASK:
Create a concise, factual consultation summary using ONLY information explicitly stated in the long summary.
- **ONLY include, critical, or clinically significant findings**.
- **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**
STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Physical Exam:[value] | Vital Signs:[value] | Key Findings:[value] | Medication:[value] | Treatments:[value] | Clinical Assessment:[value] | Plan:[value] | MMI Status:[value] | Work Status:[value] | Critical Finding:[value]

KEY RULES:
- ONLY include abnormal or critical findings for these fields (physical exam, vital signs).
- For Author never use "Dr." with it
- If a value is missing or not extractable, omit the ENTIRE key-value pair.
- NEVER output empty fields or placeholder text.
- NEVER fabricate dates, meds, findings, or recommendations.
- NO narrative sentences; use short factual fragments.
- First three fields (Report Title, Author, Date) appear without keys.
- All other fields use key-value format: Key:[value].
- Key Findings and Physical Exam details include **only abnormalities or critical observations**.
- Medications, Treatments, Plan, MMI Status, Work Status included only if explicitly stated in the summary.

CONTENT PRIORITY (only if critical and provided):
1. Report Title  
2. Author  
3. Date  
4. Body Parts  
5. Diagnosis  
6. Physical Exam (abnormal only if present)  
7. Vital Signs (abnormal only if present)  
8. Key objective findings (abnormal only)  
9. Treatment plan (given explicit only)
10. Medications (given explicit only)  
11. Recommendations (if explicitly stated) or Plan (if explicitly or stated)  
12. MMI status (if stated)  
13. Work status (if stated)  
14. Critical finding

ABSOLUTELY FORBIDDEN:
- Normal findings (ignore entirely for physical exam and vital signs)
- For Author never use "Dr." with it
- Assumptions, interpretations, inferred diagnoses
- Patient details (name, DOB, claim, MRN, etc.)
- Narrative writing
- Placeholder text or "Not provided"
- Duplicate or empty pipes (||)

Your final output MUST be between 30–60 words, single-line, pipe-delimited, and contain ONLY explicitly provided abnormal or critical information.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
    CONSULTATION LONG SUMMARY:

    {long_summary}

    Now produce a 30–60 word structured consultation summary following ALL rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({
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
                logger.warning(f"⚠️ Consultation summary out of range ({wc} words). Fixing...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your previous summary contained {wc} words. Rewrite it to be **between 30 and 60** words. "
                        "Do NOT add fabricated data. Preserve all factual elements. Maintain the key-value pipe-delimited format: [Report Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | etc."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # Re-ensure date and author after correction
                summary = ensure_date_and_author(summary, long_summary)

            logger.info(f"✅ Consultation summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Consultation summary generation failed: {e}")
            return "Summary unavailable due to processing error."
 
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract physician information
        physician_match = re.search(r'Consulting Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Consulting Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'diagnosis': r'Primary Diagnosis:\s*([^\n]+)',
            'complaint': r'Primary Complaint:\s*([^\n]+)',
            'recommendations': r'TREATMENT RECOMMENDATIONS(.*?)(?:\n\n|\n[A-Z]|$)',
            'restrictions': r'Work Restrictions:(.*?)(?:\n\n|\n[A-Z]|$)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with physician
        parts.append(f"{physician} consultation")
        
        # Add diagnosis
        if 'diagnosis' in extracted:
            parts.append(f"for {extracted['diagnosis'][:80]}")
        elif 'complaint' in extracted:
            parts.append(f"for {extracted['complaint'][:80]}")
        
        # Add recommendations
        if 'recommendations' in extracted:
            # Take first line of recommendations
            first_rec = extracted['recommendations'].split('\n')[0].replace('•', '').strip()[:60]
            if first_rec:
                parts.append(f"Recommendations: {first_rec}")
        
        # Add work restrictions
        if 'restrictions' in extracted:
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:50]
            if first_restrict:
                parts.append(f"Restrictions: {first_restrict}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["with follow-up planned", "for ongoing management", "and progress evaluation"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary

    def _safe_str(self, value, default="") -> str:
        """Convert any value to string safely - MUST return STRING"""
        try:
            if value is None:
                return str(default)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                # For dicts, convert to string representation
                return str(value)
            if isinstance(value, list):
                # For lists, join with comma
                flat = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        flat.append(str(item))
                    elif item:
                        flat.append(str(item))
                return ", ".join(flat) if flat else str(default)
            return str(value)
        except Exception as e:
            logger.error(f"Error in _safe_str: {str(e)}")
            return str(default)