"""
Imaging Reports Enhanced Extractor - 6 Critical Imaging Fields Focused

Optimized for MRI, X-ray, CT-scan, and other imaging modalities
Full-context processing with anti-hallucination rules
NO assumptions, NO self-additions, ONLY explicit information from report
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
from utils.summary_helpers import ensure_date_and_author, clean_long_summary

logger = logging.getLogger("document_ai")


class ImagingExtractorChained:
    """
    Enhanced Imaging extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - 6-FIELD IMAGING FOCUS (Header, Clinical Data, Technique, Key Findings, Impression, Recommendations)
    - ZERO tolerance for hallucination, assumptions, or self-additions
    - Only extracts EXPLICITLY STATED information from imaging reports
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex patterns for imaging specific content
        self.imaging_patterns = {
            'modality': re.compile(r'\b(MRI|CT|X-RAY|XRAY|ULTRASOUND|US|MAMMOGRAM|PET|SPECT|DEXA)\b', re.IGNORECASE),
            'body_part': re.compile(r'\b(shoulder|knee|spine|wrist|hip|ankle|elbow|hand|foot|brain|chest|abdomen|pelvis|lumbar|cervical|thoracic)\b', re.IGNORECASE),
            'contrast': re.compile(r'\b(with|without)\s+contrast\b', re.IGNORECASE),
            'finding_severity': re.compile(r'\b(mild|moderate|severe|minimal|marked|advanced|subtle|questionable|probable|likely)\b', re.IGNORECASE)
        }
        
        logger.info("✅ ImagingExtractorChained initialized (6-Field Imaging Focus)")

    # imaging_extractor.py - UPDATED with dual-context priority

    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
    ) -> Dict:
        """
        Extract imaging data with FULL CONTEXT and 6-field focus.
        
        Args:
            text: Complete document text (full OCR extraction)
            raw_text: Accurate summarized context from Document AI Summarizer (PRIMARY SOURCE)
            doc_type: Document type (MRI, CT, X-ray, Ultrasound, etc.)
            fallback_date: Fallback date if not found
        
        Returns dictionary with long_summary and short_summary like QME extractor.
        """
        logger.info("=" * 80)
        logger.info("📊 STARTING IMAGING EXTRACTION (DUAL-CONTEXT PRIORITY)")
        logger.info("=" * 80)
        
        # logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        # logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        start_time = time.time()
        
        try:
            # Step 1: Directly generate long summary with DUAL-CONTEXT (raw_text PRIMARY + text SUPPLEMENTARY)
            long_summary = self._generate_long_summary_direct(
                text=text,
                raw_text=raw_text,
                doc_type=doc_type,
                fallback_date=fallback_date
            )
            
            # Step 1.5: Clean the long summary - remove empty fields, placeholders, and instruction text
            long_summary = clean_long_summary(long_summary)
            
            # Step 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context imaging extraction completed in {elapsed_time:.2f}s")
            logger.info("=" * 80)
            logger.info("✅ IMAGING EXTRACTION COMPLETE (DUAL-CONTEXT)")
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
                "long_summary": f"Imaging extraction failed: {str(e)}",
                "short_summary": "Imaging summary not available"
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
        logger.info("🔍 Processing imaging report with DUAL-CONTEXT approach...")
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # Build system prompt with DUAL-CONTEXT PRIORITY and 6-field imaging focus
        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert radiological report specialist analyzing a COMPLETE imaging report.

    🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

    You are provided with TWO versions of the document:

    1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
    - This is the MOST ACCURATE, context-aware summary from Google's Document AI foundation model
    - It preserves CRITICAL RADIOLOGICAL CONTEXT with accurate interpretations
    - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
    - Contains CORRECT radiological impressions, accurate findings context, proper interpretations
    - **ALWAYS PRIORITIZE information from this source**

    2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
    - Complete OCR text extraction (may have formatting noise, OCR artifacts)
    - Use ONLY to fill in SPECIFIC DETAILS missing from the accurate context
    - Examples of acceptable supplementary use:
        * Exact measurements (sizes, dimensions) if not in primary
        * Specific anatomical locations if more precise in full text
        * Patient demographics in headers if not in primary
        * Technical parameters (contrast details, protocols) if missing
    - **DO NOT let this override the radiological context from the primary source**

    🚨 ABSOLUTE ANTI-FABRICATION RULE (HIGHEST PRIORITY):
    **YOU MUST ONLY EXTRACT AND SUMMARIZE INFORMATION THAT EXISTS IN THE PROVIDED SOURCES.**
    - NEVER generate, infer, assume, or fabricate ANY information
    - If information is NOT explicitly stated in either source → OMIT IT ENTIRELY
    - An incomplete summary is 100x better than a fabricated one
    - Every single piece of information in your output MUST be traceable to the source text

    ⚠️ STRICT ANTI-HALLUCINATION RULES:

    1. **ZERO FABRICATION TOLERANCE**:
    - If a field (e.g., DOB, Measurement, Finding) is NOT in either source → LEAVE IT BLANK or OMIT
    - NEVER write "likely", "probably", "typically", "usually" - these indicate fabrication
    - NEVER fill in "standard" or "typical" values - only actual extracted values

    2. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME radiological finding:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    3. **RADIOLOGICAL FINDINGS PRIORITY**:
    - PRIMARY SOURCE provides accurate radiological interpretations and significance
    - Use FULL TEXT only for exact measurements if missing
    - NEVER change radiological interpretation based on full text alone

    3. **IMPRESSION & DIAGNOSIS**:
    - PRIMARY SOURCE contains accurate impressions and diagnostic conclusions
    - Use FULL TEXT only for specific details if missing
    - DO NOT add diagnoses from full text if they contradict primary source

    4. **MEASUREMENTS & DIMENSIONS**:
    - PRIMARY SOURCE for clinical significance of measurements
    - Use FULL TEXT for exact dimensions if more precise
    - DO NOT add measurements from full text without context from primary

    5. **TECHNICAL DETAILS**:
    - PRIMARY SOURCE for technique context and protocols
    - FULL TEXT for specific parameters (contrast type, sequences) if missing
    - Check both sources for contrast status - use most clear/accurate

    6. **PATIENT DEMOGRAPHICS**:
    - Check both sources for patient name, DOB
    - FULL TEXT headers often better for exact demographics
    - Use most complete/accurate version

    7. **RADIOLOGIST/AUTHOR**:
    - Check PRIMARY SOURCE first for signing radiologist
    - If not clear, scan FULL TEXT signature blocks (usually last pages)
    - Extract ONLY from explicit sign blocks

    🔍 EXTRACTION WORKFLOW:

    Step 1: Read PRIMARY SOURCE (accurate context) thoroughly for radiological understanding
    Step 2: Extract ALL findings, impressions, diagnoses from PRIMARY SOURCE
    Step 3: Check SUPPLEMENTARY SOURCE (full text) ONLY for:
    - Exact measurements missing from primary
    - Patient demographics in headers
    - Technical parameters if missing
    - Additional specific details
    Step 4: Verify no contradictions between sources (if conflict, PRIMARY wins)

    PRIMARY PURPOSE: Generate a comprehensive, structured long summary of the 6 critical imaging fields for accurate medical documentation.

    CRITICAL ANTI-HALLUCINATION RULES (ABSOLUTE PRIORITY):

    1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
    - If NOT explicitly mentioned in PRIMARY SOURCE, check SUPPLEMENTARY
    - If still not found, return EMPTY string "" or empty list []
    - DO NOT infer, assume, extrapolate, or use medical knowledge to fill gaps

    2. **FINDINGS - ZERO TOLERANCE FOR ASSUMPTIONS**
    - Extract from PRIMARY SOURCE for accurate radiological context
    - Supplement with exact measurements from FULL TEXT if missing
    - DO NOT extract "possible" or "rule out" as confirmed findings

    3. **TECHNICAL DETAILS - EXACT WORDING**
    - PRIMARY SOURCE for technique context
    - FULL TEXT for specific parameters if missing
    - Contrast status: Use clearest version from either source

    4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
    - Better to return empty field than to guess
    - DO NOT use "Not mentioned", "Not stated", "Unknown" - just return ""

    5. **RADIOLOGIST'S EXACT LANGUAGE**
    - PRIMARY SOURCE for radiological interpretations
    - Use FULL TEXT only for exact wording if more specific
    - Preserve certainty qualifiers: "suspicious", "likely", "consistent with", "probable"

    6. **RADIOLOGIST/AUTHOR DETECTION**:
    - Check PRIMARY SOURCE first
    - If not clear, scan FULL TEXT signature blocks (last pages)
    - Extract name as explicitly signed

    6 CRITICAL IMAGING FIELDS:

    FIELD 1: HEADER & CONTEXT (Report Identity & Date)
    FIELD 2: CLINICAL DATA/INDICATION (Reason for the Study)
    FIELD 3: TECHNIQUE/PRIOR STUDIES (Methodology & Comparison)
    FIELD 4: KEY FINDINGS - POSITIVE/NEGATIVE (Evidence of Pathology)
    FIELD 5: IMPRESSION/CONCLUSION (Radiologist's Final Diagnosis)
    FIELD 6: RECOMMENDATIONS/FOLLOW-UP (Actionable Next Steps)

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

    Now analyze this COMPLETE imaging report using the DUAL-CONTEXT PRIORITY approach and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
    """)

        # Build user prompt with clear source separation
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

    Generate the long summary in this EXACT STRUCTURED FORMAT using the DUAL-CONTEXT PRIORITY rules:

    📋 IMAGING OVERVIEW
    --------------------------------------------------
    Document Type: {doc_type}
    Exam Date: [from primary, supplement if needed]
    Exam Type: [from primary]
    Radiologist: [from primary, check full text signature if unclear]
    Imaging Center: [from primary]
    Referring Physician: [from primary]

    Author:
    hint: check primary source first, then full text signature block (last pages) if unclear
    • Signature: [extracted name/title if physical or electronic signature present; otherwise omit]

    ## PATIENT INFORMATION
    - **Name:** [check both sources, use most complete]
    - **Date of Birth:** [check both sources, use most complete]
    - **Claim Number:** [check full text headers first, then primary]
    - **Date of Injury:** [from primary]
    - **Employer:** [from primary]

    ━━━ CLAIM NUMBER EXTRACTION ━━━
    - Check FULL TEXT headers/footers FIRST for exact claim numbers
    - Then check PRIMARY SOURCE if full text unclear
    - Scan for patterns: "[Claim #XXXXXXXXX]", "Claim Number:", "WC Claim:"
    - if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field

                                                            
    All Doctors Involved:
    • [extract from BOTH sources, deduplicate, prefer primary source format]

    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract from BOTH sources (primary + supplementary)
    - Deduplicate: If same doctor in both, use PRIMARY SOURCE format
    - Include: radiologist, referring doctor, ordering physician

    🎯 CLINICAL INDICATION
    --------------------------------------------------
    [FROM PRIMARY SOURCE for clinical context]

    Clinical Indication: [primary source]
    Clinical History: [primary source]
    Chief Complaint: [primary source]
    Specific Questions: [primary source]

    🔧 TECHNICAL DETAILS
    --------------------------------------------------
    [FROM PRIMARY SOURCE for technique context]
    [Supplement specific parameters from FULL TEXT if missing]

    Study Type: {doc_type}
    Body Part Imaged: [primary, supplement from full text if more specific]
    Laterality: [primary, supplement if needed]
    Contrast Used: [check both sources, use clearest]
    Contrast Type: [primary, supplement if needed]
    Prior Studies Available: [primary]
    Technical Quality: [primary]
    Limitations: [primary]

    📊 KEY FINDINGS
    --------------------------------------------------
    [FROM PRIMARY SOURCE for radiological significance]
    [Supplement with exact measurements from FULL TEXT if missing]

    Primary Finding:
    • Description: [primary for context, full text for exact measurements]
    • Location: [primary for context, full text for precision]
    • Size: [primary if stated, full text for exact dimensions]
    • Characteristics: [from primary source]
    • Acuity: [from primary source]

    Secondary Findings:
    • [from primary source, supplement measurements from full text]

    Normal Findings:
    • [from primary source, list up to 5]

    💡 IMPRESSION & CONCLUSION
    --------------------------------------------------
    [ALL FROM PRIMARY SOURCE for accurate radiological interpretation]

    Overall Impression: [primary source - radiologist's exact language]
    Primary Diagnosis: [primary source]
    Final Diagnostic Statement: [primary source]

    Differential Diagnoses:
    • [from primary source, list up to 3]

    Clinical Correlation: [primary source]

    📋 RECOMMENDATIONS & FOLLOW-UP
    --------------------------------------------------
    [FROM PRIMARY SOURCE for recommendations context]

    Follow-up Recommended: [primary source]
    Follow-up Modality: [primary source]
    Follow-up Timing: [primary source]
    Clinical Correlation Needed: [primary source]
    Specialist Consultation: [primary source]

    ⚠️ MANDATORY EXTRACTION RULES:
    1. PRIMARY SOURCE is your MAIN reference for radiological interpretations
    2. Use FULL TEXT only for exact measurements, demographics, technical parameters if missing
    3. NEVER override primary source radiological context with full text
    4. EMPTY FIELDS ARE ACCEPTABLE - Better than guessed information
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
                "doc_type": doc_type,
                "fallback_date": fallback_date
            })
            
            long_summary = result.content.strip()
            
            logger.info("✅ Generated imaging long summary with DUAL-CONTEXT PRIORITY")
            logger.info("✅ Context priority maintained: PRIMARY source used for radiological findings")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct LLM generation failed: {str(e)}")
            return self._get_fallback_long_summary(fallback_date, doc_type)

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
        Generate a precise 30–60 word structured imaging summary in key-value format.
        Zero hallucinations. Pipe-delimited. Skips missing fields.
        """

        logger.info("🎯 Generating 30–60 word structured imaging summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a radiology-report summarization specialist.

    TASK:
    Produce a concise structured summary of an imaging report using ONLY details explicitly present in the long summary.
    - **ONLY include, critical, or clinically significant findings**.
    - **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**

    STRICT REQUIREMENTS:
    1. Word count MUST be **between 30 and 60 words**.
    2. Output format MUST be EXACTLY:

    [Report Title] | [Radiologist/Physician] | [Study Date] | Body Parts:[value] | Findings:[value] | Impression:[value] | Comparison:[value] | Physical Exam:[value] | Vital Signs:[value] | Critical Finding:[value] | Recommendations:[value]

    FORMAT & RULES:
    - MUST be **30–60 words**.
    - MUST be **ONE LINE**, pipe-delimited, no line breaks.
    - For Author never use "Dr." with it
    - NEVER include empty fields. If a field is missing, SKIP that key and remove its pipe.
    - NEVER fabricate: no invented dates, findings, or recommendations.
    - NO narrative sentences. Use short factual fragments ONLY.
    - First three fields (Report Title, Radiologist, Study Date) appear without keys
    - All other fields use key-value format: Key:[value]
    - Focus on radiology-specific elements: findings, impressions, comparisons

    CONTENT PRIORITY (only if provided in the long summary):
    1. Report Title  
    2. Radiologist  
    3. Study Date  
    4. Body parts studied  
    5. Key imaging findings  
    6. Radiologist's impression  
    7. Physical Exam (only abnormal findings if mentioned)  
    8. Vital Signs (only abnormal values if mentioned)  
    9. Comparison to prior studies  (if relevant)  
    10. Critical/urgent findings  
    11. Recommendations (only if explicitly stated)

    ABSOLUTELY FORBIDDEN:
    - assumptions, interpretations, or invented findings
    - For Author never use "Dr." with it
    - narrative writing
    - placeholder text or "Not provided"
    - duplicate pipes or empty pipe fields (e.g., "||")
    - Including non-radiology fields (medications, work status, etc.)

    Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    IMAGING REPORT LONG SUMMARY:

    {long_summary}

    Create a strict 30–60 word imaging summary using the required pipe-delimited format.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary})

            summary = response.content.strip()
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # Programmatically add missing Date or Author if LLM missed them
            summary = ensure_date_and_author(summary, long_summary)
            
            # No pipe cleaning - keep pipes as generated

            # Validate 30–60 word requirement
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Imaging summary word count out of range: {wc} words. Regenerating...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior output was {wc} words. Rewrite it to be between 30–60 words, preserving only factual content, keeping the exact key-value pipe format, and adding NO fabricated details. Maintain format: [Report Title] | [Radiologist] | [Study Date] | Body Parts:[value] | Findings:[value] | etc."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])

                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # Re-ensure date and author after correction
                summary = ensure_date_and_author(summary, long_summary)

            logger.info(f"✅ Imaging summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Imaging summary generation failed: {e}")
            return "Summary unavailable due to processing error."
  
    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract radiologist information
        radiologist_match = re.search(r'Radiologist:\s*([^\n]+)', long_summary)
        radiologist = radiologist_match.group(1).strip() if radiologist_match else "Radiologist"
        
        # Extract key information using regex patterns
        patterns = {
            'modality': r'Exam Type:\s*([^\n]+)',
            'body_part': r'Body Part Imaged:\s*([^\n]+)',
            'indication': r'Clinical Indication:\s*([^\n]+)',
            'findings': r'Primary Finding:(.*?)(?:\n\n|\n[A-Z]|$)',
            'impression': r'Overall Impression:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with modality and body part
        if 'modality' in extracted and 'body_part' in extracted:
            parts.append(f"{extracted['modality']} {extracted['body_part']}")
        elif 'modality' in extracted:
            parts.append(f"{extracted['modality']} study")
        
        # Add radiologist
        parts.append(f"by {radiologist}")
        
        # Add indication
        if 'indication' in extracted:
            parts.append(f"for {extracted['indication'][:60]}")
        
        # Add findings
        if 'findings' in extracted:
            # Take first line of findings
            first_finding = extracted['findings'].split('\n')[0].replace('•', '').replace('Description:', '').strip()[:80]
            if first_finding:
                parts.append(f"Findings: {first_finding}")
        
        # Add impression
        if 'impression' in extracted:
            parts.append(f"Impression: {extracted['impression'][:80]}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["comprehensive radiological evaluation", "with diagnostic interpretation", "and clinical implications"] 
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
                return str(value)
            if isinstance(value, list):
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

    def _get_fallback_result(self, fallback_date: str, doc_type: str) -> Dict:
        """Return fallback result structure matching 6-field imaging structure"""
        return {
            "field_1_header_context": {
                "imaging_center": "",
                "exam_date": fallback_date,
                "exam_type": doc_type,
                "patient_name": "",
                "patient_dob": "",
                "referring_physician": "",
                "radiologist": {
                    "name": "",
                    "credentials": "",
                    "specialty": "Radiology"
                }
            },
            "field_2_clinical_data": {
                "clinical_indication": "",
                "clinical_history": "",
                "specific_clinical_questions": "",
                "chief_complaint": ""
            },
            "field_3_technique_prior": {
                "study_type": doc_type,
                "body_part_imaged": "",
                "laterality": "",
                "contrast_used": "",
                "contrast_type": "",
                "prior_studies_available": "",
                "prior_study_dates": [],
                "technical_quality": "",
                "limitations": ""
            },
            "field_4_key_findings": {
                "primary_finding": {
                    "description": "",
                    "location": "",
                    "size": "",
                    "characteristics": "",
                    "acuity": "",
                    "significance": ""
                },
                "secondary_findings": [],
                "normal_findings": []
            },
            "field_5_impression_conclusion": {
                "overall_impression": "",
                "primary_diagnosis": "",
                "differential_diagnoses": [],
                "clinical_correlation_statement": "",
                "final_diagnostic_statement": ""
            },
            "field_6_recommendations_followup": {
                "follow_up_recommended": "",
                "follow_up_modality": "",
                "follow_up_timing": "",
                "clinical_correlation_needed": "",
                "specialist_consultation": ""
            }
        }

    def _get_fallback_long_summary(self, fallback_date: str, doc_type: str) -> str:
        """Return fallback long summary structure"""
        fallback_text = f"""
📋 IMAGING OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Exam Date: {fallback_date}
Exam Type: {doc_type}
Radiologist: Not specified
Imaging Center: Not specified
Referring Physician: Not specified

👤 PATIENT INFORMATION
--------------------------------------------------
Name: Not specified
Date of Birth: Not specified

🎯 CLINICAL INDICATION
--------------------------------------------------
Clinical Indication: Not specified
Clinical History: Not specified
Chief Complaint: Not specified
Specific Questions: Not specified

🔧 TECHNICAL DETAILS
--------------------------------------------------
Study Type: {doc_type}
Body Part Imaged: Not specified
Laterality: Not specified
Contrast Used: Not specified
Contrast Type: Not specified
Prior Studies Available: Not specified
Technical Quality: Not specified
Limitations: Not specified

📊 KEY FINDINGS
--------------------------------------------------
Primary Finding:
  • Description: Not specified
  • Location: Not specified
  • Size: Not specified
  • Characteristics: Not specified
  • Acuity: Not specified
Secondary Findings:
• None specified
Normal Findings:
• None specified

💡 IMPRESSION & CONCLUSION
--------------------------------------------------
Overall Impression: Not specified
Primary Diagnosis: Not specified
Final Diagnostic Statement: Not specified
Differential Diagnoses:
• None specified
Clinical Correlation: Not specified

📋 RECOMMENDATIONS & FOLLOW-UP
--------------------------------------------------
Follow-up Recommended: Not specified
Follow-up Modality: Not specified
Follow-up Timing: Not specified
Clinical Correlation Needed: Not specified
Specialist Consultation: Not specified
        """
        return fallback_text.strip()