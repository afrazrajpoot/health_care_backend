"""
PR-2 Progress Report Enhanced Extractor - Full Context
Optimized for accuracy using Gemini-style full-document processing
Version: 3.0 - With Source Attribution for Legal Compliance
"""
import logging
import re
import json
import time
from typing import Dict, Optional, List

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from utils.extraction_verifier import ExtractionVerifier
from utils.summary_helpers import clean_long_summary
from helpers.short_summary_generator import generate_structured_short_summary
from models.long_summary_models import (
    PR2LongSummary,
    DoctorInfo,
    format_pr2_long_summary,
    create_fallback_pr2_summary,
    # Source-attributed models for legal compliance
    SourcedPR2LongSummary,
    SourcedField,
    SourcedListItem,
    SourcedDoctorInfo,
    format_sourced_pr2_long_summary,
    flatten_sources_from_sourced_pr2,
    create_fallback_sourced_pr2_summary,
    convert_pr2_to_unsourced
)

logger = logging.getLogger("document_ai")


class PR2ExtractorChained:
    """
    Enhanced PR-2 extractor with FULL CONTEXT processing and SOURCE ATTRIBUTION.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Chain-of-thought reasoning for clinical progress tracking
    - Optimized for PR-2 specific clinical workflow patterns
    - Direct LLM generation for long summary (removes intermediate extraction)
    - SOURCE ATTRIBUTION: Every extracted field includes exact source text for legal compliance
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.pr2_summary_parser = PydanticOutputParser(pydantic_object=PR2LongSummary)
        self.pr2_sourced_parser = PydanticOutputParser(pydantic_object=SourcedPR2LongSummary)
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex patterns for PR-2 specific content
        self.progress_patterns = {
            'status': re.compile(r'\b(improved|stable|worsened|resolved|unchanged|progressing|regressing)\b', re.IGNORECASE),
            'work_status': re.compile(r'\b(ttd|modified duty|full duty|light duty|no restrictions|work restrictions)\b', re.IGNORECASE),
            'treatment': re.compile(r'\b(pt|physical therapy|injection|medication|therapy|exercise)\b', re.IGNORECASE)
        }
        
        logger.info("✅ PR2ExtractorChained v3.0 initialized with Source Attribution for Legal Compliance")

    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
        include_sources: bool = True
    ) -> Dict:
        """
        Extract PR-2 data with FULL CONTEXT using raw text.
        Returns dictionary with long_summary, short_summary, and optionally source attribution.
        
        Args:
            text: Complete document text (layout-preserved)
            raw_text: Summarized original context from Document AI
            doc_type: Document type (PR-2 Progress Report)
            fallback_date: Fallback date if not found
            include_sources: If True, includes source attribution for legal compliance (default: True)
            
        Returns:
            Dict containing:
                - long_summary: Formatted text summary
                - short_summary: UI-ready structured summary
                - sources: (if include_sources=True) Flat dict mapping field paths to source text
                - sourced_extraction: (if include_sources=True) Full structured data with sources
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING PR-2 EXTRACTION (FULL CONTEXT + RAW TEXT + SOURCE ATTRIBUTION)")
        logger.info("=" * 80)
        logger.info(f"   📌 Source attribution enabled: {include_sources}")
        
        start_time = time.time()
        
        try:
            # Check document size
            text_length = len(raw_text)
            token_estimate = text_length // 4
            logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
            
            if token_estimate > 120000:
                logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
                logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
            
            # Stage 1: Generate long summary with source attribution
            if include_sources:
                extraction_result = self._generate_sourced_long_summary(
                    text=text,
                    raw_text=raw_text,
                    doc_type=doc_type,
                    fallback_date=fallback_date
                )
                long_summary = extraction_result['long_summary']
                sourced_extraction = extraction_result['sourced_extraction']
                sources = extraction_result['sources']
            else:
                long_summary = self._generate_long_summary_direct(
                    text=text,
                    raw_text=raw_text,
                    doc_type=doc_type,
                    fallback_date=fallback_date
                )
                sourced_extraction = None
                sources = {}
            
            # Stage 1.5: Clean the long summary - remove empty fields, placeholders, and instruction text
            long_summary = clean_long_summary(long_summary)
            
            # Stage 2: Generate short summary from long summary (like QME extractor)
            short_summary = self._generate_short_summary_from_long_summary(raw_text, doc_type, long_summary)
            
            elapsed_time = time.time() - start_time
            logger.info(f"⚡ Full-context PR-2 extraction completed in {elapsed_time:.2f}s")
            logger.info(f"✅ Extracted progress data using:")
            logger.info(f"   - PRIMARY SOURCE: {len(raw_text):,} chars")
            logger.info(f"   - SUPPLEMENTARY SOURCE: {len(text):,} chars")
            if include_sources:
                logger.info(f"   - SOURCE ATTRIBUTIONS: {len(sources)} fields with sources")
            
            logger.info("=" * 80)
            logger.info("✅ PR-2 EXTRACTION COMPLETE (2 LLM CALLS ONLY)")
            logger.info("=" * 80)
            
            # Build result dictionary
            result = {
                "long_summary": long_summary,
                "short_summary": short_summary
            }
            
            # Add source attribution if enabled
            if include_sources:
                result["sources"] = sources
                result["sourced_extraction"] = sourced_extraction
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            # Return fallback result structure
            fallback_result = {
                "long_summary": f"PR-2 extraction failed: {str(e)}",
                "short_summary": "PR-2 summary not available"
            }
            if include_sources:
                fallback_result["sources"] = {}
                fallback_result["sourced_extraction"] = None
            return fallback_result

    def _generate_sourced_long_summary(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Generate long summary WITH SOURCE ATTRIBUTION for legal compliance.
        Every extracted field includes exact quoted text from the original document.
        
        Returns:
            Dict containing:
                - long_summary: Formatted text with source annotations
                - sourced_extraction: SourcedPR2LongSummary Pydantic model
                - sources: Flat dict mapping field paths to source text
        """
        logger.info("🔍 Processing PR-2 document with SOURCE ATTRIBUTION for legal compliance...")
        
        # Get format instructions from sourced Pydantic parser
        format_instructions = self.pr2_sourced_parser.get_format_instructions()
        
        # Build system prompt with SOURCE EXTRACTION instructions
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert Workers' Compensation medical documentation specialist analyzing a COMPLETE PR-2 Progress Report.

🔒 LEGAL COMPLIANCE MODE - SOURCE ATTRIBUTION REQUIRED

For EVERY piece of information you extract, you MUST:
1. Provide the extracted/summarized VALUE
2. Provide the EXACT SOURCE TEXT from the original document (in quotation marks)

This is for LEGAL COMPLIANCE and AUDIT PURPOSES. Every extraction must be traceable.

🎯 CRITICAL CONTEXT HIERARCHY:

You are provided with TWO versions of the document:

1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
   - This is the MOST ACCURATE, context-aware summary generated by Google's Document AI
   - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
   - Extract source quotes from this when available

2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
   - Complete OCR text extraction for finding exact source quotes
   - Use to fill in specific details and locate exact source text

🚨 SOURCE ATTRIBUTION RULES (MANDATORY):

1. **EVERY FIELD MUST HAVE A SOURCE**:
   - For each extracted value, include the EXACT text from the document
   - The source should be a direct quote, verbatim from the document
   - Keep source quotes concise but complete (capture the key phrase)
   - Maximum source length: ~100 characters (truncate with ... if longer)

2. **SOURCE FORMAT**:
   - For SourcedField: {{"value": "extracted summary", "source": "exact quoted text from document"}}
   - For SourcedListItem: {{"value": "item value", "source": "exact quoted text"}}
   - For SourcedDoctorInfo: {{"name": "Dr. Smith", "title": "MD", "role": "treating", "source": "Dr. John Smith, M.D., treating physician"}}

3. **EMPTY SOURCE HANDLING**:
   - If you extract a value but cannot find exact source text → source = ""
   - If field is not found in document → value = "", source = ""
   - NEVER fabricate source quotes

4. **HIGH-PRIORITY FIELDS FOR SOURCE ATTRIBUTION**:
   - Work Status (current_status, work_limitations) - CRITICAL for claims
   - Treatment Authorization Requests - Time-sensitive
   - Diagnoses - Medical accuracy required
   - Medications - Safety critical
   - Follow-up Plan - Affects claim timeline

🚨 ABSOLUTE ANTI-FABRICATION RULE:
- ONLY extract information that EXISTS in the document
- NEVER generate, infer, or fabricate ANY information or source quotes
- Empty fields with no source are better than fabricated ones

PRIMARY PURPOSE: Generate a comprehensive, legally-defensible structured summary for workers' compensation claims.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
📋 PRIMARY SOURCE - ACCURATE CONTEXT (from Document AI Summarizer):

{primary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 SUPPLEMENTARY SOURCE - FULL TEXT EXTRACTION (use for locating exact source quotes):

{supplementary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: For EVERY extracted field, provide BOTH the value AND the exact source text from the document.

📋 OUTPUT FORMAT INSTRUCTIONS:
{format_instructions}

Generate the complete PR-2 extraction with SOURCE ATTRIBUTION for every field.
Use fallback date {fallback_date} if no report date found in the document.
Document type: {doc_type}
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for SOURCED PR-2 extraction...")
            
            # Single LLM call with FULL document context
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "primary_source": raw_text,
                "supplementary_source": text,
                "fallback_date": fallback_date,
                "doc_type": doc_type,
                "format_instructions": format_instructions
            })
            
            raw_response = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"📝 Raw LLM response length: {len(raw_response)} chars in {processing_time:.2f}s")
            
            # Parse and validate with sourced Pydantic model
            try:
                sourced_summary = self.pr2_sourced_parser.parse(raw_response)
                logger.info(f"✅ Sourced Pydantic validation successful - content_type: {sourced_summary.content_type}")
                
                # Format the sourced summary to text format with source annotations
                long_summary = format_sourced_pr2_long_summary(sourced_summary)
                
                # Flatten sources for easy lookup
                sources = flatten_sources_from_sourced_pr2(sourced_summary)
                
                logger.info(f"✅ Source attribution complete: {len(sources)} fields with sources")
                
                return {
                    "long_summary": long_summary,
                    "sourced_extraction": sourced_summary.model_dump(),
                    "sources": sources
                }
                
            except Exception as parse_error:
                logger.warning(f"⚠️ Sourced Pydantic parsing failed: {parse_error}")
                logger.warning("⚠️ Attempting JSON extraction fallback...")
                
                # Try to extract JSON from the response
                try:
                    json_match = re.search(r'\{[\s\S]*\}', raw_response)
                    if json_match:
                        json_str = json_match.group()
                        parsed_dict = json.loads(json_str)
                        sourced_summary = SourcedPR2LongSummary(**parsed_dict)
                        long_summary = format_sourced_pr2_long_summary(sourced_summary)
                        sources = flatten_sources_from_sourced_pr2(sourced_summary)
                        logger.info("✅ Successfully recovered with JSON extraction fallback")
                        return {
                            "long_summary": long_summary,
                            "sourced_extraction": sourced_summary.model_dump(),
                            "sources": sources
                        }
                except Exception as json_error:
                    logger.warning(f"⚠️ JSON extraction fallback failed: {json_error}")
                
                # Final fallback: try unsourced extraction
                logger.warning("⚠️ Falling back to unsourced extraction")
                return self._fallback_to_unsourced(text, raw_text, doc_type, fallback_date)
            
        except Exception as e:
            logger.error(f"❌ Sourced PR-2 extraction failed: {e}", exc_info=True)
            return self._fallback_to_unsourced(text, raw_text, doc_type, fallback_date)
    
    def _fallback_to_unsourced(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """Fallback to unsourced extraction when sourced extraction fails."""
        logger.warning("🔄 Attempting unsourced extraction as fallback...")
        try:
            long_summary = self._generate_long_summary_direct(
                text=text,
                raw_text=raw_text,
                doc_type=doc_type,
                fallback_date=fallback_date
            )
            return {
                "long_summary": long_summary,
                "sourced_extraction": None,
                "sources": {}
            }
        except Exception as e:
            logger.error(f"❌ Unsourced fallback also failed: {e}")
            fallback = create_fallback_sourced_pr2_summary(doc_type, fallback_date)
            return {
                "long_summary": format_sourced_pr2_long_summary(fallback),
                "sourced_extraction": fallback.model_dump(),
                "sources": {}
            }

    def _generate_long_summary_direct(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Generate long summary with PRIORITIZED context hierarchy:
        1. PRIMARY SOURCE: raw_text (accurate Document AI summarized context)
        2. SUPPLEMENTARY: text (full OCR extraction for missing details only)
        
        This ensures accurate context preservation while capturing all necessary details.
        Uses Pydantic validation for consistent, structured output without hallucination.
        """
        logger.info("🔍 Processing PR-2 document with DUAL-CONTEXT approach + Pydantic validation...")
        
        # Get format instructions from Pydantic parser
        format_instructions = self.pr2_summary_parser.get_format_instructions()
        
        # Build system prompt with CLEAR PRIORITY INSTRUCTIONS
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert Workers' Compensation medical documentation specialist analyzing a COMPLETE PR-2 Progress Report.

🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

You are provided with TWO versions of the document:

1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
   - This is the MOST ACCURATE, context-aware summary generated by Google's Document AI foundation model
   - It has been intelligently processed to preserve CRITICAL PR-2 CLINICAL CONTEXT
   - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
   - This contains the CORRECT progress interpretations, accurate findings, and proper context
   - **ALWAYS PRIORITIZE information from this source**

2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
   - This is the complete OCR text extraction (may have formatting noise, OCR artifacts)
   - Use ONLY to fill in SPECIFIC DETAILS that may be missing from the accurate context
   - Examples of acceptable supplementary use:
       * Exact medication dosages if not in primary source
       * Specific claim numbers or identifiers
       * Additional doctor names mentioned
       * Precise dates or measurements
   - **DO NOT let this override the clinical context from the primary source**

🚨 ABSOLUTE ANTI-FABRICATION RULE (HIGHEST PRIORITY):
**YOU MUST ONLY EXTRACT AND SUMMARIZE INFORMATION THAT EXISTS IN THE PROVIDED SOURCES.**
- NEVER generate, infer, assume, or fabricate ANY information
- If information is NOT explicitly stated in either source → OMIT IT ENTIRELY
- An incomplete summary is 100x better than a fabricated one
- Every single piece of information in your output MUST be traceable to the source text

⚠️ STRICT ANTI-HALLUCINATION RULES:

1. **ZERO FABRICATION TOLERANCE**:
   - If a field (e.g., DOB, Claim Number, Medication) is NOT in either source → LEAVE IT BLANK or OMIT
   - NEVER write "likely", "probably", "typically", "usually" - these indicate fabrication
   - NEVER fill in "standard" or "typical" values - only actual extracted values

2. **CONTEXT PRIORITY ENFORCEMENT**:
   - When both sources provide information about the SAME clinical finding:
     ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
     ❌ NEVER override with potentially inaccurate full text version

3. **EXTRACT ONLY EXPLICITLY STATED INFORMATION** - Empty if not mentioned
3. **MEDICATIONS - ZERO TOLERANCE FOR ASSUMPTIONS** - Only explicitly listed current medications
4. **WORK RESTRICTIONS - EXACT WORDING ONLY** - Use exact phrases from primary source
5. **EMPTY FIELDS BETTER THAN GUESSES** - Omit if not found
6. **NO CLINICAL ASSUMPTIONS OR PREDICTIONS** - Only stated facts
7. **TREATING PHYSICIAN/AUTHOR** - Check both sources for signature/author

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

PRIMARY PURPOSE: Generate a comprehensive, structured long summary for workers' compensation claims administrators to:
1. Assess work capacity and disability status
2. Process treatment authorization requests
3. Track patient progress and treatment effectiveness
4. Plan future care and return-to-work timeline

PR-2 EXTRACTION FOCUS - 4 CORE WORKERS' COMPENSATION AREAS:

I. WORK STATUS AND IMPAIRMENT (HIGHEST PRIORITY FOR WC CLAIMS)
II. TREATMENT AUTHORIZATION REQUESTS (MOST TIME-SENSITIVE)
III. PATIENT PROGRESS AND CURRENT STATUS
IV. NEXT STEPS AND PLANNING

Now analyze this COMPLETE PR-2 Progress Report using the DUAL-CONTEXT approach and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY:
""")

        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
📋 PRIMARY SOURCE - ACCURATE CONTEXT (from Document AI Summarizer):

{primary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 SUPPLEMENTARY SOURCE - FULL TEXT EXTRACTION (use ONLY for specific missing details):

{supplementary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no report date found):

📋 REPORT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Report Date: [extracted or {fallback_date}]
Visit Date: [extracted]
Treating Physician: [name]
Specialty: [extracted]
Time Since Injury: [extracted]
Time Since Last Visit: [extracted]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit ; should not the business name or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.]


👤 PATIENT INFORMATION
--------------------------------------------------
Name: [extracted]
Date of Birth: [extracted]
Age: [extracted]
Date of Injury: [extracted]
Occupation: [extracted]
Employer: [extracted]
Claims Administrator: [extracted]

🎯 CHIEF COMPLAINT
--------------------------------------------------
Primary Complaint: [extracted]
Location: [extracted]
Description: [extracted]

━━━ CLAIM NUMBER EXTRACTION PATTERNS ━━━
CRITICAL: Scan the ENTIRE document mainly (header, footer, cc: lines, letterhead) for claim numbers.

Common claim number patterns (case-insensitive) and make sure to extract EXACTLY as written and must be claim number not just random numbers (like chart numbers, or id numbers) that look similar:
- if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
- "[Claim #XXXXXXXXX]" or "[Claim #XXXXX-XXX]"
- "Claim Number: XXXXXXXXX" or "Claim #: XXXXXXXXX"
- "Claim: XXXXXXXXX" or "Claim #XXXXXXXXX"
- "WC Claim: XXXXXXXXX" or "Workers Comp Claim: XXXXXXXXX"
- "Policy/Claim: XXXXXXXXX"
- In "cc:" lines: "Broadspire [Claim #XXXXXXXXX]"
- In subject lines or reference fields: "Claim #XXXXXXX"

All Doctors Involved:
• [list all extracted doctors with names and titles]
━━━ ALL DOCTORS EXTRACTION ━━━
- Extract ALL physician/doctor names mentioned ANYWHERE in the document into the "all_doctors" list.
- Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
- Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
- Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
- Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
- If no doctors found, leave list empty [].
                                                               


💬 SUBJECTIVE ASSESSMENT
--------------------------------------------------
Current Pain Score: [extracted]
Previous Pain Score: [extracted]
Symptom Changes: [extracted]
Functional Status (Patient Reported): [extracted]
Patient Compliance: [extracted]

🔬 OBJECTIVE FINDINGS
--------------------------------------------------
Physical Exam Findings: [extracted]
ROM Measurements: [extracted]
Strength Testing: [extracted]
Gait Assessment: [extracted]
Neurological Findings: [extracted]
Functional Limitations Observed:
• [list up to 5, exact wording]

🏥 DIAGNOSIS
--------------------------------------------------
Primary Diagnosis: [extracted with ICD-10 if available]
Secondary Diagnoses:
• [list up to 3]

💊 MEDICATIONS
--------------------------------------------------
Current Medications:
• [list up to 8 with doses if stated]
New Medications:
• [list up to 3]
Dosage Changes:
• [list up to 3]

📈 TREATMENT EFFECTIVENESS
--------------------------------------------------
Patient Response: [extracted]
Functional Gains: [extracted]
Objective Improvements:
• [list up to 5]
Barriers to Progress: [extracted]

✅ TREATMENT AUTHORIZATION REQUEST
--------------------------------------------------
Primary Request: [extracted]
Secondary Requests:
• [list up to 3]
Requested Frequency: [extracted]
Requested Duration: [extracted]
Medical Necessity Rationale: [extracted]

💼 WORK STATUS
--------------------------------------------------
Current Status: [extracted]
Status Effective Date: [extracted]
Work Limitations:
• [list up to 8, exact wording]
Work Status Rationale: [extracted]
Changes from Previous Status: [extracted]
Expected Return to Work Date: [extracted]

📅 FOLLOW-UP PLAN
--------------------------------------------------
Next Appointment Date: [extracted]
Purpose of Next Visit: [extracted]
Specialist Referrals Requested:
• [list up to 3]
MMI/P&S Anticipated Date: [extracted]
Return Sooner If: [extracted]

🚨 CRITICAL FINDINGS
--------------------------------------------------
• [list up to 5 most actionable items]

⚠️ MANDATORY EXTRACTION RULES (donot include in output, for LLM use only):
1. "work_limitations": Use EXACT wording (don't add weight/time limits not stated)
2. "treatment_authorization_request": The MOST CRITICAL field - be specific
3. "critical_findings": Only 3-5 most actionable items for claims administrator
4. Empty fields are acceptable if information not stated in document - use [extracted] as placeholder only if truly empty

📋 OUTPUT FORMAT INSTRUCTIONS:
You MUST output your response as a valid JSON object following this exact schema:
{format_instructions}

⚠️ IMPORTANT JSON RULES:
- Use empty string "" for text fields with no data (NOT null, NOT "N/A", NOT "unknown", NOT "[extracted]")
- Use empty array [] for list fields with no data
- Use null ONLY for optional fields like claim_number when not present
- For all_doctors_involved: each doctor must have "name" (required), "title" (optional), "role" (optional)
- For signature_type: use "physical" or "electronic" or null if not found
- For content_type: always use "pr2"
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for structured PR-2 long summary generation with Pydantic validation...")
            
            # Single LLM call with FULL document context to generate long summary directly
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "primary_source": raw_text,
                "supplementary_source": text,
                "fallback_date": fallback_date,
                "doc_type": doc_type,
                "format_instructions": format_instructions
            })
            
            raw_response = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"📝 Raw LLM response length: {len(raw_response)} chars in {processing_time:.2f}s")
            
            # Parse and validate with Pydantic
            try:
                parsed_summary = self.pr2_summary_parser.parse(raw_response)
                logger.info(f"✅ Pydantic validation successful - content_type: {parsed_summary.content_type}")
                
                # Format the validated Pydantic model back to text format
                long_summary = format_pr2_long_summary(parsed_summary)
                
                logger.info(f"⚡ Structured PR-2 long summary generation completed with Pydantic validation")
                logger.info(f"✅ Generated long summary from complete {len(text):,} char document")
                
                return long_summary
                
            except Exception as parse_error:
                logger.warning(f"⚠️ Pydantic parsing failed: {parse_error}")
                logger.warning("⚠️ Falling back to raw LLM output or JSON extraction")
                
                # Try to extract JSON from the response and create a fallback
                try:
                    # Attempt to find JSON in the response
                    json_match = re.search(r'\{[\s\S]*\}', raw_response)
                    if json_match:
                        json_str = json_match.group()
                        parsed_dict = json.loads(json_str)
                        # Create Pydantic model with defaults for missing fields
                        parsed_summary = PR2LongSummary(**parsed_dict)
                        long_summary = format_pr2_long_summary(parsed_summary)
                        logger.info("✅ Successfully recovered with JSON extraction fallback")
                        return long_summary
                except Exception as json_error:
                    logger.warning(f"⚠️ JSON extraction fallback failed: {json_error}")
                
                # Final fallback: use raw response if it looks like formatted text
                if "📋" in raw_response or "REPORT OVERVIEW" in raw_response:
                    logger.info("✅ Using raw formatted text response")
                    return raw_response
                
                # Create minimal fallback summary
                fallback = create_fallback_pr2_summary(doc_type, fallback_date)
                return format_pr2_long_summary(fallback)
            
        except Exception as e:
            logger.error(f"❌ Direct PR-2 long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            # Fallback: Generate a minimal summary using Pydantic model
            fallback = create_fallback_pr2_summary(doc_type, fallback_date)
            return format_pr2_long_summary(fallback)

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
    
    def _generate_short_summary_from_long_summary(self, raw_text: str, doc_type: str, long_summary: str) -> dict:
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
        return generate_structured_short_summary(self.llm, raw_text, doc_type, long_summary)
   

    def _create_comprehensive_fallback_summary(self, long_summary: str) -> str:
        """Create comprehensive fallback short summary directly from long summary"""
        
        # Extract physician information
        physician_match = re.search(r'Treating Physician:\s*([^\n]+)', long_summary)
        physician = physician_match.group(1).strip() if physician_match else "Treating Physician"
        
        # Extract key information using regex patterns
        patterns = {
            'work_status': r'Current Status:\s*([^\n]+)',
            'restrictions': r'Work Limitations:(.*?)(?:\n\n|\n[A-Z]|$)',
            'progress': r'Patient Response:\s*([^\n]+)',
            'requests': r'Primary Request:\s*([^\n]+)',
            'followup': r'Next Appointment Date:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with physician
        parts.append(f"{physician} progress report")
        
        # Add work status
        if 'work_status' in extracted:
            parts.append(f"Work status: {extracted['work_status'][:40]}")
        
        # Add restrictions
        if 'restrictions' in extracted:
            # Take first line of restrictions
            first_restrict = extracted['restrictions'].split('\n')[0].replace('•', '').strip()[:50]
            if first_restrict:
                parts.append(f"Restrictions: {first_restrict}")
        
        # Add progress
        if 'progress' in extracted:
            parts.append(f"Progress: {extracted['progress'][:60]}")
        
        # Add requests
        if 'requests' in extracted:
            parts.append(f"Request: {extracted['requests'][:60]}")
        
        # Add follow-up
        if 'followup' in extracted:
            parts.append(f"Follow-up: {extracted['followup']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            # Add padding to reach 60 words
            padding = ["with ongoing clinical management", "for workers' compensation case", "and functional improvement tracking"] 
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used fallback summary: {len(summary.split())} words")
        return summary
    

















    # End of parse_medical_doc.py



    """
Pydantic Models for Long Summary Generation
Ensures consistent, structured output without hallucination.
Now includes source attribution for legal compliance and verification.
"""
from typing import List, Optional, Literal, Any, Dict, Union
from pydantic import BaseModel, Field, field_validator


def normalize_claim_number(value: Any) -> Optional[str]:
    """Convert claim_number to string if it's a list/array."""
    if value is None:
        return None
    if isinstance(value, list):
        # Join multiple claim numbers with comma and space
        return ", ".join(str(v) for v in value if v)
    return str(value) if value else None


# ============================================================================
# SOURCE-ATTRIBUTED FIELD MODELS (For Legal Compliance)
# ============================================================================

class SourcedField(BaseModel):
    """
    A field with source attribution for legal compliance and verification.
    Contains both the extracted/summarized value and the exact source text from the original document.
    """
    value: str = Field(default="", description="The extracted or summarized value")
    source: str = Field(default="", description="Exact quoted text from the original document that supports this value. Empty if not found.")


class SourcedListItem(BaseModel):
    """
    A list item with source attribution.
    Used for lists like medications, diagnoses, limitations, etc.
    """
    value: str = Field(description="The extracted item value")
    source: str = Field(default="", description="Exact quoted text from the original document that supports this item")


class SourcedDoctorInfo(BaseModel):
    """Doctor information with source attribution."""
    name: str = Field(description="Full name of the doctor")
    title: str = Field(default="", description="Title or credentials (MD, DO, NP, etc.)")
    role: str = Field(default="", description="Role in treatment (treating, consulting, referring, etc.)")
    source: str = Field(default="", description="Exact quoted text mentioning this doctor")


# ============================================================================
# MEDICAL LONG SUMMARY MODELS
# ============================================================================

class MedicalDocumentOverview(BaseModel):
    """Overview section for medical documents"""
    document_type: str = Field(default="", description="Type of the medical document")
    report_date: str = Field(default="", description="Date of the report in MM/DD/YYYY format. Use '00/00/0000' if not found")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present, otherwise null. If multiple, join with comma.")
    patient_name: str = Field(default="", description="Full name of the patient")
    provider: str = Field(default="", description="Healthcare provider or facility name")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class DoctorInfo(BaseModel):
    """Information about a doctor mentioned in the document"""
    name: str = Field(description="Full name of the doctor")
    title: str = Field(default="", description="Title or credentials (MD, DO, NP, etc.)")
    role: str = Field(default="", description="Role in treatment (treating, consulting, referring, etc.)")


class PatientClinicalInformation(BaseModel):
    """Patient and clinical information section"""
    name: str = Field(default="", description="Patient's full name")
    dob: str = Field(default="", description="Date of birth")
    chief_complaint: str = Field(default="", description="Main reason for visit or chief complaint")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="List of all doctors mentioned in the document")


class DiagnosesAssessments(BaseModel):
    """Diagnoses and assessments section"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis from the document")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="List of secondary diagnoses (up to 3)")
    lab_results: List[str] = Field(default_factory=list, description="Key lab results with values/ranges (up to 5)")
    imaging_findings: List[str] = Field(default_factory=list, description="Key imaging observations")


class TreatmentObservations(BaseModel):
    """Treatment and observations section"""
    current_medications: List[str] = Field(default_factory=list, description="List of current medications with doses if stated")
    clinical_observations: List[str] = Field(default_factory=list, description="Vital signs, exam findings, clinical observations")
    procedures_treatments: List[str] = Field(default_factory=list, description="Recent or ongoing procedures/treatments")


class StatusRecommendations(BaseModel):
    """Status and recommendations section"""
    work_status: str = Field(default="", description="Current work status")
    mmi: str = Field(default="", description="Maximum Medical Improvement status")
    recommendations: List[str] = Field(default_factory=list, description="List of recommendations/next steps (up to 5)")


class MedicalLongSummary(BaseModel):
    """Complete structured medical long summary"""
    content_type: Literal["medical"] = Field(default="medical", description="Type of content")
    document_overview: MedicalDocumentOverview = Field(default_factory=MedicalDocumentOverview, description="Document overview section")
    patient_clinical_info: PatientClinicalInformation = Field(default_factory=PatientClinicalInformation, description="Patient and clinical information")
    critical_findings: List[str] = Field(default_factory=list, description="Up to 5 critical findings from the document")
    diagnoses_assessments: DiagnosesAssessments = Field(default_factory=DiagnosesAssessments, description="Diagnoses and assessments")
    treatment_observations: TreatmentObservations = Field(default_factory=TreatmentObservations, description="Treatment and observations")
    status_recommendations: StatusRecommendations = Field(default_factory=StatusRecommendations, description="Status and recommendations")


# ============================================================================
# ADMINISTRATIVE LONG SUMMARY MODELS
# ============================================================================

class AuthorInfo(BaseModel):
    """Author/signature information - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, utilization reviewers, or other officials mentioned in the document. Leave empty if no signature found.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature if present")


class AdministrativeDocumentOverview(BaseModel):
    """Overview section for administrative documents"""
    document_type: str = Field(default="", description="Type of the administrative document")
    document_date: str = Field(default="", description="Date of the document in MM/DD/YYYY format. Use '00/00/0000' if not found")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present, otherwise null. If multiple, join with comma.")
    purpose: str = Field(default="", description="Purpose of the document")
    author: AuthorInfo = Field(default_factory=AuthorInfo, description="Author/signature information")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class KeyParties(BaseModel):
    """Key parties involved in the document"""
    patient: str = Field(default="", description="Patient name")
    provider: str = Field(default="", description="Provider name/facility")
    referring_party: str = Field(default="", description="Referring party if applicable")


class KeyInformation(BaseModel):
    """Key information section for administrative documents"""
    important_dates: str = Field(default="", description="Important dates mentioned")
    reference_numbers: str = Field(default="", description="Reference numbers (claim numbers, case numbers, etc.)")
    administrative_details: str = Field(default="", description="Other administrative details")


class ActionItems(BaseModel):
    """Action items section"""
    required_actions: List[str] = Field(default_factory=list, description="List of required actions (up to 5)")
    deadlines: str = Field(default="", description="Any deadlines mentioned")


class ContactFollowUp(BaseModel):
    """Contact and follow-up section"""
    contact_information: str = Field(default="", description="Contact information if provided")
    next_steps: str = Field(default="", description="Next steps or follow-up actions")


class AdministrativeLongSummary(BaseModel):
    """Complete structured administrative long summary"""
    content_type: Literal["administrative"] = Field(default="administrative", description="Type of content")
    document_overview: AdministrativeDocumentOverview = Field(default_factory=AdministrativeDocumentOverview, description="Document overview section")
    key_parties: KeyParties = Field(default_factory=KeyParties, description="Key parties involved")
    key_information: KeyInformation = Field(default_factory=KeyInformation, description="Key information")
    action_items: ActionItems = Field(default_factory=ActionItems, description="Action items and deadlines")
    contact_followup: ContactFollowUp = Field(default_factory=ContactFollowUp, description="Contact and follow-up information")


# ============================================================================
# UNIVERSAL LONG SUMMARY MODEL (combines both types)
# ============================================================================

class UniversalLongSummary(BaseModel):
    """
    Universal long summary that can represent either medical or administrative content.
    Uses Optional fields to handle both types flexibly.
    """
    content_type: Literal["medical", "administrative", "unknown"] = Field(
        default="unknown", 
        description="Type of content: 'medical' for clinical documents, 'administrative' for non-clinical"
    )
    
    # === COMMON FIELDS (applicable to both types) ===
    document_type: str = Field(default="", description="Type of the document")
    document_date: str = Field(default="", description="Date of the document. Use '00/00/0000' if not found")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    patient_name: str = Field(default="", description="Patient's full name")
    
    # === MEDICAL-SPECIFIC FIELDS ===
    provider: str = Field(default="", description="Healthcare provider or facility name")
    dob: str = Field(default="", description="Patient's date of birth")
    chief_complaint: str = Field(default="", description="Main reason for visit")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 5)")
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 3)")
    lab_results: List[str] = Field(default_factory=list, description="Key lab results (up to 5)")
    imaging_findings: List[str] = Field(default_factory=list, description="Key imaging findings")
    current_medications: List[str] = Field(default_factory=list, description="Current medications with doses")
    clinical_observations: List[str] = Field(default_factory=list, description="Vital signs, exam findings")
    procedures_treatments: List[str] = Field(default_factory=list, description="Recent/ongoing procedures")
    work_status: str = Field(default="", description="Work status")
    mmi: str = Field(default="", description="Maximum Medical Improvement status")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations (up to 5)")
    
    # === ADMINISTRATIVE-SPECIFIC FIELDS ===
    purpose: str = Field(default="", description="Purpose of the document")
    author_signature: str = Field(default="", description="Name/title of the person who SIGNED the report. Must be the actual signer with physical or electronic signature - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")
    referring_party: str = Field(default="", description="Referring party")
    important_dates: str = Field(default="", description="Important dates mentioned")
    reference_numbers: str = Field(default="", description="Reference numbers")
    administrative_details: str = Field(default="", description="Administrative details")
    required_actions: List[str] = Field(default_factory=list, description="Required actions (up to 5)")
    deadlines: str = Field(default="", description="Deadlines mentioned")
    contact_information: str = Field(default="", description="Contact information")
    next_steps: str = Field(default="", description="Next steps")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


# ============================================================================
# HELPER FUNCTIONS FOR FORMATTING
# ============================================================================

def format_medical_long_summary(summary: UniversalLongSummary) -> str:
    """Format a medical long summary into the expected text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 MEDICAL DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_type:
        lines.append(f"Document Type: {summary.document_type}")
    if summary.document_date:
        lines.append(f"Report Date: {summary.document_date}")
    if summary.claim_number:
        lines.append(f"Claim Number: {summary.claim_number}")
    if summary.patient_name:
        lines.append(f"Patient Name: {summary.patient_name}")
    if summary.author_signature:
        lines.append(f"Author Signature: {summary.author_signature}")
    lines.append("")
    
    # Patient & Clinical Information
    lines.append("👤 PATIENT & CLINICAL INFORMATION")
    lines.append("-" * 50)
    if summary.patient_name:
        lines.append(f"Name: {summary.patient_name}")
    if summary.dob:
        lines.append(f"DOB: {summary.dob}")
    if summary.chief_complaint:
        lines.append(f"Chief Complaint: {summary.chief_complaint}")
    if summary.clinical_history:
        lines.append(f"Clinical History: {summary.clinical_history}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:5]:
            lines.append(f"• {finding}")
        lines.append("")
    
    # Diagnoses & Assessments
    lines.append("🏥 DIAGNOSES & ASSESSMENTS")
    lines.append("-" * 50)
    if summary.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.primary_diagnosis}")
    if summary.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.secondary_diagnoses[:3]:
            lines.append(f"• {dx}")
    if summary.lab_results:
        lines.append("Lab Results:")
        for result in summary.lab_results[:5]:
            lines.append(f"• {result}")
    if summary.imaging_findings:
        lines.append("Imaging Findings:")
        for finding in summary.imaging_findings:
            lines.append(f"• {finding}")
    lines.append("")
    
    # Treatment & Observations
    lines.append("💊 TREATMENT & OBSERVATIONS")
    lines.append("-" * 50)
    if summary.current_medications:
        lines.append("Current Medications:")
        for med in summary.current_medications:
            lines.append(f"• {med}")
    if summary.clinical_observations:
        lines.append("Clinical Observations:")
        for obs in summary.clinical_observations:
            lines.append(f"• {obs}")
    if summary.procedures_treatments:
        lines.append("Procedures/Treatments:")
        for proc in summary.procedures_treatments:
            lines.append(f"• {proc}")
    lines.append("")
    
    # Status & Recommendations
    lines.append("💼 STATUS & RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.work_status:
        lines.append(f"Work Status: {summary.work_status}")
    if summary.mmi:
        lines.append(f"MMI: {summary.mmi}")
    if summary.recommendations:
        lines.append("Recommendations:")
        for rec in summary.recommendations[:5]:
            lines.append(f"• {rec}")
    
    return "\n".join(lines)


def format_administrative_long_summary(summary: UniversalLongSummary) -> str:
    """Format an administrative long summary into the expected text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 ADMINISTRATIVE DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_type:
        lines.append(f"Document Type: {summary.document_type}")
    if summary.document_date:
        lines.append(f"Document Date: {summary.document_date}")
    if summary.claim_number:
        lines.append(f"Claim Number: {summary.claim_number}")
    if summary.purpose:
        lines.append(f"Purpose: {summary.purpose}")
    if summary.author_signature:
        sig_type = f" ({summary.signature_type})" if summary.signature_type else ""
        lines.append(f"Author:")
        lines.append(f"• Signature: {summary.author_signature}{sig_type}")
    lines.append("")
    
    # Key Parties
    lines.append("👥 KEY PARTIES")
    lines.append("-" * 50)
    if summary.patient_name:
        lines.append(f"Patient: {summary.patient_name}")
    if summary.provider:
        lines.append(f"Provider: {summary.provider}")
    if summary.referring_party:
        lines.append(f"Referring Party: {summary.referring_party}")
    lines.append("")
    
    # Key Information
    lines.append("📄 KEY INFORMATION")
    lines.append("-" * 50)
    if summary.important_dates:
        lines.append(f"Important Dates: {summary.important_dates}")
    if summary.reference_numbers:
        lines.append(f"Reference Numbers: {summary.reference_numbers}")
    if summary.administrative_details:
        lines.append(f"Administrative Details: {summary.administrative_details}")
    lines.append("")
    
    # Action Items
    lines.append("✅ ACTION ITEMS")
    lines.append("-" * 50)
    if summary.required_actions:
        lines.append("Required Actions:")
        for action in summary.required_actions[:5]:
            lines.append(f"• {action}")
    if summary.deadlines:
        lines.append(f"Deadlines: {summary.deadlines}")
    lines.append("")
    
    # Contact & Follow-Up
    lines.append("📞 CONTACT & FOLLOW-UP")
    lines.append("-" * 50)
    if summary.contact_information:
        lines.append(f"Contact Information: {summary.contact_information}")
    if summary.next_steps:
        lines.append(f"Next Steps: {summary.next_steps}")
    
    return "\n".join(lines)


def format_universal_long_summary(summary: UniversalLongSummary) -> str:
    """
    Format a universal long summary into text based on content type.
    """
    if summary.content_type == "medical":
        return format_medical_long_summary(summary)
    elif summary.content_type == "administrative":
        return format_administrative_long_summary(summary)
    else:
        # Default to medical format if unknown
        return format_medical_long_summary(summary)


def create_fallback_long_summary(doc_type: str, fallback_date: str) -> UniversalLongSummary:
    """Create a fallback long summary when extraction fails."""
    return UniversalLongSummary(
        content_type="unknown",
        document_type=doc_type,
        document_date=fallback_date,
        patient_name="",
        claim_number=None
    )


# ============================================================================
# PR-2 PROGRESS REPORT SPECIFIC MODELS
# ============================================================================

class PR2ReportOverview(BaseModel):
    """Report overview section for PR-2 Progress Reports"""
    document_type: str = Field(default="PR-2 Progress Report", description="Type of document")
    report_date: str = Field(default="", description="Date of the report. Use fallback date if not found")
    visit_date: str = Field(default="", description="Date of the patient visit")
    treating_physician: str = Field(default="", description="Name of the treating physician")
    specialty: str = Field(default="", description="Physician's specialty")
    time_since_injury: str = Field(default="", description="Time elapsed since injury")
    time_since_last_visit: str = Field(default="", description="Time since last visit")
    author_signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class PR2PatientInformation(BaseModel):
    """Patient information section for PR-2"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    age: str = Field(default="", description="Patient's age")
    date_of_injury: str = Field(default="", description="Date of injury")
    occupation: str = Field(default="", description="Patient's occupation")
    employer: str = Field(default="", description="Patient's employer")
    claims_administrator: str = Field(default="", description="Claims administrator name")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class PR2ChiefComplaint(BaseModel):
    """Chief complaint section for PR-2"""
    primary_complaint: str = Field(default="", description="Primary complaint")
    location: str = Field(default="", description="Location of complaint/pain")
    description: str = Field(default="", description="Description of complaint")


class PR2SubjectiveAssessment(BaseModel):
    """Subjective assessment section for PR-2"""
    current_pain_score: str = Field(default="", description="Current pain score (0-10)")
    previous_pain_score: str = Field(default="", description="Previous pain score")
    symptom_changes: str = Field(default="", description="Changes in symptoms")
    functional_status_patient_reported: str = Field(default="", description="Patient reported functional status")
    patient_compliance: str = Field(default="", description="Patient compliance with treatment")


class PR2ObjectiveFindings(BaseModel):
    """Objective findings section for PR-2"""
    physical_exam_findings: str = Field(default="", description="Physical examination findings")
    rom_measurements: str = Field(default="", description="Range of motion measurements")
    strength_testing: str = Field(default="", description="Strength testing results")
    gait_assessment: str = Field(default="", description="Gait assessment findings")
    neurological_findings: str = Field(default="", description="Neurological findings")
    functional_limitations_observed: List[str] = Field(default_factory=list, description="Observed functional limitations (up to 5)")


class PR2Diagnosis(BaseModel):
    """Diagnosis section for PR-2"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis with ICD-10 if available")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 3)")


class PR2Medications(BaseModel):
    """Medications section for PR-2"""
    current_medications: List[str] = Field(default_factory=list, description="Current medications with doses (up to 8)")
    new_medications: List[str] = Field(default_factory=list, description="Newly prescribed medications (up to 3)")
    dosage_changes: List[str] = Field(default_factory=list, description="Medication dosage changes (up to 3)")


class PR2TreatmentEffectiveness(BaseModel):
    """Treatment effectiveness section for PR-2"""
    patient_response: str = Field(default="", description="Patient's response to treatment")
    functional_gains: str = Field(default="", description="Functional gains achieved")
    objective_improvements: List[str] = Field(default_factory=list, description="Objective improvements noted (up to 5)")
    barriers_to_progress: str = Field(default="", description="Barriers to progress")


class PR2TreatmentAuthorizationRequest(BaseModel):
    """Treatment authorization request section for PR-2"""
    primary_request: str = Field(default="", description="Primary treatment authorization request")
    secondary_requests: List[str] = Field(default_factory=list, description="Secondary requests (up to 3)")
    requested_frequency: str = Field(default="", description="Requested treatment frequency")
    requested_duration: str = Field(default="", description="Requested treatment duration")
    medical_necessity_rationale: str = Field(default="", description="Medical necessity rationale")


class PR2WorkStatus(BaseModel):
    """Work status section for PR-2"""
    current_status: str = Field(default="", description="Current work status (TTD, Modified Duty, Full Duty, etc.)")
    status_effective_date: str = Field(default="", description="Date work status is effective")
    work_limitations: List[str] = Field(default_factory=list, description="Work limitations/restrictions (up to 8)")
    work_status_rationale: str = Field(default="", description="Rationale for work status")
    changes_from_previous_status: str = Field(default="", description="Changes from previous work status")
    expected_return_to_work_date: str = Field(default="", description="Expected return to work date")


class PR2FollowUpPlan(BaseModel):
    """Follow-up plan section for PR-2"""
    next_appointment_date: str = Field(default="", description="Date of next appointment")
    purpose_of_next_visit: str = Field(default="", description="Purpose of next visit")
    specialist_referrals_requested: List[str] = Field(default_factory=list, description="Specialist referrals requested (up to 3)")
    mmi_ps_anticipated_date: str = Field(default="", description="Anticipated MMI/P&S date")
    return_sooner_if: str = Field(default="", description="Conditions to return sooner")


class PR2LongSummary(BaseModel):
    """
    Complete structured PR-2 Progress Report long summary.
    Designed for Workers' Compensation claims processing.
    """
    content_type: Literal["pr2"] = Field(default="pr2", description="Content type for PR-2 documents")
    
    # Main sections
    report_overview: PR2ReportOverview = Field(default_factory=PR2ReportOverview, description="Report overview")
    patient_information: PR2PatientInformation = Field(default_factory=PR2PatientInformation, description="Patient information")
    chief_complaint: PR2ChiefComplaint = Field(default_factory=PR2ChiefComplaint, description="Chief complaint")
    subjective_assessment: PR2SubjectiveAssessment = Field(default_factory=PR2SubjectiveAssessment, description="Subjective assessment")
    objective_findings: PR2ObjectiveFindings = Field(default_factory=PR2ObjectiveFindings, description="Objective findings")
    diagnosis: PR2Diagnosis = Field(default_factory=PR2Diagnosis, description="Diagnosis")
    medications: PR2Medications = Field(default_factory=PR2Medications, description="Medications")
    treatment_effectiveness: PR2TreatmentEffectiveness = Field(default_factory=PR2TreatmentEffectiveness, description="Treatment effectiveness")
    treatment_authorization_request: PR2TreatmentAuthorizationRequest = Field(default_factory=PR2TreatmentAuthorizationRequest, description="Treatment authorization request")
    work_status: PR2WorkStatus = Field(default_factory=PR2WorkStatus, description="Work status")
    follow_up_plan: PR2FollowUpPlan = Field(default_factory=PR2FollowUpPlan, description="Follow-up plan")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 5)")


def format_pr2_long_summary(summary: PR2LongSummary) -> str:
    """Format a PR-2 long summary into the expected text format."""
    lines = []
    
    # Report Overview
    lines.append("📋 REPORT OVERVIEW")
    lines.append("-" * 50)
    if summary.report_overview.document_type:
        lines.append(f"Document Type: {summary.report_overview.document_type}")
    if summary.report_overview.report_date:
        lines.append(f"Report Date: {summary.report_overview.report_date}")
    if summary.report_overview.visit_date:
        lines.append(f"Visit Date: {summary.report_overview.visit_date}")
    if summary.report_overview.treating_physician:
        lines.append(f"Treating Physician: {summary.report_overview.treating_physician}")
    if summary.report_overview.specialty:
        lines.append(f"Specialty: {summary.report_overview.specialty}")
    if summary.report_overview.time_since_injury:
        lines.append(f"Time Since Injury: {summary.report_overview.time_since_injury}")
    if summary.report_overview.time_since_last_visit:
        lines.append(f"Time Since Last Visit: {summary.report_overview.time_since_last_visit}")
    if summary.report_overview.author_signature:
        sig_type = f" ({summary.report_overview.signature_type})" if summary.report_overview.signature_type else ""
        lines.append(f"Author:")
        lines.append(f"• Signature: {summary.report_overview.author_signature}{sig_type}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_information.name:
        lines.append(f"Name: {summary.patient_information.name}")
    if summary.patient_information.date_of_birth:
        lines.append(f"Date of Birth: {summary.patient_information.date_of_birth}")
    if summary.patient_information.age:
        lines.append(f"Age: {summary.patient_information.age}")
    if summary.patient_information.date_of_injury:
        lines.append(f"Date of Injury: {summary.patient_information.date_of_injury}")
    if summary.patient_information.occupation:
        lines.append(f"Occupation: {summary.patient_information.occupation}")
    if summary.patient_information.employer:
        lines.append(f"Employer: {summary.patient_information.employer}")
    if summary.patient_information.claims_administrator:
        lines.append(f"Claims Administrator: {summary.patient_information.claims_administrator}")
    if summary.patient_information.claim_number:
        lines.append(f"Claim Number: {summary.patient_information.claim_number}")
    if summary.patient_information.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.patient_information.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Chief Complaint
    lines.append("🎯 CHIEF COMPLAINT")
    lines.append("-" * 50)
    if summary.chief_complaint.primary_complaint:
        lines.append(f"Primary Complaint: {summary.chief_complaint.primary_complaint}")
    if summary.chief_complaint.location:
        lines.append(f"Location: {summary.chief_complaint.location}")
    if summary.chief_complaint.description:
        lines.append(f"Description: {summary.chief_complaint.description}")
    lines.append("")
    
    # Subjective Assessment
    lines.append("💬 SUBJECTIVE ASSESSMENT")
    lines.append("-" * 50)
    if summary.subjective_assessment.current_pain_score:
        lines.append(f"Current Pain Score: {summary.subjective_assessment.current_pain_score}")
    if summary.subjective_assessment.previous_pain_score:
        lines.append(f"Previous Pain Score: {summary.subjective_assessment.previous_pain_score}")
    if summary.subjective_assessment.symptom_changes:
        lines.append(f"Symptom Changes: {summary.subjective_assessment.symptom_changes}")
    if summary.subjective_assessment.functional_status_patient_reported:
        lines.append(f"Functional Status (Patient Reported): {summary.subjective_assessment.functional_status_patient_reported}")
    if summary.subjective_assessment.patient_compliance:
        lines.append(f"Patient Compliance: {summary.subjective_assessment.patient_compliance}")
    lines.append("")
    
    # Objective Findings
    lines.append("🔬 OBJECTIVE FINDINGS")
    lines.append("-" * 50)
    if summary.objective_findings.physical_exam_findings:
        lines.append(f"Physical Exam Findings: {summary.objective_findings.physical_exam_findings}")
    if summary.objective_findings.rom_measurements:
        lines.append(f"ROM Measurements: {summary.objective_findings.rom_measurements}")
    if summary.objective_findings.strength_testing:
        lines.append(f"Strength Testing: {summary.objective_findings.strength_testing}")
    if summary.objective_findings.gait_assessment:
        lines.append(f"Gait Assessment: {summary.objective_findings.gait_assessment}")
    if summary.objective_findings.neurological_findings:
        lines.append(f"Neurological Findings: {summary.objective_findings.neurological_findings}")
    if summary.objective_findings.functional_limitations_observed:
        lines.append("Functional Limitations Observed:")
        for limitation in summary.objective_findings.functional_limitations_observed[:5]:
            lines.append(f"• {limitation}")
    lines.append("")
    
    # Diagnosis
    lines.append("🏥 DIAGNOSIS")
    lines.append("-" * 50)
    if summary.diagnosis.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.diagnosis.primary_diagnosis}")
    if summary.diagnosis.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.diagnosis.secondary_diagnoses[:3]:
            lines.append(f"• {dx}")
    lines.append("")
    
    # Medications
    lines.append("💊 MEDICATIONS")
    lines.append("-" * 50)
    if summary.medications.current_medications:
        lines.append("Current Medications:")
        for med in summary.medications.current_medications[:8]:
            lines.append(f"• {med}")
    if summary.medications.new_medications:
        lines.append("New Medications:")
        for med in summary.medications.new_medications[:3]:
            lines.append(f"• {med}")
    if summary.medications.dosage_changes:
        lines.append("Dosage Changes:")
        for change in summary.medications.dosage_changes[:3]:
            lines.append(f"• {change}")
    lines.append("")
    
    # Treatment Effectiveness
    lines.append("📈 TREATMENT EFFECTIVENESS")
    lines.append("-" * 50)
    if summary.treatment_effectiveness.patient_response:
        lines.append(f"Patient Response: {summary.treatment_effectiveness.patient_response}")
    if summary.treatment_effectiveness.functional_gains:
        lines.append(f"Functional Gains: {summary.treatment_effectiveness.functional_gains}")
    if summary.treatment_effectiveness.objective_improvements:
        lines.append("Objective Improvements:")
        for improvement in summary.treatment_effectiveness.objective_improvements[:5]:
            lines.append(f"• {improvement}")
    if summary.treatment_effectiveness.barriers_to_progress:
        lines.append(f"Barriers to Progress: {summary.treatment_effectiveness.barriers_to_progress}")
    lines.append("")
    
    # Treatment Authorization Request
    lines.append("✅ TREATMENT AUTHORIZATION REQUEST")
    lines.append("-" * 50)
    if summary.treatment_authorization_request.primary_request:
        lines.append(f"Primary Request: {summary.treatment_authorization_request.primary_request}")
    if summary.treatment_authorization_request.secondary_requests:
        lines.append("Secondary Requests:")
        for req in summary.treatment_authorization_request.secondary_requests[:3]:
            lines.append(f"• {req}")
    if summary.treatment_authorization_request.requested_frequency:
        lines.append(f"Requested Frequency: {summary.treatment_authorization_request.requested_frequency}")
    if summary.treatment_authorization_request.requested_duration:
        lines.append(f"Requested Duration: {summary.treatment_authorization_request.requested_duration}")
    if summary.treatment_authorization_request.medical_necessity_rationale:
        lines.append(f"Medical Necessity Rationale: {summary.treatment_authorization_request.medical_necessity_rationale}")
    lines.append("")
    
    # Work Status
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    if summary.work_status.current_status:
        lines.append(f"Current Status: {summary.work_status.current_status}")
    if summary.work_status.status_effective_date:
        lines.append(f"Status Effective Date: {summary.work_status.status_effective_date}")
    if summary.work_status.work_limitations:
        lines.append("Work Limitations:")
        for limitation in summary.work_status.work_limitations[:8]:
            lines.append(f"• {limitation}")
    if summary.work_status.work_status_rationale:
        lines.append(f"Work Status Rationale: {summary.work_status.work_status_rationale}")
    if summary.work_status.changes_from_previous_status:
        lines.append(f"Changes from Previous Status: {summary.work_status.changes_from_previous_status}")
    if summary.work_status.expected_return_to_work_date:
        lines.append(f"Expected Return to Work Date: {summary.work_status.expected_return_to_work_date}")
    lines.append("")
    
    # Follow-Up Plan
    lines.append("📅 FOLLOW-UP PLAN")
    lines.append("-" * 50)
    if summary.follow_up_plan.next_appointment_date:
        lines.append(f"Next Appointment Date: {summary.follow_up_plan.next_appointment_date}")
    if summary.follow_up_plan.purpose_of_next_visit:
        lines.append(f"Purpose of Next Visit: {summary.follow_up_plan.purpose_of_next_visit}")
    if summary.follow_up_plan.specialist_referrals_requested:
        lines.append("Specialist Referrals Requested:")
        for referral in summary.follow_up_plan.specialist_referrals_requested[:3]:
            lines.append(f"• {referral}")
    if summary.follow_up_plan.mmi_ps_anticipated_date:
        lines.append(f"MMI/P&S Anticipated Date: {summary.follow_up_plan.mmi_ps_anticipated_date}")
    if summary.follow_up_plan.return_sooner_if:
        lines.append(f"Return Sooner If: {summary.follow_up_plan.return_sooner_if}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:5]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_pr2_summary(doc_type: str, fallback_date: str) -> PR2LongSummary:
    """Create a fallback PR-2 long summary when extraction fails."""
    return PR2LongSummary(
        content_type="pr2",
        report_overview=PR2ReportOverview(
            document_type=doc_type,
            report_date=fallback_date
        )
    )


# ============================================================================
# PR-2 SOURCE-ATTRIBUTED MODELS (For Legal Compliance)
# ============================================================================

class SourcedPR2ReportOverview(BaseModel):
    """Report overview section for PR-2 with source attribution"""
    document_type: SourcedField = Field(default_factory=SourcedField, description="Type of document with source")
    report_date: SourcedField = Field(default_factory=SourcedField, description="Date of the report with source")
    visit_date: SourcedField = Field(default_factory=SourcedField, description="Date of patient visit with source")
    treating_physician: SourcedField = Field(default_factory=SourcedField, description="Treating physician with source")
    specialty: SourcedField = Field(default_factory=SourcedField, description="Physician specialty with source")
    time_since_injury: SourcedField = Field(default_factory=SourcedField, description="Time since injury with source")
    time_since_last_visit: SourcedField = Field(default_factory=SourcedField, description="Time since last visit with source")
    author_signature: SourcedField = Field(default_factory=SourcedField, description="Author signature with source")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class SourcedPR2PatientInformation(BaseModel):
    """Patient information section for PR-2 with source attribution"""
    name: SourcedField = Field(default_factory=SourcedField, description="Patient name with source")
    date_of_birth: SourcedField = Field(default_factory=SourcedField, description="DOB with source")
    age: SourcedField = Field(default_factory=SourcedField, description="Age with source")
    date_of_injury: SourcedField = Field(default_factory=SourcedField, description="DOI with source")
    occupation: SourcedField = Field(default_factory=SourcedField, description="Occupation with source")
    employer: SourcedField = Field(default_factory=SourcedField, description="Employer with source")
    claims_administrator: SourcedField = Field(default_factory=SourcedField, description="Claims admin with source")
    claim_number: SourcedField = Field(default_factory=SourcedField, description="Claim number with source")
    all_doctors_involved: List[SourcedDoctorInfo] = Field(default_factory=list, description="All doctors with sources")


class SourcedPR2ChiefComplaint(BaseModel):
    """Chief complaint section for PR-2 with source attribution"""
    primary_complaint: SourcedField = Field(default_factory=SourcedField, description="Primary complaint with source")
    location: SourcedField = Field(default_factory=SourcedField, description="Location with source")
    description: SourcedField = Field(default_factory=SourcedField, description="Description with source")


class SourcedPR2SubjectiveAssessment(BaseModel):
    """Subjective assessment section for PR-2 with source attribution"""
    current_pain_score: SourcedField = Field(default_factory=SourcedField, description="Current pain score with source")
    previous_pain_score: SourcedField = Field(default_factory=SourcedField, description="Previous pain score with source")
    symptom_changes: SourcedField = Field(default_factory=SourcedField, description="Symptom changes with source")
    functional_status_patient_reported: SourcedField = Field(default_factory=SourcedField, description="Patient-reported functional status with source")
    patient_compliance: SourcedField = Field(default_factory=SourcedField, description="Patient compliance with source")


class SourcedPR2ObjectiveFindings(BaseModel):
    """Objective findings section for PR-2 with source attribution"""
    physical_exam_findings: SourcedField = Field(default_factory=SourcedField, description="Physical exam findings with source")
    rom_measurements: SourcedField = Field(default_factory=SourcedField, description="ROM measurements with source")
    strength_testing: SourcedField = Field(default_factory=SourcedField, description="Strength testing with source")
    gait_assessment: SourcedField = Field(default_factory=SourcedField, description="Gait assessment with source")
    neurological_findings: SourcedField = Field(default_factory=SourcedField, description="Neurological findings with source")
    functional_limitations_observed: List[SourcedListItem] = Field(default_factory=list, description="Observed functional limitations with sources")


class SourcedPR2Diagnosis(BaseModel):
    """Diagnosis section for PR-2 with source attribution"""
    primary_diagnosis: SourcedField = Field(default_factory=SourcedField, description="Primary diagnosis with source")
    secondary_diagnoses: List[SourcedListItem] = Field(default_factory=list, description="Secondary diagnoses with sources")


class SourcedPR2Medications(BaseModel):
    """Medications section for PR-2 with source attribution"""
    current_medications: List[SourcedListItem] = Field(default_factory=list, description="Current medications with sources")
    new_medications: List[SourcedListItem] = Field(default_factory=list, description="New medications with sources")
    dosage_changes: List[SourcedListItem] = Field(default_factory=list, description="Dosage changes with sources")


class SourcedPR2TreatmentEffectiveness(BaseModel):
    """Treatment effectiveness section for PR-2 with source attribution"""
    patient_response: SourcedField = Field(default_factory=SourcedField, description="Patient response with source")
    functional_gains: SourcedField = Field(default_factory=SourcedField, description="Functional gains with source")
    objective_improvements: List[SourcedListItem] = Field(default_factory=list, description="Objective improvements with sources")
    barriers_to_progress: SourcedField = Field(default_factory=SourcedField, description="Barriers to progress with source")


class SourcedPR2TreatmentAuthorizationRequest(BaseModel):
    """Treatment authorization request section for PR-2 with source attribution"""
    primary_request: SourcedField = Field(default_factory=SourcedField, description="Primary request with source")
    secondary_requests: List[SourcedListItem] = Field(default_factory=list, description="Secondary requests with sources")
    requested_frequency: SourcedField = Field(default_factory=SourcedField, description="Requested frequency with source")
    requested_duration: SourcedField = Field(default_factory=SourcedField, description="Requested duration with source")
    medical_necessity_rationale: SourcedField = Field(default_factory=SourcedField, description="Medical necessity rationale with source")


class SourcedPR2WorkStatus(BaseModel):
    """Work status section for PR-2 with source attribution - CRITICAL FOR WC CLAIMS"""
    current_status: SourcedField = Field(default_factory=SourcedField, description="Current work status with source")
    status_effective_date: SourcedField = Field(default_factory=SourcedField, description="Status effective date with source")
    work_limitations: List[SourcedListItem] = Field(default_factory=list, description="Work limitations with sources - CRITICAL")
    work_status_rationale: SourcedField = Field(default_factory=SourcedField, description="Work status rationale with source")
    changes_from_previous_status: SourcedField = Field(default_factory=SourcedField, description="Changes from previous with source")
    expected_return_to_work_date: SourcedField = Field(default_factory=SourcedField, description="Expected RTW date with source")


class SourcedPR2FollowUpPlan(BaseModel):
    """Follow-up plan section for PR-2 with source attribution"""
    next_appointment_date: SourcedField = Field(default_factory=SourcedField, description="Next appointment with source")
    purpose_of_next_visit: SourcedField = Field(default_factory=SourcedField, description="Purpose of next visit with source")
    specialist_referrals_requested: List[SourcedListItem] = Field(default_factory=list, description="Specialist referrals with sources")
    mmi_ps_anticipated_date: SourcedField = Field(default_factory=SourcedField, description="MMI/P&S date with source")
    return_sooner_if: SourcedField = Field(default_factory=SourcedField, description="Return sooner conditions with source")


class SourcedPR2LongSummary(BaseModel):
    """
    Complete structured PR-2 Progress Report long summary WITH SOURCE ATTRIBUTION.
    Designed for Workers' Compensation claims processing with legal compliance.
    Every extracted field includes the exact source text from the original document.
    """
    content_type: Literal["pr2_sourced"] = Field(default="pr2_sourced", description="Content type for sourced PR-2 documents")
    
    # Main sections with source attribution
    report_overview: SourcedPR2ReportOverview = Field(default_factory=SourcedPR2ReportOverview, description="Report overview with sources")
    patient_information: SourcedPR2PatientInformation = Field(default_factory=SourcedPR2PatientInformation, description="Patient information with sources")
    chief_complaint: SourcedPR2ChiefComplaint = Field(default_factory=SourcedPR2ChiefComplaint, description="Chief complaint with sources")
    subjective_assessment: SourcedPR2SubjectiveAssessment = Field(default_factory=SourcedPR2SubjectiveAssessment, description="Subjective assessment with sources")
    objective_findings: SourcedPR2ObjectiveFindings = Field(default_factory=SourcedPR2ObjectiveFindings, description="Objective findings with sources")
    diagnosis: SourcedPR2Diagnosis = Field(default_factory=SourcedPR2Diagnosis, description="Diagnosis with sources")
    medications: SourcedPR2Medications = Field(default_factory=SourcedPR2Medications, description="Medications with sources")
    treatment_effectiveness: SourcedPR2TreatmentEffectiveness = Field(default_factory=SourcedPR2TreatmentEffectiveness, description="Treatment effectiveness with sources")
    treatment_authorization_request: SourcedPR2TreatmentAuthorizationRequest = Field(default_factory=SourcedPR2TreatmentAuthorizationRequest, description="Treatment auth request with sources")
    work_status: SourcedPR2WorkStatus = Field(default_factory=SourcedPR2WorkStatus, description="Work status with sources - CRITICAL")
    follow_up_plan: SourcedPR2FollowUpPlan = Field(default_factory=SourcedPR2FollowUpPlan, description="Follow-up plan with sources")
    critical_findings: List[SourcedListItem] = Field(default_factory=list, description="Critical findings with sources")


def format_sourced_pr2_long_summary(summary: SourcedPR2LongSummary) -> str:
    """
    Format a sourced PR-2 long summary into the expected text format.
    Includes source attributions in a verifiable format.
    """
    lines = []
    
    def format_sourced_field(label: str, field: SourcedField) -> List[str]:
        """Format a sourced field with its source attribution."""
        result = []
        if field.value:
            result.append(f"{label}: {field.value}")
            if field.source:
                result.append(f"  📍 Source: \"{field.source}\"")
        return result
    
    def format_sourced_list(items: List[SourcedListItem]) -> List[str]:
        """Format a list of sourced items."""
        result = []
        for item in items:
            if item.value:
                result.append(f"• {item.value}")
                if item.source:
                    result.append(f"  📍 Source: \"{item.source}\"")
        return result
    
    # Report Overview
    lines.append("📋 REPORT OVERVIEW")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Document Type", summary.report_overview.document_type))
    lines.extend(format_sourced_field("Report Date", summary.report_overview.report_date))
    lines.extend(format_sourced_field("Visit Date", summary.report_overview.visit_date))
    lines.extend(format_sourced_field("Treating Physician", summary.report_overview.treating_physician))
    lines.extend(format_sourced_field("Specialty", summary.report_overview.specialty))
    lines.extend(format_sourced_field("Time Since Injury", summary.report_overview.time_since_injury))
    lines.extend(format_sourced_field("Time Since Last Visit", summary.report_overview.time_since_last_visit))
    if summary.report_overview.author_signature.value:
        sig_type = f" ({summary.report_overview.signature_type})" if summary.report_overview.signature_type else ""
        lines.append(f"Author:")
        lines.append(f"• Signature: {summary.report_overview.author_signature.value}{sig_type}")
        if summary.report_overview.author_signature.source:
            lines.append(f"  📍 Source: \"{summary.report_overview.author_signature.source}\"")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Name", summary.patient_information.name))
    lines.extend(format_sourced_field("Date of Birth", summary.patient_information.date_of_birth))
    lines.extend(format_sourced_field("Age", summary.patient_information.age))
    lines.extend(format_sourced_field("Date of Injury", summary.patient_information.date_of_injury))
    lines.extend(format_sourced_field("Occupation", summary.patient_information.occupation))
    lines.extend(format_sourced_field("Employer", summary.patient_information.employer))
    lines.extend(format_sourced_field("Claims Administrator", summary.patient_information.claims_administrator))
    lines.extend(format_sourced_field("Claim Number", summary.patient_information.claim_number))
    if summary.patient_information.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.patient_information.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
            if doctor.source:
                lines.append(f"  📍 Source: \"{doctor.source}\"")
    lines.append("")
    
    # Chief Complaint
    lines.append("🎯 CHIEF COMPLAINT")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Primary Complaint", summary.chief_complaint.primary_complaint))
    lines.extend(format_sourced_field("Location", summary.chief_complaint.location))
    lines.extend(format_sourced_field("Description", summary.chief_complaint.description))
    lines.append("")
    
    # Subjective Assessment
    lines.append("💬 SUBJECTIVE ASSESSMENT")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Current Pain Score", summary.subjective_assessment.current_pain_score))
    lines.extend(format_sourced_field("Previous Pain Score", summary.subjective_assessment.previous_pain_score))
    lines.extend(format_sourced_field("Symptom Changes", summary.subjective_assessment.symptom_changes))
    lines.extend(format_sourced_field("Functional Status (Patient Reported)", summary.subjective_assessment.functional_status_patient_reported))
    lines.extend(format_sourced_field("Patient Compliance", summary.subjective_assessment.patient_compliance))
    lines.append("")
    
    # Objective Findings
    lines.append("🔬 OBJECTIVE FINDINGS")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Physical Exam Findings", summary.objective_findings.physical_exam_findings))
    lines.extend(format_sourced_field("ROM Measurements", summary.objective_findings.rom_measurements))
    lines.extend(format_sourced_field("Strength Testing", summary.objective_findings.strength_testing))
    lines.extend(format_sourced_field("Gait Assessment", summary.objective_findings.gait_assessment))
    lines.extend(format_sourced_field("Neurological Findings", summary.objective_findings.neurological_findings))
    if summary.objective_findings.functional_limitations_observed:
        lines.append("Functional Limitations Observed:")
        lines.extend(format_sourced_list(summary.objective_findings.functional_limitations_observed[:5]))
    lines.append("")
    
    # Diagnosis
    lines.append("🏥 DIAGNOSIS")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Primary Diagnosis", summary.diagnosis.primary_diagnosis))
    if summary.diagnosis.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        lines.extend(format_sourced_list(summary.diagnosis.secondary_diagnoses[:3]))
    lines.append("")
    
    # Medications
    lines.append("💊 MEDICATIONS")
    lines.append("-" * 50)
    if summary.medications.current_medications:
        lines.append("Current Medications:")
        lines.extend(format_sourced_list(summary.medications.current_medications[:8]))
    if summary.medications.new_medications:
        lines.append("New Medications:")
        lines.extend(format_sourced_list(summary.medications.new_medications[:3]))
    if summary.medications.dosage_changes:
        lines.append("Dosage Changes:")
        lines.extend(format_sourced_list(summary.medications.dosage_changes[:3]))
    lines.append("")
    
    # Treatment Effectiveness
    lines.append("📈 TREATMENT EFFECTIVENESS")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Patient Response", summary.treatment_effectiveness.patient_response))
    lines.extend(format_sourced_field("Functional Gains", summary.treatment_effectiveness.functional_gains))
    if summary.treatment_effectiveness.objective_improvements:
        lines.append("Objective Improvements:")
        lines.extend(format_sourced_list(summary.treatment_effectiveness.objective_improvements[:5]))
    lines.extend(format_sourced_field("Barriers to Progress", summary.treatment_effectiveness.barriers_to_progress))
    lines.append("")
    
    # Treatment Authorization Request
    lines.append("✅ TREATMENT AUTHORIZATION REQUEST")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Primary Request", summary.treatment_authorization_request.primary_request))
    if summary.treatment_authorization_request.secondary_requests:
        lines.append("Secondary Requests:")
        lines.extend(format_sourced_list(summary.treatment_authorization_request.secondary_requests[:3]))
    lines.extend(format_sourced_field("Requested Frequency", summary.treatment_authorization_request.requested_frequency))
    lines.extend(format_sourced_field("Requested Duration", summary.treatment_authorization_request.requested_duration))
    lines.extend(format_sourced_field("Medical Necessity Rationale", summary.treatment_authorization_request.medical_necessity_rationale))
    lines.append("")
    
    # Work Status - CRITICAL SECTION
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Current Status", summary.work_status.current_status))
    lines.extend(format_sourced_field("Status Effective Date", summary.work_status.status_effective_date))
    if summary.work_status.work_limitations:
        lines.append("Work Limitations:")
        lines.extend(format_sourced_list(summary.work_status.work_limitations[:8]))
    lines.extend(format_sourced_field("Work Status Rationale", summary.work_status.work_status_rationale))
    lines.extend(format_sourced_field("Changes from Previous Status", summary.work_status.changes_from_previous_status))
    lines.extend(format_sourced_field("Expected Return to Work Date", summary.work_status.expected_return_to_work_date))
    lines.append("")
    
    # Follow-Up Plan
    lines.append("📅 FOLLOW-UP PLAN")
    lines.append("-" * 50)
    lines.extend(format_sourced_field("Next Appointment Date", summary.follow_up_plan.next_appointment_date))
    lines.extend(format_sourced_field("Purpose of Next Visit", summary.follow_up_plan.purpose_of_next_visit))
    if summary.follow_up_plan.specialist_referrals_requested:
        lines.append("Specialist Referrals Requested:")
        lines.extend(format_sourced_list(summary.follow_up_plan.specialist_referrals_requested[:3]))
    lines.extend(format_sourced_field("MMI/P&S Anticipated Date", summary.follow_up_plan.mmi_ps_anticipated_date))
    lines.extend(format_sourced_field("Return Sooner If", summary.follow_up_plan.return_sooner_if))
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        lines.extend(format_sourced_list(summary.critical_findings[:5]))
    
    return "\n".join(lines)


def flatten_sources_from_sourced_pr2(summary: SourcedPR2LongSummary) -> Dict[str, str]:
    """
    Create a flat dictionary mapping field paths to their source texts.
    Useful for quick lookup and verification.
    
    Returns:
        Dict mapping field paths (e.g., "work_status.current_status") to source text
    """
    sources = {}
    
    def add_sourced_field(path: str, field: SourcedField):
        if field.source:
            sources[path] = field.source
    
    def add_sourced_list(path: str, items: List[SourcedListItem]):
        for i, item in enumerate(items):
            if item.source:
                sources[f"{path}[{i}]"] = item.source
    
    # Report Overview
    add_sourced_field("report_overview.document_type", summary.report_overview.document_type)
    add_sourced_field("report_overview.report_date", summary.report_overview.report_date)
    add_sourced_field("report_overview.visit_date", summary.report_overview.visit_date)
    add_sourced_field("report_overview.treating_physician", summary.report_overview.treating_physician)
    add_sourced_field("report_overview.specialty", summary.report_overview.specialty)
    add_sourced_field("report_overview.time_since_injury", summary.report_overview.time_since_injury)
    add_sourced_field("report_overview.time_since_last_visit", summary.report_overview.time_since_last_visit)
    add_sourced_field("report_overview.author_signature", summary.report_overview.author_signature)
    
    # Patient Information
    add_sourced_field("patient_information.name", summary.patient_information.name)
    add_sourced_field("patient_information.date_of_birth", summary.patient_information.date_of_birth)
    add_sourced_field("patient_information.age", summary.patient_information.age)
    add_sourced_field("patient_information.date_of_injury", summary.patient_information.date_of_injury)
    add_sourced_field("patient_information.occupation", summary.patient_information.occupation)
    add_sourced_field("patient_information.employer", summary.patient_information.employer)
    add_sourced_field("patient_information.claims_administrator", summary.patient_information.claims_administrator)
    add_sourced_field("patient_information.claim_number", summary.patient_information.claim_number)
    for i, doctor in enumerate(summary.patient_information.all_doctors_involved):
        if doctor.source:
            sources[f"patient_information.all_doctors_involved[{i}]"] = doctor.source
    
    # Chief Complaint
    add_sourced_field("chief_complaint.primary_complaint", summary.chief_complaint.primary_complaint)
    add_sourced_field("chief_complaint.location", summary.chief_complaint.location)
    add_sourced_field("chief_complaint.description", summary.chief_complaint.description)
    
    # Subjective Assessment
    add_sourced_field("subjective_assessment.current_pain_score", summary.subjective_assessment.current_pain_score)
    add_sourced_field("subjective_assessment.previous_pain_score", summary.subjective_assessment.previous_pain_score)
    add_sourced_field("subjective_assessment.symptom_changes", summary.subjective_assessment.symptom_changes)
    add_sourced_field("subjective_assessment.functional_status_patient_reported", summary.subjective_assessment.functional_status_patient_reported)
    add_sourced_field("subjective_assessment.patient_compliance", summary.subjective_assessment.patient_compliance)
    
    # Objective Findings
    add_sourced_field("objective_findings.physical_exam_findings", summary.objective_findings.physical_exam_findings)
    add_sourced_field("objective_findings.rom_measurements", summary.objective_findings.rom_measurements)
    add_sourced_field("objective_findings.strength_testing", summary.objective_findings.strength_testing)
    add_sourced_field("objective_findings.gait_assessment", summary.objective_findings.gait_assessment)
    add_sourced_field("objective_findings.neurological_findings", summary.objective_findings.neurological_findings)
    add_sourced_list("objective_findings.functional_limitations_observed", summary.objective_findings.functional_limitations_observed)
    
    # Diagnosis
    add_sourced_field("diagnosis.primary_diagnosis", summary.diagnosis.primary_diagnosis)
    add_sourced_list("diagnosis.secondary_diagnoses", summary.diagnosis.secondary_diagnoses)
    
    # Medications
    add_sourced_list("medications.current_medications", summary.medications.current_medications)
    add_sourced_list("medications.new_medications", summary.medications.new_medications)
    add_sourced_list("medications.dosage_changes", summary.medications.dosage_changes)
    
    # Treatment Effectiveness
    add_sourced_field("treatment_effectiveness.patient_response", summary.treatment_effectiveness.patient_response)
    add_sourced_field("treatment_effectiveness.functional_gains", summary.treatment_effectiveness.functional_gains)
    add_sourced_list("treatment_effectiveness.objective_improvements", summary.treatment_effectiveness.objective_improvements)
    add_sourced_field("treatment_effectiveness.barriers_to_progress", summary.treatment_effectiveness.barriers_to_progress)
    
    # Treatment Authorization Request
    add_sourced_field("treatment_authorization_request.primary_request", summary.treatment_authorization_request.primary_request)
    add_sourced_list("treatment_authorization_request.secondary_requests", summary.treatment_authorization_request.secondary_requests)
    add_sourced_field("treatment_authorization_request.requested_frequency", summary.treatment_authorization_request.requested_frequency)
    add_sourced_field("treatment_authorization_request.requested_duration", summary.treatment_authorization_request.requested_duration)
    add_sourced_field("treatment_authorization_request.medical_necessity_rationale", summary.treatment_authorization_request.medical_necessity_rationale)
    
    # Work Status - CRITICAL
    add_sourced_field("work_status.current_status", summary.work_status.current_status)
    add_sourced_field("work_status.status_effective_date", summary.work_status.status_effective_date)
    add_sourced_list("work_status.work_limitations", summary.work_status.work_limitations)
    add_sourced_field("work_status.work_status_rationale", summary.work_status.work_status_rationale)
    add_sourced_field("work_status.changes_from_previous_status", summary.work_status.changes_from_previous_status)
    add_sourced_field("work_status.expected_return_to_work_date", summary.work_status.expected_return_to_work_date)
    
    # Follow-Up Plan
    add_sourced_field("follow_up_plan.next_appointment_date", summary.follow_up_plan.next_appointment_date)
    add_sourced_field("follow_up_plan.purpose_of_next_visit", summary.follow_up_plan.purpose_of_next_visit)
    add_sourced_list("follow_up_plan.specialist_referrals_requested", summary.follow_up_plan.specialist_referrals_requested)
    add_sourced_field("follow_up_plan.mmi_ps_anticipated_date", summary.follow_up_plan.mmi_ps_anticipated_date)
    add_sourced_field("follow_up_plan.return_sooner_if", summary.follow_up_plan.return_sooner_if)
    
    # Critical Findings
    add_sourced_list("critical_findings", summary.critical_findings)
    
    return sources


def create_fallback_sourced_pr2_summary(doc_type: str, fallback_date: str) -> SourcedPR2LongSummary:
    """Create a fallback sourced PR-2 long summary when extraction fails."""
    return SourcedPR2LongSummary(
        content_type="pr2_sourced",
        report_overview=SourcedPR2ReportOverview(
            document_type=SourcedField(value=doc_type, source=""),
            report_date=SourcedField(value=fallback_date, source="")
        )
    )


def convert_pr2_to_unsourced(sourced_summary: SourcedPR2LongSummary) -> PR2LongSummary:
    """
    Convert a sourced PR2 summary to the original unsourced format.
    Useful for backward compatibility with existing code.
    """
    def extract_value(field: SourcedField) -> str:
        return field.value if field else ""
    
    def extract_list_values(items: List[SourcedListItem]) -> List[str]:
        return [item.value for item in items if item.value]
    
    def convert_doctors(sourced_doctors: List[SourcedDoctorInfo]) -> List[DoctorInfo]:
        return [
            DoctorInfo(name=d.name, title=d.title, role=d.role)
            for d in sourced_doctors
        ]
    
    return PR2LongSummary(
        content_type="pr2",
        report_overview=PR2ReportOverview(
            document_type=extract_value(sourced_summary.report_overview.document_type),
            report_date=extract_value(sourced_summary.report_overview.report_date),
            visit_date=extract_value(sourced_summary.report_overview.visit_date),
            treating_physician=extract_value(sourced_summary.report_overview.treating_physician),
            specialty=extract_value(sourced_summary.report_overview.specialty),
            time_since_injury=extract_value(sourced_summary.report_overview.time_since_injury),
            time_since_last_visit=extract_value(sourced_summary.report_overview.time_since_last_visit),
            author_signature=extract_value(sourced_summary.report_overview.author_signature),
            signature_type=sourced_summary.report_overview.signature_type
        ),
        patient_information=PR2PatientInformation(
            name=extract_value(sourced_summary.patient_information.name),
            date_of_birth=extract_value(sourced_summary.patient_information.date_of_birth),
            age=extract_value(sourced_summary.patient_information.age),
            date_of_injury=extract_value(sourced_summary.patient_information.date_of_injury),
            occupation=extract_value(sourced_summary.patient_information.occupation),
            employer=extract_value(sourced_summary.patient_information.employer),
            claims_administrator=extract_value(sourced_summary.patient_information.claims_administrator),
            claim_number=extract_value(sourced_summary.patient_information.claim_number) or None,
            all_doctors_involved=convert_doctors(sourced_summary.patient_information.all_doctors_involved)
        ),
        chief_complaint=PR2ChiefComplaint(
            primary_complaint=extract_value(sourced_summary.chief_complaint.primary_complaint),
            location=extract_value(sourced_summary.chief_complaint.location),
            description=extract_value(sourced_summary.chief_complaint.description)
        ),
        subjective_assessment=PR2SubjectiveAssessment(
            current_pain_score=extract_value(sourced_summary.subjective_assessment.current_pain_score),
            previous_pain_score=extract_value(sourced_summary.subjective_assessment.previous_pain_score),
            symptom_changes=extract_value(sourced_summary.subjective_assessment.symptom_changes),
            functional_status_patient_reported=extract_value(sourced_summary.subjective_assessment.functional_status_patient_reported),
            patient_compliance=extract_value(sourced_summary.subjective_assessment.patient_compliance)
        ),
        objective_findings=PR2ObjectiveFindings(
            physical_exam_findings=extract_value(sourced_summary.objective_findings.physical_exam_findings),
            rom_measurements=extract_value(sourced_summary.objective_findings.rom_measurements),
            strength_testing=extract_value(sourced_summary.objective_findings.strength_testing),
            gait_assessment=extract_value(sourced_summary.objective_findings.gait_assessment),
            neurological_findings=extract_value(sourced_summary.objective_findings.neurological_findings),
            functional_limitations_observed=extract_list_values(sourced_summary.objective_findings.functional_limitations_observed)
        ),
        diagnosis=PR2Diagnosis(
            primary_diagnosis=extract_value(sourced_summary.diagnosis.primary_diagnosis),
            secondary_diagnoses=extract_list_values(sourced_summary.diagnosis.secondary_diagnoses)
        ),
        medications=PR2Medications(
            current_medications=extract_list_values(sourced_summary.medications.current_medications),
            new_medications=extract_list_values(sourced_summary.medications.new_medications),
            dosage_changes=extract_list_values(sourced_summary.medications.dosage_changes)
        ),
        treatment_effectiveness=PR2TreatmentEffectiveness(
            patient_response=extract_value(sourced_summary.treatment_effectiveness.patient_response),
            functional_gains=extract_value(sourced_summary.treatment_effectiveness.functional_gains),
            objective_improvements=extract_list_values(sourced_summary.treatment_effectiveness.objective_improvements),
            barriers_to_progress=extract_value(sourced_summary.treatment_effectiveness.barriers_to_progress)
        ),
        treatment_authorization_request=PR2TreatmentAuthorizationRequest(
            primary_request=extract_value(sourced_summary.treatment_authorization_request.primary_request),
            secondary_requests=extract_list_values(sourced_summary.treatment_authorization_request.secondary_requests),
            requested_frequency=extract_value(sourced_summary.treatment_authorization_request.requested_frequency),
            requested_duration=extract_value(sourced_summary.treatment_authorization_request.requested_duration),
            medical_necessity_rationale=extract_value(sourced_summary.treatment_authorization_request.medical_necessity_rationale)
        ),
        work_status=PR2WorkStatus(
            current_status=extract_value(sourced_summary.work_status.current_status),
            status_effective_date=extract_value(sourced_summary.work_status.status_effective_date),
            work_limitations=extract_list_values(sourced_summary.work_status.work_limitations),
            work_status_rationale=extract_value(sourced_summary.work_status.work_status_rationale),
            changes_from_previous_status=extract_value(sourced_summary.work_status.changes_from_previous_status),
            expected_return_to_work_date=extract_value(sourced_summary.work_status.expected_return_to_work_date)
        ),
        follow_up_plan=PR2FollowUpPlan(
            next_appointment_date=extract_value(sourced_summary.follow_up_plan.next_appointment_date),
            purpose_of_next_visit=extract_value(sourced_summary.follow_up_plan.purpose_of_next_visit),
            specialist_referrals_requested=extract_list_values(sourced_summary.follow_up_plan.specialist_referrals_requested),
            mmi_ps_anticipated_date=extract_value(sourced_summary.follow_up_plan.mmi_ps_anticipated_date),
            return_sooner_if=extract_value(sourced_summary.follow_up_plan.return_sooner_if)
        ),
        critical_findings=extract_list_values(sourced_summary.critical_findings)
    )


# ============================================================================
# ADMINISTRATIVE/LEGAL DOCUMENT SPECIFIC MODELS
# ============================================================================

class AdminDocumentOverview(BaseModel):
    """Document overview section for administrative/legal documents"""
    document_type: str = Field(default="", description="Type of administrative document")
    document_date: str = Field(default="", description="Date of the document")
    subject: str = Field(default="", description="Subject of the document")
    purpose: str = Field(default="", description="Purpose of the document")
    document_id: str = Field(default="", description="Document ID from headers/footers")


class AdminPartyInfo(BaseModel):
    """Information about a party (sender/recipient) in the document"""
    name: str = Field(default="", description="Name of the party")
    organization: str = Field(default="", description="Organization name")
    title: str = Field(default="", description="Title/position")


class AdminLegalRepresentation(BaseModel):
    """Legal representation information"""
    representative: str = Field(default="", description="Legal representative name")
    firm: str = Field(default="", description="Law firm name")


class AdminPartiesInvolved(BaseModel):
    """All parties involved in the administrative document"""
    patient_details: str = Field(default="", description="Patient details if applicable")
    from_party: AdminPartyInfo = Field(default_factory=AdminPartyInfo, description="Sender information")
    to_party: AdminPartyInfo = Field(default_factory=AdminPartyInfo, description="Recipient information")
    author_signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, insurance representatives, or other officials.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")
    legal_representation: AdminLegalRepresentation = Field(default_factory=AdminLegalRepresentation, description="Legal representation")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class AdminKeyDatesDeadlines(BaseModel):
    """Key dates and deadlines section"""
    response_deadline: str = Field(default="", description="Response deadline")
    hearing_date: str = Field(default="", description="Hearing date if applicable")
    appointment_date: str = Field(default="", description="Appointment date if applicable")
    time_sensitive_requirements: List[str] = Field(default_factory=list, description="Time-sensitive requirements (up to 3)")


class AdminContent(BaseModel):
    """Administrative content section"""
    primary_subject: str = Field(default="", description="Primary subject matter")
    key_points: str = Field(default="", description="Key points from the document")
    current_status: str = Field(default="", description="Current status")
    incident_details: str = Field(default="", description="Incident details if applicable (truncated to 200 chars)")


class AdminActionItemsRequirements(BaseModel):
    """Action items and requirements section"""
    required_responses: List[str] = Field(default_factory=list, description="Required responses (up to 5)")
    documentation_required: List[str] = Field(default_factory=list, description="Documentation required (up to 5)")
    specific_actions: List[str] = Field(default_factory=list, description="Specific actions needed (up to 5)")


class AdminLegalProceduralElements(BaseModel):
    """Legal and procedural elements section"""
    legal_demands: List[str] = Field(default_factory=list, description="Legal demands (up to 3)")
    next_steps: List[str] = Field(default_factory=list, description="Next steps (up to 3)")
    consequences_of_non_compliance: str = Field(default="", description="Consequences of non-compliance")


class AdminMedicalClaimInfo(BaseModel):
    """Medical and claim information section for administrative documents"""
    claim_number: Optional[str] = Field(default=None, description="Claim number. If multiple, join with comma.")
    case_number: str = Field(default="", description="Case number")
    work_status: str = Field(default="", description="Work status")
    disability_information: str = Field(default="", description="Disability information")
    treatment_authorizations: List[str] = Field(default_factory=list, description="Treatment authorizations (up to 3)")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class AdminContactFollowUp(BaseModel):
    """Contact and follow-up section for administrative documents"""
    contact_person: str = Field(default="", description="Contact person name")
    phone: str = Field(default="", description="Phone number")
    email: str = Field(default="", description="Email address")
    submission_address: str = Field(default="", description="Submission/mailing address")
    response_format: str = Field(default="", description="Required response format")


class AdminLongSummary(BaseModel):
    """
    Complete structured Administrative/Legal document long summary.
    Designed for attorney letters, NCM notes, employer reports, disability forms, legal correspondence.
    """
    content_type: Literal["administrative"] = Field(default="administrative", description="Content type for administrative documents")
    
    # Main sections
    document_overview: AdminDocumentOverview = Field(default_factory=AdminDocumentOverview, description="Document overview")
    parties_involved: AdminPartiesInvolved = Field(default_factory=AdminPartiesInvolved, description="Parties involved")
    key_dates_deadlines: AdminKeyDatesDeadlines = Field(default_factory=AdminKeyDatesDeadlines, description="Key dates and deadlines")
    administrative_content: AdminContent = Field(default_factory=AdminContent, description="Administrative content")
    action_items_requirements: AdminActionItemsRequirements = Field(default_factory=AdminActionItemsRequirements, description="Action items and requirements")
    legal_procedural_elements: AdminLegalProceduralElements = Field(default_factory=AdminLegalProceduralElements, description="Legal and procedural elements")
    medical_claim_info: AdminMedicalClaimInfo = Field(default_factory=AdminMedicalClaimInfo, description="Medical and claim information")
    contact_followup: AdminContactFollowUp = Field(default_factory=AdminContactFollowUp, description="Contact and follow-up")
    critical_administrative_findings: List[str] = Field(default_factory=list, description="Critical administrative findings (up to 8)")


def format_admin_long_summary(summary: AdminLongSummary) -> str:
    """Format an administrative long summary into the expected text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 ADMINISTRATIVE DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_overview.document_type:
        lines.append(f"Document Type: {summary.document_overview.document_type}")
    if summary.document_overview.document_date:
        lines.append(f"Document Date: {summary.document_overview.document_date}")
    if summary.document_overview.subject:
        lines.append(f"Subject: {summary.document_overview.subject}")
    if summary.document_overview.purpose:
        lines.append(f"Purpose: {summary.document_overview.purpose}")
    if summary.document_overview.document_id:
        lines.append(f"Document ID: {summary.document_overview.document_id}")
    lines.append("")
    
    # Parties Involved
    lines.append("👥 PARTIES INVOLVED")
    lines.append("-" * 50)
    if summary.parties_involved.patient_details:
        lines.append(f"Patient Details: {summary.parties_involved.patient_details}")
    lines.append("")
    if summary.parties_involved.from_party.name or summary.parties_involved.from_party.organization:
        lines.append("From:")
        if summary.parties_involved.from_party.name:
            lines.append(f"  Name: {summary.parties_involved.from_party.name}")
        if summary.parties_involved.from_party.organization:
            lines.append(f"  Organization: {summary.parties_involved.from_party.organization}")
        if summary.parties_involved.from_party.title:
            lines.append(f"  Title: {summary.parties_involved.from_party.title}")
    if summary.parties_involved.to_party.name or summary.parties_involved.to_party.organization:
        lines.append("To:")
        if summary.parties_involved.to_party.name:
            lines.append(f"  Name: {summary.parties_involved.to_party.name}")
        if summary.parties_involved.to_party.organization:
            lines.append(f"  Organization: {summary.parties_involved.to_party.organization}")
    if summary.parties_involved.author_signature:
        sig_type = f" ({summary.parties_involved.signature_type})" if summary.parties_involved.signature_type else ""
        lines.append("Author:")
        lines.append(f"• Signature: {summary.parties_involved.author_signature}{sig_type}")
    if summary.parties_involved.legal_representation.representative or summary.parties_involved.legal_representation.firm:
        lines.append("Legal Representation:")
        if summary.parties_involved.legal_representation.representative:
            lines.append(f"  Representative: {summary.parties_involved.legal_representation.representative}")
        if summary.parties_involved.legal_representation.firm:
            lines.append(f"  Firm: {summary.parties_involved.legal_representation.firm}")
    if summary.parties_involved.claim_number:
        lines.append(f"Claim Number: {summary.parties_involved.claim_number}")
    if summary.parties_involved.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.parties_involved.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Key Dates & Deadlines
    lines.append("📅 KEY DATES & DEADLINES")
    lines.append("-" * 50)
    if summary.key_dates_deadlines.response_deadline:
        lines.append(f"Response Deadline: {summary.key_dates_deadlines.response_deadline}")
    if summary.key_dates_deadlines.hearing_date:
        lines.append(f"Hearing Date: {summary.key_dates_deadlines.hearing_date}")
    if summary.key_dates_deadlines.appointment_date:
        lines.append(f"Appointment Date: {summary.key_dates_deadlines.appointment_date}")
    if summary.key_dates_deadlines.time_sensitive_requirements:
        lines.append("Time-Sensitive Requirements:")
        for req in summary.key_dates_deadlines.time_sensitive_requirements[:3]:
            lines.append(f"• {req}")
    lines.append("")
    
    # Administrative Content
    lines.append("📄 ADMINISTRATIVE CONTENT")
    lines.append("-" * 50)
    if summary.administrative_content.primary_subject:
        lines.append(f"Primary Subject: {summary.administrative_content.primary_subject}")
    if summary.administrative_content.key_points:
        lines.append(f"Key Points: {summary.administrative_content.key_points}")
    if summary.administrative_content.current_status:
        lines.append(f"Current Status: {summary.administrative_content.current_status}")
    if summary.administrative_content.incident_details:
        lines.append(f"Incident Details: {summary.administrative_content.incident_details}")
    lines.append("")
    
    # Action Items & Requirements
    lines.append("✅ ACTION ITEMS & REQUIREMENTS")
    lines.append("-" * 50)
    if summary.action_items_requirements.required_responses:
        lines.append("Required Responses:")
        for resp in summary.action_items_requirements.required_responses[:5]:
            lines.append(f"• {resp}")
    if summary.action_items_requirements.documentation_required:
        lines.append("Documentation Required:")
        for doc in summary.action_items_requirements.documentation_required[:5]:
            lines.append(f"• {doc}")
    if summary.action_items_requirements.specific_actions:
        lines.append("Specific Actions:")
        for action in summary.action_items_requirements.specific_actions[:5]:
            lines.append(f"• {action}")
    lines.append("")
    
    # Legal & Procedural Elements
    lines.append("⚖️ LEGAL & PROCEDURAL ELEMENTS")
    lines.append("-" * 50)
    if summary.legal_procedural_elements.legal_demands:
        lines.append("Legal Demands:")
        for demand in summary.legal_procedural_elements.legal_demands[:3]:
            lines.append(f"• {demand}")
    if summary.legal_procedural_elements.next_steps:
        lines.append("Next Steps:")
        for step in summary.legal_procedural_elements.next_steps[:3]:
            lines.append(f"• {step}")
    if summary.legal_procedural_elements.consequences_of_non_compliance:
        lines.append(f"Consequences of Non-Compliance: {summary.legal_procedural_elements.consequences_of_non_compliance}")
    lines.append("")
    
    # Medical & Claim Information
    lines.append("🏥 MEDICAL & CLAIM INFORMATION")
    lines.append("-" * 50)
    if summary.medical_claim_info.claim_number:
        lines.append(f"Claim Number: {summary.medical_claim_info.claim_number}")
    if summary.medical_claim_info.case_number:
        lines.append(f"Case Number: {summary.medical_claim_info.case_number}")
    if summary.medical_claim_info.work_status:
        lines.append(f"Work Status: {summary.medical_claim_info.work_status}")
    if summary.medical_claim_info.disability_information:
        lines.append(f"Disability Information: {summary.medical_claim_info.disability_information}")
    if summary.medical_claim_info.treatment_authorizations:
        lines.append("Treatment Authorizations:")
        for auth in summary.medical_claim_info.treatment_authorizations[:3]:
            lines.append(f"• {auth}")
    lines.append("")
    
    # Contact & Follow-Up
    lines.append("📞 CONTACT & FOLLOW-UP")
    lines.append("-" * 50)
    if summary.contact_followup.contact_person:
        lines.append(f"Contact Person: {summary.contact_followup.contact_person}")
    if summary.contact_followup.phone:
        lines.append(f"Phone: {summary.contact_followup.phone}")
    if summary.contact_followup.email:
        lines.append(f"Email: {summary.contact_followup.email}")
    if summary.contact_followup.submission_address:
        lines.append(f"Submission Address: {summary.contact_followup.submission_address}")
    if summary.contact_followup.response_format:
        lines.append(f"Response Format: {summary.contact_followup.response_format}")
    lines.append("")
    
    # Critical Administrative Findings
    if summary.critical_administrative_findings:
        lines.append("🚨 CRITICAL ADMINISTRATIVE FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_administrative_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_admin_summary(doc_type: str, fallback_date: str) -> AdminLongSummary:
    """Create a fallback administrative long summary when extraction fails."""
    return AdminLongSummary(
        content_type="administrative",
        document_overview=AdminDocumentOverview(
            document_type=doc_type,
            document_date=fallback_date
        )
    )

# ============================================================================
# CLINICAL PROGRESS NOTE / THERAPY REPORT SPECIFIC MODELS
# ============================================================================

class ClinicalEncounterOverview(BaseModel):
    """Clinical encounter overview section"""
    note_type: str = Field(default="", description="Type of clinical note (Progress Note, PT, OT, Chiropractic, etc.)")
    visit_date: str = Field(default="", description="Date of the visit")
    visit_type: str = Field(default="", description="Type of visit (initial, follow-up, re-evaluation, etc.)")
    duration: str = Field(default="", description="Duration of the visit")
    facility: str = Field(default="", description="Facility/clinic name")


class ClinicalProviderInfo(BaseModel):
    """Provider information section"""
    treating_provider: str = Field(default="", description="Treating provider name")
    credentials: str = Field(default="", description="Provider credentials (PT, OT, DC, MD, etc.)")
    specialty: str = Field(default="", description="Provider specialty")


class ClinicalSubjectiveFindings(BaseModel):
    """Subjective findings section"""
    chief_complaint: str = Field(default="", description="Main reason for visit/chief complaint")
    pain_description: str = Field(default="", description="Pain description including location, intensity, character")
    functional_limitations: List[str] = Field(default_factory=list, description="Functional limitations reported by patient (up to 5)")


class ClinicalObjectiveExamination(BaseModel):
    """Objective examination findings section"""
    range_of_motion: List[str] = Field(default_factory=list, description="Range of motion findings (up to 5)")
    manual_muscle_testing: List[str] = Field(default_factory=list, description="Manual muscle testing results (up to 5)")
    special_tests: List[str] = Field(default_factory=list, description="Special tests performed with results (up to 3)")


class ClinicalTreatmentProvided(BaseModel):
    """Treatment provided section"""
    treatment_techniques: List[str] = Field(default_factory=list, description="Treatment techniques used with CPT codes if available (up to 5)")
    therapeutic_exercises: List[str] = Field(default_factory=list, description="Therapeutic exercises performed (up to 5)")
    modalities_used: List[str] = Field(default_factory=list, description="Modalities used with parameters (up to 5)")


class ClinicalAssessment(BaseModel):
    """Clinical assessment section"""
    assessment: str = Field(default="", description="Clinical assessment")
    progress: str = Field(default="", description="Progress towards goals")
    clinical_impression: str = Field(default="", description="Clinical impression")
    prognosis: str = Field(default="", description="Prognosis for recovery")


class ClinicalTreatmentPlan(BaseModel):
    """Treatment plan section"""
    short_term_goals: List[str] = Field(default_factory=list, description="Short-term goals (up to 3)")
    home_exercise_program: List[str] = Field(default_factory=list, description="Home exercise program (up to 3)")
    frequency_duration: str = Field(default="", description="Recommended treatment frequency and duration")
    next_appointment: str = Field(default="", description="Next appointment date/time")


class ClinicalWorkStatus(BaseModel):
    """Work status and functional capacity section"""
    current_status: str = Field(default="", description="Current work status (full duty, modified, off work, etc.)")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions (up to 5)")
    functional_capacity: str = Field(default="", description="Functional capacity assessment")


class ClinicalOutcomeMeasures(BaseModel):
    """Outcome measures and progress tracking section"""
    pain_scale: str = Field(default="", description="Pain scale rating (e.g., 7/10)")
    functional_scores: List[str] = Field(default_factory=list, description="Functional scores/outcome measures (up to 3)")


class ClinicalSignatureAuthor(BaseModel):
    """Signature and author section - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class ClinicalLongSummary(BaseModel):
    """
    Complete structured Clinical Progress Note / Therapy Report long summary.
    Designed for PT, OT, Chiropractic, Pain Management, Psychiatry, Nursing notes.
    """
    content_type: Literal["clinical"] = Field(default="clinical", description="Content type for clinical documents")
    
    # Main sections matching the clinical extractor prompt
    encounter_overview: ClinicalEncounterOverview = Field(default_factory=ClinicalEncounterOverview, description="Clinical encounter overview")
    provider_info: ClinicalProviderInfo = Field(default_factory=ClinicalProviderInfo, description="Provider information")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")
    subjective_findings: ClinicalSubjectiveFindings = Field(default_factory=ClinicalSubjectiveFindings, description="Subjective findings")
    objective_examination: ClinicalObjectiveExamination = Field(default_factory=ClinicalObjectiveExamination, description="Objective examination findings")
    treatment_provided: ClinicalTreatmentProvided = Field(default_factory=ClinicalTreatmentProvided, description="Treatment provided")
    clinical_assessment: ClinicalAssessment = Field(default_factory=ClinicalAssessment, description="Clinical assessment")
    treatment_plan: ClinicalTreatmentPlan = Field(default_factory=ClinicalTreatmentPlan, description="Treatment plan")
    work_status: ClinicalWorkStatus = Field(default_factory=ClinicalWorkStatus, description="Work status and functional capacity")
    outcome_measures: ClinicalOutcomeMeasures = Field(default_factory=ClinicalOutcomeMeasures, description="Outcome measures")
    signature_author: ClinicalSignatureAuthor = Field(default_factory=ClinicalSignatureAuthor, description="Signature and author")
    critical_clinical_findings: List[str] = Field(default_factory=list, description="Critical clinical findings (up to 8)")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


def format_clinical_long_summary(summary: ClinicalLongSummary) -> str:
    """Format a clinical long summary into the expected text format."""
    lines = []
    
    # Clinical Encounter Overview
    lines.append("📋 CLINICAL ENCOUNTER OVERVIEW")
    lines.append("-" * 50)
    if summary.encounter_overview.note_type:
        lines.append(f"Note Type: {summary.encounter_overview.note_type}")
    if summary.encounter_overview.visit_date:
        lines.append(f"Visit Date: {summary.encounter_overview.visit_date}")
    if summary.encounter_overview.visit_type:
        lines.append(f"Visit Type: {summary.encounter_overview.visit_type}")
    if summary.encounter_overview.duration:
        lines.append(f"Duration: {summary.encounter_overview.duration}")
    if summary.encounter_overview.facility:
        lines.append(f"Facility: {summary.encounter_overview.facility}")
    lines.append("")
    
    # Provider Information
    lines.append("👨‍⚕️ PROVIDER INFORMATION")
    lines.append("-" * 50)
    if summary.provider_info.treating_provider:
        lines.append(f"Treating Provider: {summary.provider_info.treating_provider}")
    if summary.provider_info.credentials:
        lines.append(f"Credentials: {summary.provider_info.credentials}")
    if summary.provider_info.specialty:
        lines.append(f"Specialty: {summary.provider_info.specialty}")
    if summary.claim_number:
        lines.append(f"Claim Number: {summary.claim_number}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Subjective Findings
    lines.append("🗣️ SUBJECTIVE FINDINGS")
    lines.append("-" * 50)
    if summary.subjective_findings.chief_complaint:
        lines.append(f"Chief Complaint: {summary.subjective_findings.chief_complaint}")
    if summary.subjective_findings.pain_description:
        lines.append(f"Pain: {summary.subjective_findings.pain_description}")
    if summary.subjective_findings.functional_limitations:
        lines.append("Functional Limitations:")
        for limitation in summary.subjective_findings.functional_limitations[:5]:
            lines.append(f"• {limitation}")
    lines.append("")
    
    # Objective Examination
    lines.append("🔍 OBJECTIVE EXAMINATION")
    lines.append("-" * 50)
    if summary.objective_examination.range_of_motion:
        lines.append("Range of Motion:")
        for rom in summary.objective_examination.range_of_motion[:5]:
            lines.append(f"• {rom}")
    if summary.objective_examination.manual_muscle_testing:
        lines.append("Manual Muscle Testing:")
        for mmt in summary.objective_examination.manual_muscle_testing[:5]:
            lines.append(f"• {mmt}")
    if summary.objective_examination.special_tests:
        lines.append("Special Tests:")
        for test in summary.objective_examination.special_tests[:3]:
            lines.append(f"• {test}")
    lines.append("")
    
    # Treatment Provided
    lines.append("💆 TREATMENT PROVIDED")
    lines.append("-" * 50)
    if summary.treatment_provided.treatment_techniques:
        lines.append("Treatment Techniques:")
        for tech in summary.treatment_provided.treatment_techniques[:5]:
            lines.append(f"• {tech}")
    if summary.treatment_provided.therapeutic_exercises:
        lines.append("Therapeutic Exercises:")
        for exercise in summary.treatment_provided.therapeutic_exercises[:5]:
            lines.append(f"• {exercise}")
    if summary.treatment_provided.modalities_used:
        lines.append("Modalities Used:")
        for mod in summary.treatment_provided.modalities_used[:5]:
            lines.append(f"• {mod}")
    lines.append("")
    
    # Clinical Assessment
    lines.append("🏥 CLINICAL ASSESSMENT")
    lines.append("-" * 50)
    if summary.clinical_assessment.assessment:
        lines.append(f"Assessment: {summary.clinical_assessment.assessment}")
    if summary.clinical_assessment.progress:
        lines.append(f"Progress: {summary.clinical_assessment.progress}")
    if summary.clinical_assessment.clinical_impression:
        lines.append(f"Clinical Impression: {summary.clinical_assessment.clinical_impression}")
    if summary.clinical_assessment.prognosis:
        lines.append(f"Prognosis: {summary.clinical_assessment.prognosis}")
    lines.append("")
    
    # Treatment Plan
    lines.append("🎯 TREATMENT PLAN")
    lines.append("-" * 50)
    if summary.treatment_plan.short_term_goals:
        lines.append("Short-term Goals:")
        for goal in summary.treatment_plan.short_term_goals[:3]:
            lines.append(f"• {goal}")
    if summary.treatment_plan.home_exercise_program:
        lines.append("Home Exercise Program:")
        for exercise in summary.treatment_plan.home_exercise_program[:3]:
            lines.append(f"• {exercise}")
    if summary.treatment_plan.frequency_duration:
        lines.append(f"Frequency/Duration: {summary.treatment_plan.frequency_duration}")
    if summary.treatment_plan.next_appointment:
        lines.append(f"Next Appointment: {summary.treatment_plan.next_appointment}")
    lines.append("")
    
    # Work Status
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    if summary.work_status.current_status:
        lines.append(f"Current Status: {summary.work_status.current_status}")
    if summary.work_status.work_restrictions:
        lines.append("Work Restrictions:")
        for restriction in summary.work_status.work_restrictions[:5]:
            lines.append(f"• {restriction}")
    if summary.work_status.functional_capacity:
        lines.append(f"Functional Capacity: {summary.work_status.functional_capacity}")
    lines.append("")
    
    # Outcome Measures
    lines.append("📊 OUTCOME MEASURES")
    lines.append("-" * 50)
    if summary.outcome_measures.pain_scale:
        lines.append(f"Pain Scale: {summary.outcome_measures.pain_scale}")
    if summary.outcome_measures.functional_scores:
        lines.append("Functional Scores:")
        for score in summary.outcome_measures.functional_scores[:3]:
            lines.append(f"• {score}")
    lines.append("")
    
    # Signature & Author
    lines.append("✍️ SIGNATURE & AUTHOR")
    lines.append("-" * 50)
    if summary.signature_author.signature:
        sig_type = f" ({summary.signature_author.signature_type})" if summary.signature_author.signature_type else ""
        lines.append("Author:")
        lines.append(f"• Signature: {summary.signature_author.signature}{sig_type}")
    lines.append("")
    
    # Critical Clinical Findings
    if summary.critical_clinical_findings:
        lines.append("🚨 CRITICAL CLINICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_clinical_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_clinical_summary(doc_type: str, fallback_date: str) -> ClinicalLongSummary:
    """Create a fallback clinical long summary when extraction fails."""
    return ClinicalLongSummary(
        content_type="clinical",
        encounter_overview=ClinicalEncounterOverview(
            note_type=doc_type,
            visit_date=fallback_date
        )
    )


# ============================================================================
# SPECIALIST CONSULTATION REPORT SPECIFIC MODELS
# ============================================================================

class ConsultOverview(BaseModel):
    """Consultation overview section"""
    document_type: str = Field(default="", description="Type of consultation document")
    consultation_date: str = Field(default="", description="Date of the consultation")
    consulting_physician: str = Field(default="", description="Name of the consulting physician")
    specialty: str = Field(default="", description="Specialty of the consulting physician")
    referring_physician: str = Field(default="", description="Name of the referring physician")


class ConsultPatientInfo(BaseModel):
    """Patient information section for consultation"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    date_of_injury: str = Field(default="", description="Date of injury")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class ConsultChiefComplaint(BaseModel):
    """Chief complaint section"""
    primary_complaint: str = Field(default="", description="Primary complaint from the patient")
    location: str = Field(default="", description="Location of the complaint/pain")
    duration: str = Field(default="", description="Duration of the complaint")
    radiation_pattern: str = Field(default="", description="Radiation pattern of the pain if applicable")


class ConsultDiagnosis(BaseModel):
    """Diagnosis and assessment section"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    icd_10_code: str = Field(default="", description="ICD-10 code for primary diagnosis")
    certainty: str = Field(default="", description="Diagnostic certainty level")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 5)")
    causation: str = Field(default="", description="Causation statement linking injury to diagnosis")


class ConsultClinicalHistory(BaseModel):
    """Clinical history and symptoms section"""
    pain_quality: str = Field(default="", description="Quality/character of pain")
    pain_location: str = Field(default="", description="Location of pain")
    radiation: str = Field(default="", description="Radiation pattern of pain")
    aggravating_factors: List[str] = Field(default_factory=list, description="Factors that aggravate symptoms (up to 5)")
    alleviating_factors: List[str] = Field(default_factory=list, description="Factors that alleviate symptoms (up to 5)")


class ConsultPriorTreatment(BaseModel):
    """Prior treatment and efficacy section"""
    prior_treatments: List[str] = Field(default_factory=list, description="Prior treatments received (up to 8)")
    level_of_relief: List[str] = Field(default_factory=list, description="Level of relief from each treatment (up to 5)")
    treatment_failure_statement: str = Field(default="", description="Statement about failure of conservative care")


class ConsultObjectiveFindings(BaseModel):
    """Objective findings section"""
    physical_examination: List[str] = Field(default_factory=list, description="Physical examination findings (up to 8)")
    imaging_review: List[str] = Field(default_factory=list, description="Imaging review findings (up to 5)")


class ConsultTreatmentRecommendations(BaseModel):
    """Treatment recommendations section - most critical for authorization"""
    injections_requested: List[str] = Field(default_factory=list, description="Injections requested with justification (up to 5)")
    procedures_requested: List[str] = Field(default_factory=list, description="Procedures requested with reasons (up to 5)")
    surgery_recommended: List[str] = Field(default_factory=list, description="Surgery recommended with urgency (up to 3)")
    diagnostics_ordered: List[str] = Field(default_factory=list, description="Diagnostics ordered with reasons (up to 5)")
    medication_changes: List[str] = Field(default_factory=list, description="Medication changes with dosages (up to 5)")
    therapy_recommendations: List[str] = Field(default_factory=list, description="Therapy recommendations with frequency (up to 5)")


class ConsultWorkStatus(BaseModel):
    """Work status and impairment section"""
    current_work_status: str = Field(default="", description="Current work status (full duty, modified, off work, etc.)")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions (up to 8)")
    restriction_duration: str = Field(default="", description="Duration of restrictions")
    return_to_work_plan: str = Field(default="", description="Plan for return to work")


class ConsultSignatureAuthor(BaseModel):
    """Signature and author section - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class ConsultLongSummary(BaseModel):
    """
    Complete structured Specialist Consultation Report long summary.
    Designed for Workers' Compensation consultation analysis with 8 critical fields.
    """
    content_type: Literal["consultation"] = Field(default="consultation", description="Content type for consultation documents")
    
    # Main sections matching the consult extractor prompt
    consultation_overview: ConsultOverview = Field(default_factory=ConsultOverview, description="Consultation overview")
    patient_info: ConsultPatientInfo = Field(default_factory=ConsultPatientInfo, description="Patient information")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")
    chief_complaint: ConsultChiefComplaint = Field(default_factory=ConsultChiefComplaint, description="Chief complaint")
    signature_author: ConsultSignatureAuthor = Field(default_factory=ConsultSignatureAuthor, description="Signature and author")
    diagnosis_assessment: ConsultDiagnosis = Field(default_factory=ConsultDiagnosis, description="Diagnosis and assessment")
    clinical_history: ConsultClinicalHistory = Field(default_factory=ConsultClinicalHistory, description="Clinical history and symptoms")
    prior_treatment: ConsultPriorTreatment = Field(default_factory=ConsultPriorTreatment, description="Prior treatment and efficacy")
    objective_findings: ConsultObjectiveFindings = Field(default_factory=ConsultObjectiveFindings, description="Objective findings")
    treatment_recommendations: ConsultTreatmentRecommendations = Field(default_factory=ConsultTreatmentRecommendations, description="Treatment recommendations")
    work_status: ConsultWorkStatus = Field(default_factory=ConsultWorkStatus, description="Work status and impairment")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 8)")


def format_consult_long_summary(summary: ConsultLongSummary) -> str:
    """Format a consultation long summary into the expected text format."""
    lines = []
    
    # Consultation Overview
    lines.append("📋 CONSULTATION OVERVIEW")
    lines.append("-" * 50)
    if summary.consultation_overview.document_type:
        lines.append(f"Document Type: {summary.consultation_overview.document_type}")
    if summary.consultation_overview.consultation_date:
        lines.append(f"Consultation Date: {summary.consultation_overview.consultation_date}")
    if summary.consultation_overview.consulting_physician:
        lines.append(f"Consulting Physician: {summary.consultation_overview.consulting_physician}")
    if summary.consultation_overview.specialty:
        lines.append(f"Specialty: {summary.consultation_overview.specialty}")
    if summary.consultation_overview.referring_physician:
        lines.append(f"Referring Physician: {summary.consultation_overview.referring_physician}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"Name: {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"Date of Birth: {summary.patient_info.date_of_birth}")
    if summary.patient_info.date_of_injury:
        lines.append(f"Date of Injury: {summary.patient_info.date_of_injury}")
    if summary.patient_info.claim_number:
        lines.append(f"Claim Number: {summary.patient_info.claim_number}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Chief Complaint
    lines.append("🎯 CHIEF COMPLAINT")
    lines.append("-" * 50)
    if summary.chief_complaint.primary_complaint:
        lines.append(f"Primary Complaint: {summary.chief_complaint.primary_complaint}")
    if summary.chief_complaint.location:
        lines.append(f"Location: {summary.chief_complaint.location}")
    if summary.chief_complaint.duration:
        lines.append(f"Duration: {summary.chief_complaint.duration}")
    if summary.chief_complaint.radiation_pattern:
        lines.append(f"Radiation Pattern: {summary.chief_complaint.radiation_pattern}")
    if summary.signature_author.signature:
        sig_type = f" ({summary.signature_author.signature_type})" if summary.signature_author.signature_type else ""
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.signature_author.signature}{sig_type}")
    lines.append("")
    
    # Diagnosis & Assessment
    lines.append("🏥 DIAGNOSIS & ASSESSMENT")
    lines.append("-" * 50)
    if summary.diagnosis_assessment.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.diagnosis_assessment.primary_diagnosis}")
    if summary.diagnosis_assessment.icd_10_code:
        lines.append(f"- ICD-10: {summary.diagnosis_assessment.icd_10_code}")
    if summary.diagnosis_assessment.certainty:
        lines.append(f"- Certainty: {summary.diagnosis_assessment.certainty}")
    if summary.diagnosis_assessment.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.diagnosis_assessment.secondary_diagnoses[:5]:
            lines.append(f"• {dx}")
    if summary.diagnosis_assessment.causation:
        lines.append(f"Causation: {summary.diagnosis_assessment.causation}")
    lines.append("")
    
    # Clinical History & Symptoms
    lines.append("🔬 CLINICAL HISTORY & SYMPTOMS")
    lines.append("-" * 50)
    if summary.clinical_history.pain_quality:
        lines.append(f"Pain Quality: {summary.clinical_history.pain_quality}")
    if summary.clinical_history.pain_location:
        lines.append(f"Pain Location: {summary.clinical_history.pain_location}")
    if summary.clinical_history.radiation:
        lines.append(f"Radiation: {summary.clinical_history.radiation}")
    if summary.clinical_history.aggravating_factors:
        lines.append("Aggravating Factors:")
        for factor in summary.clinical_history.aggravating_factors[:5]:
            lines.append(f"• {factor}")
    if summary.clinical_history.alleviating_factors:
        lines.append("Alleviating Factors:")
        for factor in summary.clinical_history.alleviating_factors[:5]:
            lines.append(f"• {factor}")
    lines.append("")
    
    # Prior Treatment & Efficacy
    lines.append("💊 PRIOR TREATMENT & EFFICACY")
    lines.append("-" * 50)
    if summary.prior_treatment.prior_treatments:
        lines.append("Prior Treatments Received:")
        for treatment in summary.prior_treatment.prior_treatments[:8]:
            lines.append(f"• {treatment}")
    if summary.prior_treatment.level_of_relief:
        lines.append("Level of Relief:")
        for relief in summary.prior_treatment.level_of_relief[:5]:
            lines.append(f"• {relief}")
    if summary.prior_treatment.treatment_failure_statement:
        lines.append(f"Treatment Failure Statement: {summary.prior_treatment.treatment_failure_statement}")
    lines.append("")
    
    # Objective Findings
    lines.append("📊 OBJECTIVE FINDINGS")
    lines.append("-" * 50)
    if summary.objective_findings.physical_examination:
        lines.append("Physical Examination:")
        for finding in summary.objective_findings.physical_examination[:8]:
            lines.append(f"• {finding}")
    if summary.objective_findings.imaging_review:
        lines.append("Imaging Review:")
        for finding in summary.objective_findings.imaging_review[:5]:
            lines.append(f"• {finding}")
    lines.append("")
    
    # Treatment Recommendations
    lines.append("🎯 TREATMENT RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.treatment_recommendations.injections_requested:
        lines.append("Injections Requested:")
        for injection in summary.treatment_recommendations.injections_requested[:5]:
            lines.append(f"• {injection}")
    if summary.treatment_recommendations.procedures_requested:
        lines.append("Procedures Requested:")
        for procedure in summary.treatment_recommendations.procedures_requested[:5]:
            lines.append(f"• {procedure}")
    if summary.treatment_recommendations.surgery_recommended:
        lines.append("Surgery Recommended:")
        for surgery in summary.treatment_recommendations.surgery_recommended[:3]:
            lines.append(f"• {surgery}")
    if summary.treatment_recommendations.diagnostics_ordered:
        lines.append("Diagnostics Ordered:")
        for diagnostic in summary.treatment_recommendations.diagnostics_ordered[:5]:
            lines.append(f"• {diagnostic}")
    if summary.treatment_recommendations.medication_changes:
        lines.append("Medication Changes:")
        for med in summary.treatment_recommendations.medication_changes[:5]:
            lines.append(f"• {med}")
    if summary.treatment_recommendations.therapy_recommendations:
        lines.append("Therapy Recommendations:")
        for therapy in summary.treatment_recommendations.therapy_recommendations[:5]:
            lines.append(f"• {therapy}")
    lines.append("")
    
    # Work Status & Impairment
    lines.append("💼 WORK STATUS & IMPAIRMENT")
    lines.append("-" * 50)
    if summary.work_status.current_work_status:
        lines.append(f"Current Work Status: {summary.work_status.current_work_status}")
    if summary.work_status.work_restrictions:
        lines.append("Work Restrictions:")
        for restriction in summary.work_status.work_restrictions[:8]:
            lines.append(f"• {restriction}")
    if summary.work_status.restriction_duration:
        lines.append(f"Restriction Duration: {summary.work_status.restriction_duration}")
    if summary.work_status.return_to_work_plan:
        lines.append(f"Return to Work Plan: {summary.work_status.return_to_work_plan}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_consult_summary(doc_type: str, fallback_date: str) -> ConsultLongSummary:
    """Create a fallback consultation long summary when extraction fails."""
    return ConsultLongSummary(
        content_type="consultation",
        consultation_overview=ConsultOverview(
            document_type=doc_type,
            consultation_date=fallback_date
        )
    )


# ============================================================================
# FORMAL MEDICAL REPORT SPECIFIC MODELS
# (Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, Endoscopy, Genetics, Discharge)
# ============================================================================

class FormalMedicalPatientInfo(BaseModel):
    """Patient information section for formal medical reports"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    date_of_injury: str = Field(default="", description="Date of injury")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    employer: str = Field(default="", description="Employer name if applicable")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class FormalMedicalProviders(BaseModel):
    """Healthcare providers section"""
    performing_physician: str = Field(default="", description="Name of performing physician/surgeon")
    specialty: str = Field(default="", description="Physician's specialty")
    ordering_physician: str = Field(default="", description="Name of ordering/referring physician")
    anesthesiologist: str = Field(default="", description="Name of anesthesiologist if applicable")
    assistant_surgeon: str = Field(default="", description="Name of assistant surgeon if applicable")


class FormalMedicalSignatureAuthor(BaseModel):
    """Signature and author section - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT providers, claim adjusters, requesting physicians, or other officials mentioned in the document.")
    signature_type: Optional[Literal["physical", "electronic"]] = Field(default=None, description="Type of signature")


class FormalMedicalClinicalIndications(BaseModel):
    """Clinical indications and pre-procedure information"""
    pre_procedure_diagnosis: str = Field(default="", description="Pre-procedure/pre-operative diagnosis")
    chief_complaint: str = Field(default="", description="Chief complaint or presenting symptoms")
    indications: List[str] = Field(default_factory=list, description="Indications for the procedure (up to 5)")
    relevant_history: str = Field(default="", description="Relevant medical history")


class FormalMedicalProcedureDetails(BaseModel):
    """Procedure details section"""
    procedure_name: str = Field(default="", description="Name of the procedure performed")
    procedure_date: str = Field(default="", description="Date of the procedure")
    cpt_codes: List[str] = Field(default_factory=list, description="CPT codes for procedures (up to 5)")
    technique: str = Field(default="", description="Technique or approach used")
    duration: str = Field(default="", description="Duration of the procedure")
    anesthesia_type: str = Field(default="", description="Type of anesthesia used")
    implants_devices: List[str] = Field(default_factory=list, description="Implants or devices used (up to 5)")


class FormalMedicalFindings(BaseModel):
    """Findings section (intraoperative, study results, etc.)"""
    intraoperative_findings: List[str] = Field(default_factory=list, description="Intraoperative or study findings (up to 8)")
    specimens_collected: List[str] = Field(default_factory=list, description="Specimens collected (up to 5)")
    estimated_blood_loss: str = Field(default="", description="Estimated blood loss if applicable")
    complications: str = Field(default="", description="Any complications noted")
    condition_at_completion: str = Field(default="", description="Patient's condition at end of procedure")


class FormalMedicalDiagnosis(BaseModel):
    """Diagnosis section"""
    final_diagnosis: str = Field(default="", description="Final or post-procedure diagnosis")
    icd_10_codes: List[str] = Field(default_factory=list, description="ICD-10 codes (up to 5)")
    pathological_diagnosis: str = Field(default="", description="Pathological diagnosis if applicable")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses (up to 5)")


class FormalMedicalRecommendations(BaseModel):
    """Post-procedure recommendations section"""
    post_procedure_care: List[str] = Field(default_factory=list, description="Post-procedure care instructions (up to 5)")
    medications: List[str] = Field(default_factory=list, description="Medications prescribed (up to 8)")
    activity_restrictions: List[str] = Field(default_factory=list, description="Activity restrictions (up to 5)")
    follow_up: str = Field(default="", description="Follow-up appointment/instructions")
    referrals: List[str] = Field(default_factory=list, description="Referrals made (up to 3)")
    additional_procedures: List[str] = Field(default_factory=list, description="Additional procedures recommended (up to 3)")


class FormalMedicalWorkStatus(BaseModel):
    """Work status section for formal medical reports"""
    current_work_status: str = Field(default="", description="Current work status")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions (up to 5)")
    restriction_duration: str = Field(default="", description="Duration of restrictions")
    return_to_work_date: str = Field(default="", description="Expected return to work date")


class FormalMedicalLongSummary(BaseModel):
    """
    Complete structured Formal Medical Report long summary.
    Designed for Surgery, Anesthesia, EMG/NCS, Pathology, Cardiology, Sleep Studies, 
    Endoscopy, Genetics, Discharge Summaries and similar comprehensive medical reports.
    """
    content_type: Literal["formal_medical"] = Field(default="formal_medical", description="Content type for formal medical documents")
    
    # Report identification
    report_type: str = Field(default="", description="Type of medical report (Surgery, EMG, Pathology, etc.)")
    report_date: str = Field(default="", description="Date of the report")
    facility: str = Field(default="", description="Facility/hospital name")
    
    # Main sections
    patient_info: FormalMedicalPatientInfo = Field(default_factory=FormalMedicalPatientInfo, description="Patient information")
    all_doctors_involved: List[DoctorInfo] = Field(default_factory=list, description="All doctors mentioned in the document")
    providers: FormalMedicalProviders = Field(default_factory=FormalMedicalProviders, description="Healthcare providers")
    signature_author: FormalMedicalSignatureAuthor = Field(default_factory=FormalMedicalSignatureAuthor, description="Signature and author")
    clinical_indications: FormalMedicalClinicalIndications = Field(default_factory=FormalMedicalClinicalIndications, description="Clinical indications")
    procedure_details: FormalMedicalProcedureDetails = Field(default_factory=FormalMedicalProcedureDetails, description="Procedure details")
    findings: FormalMedicalFindings = Field(default_factory=FormalMedicalFindings, description="Findings")
    diagnosis: FormalMedicalDiagnosis = Field(default_factory=FormalMedicalDiagnosis, description="Diagnosis")
    recommendations: FormalMedicalRecommendations = Field(default_factory=FormalMedicalRecommendations, description="Recommendations")
    work_status: FormalMedicalWorkStatus = Field(default_factory=FormalMedicalWorkStatus, description="Work status")
    critical_findings: List[str] = Field(default_factory=list, description="Critical findings (up to 8)")


def format_formal_medical_long_summary(summary: FormalMedicalLongSummary) -> str:
    """Format a formal medical long summary into the expected text format."""
    lines = []
    
    # Report Overview
    lines.append("📋 FORMAL MEDICAL REPORT OVERVIEW")
    lines.append("-" * 50)
    if summary.report_type:
        lines.append(f"Report Type: {summary.report_type}")
    if summary.report_date:
        lines.append(f"Report Date: {summary.report_date}")
    if summary.facility:
        lines.append(f"Facility: {summary.facility}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"Name: {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"DOB: {summary.patient_info.date_of_birth}")
    if summary.patient_info.date_of_injury:
        lines.append(f"DOI: {summary.patient_info.date_of_injury}")
    if summary.patient_info.claim_number:
        lines.append(f"Claim Number: {summary.patient_info.claim_number}")
    if summary.patient_info.employer:
        lines.append(f"Employer: {summary.patient_info.employer}")
    lines.append("")
    
    # Healthcare Providers
    lines.append("👨‍⚕️ HEALTHCARE PROVIDERS")
    lines.append("-" * 50)
    if summary.providers.performing_physician:
        lines.append(f"Performing Physician: {summary.providers.performing_physician}")
    if summary.providers.specialty:
        lines.append(f"Specialty: {summary.providers.specialty}")
    if summary.providers.ordering_physician:
        lines.append(f"Ordering Physician: {summary.providers.ordering_physician}")
    if summary.providers.anesthesiologist:
        lines.append(f"Anesthesiologist: {summary.providers.anesthesiologist}")
    if summary.providers.assistant_surgeon:
        lines.append(f"Assistant Surgeon: {summary.providers.assistant_surgeon}")
    if summary.signature_author.signature:
        sig_type = f" ({summary.signature_author.signature_type})" if summary.signature_author.signature_type else ""
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.signature_author.signature}{sig_type}")
    if summary.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.all_doctors_involved:
            doctor_str = doctor.name
            if doctor.title:
                doctor_str += f", {doctor.title}"
            if doctor.role:
                doctor_str += f" ({doctor.role})"
            lines.append(f"• {doctor_str}")
    lines.append("")
    
    # Clinical Indications
    lines.append("🎯 CLINICAL INDICATIONS")
    lines.append("-" * 50)
    if summary.clinical_indications.pre_procedure_diagnosis:
        lines.append(f"Pre-Procedure Diagnosis: {summary.clinical_indications.pre_procedure_diagnosis}")
    if summary.clinical_indications.chief_complaint:
        lines.append(f"Chief Complaint: {summary.clinical_indications.chief_complaint}")
    if summary.clinical_indications.indications:
        lines.append("Indications:")
        for indication in summary.clinical_indications.indications[:5]:
            lines.append(f"• {indication}")
    if summary.clinical_indications.relevant_history:
        lines.append(f"Relevant History: {summary.clinical_indications.relevant_history}")
    lines.append("")
    
    # Procedure Details
    lines.append("🔧 PROCEDURE DETAILS")
    lines.append("-" * 50)
    if summary.procedure_details.procedure_name:
        lines.append(f"Procedure: {summary.procedure_details.procedure_name}")
    if summary.procedure_details.procedure_date:
        lines.append(f"Procedure Date: {summary.procedure_details.procedure_date}")
    if summary.procedure_details.cpt_codes:
        lines.append(f"CPT Codes: {', '.join(summary.procedure_details.cpt_codes[:5])}")
    if summary.procedure_details.technique:
        lines.append(f"Technique: {summary.procedure_details.technique}")
    if summary.procedure_details.duration:
        lines.append(f"Duration: {summary.procedure_details.duration}")
    if summary.procedure_details.anesthesia_type:
        lines.append(f"Anesthesia: {summary.procedure_details.anesthesia_type}")
    if summary.procedure_details.implants_devices:
        lines.append("Implants/Devices:")
        for device in summary.procedure_details.implants_devices[:5]:
            lines.append(f"• {device}")
    lines.append("")
    
    # Findings
    lines.append("🔬 FINDINGS")
    lines.append("-" * 50)
    if summary.findings.intraoperative_findings:
        lines.append("Intraoperative/Study Findings:")
        for finding in summary.findings.intraoperative_findings[:8]:
            lines.append(f"• {finding}")
    if summary.findings.specimens_collected:
        lines.append("Specimens Collected:")
        for specimen in summary.findings.specimens_collected[:5]:
            lines.append(f"• {specimen}")
    if summary.findings.estimated_blood_loss:
        lines.append(f"Estimated Blood Loss: {summary.findings.estimated_blood_loss}")
    if summary.findings.complications:
        lines.append(f"Complications: {summary.findings.complications}")
    if summary.findings.condition_at_completion:
        lines.append(f"Condition at Completion: {summary.findings.condition_at_completion}")
    lines.append("")
    
    # Diagnosis
    lines.append("🏥 DIAGNOSIS")
    lines.append("-" * 50)
    if summary.diagnosis.final_diagnosis:
        lines.append(f"Final Diagnosis: {summary.diagnosis.final_diagnosis}")
    if summary.diagnosis.icd_10_codes:
        lines.append(f"ICD-10 Codes: {', '.join(summary.diagnosis.icd_10_codes[:5])}")
    if summary.diagnosis.pathological_diagnosis:
        lines.append(f"Pathological Diagnosis: {summary.diagnosis.pathological_diagnosis}")
    if summary.diagnosis.secondary_diagnoses:
        lines.append("Secondary Diagnoses:")
        for dx in summary.diagnosis.secondary_diagnoses[:5]:
            lines.append(f"• {dx}")
    lines.append("")
    
    # Recommendations
    lines.append("📋 RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.recommendations.post_procedure_care:
        lines.append("Post-Procedure Care:")
        for care in summary.recommendations.post_procedure_care[:5]:
            lines.append(f"• {care}")
    if summary.recommendations.medications:
        lines.append("Medications:")
        for med in summary.recommendations.medications[:8]:
            lines.append(f"• {med}")
    if summary.recommendations.activity_restrictions:
        lines.append("Activity Restrictions:")
        for restriction in summary.recommendations.activity_restrictions[:5]:
            lines.append(f"• {restriction}")
    if summary.recommendations.follow_up:
        lines.append(f"Follow-Up: {summary.recommendations.follow_up}")
    if summary.recommendations.referrals:
        lines.append("Referrals:")
        for referral in summary.recommendations.referrals[:3]:
            lines.append(f"• {referral}")
    if summary.recommendations.additional_procedures:
        lines.append("Additional Procedures Recommended:")
        for proc in summary.recommendations.additional_procedures[:3]:
            lines.append(f"• {proc}")
    lines.append("")
    
    # Work Status
    lines.append("💼 WORK STATUS")
    lines.append("-" * 50)
    if summary.work_status.current_work_status:
        lines.append(f"Current Work Status: {summary.work_status.current_work_status}")
    if summary.work_status.work_restrictions:
        lines.append("Work Restrictions:")
        for restriction in summary.work_status.work_restrictions[:5]:
            lines.append(f"• {restriction}")
    if summary.work_status.restriction_duration:
        lines.append(f"Restriction Duration: {summary.work_status.restriction_duration}")
    if summary.work_status.return_to_work_date:
        lines.append(f"Return to Work Date: {summary.work_status.return_to_work_date}")
    lines.append("")
    
    # Critical Findings
    if summary.critical_findings:
        lines.append("🚨 CRITICAL FINDINGS")
        lines.append("-" * 50)
        for finding in summary.critical_findings[:8]:
            lines.append(f"• {finding}")
    
    return "\n".join(lines)


def create_fallback_formal_medical_summary(doc_type: str, fallback_date: str) -> FormalMedicalLongSummary:
    """Create a fallback formal medical long summary when extraction fails."""
    return FormalMedicalLongSummary(
        content_type="formal_medical",
        report_type=doc_type,
        report_date=fallback_date
    )


# ============================================================================
# IMAGING LONG SUMMARY MODELS
# ============================================================================

class ImagingAuthorInfo(BaseModel):
    """Author/signature information for imaging reports - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the report (physical or electronic signature). Must be the actual signer - NOT referring physicians, ordering providers, technologists, or other officials mentioned in the document.")


class ImagingRadiologistInfo(BaseModel):
    """Radiologist information"""
    name: str = Field(default="", description="Radiologist's name")
    credentials: str = Field(default="", description="Credentials (MD, DO, etc.)")
    specialty: str = Field(default="Radiology", description="Specialty")


class ImagingDoctorInfo(BaseModel):
    """Doctor information for imaging reports"""
    name: str = Field(default="", description="Doctor's name")
    title: str = Field(default="", description="Title or credentials")
    role: str = Field(default="", description="Role (radiologist, referring, ordering)")


class ImagingOverview(BaseModel):
    """Imaging overview section (Field 1)"""
    document_type: str = Field(default="", description="Type of imaging document")
    exam_date: str = Field(default="", description="Date of the imaging exam")
    exam_type: str = Field(default="", description="Type of exam (MRI, CT, X-ray, Ultrasound)")
    radiologist: str = Field(default="", description="Radiologist name")
    imaging_center: str = Field(default="", description="Imaging center/facility")
    referring_physician: str = Field(default="", description="Referring physician name")
    author: ImagingAuthorInfo = Field(default_factory=ImagingAuthorInfo, description="Author/signature info")


class ImagingPatientInfo(BaseModel):
    """Patient information section"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    date_of_injury: str = Field(default="", description="Date of injury")
    employer: str = Field(default="", description="Employer name")
    all_doctors_involved: List[ImagingDoctorInfo] = Field(default_factory=list, description="All doctors mentioned")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class ImagingClinicalIndication(BaseModel):
    """Clinical indication section (Field 2)"""
    clinical_indication: str = Field(default="", description="Reason for the imaging study")
    clinical_history: str = Field(default="", description="Relevant clinical history")
    chief_complaint: str = Field(default="", description="Chief complaint")
    specific_questions: str = Field(default="", description="Specific clinical questions")


class ImagingTechnicalDetails(BaseModel):
    """Technical details section (Field 3)"""
    study_type: str = Field(default="", description="Type of study")
    body_part_imaged: str = Field(default="", description="Body part being imaged")
    laterality: str = Field(default="", description="Left, right, or bilateral")
    contrast_used: str = Field(default="", description="Whether contrast was used (with/without)")
    contrast_type: str = Field(default="", description="Type of contrast if used")
    prior_studies_available: str = Field(default="", description="Prior studies available for comparison")
    technical_quality: str = Field(default="", description="Technical quality of the study")
    limitations: str = Field(default="", description="Any technical limitations")


class ImagingPrimaryFinding(BaseModel):
    """Primary finding details"""
    description: str = Field(default="", description="Description of primary finding")
    location: str = Field(default="", description="Anatomical location")
    size: str = Field(default="", description="Size/dimensions")
    characteristics: str = Field(default="", description="Imaging characteristics")
    acuity: str = Field(default="", description="Acuity (acute, subacute, chronic)")


class ImagingKeyFindings(BaseModel):
    """Key findings section (Field 4)"""
    primary_finding: ImagingPrimaryFinding = Field(default_factory=ImagingPrimaryFinding, description="Primary finding")
    secondary_findings: List[str] = Field(default_factory=list, description="Secondary findings")
    normal_findings: List[str] = Field(default_factory=list, description="Normal findings (up to 5)")


class ImagingImpression(BaseModel):
    """Impression and conclusion section (Field 5)"""
    overall_impression: str = Field(default="", description="Radiologist's overall impression")
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    final_diagnostic_statement: str = Field(default="", description="Final diagnostic statement")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Differential diagnoses (up to 3)")
    clinical_correlation: str = Field(default="", description="Clinical correlation statement")


class ImagingRecommendations(BaseModel):
    """Recommendations and follow-up section (Field 6)"""
    follow_up_recommended: str = Field(default="", description="Whether follow-up is recommended")
    follow_up_modality: str = Field(default="", description="Recommended follow-up imaging modality")
    follow_up_timing: str = Field(default="", description="Timing for follow-up")
    clinical_correlation_needed: str = Field(default="", description="Whether clinical correlation is needed")
    specialist_consultation: str = Field(default="", description="Specialist consultation recommendation")


class ImagingLongSummary(BaseModel):
    """Complete structured imaging long summary"""
    content_type: Literal["imaging"] = Field(default="imaging", description="Type of content")
    imaging_overview: ImagingOverview = Field(default_factory=ImagingOverview, description="Imaging overview section")
    patient_info: ImagingPatientInfo = Field(default_factory=ImagingPatientInfo, description="Patient information")
    clinical_indication: ImagingClinicalIndication = Field(default_factory=ImagingClinicalIndication, description="Clinical indication")
    technical_details: ImagingTechnicalDetails = Field(default_factory=ImagingTechnicalDetails, description="Technical details")
    key_findings: ImagingKeyFindings = Field(default_factory=ImagingKeyFindings, description="Key findings")
    impression: ImagingImpression = Field(default_factory=ImagingImpression, description="Impression and conclusion")
    recommendations: ImagingRecommendations = Field(default_factory=ImagingRecommendations, description="Recommendations and follow-up")


def format_imaging_long_summary(summary: ImagingLongSummary) -> str:
    """Format ImagingLongSummary Pydantic model to readable text format."""
    lines = []
    
    # Imaging Overview
    lines.append("📋 IMAGING OVERVIEW")
    lines.append("-" * 50)
    if summary.imaging_overview.document_type:
        lines.append(f"Document Type: {summary.imaging_overview.document_type}")
    if summary.imaging_overview.exam_date:
        lines.append(f"Exam Date: {summary.imaging_overview.exam_date}")
    if summary.imaging_overview.exam_type:
        lines.append(f"Exam Type: {summary.imaging_overview.exam_type}")
    if summary.imaging_overview.radiologist:
        lines.append(f"Radiologist: {summary.imaging_overview.radiologist}")
    if summary.imaging_overview.imaging_center:
        lines.append(f"Imaging Center: {summary.imaging_overview.imaging_center}")
    if summary.imaging_overview.referring_physician:
        lines.append(f"Referring Physician: {summary.imaging_overview.referring_physician}")
    if summary.imaging_overview.author.signature:
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.imaging_overview.author.signature}")
    lines.append("")
    
    # Patient Information
    lines.append("👤 PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"Name: {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"Date of Birth: {summary.patient_info.date_of_birth}")
    if summary.patient_info.claim_number:
        lines.append(f"Claim Number: {summary.patient_info.claim_number}")
    if summary.patient_info.date_of_injury:
        lines.append(f"Date of Injury: {summary.patient_info.date_of_injury}")
    if summary.patient_info.employer:
        lines.append(f"Employer: {summary.patient_info.employer}")
    if summary.patient_info.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.patient_info.all_doctors_involved[:5]:
            role_info = f" ({doctor.role})" if doctor.role else ""
            title_info = f", {doctor.title}" if doctor.title else ""
            lines.append(f"• {doctor.name}{title_info}{role_info}")
    lines.append("")
    
    # Clinical Indication
    lines.append("🎯 CLINICAL INDICATION")
    lines.append("-" * 50)
    if summary.clinical_indication.clinical_indication:
        lines.append(f"Clinical Indication: {summary.clinical_indication.clinical_indication}")
    if summary.clinical_indication.clinical_history:
        lines.append(f"Clinical History: {summary.clinical_indication.clinical_history}")
    if summary.clinical_indication.chief_complaint:
        lines.append(f"Chief Complaint: {summary.clinical_indication.chief_complaint}")
    if summary.clinical_indication.specific_questions:
        lines.append(f"Specific Questions: {summary.clinical_indication.specific_questions}")
    lines.append("")
    
    # Technical Details
    lines.append("🔧 TECHNICAL DETAILS")
    lines.append("-" * 50)
    if summary.technical_details.study_type:
        lines.append(f"Study Type: {summary.technical_details.study_type}")
    if summary.technical_details.body_part_imaged:
        lines.append(f"Body Part Imaged: {summary.technical_details.body_part_imaged}")
    if summary.technical_details.laterality:
        lines.append(f"Laterality: {summary.technical_details.laterality}")
    if summary.technical_details.contrast_used:
        lines.append(f"Contrast Used: {summary.technical_details.contrast_used}")
    if summary.technical_details.contrast_type:
        lines.append(f"Contrast Type: {summary.technical_details.contrast_type}")
    if summary.technical_details.prior_studies_available:
        lines.append(f"Prior Studies Available: {summary.technical_details.prior_studies_available}")
    if summary.technical_details.technical_quality:
        lines.append(f"Technical Quality: {summary.technical_details.technical_quality}")
    if summary.technical_details.limitations:
        lines.append(f"Limitations: {summary.technical_details.limitations}")
    lines.append("")
    
    # Key Findings
    lines.append("📊 KEY FINDINGS")
    lines.append("-" * 50)
    pf = summary.key_findings.primary_finding
    has_primary = any([pf.description, pf.location, pf.size, pf.characteristics, pf.acuity])
    if has_primary:
        lines.append("Primary Finding:")
        if pf.description:
            lines.append(f"• Description: {pf.description}")
        if pf.location:
            lines.append(f"• Location: {pf.location}")
        if pf.size:
            lines.append(f"• Size: {pf.size}")
        if pf.characteristics:
            lines.append(f"• Characteristics: {pf.characteristics}")
        if pf.acuity:
            lines.append(f"• Acuity: {pf.acuity}")
    if summary.key_findings.secondary_findings:
        lines.append("")
        lines.append("Secondary Findings:")
        for finding in summary.key_findings.secondary_findings[:5]:
            lines.append(f"• {finding}")
    if summary.key_findings.normal_findings:
        lines.append("")
        lines.append("Normal Findings:")
        for finding in summary.key_findings.normal_findings[:5]:
            lines.append(f"• {finding}")
    lines.append("")
    
    # Impression & Conclusion
    lines.append("💡 IMPRESSION & CONCLUSION")
    lines.append("-" * 50)
    if summary.impression.overall_impression:
        lines.append(f"Overall Impression: {summary.impression.overall_impression}")
    if summary.impression.primary_diagnosis:
        lines.append(f"Primary Diagnosis: {summary.impression.primary_diagnosis}")
    if summary.impression.final_diagnostic_statement:
        lines.append(f"Final Diagnostic Statement: {summary.impression.final_diagnostic_statement}")
    if summary.impression.differential_diagnoses:
        lines.append("")
        lines.append("Differential Diagnoses:")
        for dx in summary.impression.differential_diagnoses[:3]:
            lines.append(f"• {dx}")
    if summary.impression.clinical_correlation:
        lines.append(f"Clinical Correlation: {summary.impression.clinical_correlation}")
    lines.append("")
    
    # Recommendations & Follow-Up
    lines.append("📋 RECOMMENDATIONS & FOLLOW-UP")
    lines.append("-" * 50)
    if summary.recommendations.follow_up_recommended:
        lines.append(f"Follow-up Recommended: {summary.recommendations.follow_up_recommended}")
    if summary.recommendations.follow_up_modality:
        lines.append(f"Follow-up Modality: {summary.recommendations.follow_up_modality}")
    if summary.recommendations.follow_up_timing:
        lines.append(f"Follow-up Timing: {summary.recommendations.follow_up_timing}")
    if summary.recommendations.clinical_correlation_needed:
        lines.append(f"Clinical Correlation Needed: {summary.recommendations.clinical_correlation_needed}")
    if summary.recommendations.specialist_consultation:
        lines.append(f"Specialist Consultation: {summary.recommendations.specialist_consultation}")
    
    return "\n".join(lines)


def create_fallback_imaging_summary(doc_type: str, fallback_date: str) -> ImagingLongSummary:
    """Create a fallback imaging long summary when extraction fails."""
    return ImagingLongSummary(
        content_type="imaging",
        imaging_overview=ImagingOverview(
            document_type=doc_type,
            exam_date=fallback_date,
            exam_type=doc_type
        ),
        technical_details=ImagingTechnicalDetails(
            study_type=doc_type
        )
    )


# ============================================================================
# QME/AME/IME LONG SUMMARY MODELS
# ============================================================================

class QMEAuthorInfo(BaseModel):
    """Author/signature information for QME reports - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the QME report (physical or electronic signature). Must be the actual signer/evaluating physician who signed - NOT providers, claim adjusters, requesting physicians, defense attorneys, applicant attorneys, or other officials mentioned in the document.")


class QMEDoctorInfo(BaseModel):
    """Doctor information for QME reports"""
    name: str = Field(default="", description="Doctor's full name")
    credentials: str = Field(default="", description="Credentials (MD, DO, DC, etc.)")
    role: str = Field(default="", description="Role (evaluating, treating, consulting)")


class QMEPatientInfo(BaseModel):
    """Patient information section for QME reports"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    claim_number: Optional[str] = Field(default=None, description="Claim number if present. If multiple, join with comma.")
    date_of_injury: str = Field(default="", description="Date of injury")
    employer: str = Field(default="", description="Employer name")

    @field_validator('claim_number', mode='before')
    @classmethod
    def normalize_claim_number_field(cls, v: Any) -> Optional[str]:
        return normalize_claim_number(v)


class QMEReportDetails(BaseModel):
    """Report details section for QME reports"""
    report_type: str = Field(default="", description="Type of report (QME/AME/IME)")
    report_date: str = Field(default="", description="Date of the report")
    evaluating_physician: str = Field(default="", description="Name of evaluating physician")
    author: QMEAuthorInfo = Field(default_factory=QMEAuthorInfo, description="Author/signature info")
    all_doctors_involved: List[QMEDoctorInfo] = Field(default_factory=list, description="All doctors mentioned in report")


class QMEDiagnosis(BaseModel):
    """Diagnosis section for QME reports"""
    primary_diagnosis: str = Field(default="", description="Primary diagnosis")
    icd_10_codes: List[str] = Field(default_factory=list, description="ICD-10 codes")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Secondary diagnoses")
    body_parts_affected: List[str] = Field(default_factory=list, description="Body parts affected")


class QMEPhysicalExamFindings(BaseModel):
    """Physical examination findings section"""
    general_findings: List[str] = Field(default_factory=list, description="General clinical findings")
    range_of_motion: List[str] = Field(default_factory=list, description="Range of motion measurements")
    strength_testing: List[str] = Field(default_factory=list, description="Strength/motor testing results")
    sensory_findings: List[str] = Field(default_factory=list, description="Sensory examination findings")
    special_tests: List[str] = Field(default_factory=list, description="Special tests performed and results")


class QMEClinicalStatus(BaseModel):
    """Clinical status section"""
    current_condition: str = Field(default="", description="Current clinical condition")
    pain_level: str = Field(default="", description="Pain level/score if documented")
    functional_limitations: List[str] = Field(default_factory=list, description="Functional limitations")
    subjective_complaints: List[str] = Field(default_factory=list, description="Patient's subjective complaints")


class QMEMedications(BaseModel):
    """Medications section for QME reports"""
    current_medications: List[str] = Field(default_factory=list, description="Current medications with dosages")
    previous_medications: List[str] = Field(default_factory=list, description="Previous medications")
    future_medications: List[str] = Field(default_factory=list, description="Recommended future medications")


class QMEMedicalLegalConclusions(BaseModel):
    """Medical-legal conclusions section - CRITICAL for QME"""
    mmi_status: str = Field(default="", description="Maximum Medical Improvement status")
    mmi_date: str = Field(default="", description="Date of MMI if reached")
    wpi_rating: str = Field(default="", description="Whole Person Impairment rating/percentage")
    apportionment: str = Field(default="", description="Apportionment determination")
    work_restrictions: List[str] = Field(default_factory=list, description="Work restrictions")
    work_status: str = Field(default="", description="Current work status")
    causation_opinion: str = Field(default="", description="Causation opinion")
    future_medical_care: str = Field(default="", description="Need for future medical care")


class QMERecommendations(BaseModel):
    """Recommendations section for QME reports"""
    treatment_recommendations: List[str] = Field(default_factory=list, description="Treatment recommendations")
    diagnostic_recommendations: List[str] = Field(default_factory=list, description="Diagnostic test recommendations")
    specialist_referrals: List[str] = Field(default_factory=list, description="Specialist referral recommendations")
    follow_up: str = Field(default="", description="Follow-up recommendations")


class QMELongSummary(BaseModel):
    """Complete structured QME/AME/IME long summary"""
    content_type: Literal["qme"] = Field(default="qme", description="Type of content")
    patient_info: QMEPatientInfo = Field(default_factory=QMEPatientInfo, description="Patient information")
    report_details: QMEReportDetails = Field(default_factory=QMEReportDetails, description="Report details")
    diagnosis: QMEDiagnosis = Field(default_factory=QMEDiagnosis, description="Diagnosis information")
    physical_exam_findings: QMEPhysicalExamFindings = Field(default_factory=QMEPhysicalExamFindings, description="Physical exam findings")
    clinical_status: QMEClinicalStatus = Field(default_factory=QMEClinicalStatus, description="Clinical status")
    medications: QMEMedications = Field(default_factory=QMEMedications, description="Medications")
    medical_legal_conclusions: QMEMedicalLegalConclusions = Field(default_factory=QMEMedicalLegalConclusions, description="Medical-legal conclusions")
    recommendations: QMERecommendations = Field(default_factory=QMERecommendations, description="Recommendations")


def format_qme_long_summary(summary: QMELongSummary) -> str:
    """Format QMELongSummary Pydantic model to readable text format."""
    lines = []
    
    # Patient Information
    lines.append("## PATIENT INFORMATION")
    lines.append("-" * 50)
    if summary.patient_info.name:
        lines.append(f"- **Name:** {summary.patient_info.name}")
    if summary.patient_info.date_of_birth:
        lines.append(f"- **Date of Birth:** {summary.patient_info.date_of_birth}")
    if summary.patient_info.claim_number:
        lines.append(f"- **Claim Number:** {summary.patient_info.claim_number}")
    if summary.patient_info.date_of_injury:
        lines.append(f"- **Date of Injury:** {summary.patient_info.date_of_injury}")
    if summary.patient_info.employer:
        lines.append(f"- **Employer:** {summary.patient_info.employer}")
    lines.append("")
    
    # Report Details
    lines.append("## REPORT DETAILS")
    lines.append("-" * 50)
    if summary.report_details.report_type:
        lines.append(f"- **Report Type:** {summary.report_details.report_type}")
    if summary.report_details.report_date:
        lines.append(f"- **Report Date:** {summary.report_details.report_date}")
    if summary.report_details.evaluating_physician:
        lines.append(f"- **Evaluating Physician:** {summary.report_details.evaluating_physician}")
    if summary.report_details.author.signature:
        lines.append("")
        lines.append("Author:")
        lines.append(f"• Signature: {summary.report_details.author.signature}")
    lines.append("")
    
    # All Doctors Involved
    if summary.report_details.all_doctors_involved:
        lines.append("## All Doctors Involved")
        lines.append("-" * 50)
        for doctor in summary.report_details.all_doctors_involved[:10]:
            creds = f", {doctor.credentials}" if doctor.credentials else ""
            role = f" ({doctor.role})" if doctor.role else ""
            lines.append(f"• {doctor.name}{creds}{role}")
        lines.append("")
    
    # Diagnosis
    lines.append("## DIAGNOSIS")
    lines.append("-" * 50)
    if summary.diagnosis.primary_diagnosis:
        lines.append(f"- **Primary Diagnosis:** {summary.diagnosis.primary_diagnosis}")
    if summary.diagnosis.icd_10_codes:
        lines.append(f"- **ICD-10 Codes:** {', '.join(summary.diagnosis.icd_10_codes[:5])}")
    if summary.diagnosis.secondary_diagnoses:
        lines.append("- **Secondary Diagnoses:**")
        for dx in summary.diagnosis.secondary_diagnoses[:5]:
            lines.append(f"  • {dx}")
    if summary.diagnosis.body_parts_affected:
        lines.append(f"- **Body Parts Affected:** {', '.join(summary.diagnosis.body_parts_affected[:5])}")
    lines.append("")
    
    # Physical Examination Findings
    lines.append("## PHYSICAL EXAMINATION FINDINGS")
    lines.append("-" * 50)
    if summary.physical_exam_findings.general_findings:
        lines.append("**General Findings:**")
        for finding in summary.physical_exam_findings.general_findings[:5]:
            lines.append(f"• {finding}")
    if summary.physical_exam_findings.range_of_motion:
        lines.append("**Range of Motion:**")
        for rom in summary.physical_exam_findings.range_of_motion[:5]:
            lines.append(f"• {rom}")
    if summary.physical_exam_findings.strength_testing:
        lines.append("**Strength Testing:**")
        for strength in summary.physical_exam_findings.strength_testing[:5]:
            lines.append(f"• {strength}")
    if summary.physical_exam_findings.sensory_findings:
        lines.append("**Sensory Findings:**")
        for sensory in summary.physical_exam_findings.sensory_findings[:5]:
            lines.append(f"• {sensory}")
    if summary.physical_exam_findings.special_tests:
        lines.append("**Special Tests:**")
        for test in summary.physical_exam_findings.special_tests[:5]:
            lines.append(f"• {test}")
    lines.append("")
    
    # Clinical Status
    lines.append("## CLINICAL STATUS")
    lines.append("-" * 50)
    if summary.clinical_status.current_condition:
        lines.append(f"- **Current Condition:** {summary.clinical_status.current_condition}")
    if summary.clinical_status.pain_level:
        lines.append(f"- **Pain Level:** {summary.clinical_status.pain_level}")
    if summary.clinical_status.functional_limitations:
        lines.append("- **Functional Limitations:**")
        for limitation in summary.clinical_status.functional_limitations[:5]:
            lines.append(f"  • {limitation}")
    if summary.clinical_status.subjective_complaints:
        lines.append("- **Subjective Complaints:**")
        for complaint in summary.clinical_status.subjective_complaints[:5]:
            lines.append(f"  • {complaint}")
    lines.append("")
    
    # Medications
    lines.append("## MEDICATIONS")
    lines.append("-" * 50)
    if summary.medications.current_medications:
        lines.append("- **Current Medications:**")
        for med in summary.medications.current_medications[:10]:
            lines.append(f"  • {med}")
    if summary.medications.previous_medications:
        lines.append("- **Previous Medications:**")
        for med in summary.medications.previous_medications[:5]:
            lines.append(f"  • {med}")
    if summary.medications.future_medications:
        lines.append("- **Future/Recommended Medications:**")
        for med in summary.medications.future_medications[:5]:
            lines.append(f"  • {med}")
    lines.append("")
    
    # Medical-Legal Conclusions (CRITICAL)
    lines.append("## MEDICAL-LEGAL CONCLUSIONS")
    lines.append("-" * 50)
    if summary.medical_legal_conclusions.mmi_status:
        lines.append(f"- **MMI Status:** {summary.medical_legal_conclusions.mmi_status}")
    if summary.medical_legal_conclusions.mmi_date:
        lines.append(f"- **MMI Date:** {summary.medical_legal_conclusions.mmi_date}")
    if summary.medical_legal_conclusions.wpi_rating:
        lines.append(f"- **WPI Rating:** {summary.medical_legal_conclusions.wpi_rating}")
    if summary.medical_legal_conclusions.apportionment:
        lines.append(f"- **Apportionment:** {summary.medical_legal_conclusions.apportionment}")
    if summary.medical_legal_conclusions.work_status:
        lines.append(f"- **Work Status:** {summary.medical_legal_conclusions.work_status}")
    if summary.medical_legal_conclusions.work_restrictions:
        lines.append("- **Work Restrictions:**")
        for restriction in summary.medical_legal_conclusions.work_restrictions[:5]:
            lines.append(f"  • {restriction}")
    if summary.medical_legal_conclusions.causation_opinion:
        lines.append(f"- **Causation Opinion:** {summary.medical_legal_conclusions.causation_opinion}")
    if summary.medical_legal_conclusions.future_medical_care:
        lines.append(f"- **Future Medical Care:** {summary.medical_legal_conclusions.future_medical_care}")
    lines.append("")
    
    # Recommendations
    lines.append("## RECOMMENDATIONS")
    lines.append("-" * 50)
    if summary.recommendations.treatment_recommendations:
        lines.append("- **Treatment Recommendations:**")
        for rec in summary.recommendations.treatment_recommendations[:5]:
            lines.append(f"  • {rec}")
    if summary.recommendations.diagnostic_recommendations:
        lines.append("- **Diagnostic Recommendations:**")
        for rec in summary.recommendations.diagnostic_recommendations[:5]:
            lines.append(f"  • {rec}")
    if summary.recommendations.specialist_referrals:
        lines.append("- **Specialist Referrals:**")
        for ref in summary.recommendations.specialist_referrals[:3]:
            lines.append(f"  • {ref}")
    if summary.recommendations.follow_up:
        lines.append(f"- **Follow-Up:** {summary.recommendations.follow_up}")
    
    return "\n".join(lines)


def create_fallback_qme_summary(doc_type: str, fallback_date: str) -> QMELongSummary:
    """Create a fallback QME long summary when extraction fails."""
    return QMELongSummary(
        content_type="qme",
        report_details=QMEReportDetails(
            report_type=doc_type,
            report_date=fallback_date
        )
    )


# ============================================================================
# UR/IMR DECISION DOCUMENT LONG SUMMARY MODELS
# ============================================================================

class URAuthorInfo(BaseModel):
    """Author/signature information for UR decision documents - ONLY the person who signed the report"""
    signature: str = Field(default="", description="Name/title of the person who SIGNED the UR decision document (physical or electronic signature). Must be the actual signer/reviewing physician who signed - NOT requesting providers, claim adjusters, utilization review organizations, insurance representatives, or other officials mentioned in the document.")


class URDoctorInfo(BaseModel):
    """Doctor information for UR documents"""
    name: str = Field(default="", description="Doctor's full name")
    title: str = Field(default="", description="Title or credentials (MD, DO, etc.)")
    role: str = Field(default="", description="Role (requesting, reviewing, consulting)")


class URDocumentOverview(BaseModel):
    """Document overview section for UR decision documents"""
    document_type: str = Field(default="", description="Type of decision document")
    document_date: str = Field(default="", description="Date of the document")
    decision_date: str = Field(default="", description="Date of the decision")
    document_id: str = Field(default="", description="Document identifier")
    claim_case_number: Optional[str] = Field(default=None, description="Claim or case number")
    jurisdiction: str = Field(default="", description="Jurisdiction")
    author: URAuthorInfo = Field(default_factory=URAuthorInfo, description="Author/signature info")


class URPatientInfo(BaseModel):
    """Patient information for UR documents"""
    name: str = Field(default="", description="Patient's full name")
    date_of_birth: str = Field(default="", description="Patient's date of birth")
    member_id: str = Field(default="", description="Member ID")


class URRequestingProvider(BaseModel):
    """Requesting provider information"""
    name: str = Field(default="", description="Provider's name")
    specialty: str = Field(default="", description="Provider's specialty")
    npi: str = Field(default="", description="NPI number")


class URReviewingEntity(BaseModel):
    """Reviewing entity information"""
    name: str = Field(default="", description="Reviewing entity name")
    reviewer: str = Field(default="", description="Reviewer name")
    credentials: str = Field(default="", description="Reviewer credentials")


class URPartiesInvolved(BaseModel):
    """All parties involved in the UR decision"""
    patient: URPatientInfo = Field(default_factory=URPatientInfo, description="Patient information")
    requesting_provider: URRequestingProvider = Field(default_factory=URRequestingProvider, description="Requesting provider")
    reviewing_entity: URReviewingEntity = Field(default_factory=URReviewingEntity, description="Reviewing entity")
    claims_administrator: str = Field(default="", description="Claims administrator name")
    all_doctors_involved: List[URDoctorInfo] = Field(default_factory=list, description="All doctors mentioned")


class URRequestDetails(BaseModel):
    """Request details section"""
    date_of_service_requested: str = Field(default="", description="Date of service requested")
    request_received: str = Field(default="", description="Date request was received")
    requested_services: List[str] = Field(default_factory=list, description="List of requested services with CPT codes")
    clinical_reason: str = Field(default="", description="Clinical reason for request")


class URPartialDecision(BaseModel):
    """Partial decision breakdown item"""
    service: str = Field(default="", description="Service name")
    decision: str = Field(default="", description="Decision (approved/denied)")
    quantity: str = Field(default="", description="Quantity approved/denied")


class URDecisionOutcome(BaseModel):
    """Decision outcome section"""
    overall_decision: str = Field(default="", description="Overall decision (APPROVED/DENIED/PARTIALLY APPROVED/PENDING)")
    decision_details: str = Field(default="", description="Details of the decision")
    partial_decision_breakdown: List[URPartialDecision] = Field(default_factory=list, description="Breakdown for partial decisions")
    effective_dates: str = Field(default="", description="Effective start/end dates")


class URMedicalNecessity(BaseModel):
    """Medical necessity determination section"""
    medical_necessity: str = Field(default="", description="Medical necessity determination")
    criteria_applied: str = Field(default="", description="Criteria applied (ODG, MTUS, ACOEM, etc.)")
    clinical_rationale: str = Field(default="", description="Clinical rationale for decision")
    guidelines_referenced: List[str] = Field(default_factory=list, description="Guidelines referenced")


class URReviewerAnalysis(BaseModel):
    """Reviewer analysis section"""
    clinical_summary_reviewed: str = Field(default="", description="Clinical summary that was reviewed")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from review")
    documentation_gaps: List[str] = Field(default_factory=list, description="Documentation gaps noted")


class URAppealInformation(BaseModel):
    """Appeal information section"""
    appeal_deadline: str = Field(default="", description="Deadline for appeal")
    appeal_procedures: str = Field(default="", description="Appeal procedures")
    required_documentation: List[str] = Field(default_factory=list, description="Required documentation for appeal")
    timeframe_for_response: str = Field(default="", description="Timeframe for response")


class URLongSummary(BaseModel):
    """Complete structured UR/IMR decision document long summary"""
    content_type: Literal["ur_decision"] = Field(default="ur_decision", description="Type of content")
    document_overview: URDocumentOverview = Field(default_factory=URDocumentOverview, description="Document overview")
    parties_involved: URPartiesInvolved = Field(default_factory=URPartiesInvolved, description="Parties involved")
    request_details: URRequestDetails = Field(default_factory=URRequestDetails, description="Request details")
    decision_outcome: URDecisionOutcome = Field(default_factory=URDecisionOutcome, description="Decision outcome")
    medical_necessity: URMedicalNecessity = Field(default_factory=URMedicalNecessity, description="Medical necessity determination")
    reviewer_analysis: URReviewerAnalysis = Field(default_factory=URReviewerAnalysis, description="Reviewer analysis")
    appeal_information: URAppealInformation = Field(default_factory=URAppealInformation, description="Appeal information")
    critical_actions_required: List[str] = Field(default_factory=list, description="Critical time-sensitive actions")


def format_ur_long_summary(summary: URLongSummary) -> str:
    """Format URLongSummary Pydantic model to readable text format."""
    lines = []
    
    # Document Overview
    lines.append("📋 DECISION DOCUMENT OVERVIEW")
    lines.append("-" * 50)
    if summary.document_overview.document_type:
        lines.append(f"Document Type: {summary.document_overview.document_type}")
    if summary.document_overview.document_date:
        lines.append(f"Document Date: {summary.document_overview.document_date}")
    if summary.document_overview.decision_date:
        lines.append(f"Decision Date: {summary.document_overview.decision_date}")
    if summary.document_overview.document_id:
        lines.append(f"Document ID: {summary.document_overview.document_id}")
    if summary.document_overview.claim_case_number:
        lines.append(f"Claim/Case Number: {summary.document_overview.claim_case_number}")
    if summary.document_overview.jurisdiction:
        lines.append(f"Jurisdiction: {summary.document_overview.jurisdiction}")
    if summary.document_overview.author.signature:
        lines.append("Author:")
        lines.append(f"• Signature: {summary.document_overview.author.signature}")
    lines.append("")
    
    # Parties Involved
    lines.append("👥 PARTIES INVOLVED")
    lines.append("-" * 50)
    patient = summary.parties_involved.patient
    if patient.name:
        lines.append(f"Patient: {patient.name}")
        if patient.date_of_birth:
            lines.append(f"  DOB: {patient.date_of_birth}")
        if patient.member_id:
            lines.append(f"  Member ID: {patient.member_id}")
    
    req_provider = summary.parties_involved.requesting_provider
    if req_provider.name:
        lines.append(f"Requesting Provider: {req_provider.name}")
        if req_provider.specialty:
            lines.append(f"  Specialty: {req_provider.specialty}")
        if req_provider.npi:
            lines.append(f"  NPI: {req_provider.npi}")
    
    rev_entity = summary.parties_involved.reviewing_entity
    if rev_entity.name:
        lines.append(f"Reviewing Entity: {rev_entity.name}")
        if rev_entity.reviewer:
            lines.append(f"  Reviewer: {rev_entity.reviewer}")
        if rev_entity.credentials:
            lines.append(f"  Credentials: {rev_entity.credentials}")
    
    if summary.parties_involved.claims_administrator:
        lines.append(f"Claims Administrator: {summary.parties_involved.claims_administrator}")
    
    if summary.parties_involved.all_doctors_involved:
        lines.append("")
        lines.append("All Doctors Involved:")
        for doctor in summary.parties_involved.all_doctors_involved[:10]:
            title = f", {doctor.title}" if doctor.title else ""
            role = f" ({doctor.role})" if doctor.role else ""
            lines.append(f"• {doctor.name}{title}{role}")
    lines.append("")
    
    # Request Details
    lines.append("📋 REQUEST DETAILS")
    lines.append("-" * 50)
    if summary.request_details.date_of_service_requested:
        lines.append(f"Date of Service Requested: {summary.request_details.date_of_service_requested}")
    if summary.request_details.request_received:
        lines.append(f"Request Received: {summary.request_details.request_received}")
    if summary.request_details.requested_services:
        lines.append("Requested Services:")
        for service in summary.request_details.requested_services[:10]:
            lines.append(f"• {service}")
    if summary.request_details.clinical_reason:
        lines.append(f"Clinical Reason: {summary.request_details.clinical_reason}")
    lines.append("")
    
    # Decision Outcome
    lines.append("⚖️ DECISION OUTCOME")
    lines.append("-" * 50)
    if summary.decision_outcome.overall_decision:
        lines.append(f"Overall Decision: {summary.decision_outcome.overall_decision}")
    if summary.decision_outcome.decision_details:
        lines.append(f"Decision Details: {summary.decision_outcome.decision_details}")
    if summary.decision_outcome.partial_decision_breakdown:
        lines.append("Partial Decision Breakdown:")
        for item in summary.decision_outcome.partial_decision_breakdown[:5]:
            qty = f" ({item.quantity})" if item.quantity else ""
            lines.append(f"• {item.service}: {item.decision}{qty}")
    if summary.decision_outcome.effective_dates:
        lines.append(f"Effective Dates: {summary.decision_outcome.effective_dates}")
    lines.append("")
    
    # Medical Necessity
    lines.append("🏥 MEDICAL NECESSITY DETERMINATION")
    lines.append("-" * 50)
    if summary.medical_necessity.medical_necessity:
        lines.append(f"Medical Necessity: {summary.medical_necessity.medical_necessity}")
    if summary.medical_necessity.criteria_applied:
        lines.append(f"Criteria Applied: {summary.medical_necessity.criteria_applied}")
    if summary.medical_necessity.clinical_rationale:
        lines.append(f"Clinical Rationale: {summary.medical_necessity.clinical_rationale}")
    if summary.medical_necessity.guidelines_referenced:
        lines.append("Guidelines Referenced:")
        for guideline in summary.medical_necessity.guidelines_referenced[:5]:
            lines.append(f"• {guideline}")
    lines.append("")
    
    # Reviewer Analysis
    lines.append("🔍 REVIEWER ANALYSIS")
    lines.append("-" * 50)
    if summary.reviewer_analysis.clinical_summary_reviewed:
        lines.append(f"Clinical Summary Reviewed: {summary.reviewer_analysis.clinical_summary_reviewed}")
    if summary.reviewer_analysis.key_findings:
        lines.append("Key Findings:")
        for finding in summary.reviewer_analysis.key_findings[:5]:
            lines.append(f"• {finding}")
    if summary.reviewer_analysis.documentation_gaps:
        lines.append("Documentation Gaps:")
        for gap in summary.reviewer_analysis.documentation_gaps[:3]:
            lines.append(f"• {gap}")
    lines.append("")
    
    # Appeal Information
    lines.append("🔄 APPEAL INFORMATION")
    lines.append("-" * 50)
    if summary.appeal_information.appeal_deadline:
        lines.append(f"Appeal Deadline: {summary.appeal_information.appeal_deadline}")
    if summary.appeal_information.appeal_procedures:
        lines.append(f"Appeal Procedures: {summary.appeal_information.appeal_procedures}")
    if summary.appeal_information.required_documentation:
        lines.append("Required Documentation:")
        for doc in summary.appeal_information.required_documentation[:5]:
            lines.append(f"• {doc}")
    if summary.appeal_information.timeframe_for_response:
        lines.append(f"Timeframe for Response: {summary.appeal_information.timeframe_for_response}")
    lines.append("")
    
    # Critical Actions Required
    if summary.critical_actions_required:
        lines.append("🚨 CRITICAL ACTIONS REQUIRED")
        lines.append("-" * 50)
        for action in summary.critical_actions_required[:8]:
            lines.append(f"• {action}")
    
    return "\n".join(lines)


def create_fallback_ur_summary(doc_type: str, fallback_date: str) -> URLongSummary:
    """Create a fallback UR long summary when extraction fails."""
    return URLongSummary(
        content_type="ur_decision",
        document_overview=URDocumentOverview(
            document_type=doc_type,
            document_date=fallback_date
        )
    )
