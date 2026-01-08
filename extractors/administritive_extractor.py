"""
AdministrativeExtractor - Enhanced Extractor for Administrative and Legal Documents
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
Version: 2.0 - With Pydantic Validation for Consistent Output
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
from utils.summary_helpers import ensure_date_and_author, clean_long_summary
from helpers.short_summary_generator import generate_structured_short_summary
from models.long_summary_models import (
    AdminLongSummary,
    DoctorInfo,
    format_admin_long_summary,
    create_fallback_admin_summary
)


logger = logging.getLogger("document_ai")


class AdministrativeExtractor:
    """
    Enhanced Administrative Document extractor with FULL CONTEXT processing and contextual awareness.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different administrative document types
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports Attorney Letters, NCM Notes, Employer Reports, Disability Forms, Legal Correspondence
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.admin_summary_parser = PydanticOutputParser(pydantic_object=AdminLongSummary)
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.admin_type_patterns = {
            'attorney': re.compile(r'\b(attorney|lawyer|counsel|legal|subpoena|deposition)\b', re.IGNORECASE),
            'ncm': re.compile(r'\b(ncm|nurse case manager|case management|utilization review)\b', re.IGNORECASE),
            'employer': re.compile(r'\b(employer|supervisor|incident report|work injury|workers comp)\b', re.IGNORECASE),
            'disability': re.compile(r'\b(disability|claim form|dwc|workers compensation|benefits)\b', re.IGNORECASE),
            'job_analysis': re.compile(r'\b(job analysis|physical demands|work capacity|vocational)\b', re.IGNORECASE),
            'medication': re.compile(r'\b(medication administration|mar|pharmacy|prescription log)\b', re.IGNORECASE),
            'pharmacy': re.compile(r'\b(pharmacy|prescription|drug|medication list|pharmacist)\b', re.IGNORECASE),
            'telemedicine': re.compile(r'\b(telemedicine|telehealth|virtual visit|remote consultation)\b', re.IGNORECASE),
            'legal': re.compile(r'\b(legal correspondence|demand letter|settlement|liability)\b', re.IGNORECASE)
        }
        
        # Administrative patterns
        self.admin_patterns = {
            'claim_numbers': re.compile(r'\b(claim\s*(?:number|no|#)?[:\s]*([A-Z0-9\-]+))', re.IGNORECASE),
            'dates': re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[-]\d{2}[-]\d{2})\b'),
            'deadlines': re.compile(r'\b(deadline|due date|response required by|must respond by)\s*[:]?\s*([^\.]+?)(?=\.|\n|$)', re.IGNORECASE),
            'contact_info': re.compile(r'\b(phone|tel|fax|email|@)\s*[:\-]?\s*([^\s,]+)', re.IGNORECASE)
        }
        
        logger.info("✅ AdministrativeExtractor v2.0 initialized with Pydantic Validation")


    def _detect_admin_type(self, text: str, original_type: str) -> str:
        """
        Auto-detect specific administrative document type based on content patterns.
        Falls back to original type if detection is uncertain.
        """
        if original_type and original_type.lower() not in ['unknown', 'auto', '']:
            return original_type
        
        type_scores = {}
        
        for admin_type, pattern in self.admin_type_patterns.items():
            matches = pattern.findall(text)
            type_scores[admin_type] = len(matches)
        
        # Boost scores for administrative-specific terminology
        if self.admin_patterns['claim_numbers'].search(text):
            for admin_type in ['disability', 'ncm', 'employer']:
                type_scores[admin_type] = type_scores.get(admin_type, 0) + 2
        
        if self.admin_patterns['deadlines'].search(text):
            for admin_type in ['attorney', 'legal']:
                type_scores[admin_type] = type_scores.get(admin_type, 0) + 1
        
        # Get highest scoring type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:  # Only return if we found matches
                detected_type = best_type[0].replace('_', ' ').title()
                logger.info(f"🔍 Auto-detected administrative type: {detected_type} (score: {best_type[1]})")
                return detected_type
        
        logger.info(f"🔍 Could not auto-detect administrative type, using: {original_type}")
        return original_type or "Administrative Document"


    # administritive_extractor.py - UPDATED with dual-context priority

    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
    ) -> Dict:
        """
        Extract Administrative Document data with FULL CONTEXT.
        
        Args:
            text: Complete document text (full OCR extraction)
            raw_text: Accurate summarized context from Document AI Summarizer (PRIMARY SOURCE)
            doc_type: Document type (Attorney, NCM, Employer, Disability, etc.)
            fallback_date: Fallback date if not found
        
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("📋 STARTING ADMINISTRATIVE DOCUMENT EXTRACTION (DUAL-CONTEXT PRIORITY)")
        logger.info("=" * 80)
        
        # Auto-detect specific administrative type if not specified
        detected_type = self._detect_admin_type(text, doc_type)
        logger.info(f"📋 Administrative Type: {detected_type} (original: {doc_type})")
        
        # Check document size
        # logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        # logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        text_length = len(raw_text)
        token_estimate = text_length // 4
        # logger.info(f"📄 PRIMARY source size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
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
        
        # Stage 1.5: Clean the long summary - remove empty fields, placeholders, and instruction text
        long_summary = clean_long_summary(long_summary)
        
        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(raw_text, detected_type, long_summary)
        
        logger.info("=" * 80)
        logger.info("✅ ADMINISTRATIVE DOCUMENT EXTRACTION COMPLETE (DUAL-CONTEXT)")
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
        Uses Pydantic validation for consistent, structured output without hallucination.
        """
        logger.info("🔍 Processing administrative document with DUAL-CONTEXT approach + Pydantic validation...")
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # Get format instructions from Pydantic parser
        format_instructions = self.admin_summary_parser.get_format_instructions()
        
        # Updated System Prompt with DUAL-CONTEXT PRIORITY
        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert administrative and legal documentation specialist analyzing a COMPLETE {doc_type}.

    🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

    You are provided with TWO versions of the document:

    1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
    - This is the MOST ACCURATE, context-aware summary from Google's Document AI foundation model
    - It preserves CRITICAL ADMINISTRATIVE & LEGAL CONTEXT with accurate interpretations
    - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
    - Contains CORRECT administrative conclusions, accurate legal interpretations, proper context
    - **ALWAYS PRIORITIZE information from this source**

    2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
    - Complete OCR text extraction (may have formatting noise, OCR artifacts)
    - Use ONLY to fill in SPECIFIC DETAILS missing from the accurate context
    - Examples of acceptable supplementary use:
        * Exact claim numbers or case numbers in headers/footers
        * Complete contact information (phone, fax, email)
        * Specific dates in document headers
        * Additional party names mentioned
        * Exact deadline wording if more specific in full text
    - **DO NOT let this override the administrative context from the primary source**

    🚨 ABSOLUTE ANTI-FABRICATION RULE (HIGHEST PRIORITY):
    **YOU MUST ONLY EXTRACT AND SUMMARIZE INFORMATION THAT EXISTS IN THE PROVIDED SOURCES.**
    - NEVER generate, infer, assume, or fabricate ANY information
    - If information is NOT explicitly stated in either source → OMIT IT ENTIRELY
    - An incomplete summary is 100x better than a fabricated one
    - Every single piece of information in your output MUST be traceable to the source text

    ⚠️ STRICT ANTI-HALLUCINATION RULES:

    1. **ZERO FABRICATION TOLERANCE**:
    - If a field (e.g., DOB, Claim Number, Deadline) is NOT in either source → LEAVE IT BLANK or OMIT
    - NEVER write "likely", "probably", "typically", "usually" - these indicate fabrication
    - NEVER fill in "standard" or "typical" values - only actual extracted values

    2. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME administrative element:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    3. **LEGAL DEMANDS & REQUIREMENTS PRIORITY**:
    - PRIMARY SOURCE provides accurate legal context and interpretations
    - Use FULL TEXT only for exact legal wording if more specific
    - NEVER change legal interpretation based on full text alone

    3. **DEADLINES & ACTION ITEMS**:
    - PRIMARY SOURCE contains accurate, contextually appropriate deadlines
    - Use FULL TEXT for exact deadline wording only if more specific
    - DO NOT add action items from full text if they contradict primary source

    4. **CONTACT INFORMATION**:
    - FULL TEXT often better for exact contact info (headers/footers)
    - Check FULL TEXT first for phone numbers, addresses, emails
    - Use PRIMARY SOURCE if full text is unclear or missing

    5. **CLAIM NUMBERS & IDENTIFIERS**:
    - if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
    - These are often in headers/footers (better in FULL TEXT)
    - Check FULL TEXT first for exact claim/case numbers
    - Use PRIMARY SOURCE if full text is unclear

    6. **PARTIES & ROLES**:
    - PRIMARY SOURCE provides accurate party relationships and roles
    - Use FULL TEXT only to supplement with additional names if missing
    - DO NOT change party interpretations from full text

    🔍 EXTRACTION WORKFLOW:

    Step 1: Read PRIMARY SOURCE (accurate context) thoroughly for administrative understanding
    Step 2: Extract ALL administrative findings, legal demands, requirements from PRIMARY SOURCE
    Step 3: Check SUPPLEMENTARY SOURCE (full text) ONLY for:
    - Specific details missing from primary (exact claim numbers, contact info)
    - Administrative identifiers in headers/footers
    - Additional party names
    Step 4: Verify no contradictions between sources (if conflict, PRIMARY wins)

    ⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY) (donot include in output, for LLM use only):

    1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
    - If NOT explicitly mentioned in PRIMARY SOURCE, check SUPPLEMENTARY
    - If still not found, return EMPTY string "" or empty list []
    - DO NOT infer, assume, or extrapolate administrative information

    2. **DATES & DEADLINES - EXACT WORDING ONLY**
    - Extract deadlines from PRIMARY SOURCE for accurate context
    - Supplement with exact wording from FULL TEXT if more specific
    - DO NOT interpret or calculate dates

    3. **CONTACT INFORMATION - VERBATIM EXTRACTION**
    - Check FULL TEXT first for exact contact details
    - Extract EXACTLY as written without formatting
    - Use PRIMARY SOURCE if full text unclear

    4. **LEGAL & ADMINISTRATIVE TERMS - PRECISE EXTRACTION**
    - PRIMARY SOURCE for legal context and interpretation
    - FULL TEXT only for exact legal wording if needed
    - DO NOT interpret legal implications

    5. **ACTION ITEMS & REQUIREMENTS - SPECIFIC DETAILS ONLY**
    - PRIMARY SOURCE for accurate requirements context
    - FULL TEXT for specific details if missing
    - DO NOT add typical administrative procedures

    6. **AUTHOR/SIGNER DETECTION**:
    - Check PRIMARY SOURCE first for document author/signer
    - If not clear, scan FULL TEXT signature blocks (usually last pages)
    - Extract name as explicitly signed

    7. **ALL DOCTORS EXTRACTION**:
    - Extract from BOTH sources (primary + supplementary)
    - Deduplicate: If same doctor in both, use primary source version
    - Include all physicians with credentials

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

    EXTRACTION FOCUS - 8 CRITICAL ADMINISTRATIVE CATEGORIES:

    I. DOCUMENT IDENTITY & CONTEXT
    II. PARTIES INVOLVED
    III. KEY DATES & DEADLINES (MOST CRITICAL)
    IV. ADMINISTRATIVE CONTENT & SUBJECT
    V. ACTION ITEMS & REQUIREMENTS
    VI. LEGAL & PROCEDURAL ELEMENTS
    VII. MEDICAL & CLAIM INFORMATION (if applicable)
    VIII. CONTACT & FOLLOW-UP PROCEDURES

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
    Report Date: [extracted date of report]

    Generate the long summary in this EXACT STRUCTURED FORMAT using the DUAL-CONTEXT PRIORITY rules:

    📋 ADMINISTRATIVE DOCUMENT OVERVIEW
    --------------------------------------------------
    Document Type: {doc_type}
    Document Date: [from primary source, supplement if needed]
    Subject: [from primary source]
    Purpose: [from primary source]
    Document ID: [from full text headers/footers first]

    👥 PARTIES INVOLVED
    --------------------------------------------------
    Patient Details: [from primary source, supplement if needed]

    From: [from primary source]
    Organization: [from primary source]
    Title: [from primary source]

    To: [from primary source]
    Organization: [from primary source]

    Author:
    hint: check primary source first, then full text signature block (last pages) if unclear
    • Signature: [extracted name/title; should not be business name or generic title]

    Legal Representation: [from primary source]
    Firm: [from primary source]

    ━━━ CLAIM NUMBER EXTRACTION ━━━
    - Check FULL TEXT headers/footers FIRST for exact claim numbers
    - Scan for patterns: "[Claim #XXXXXXXXX]", "Claim Number:", "WC Claim:"
    - Verify claim number is actual claim (not chart number or other ID)
    - Use PRIMARY SOURCE if full text is unclear

    Claim Number: [from full text headers/footers first, primary source if unclear]

    All Doctors Involved:
    • [extract from BOTH sources, deduplicate, prefer primary source format]

    ━━━ ALL DOCTORS EXTRACTION ━━━
    - Extract ALL physician/doctor names from BOTH sources
    - Deduplicate: If same doctor appears in both, use PRIMARY SOURCE format
    - Include: consulting doctor, referring doctor, treating physician, etc.
    - Include names with credentials (MD, DO, DPM, DC, NP, PA)
    - Extract ONLY actual person names, NOT business names
    - If no doctors found, leave list empty []

    📅 KEY DATES & DEADLINES
    --------------------------------------------------
    [FROM PRIMARY SOURCE for accurate context]
    [Supplement with exact wording from FULL TEXT if more specific]

    Response Deadline: [primary source context, full text for exact wording]
    Hearing Date: [primary source]
    Appointment Date: [primary source]

    Time-Sensitive Requirements:
    • [from primary source, list up to 3, exact wording]

    📄 ADMINISTRATIVE CONTENT
    --------------------------------------------------
    [ALL FROM PRIMARY SOURCE for accurate administrative context]

    Primary Subject: [primary source]
    Key Points: [primary source]
    Current Status: [primary source]
    Incident Details: [primary source, truncate if >200 chars]

    ✅ ACTION ITEMS & REQUIREMENTS
    --------------------------------------------------
    [FROM PRIMARY SOURCE for accurate requirements context]

    Required Responses:
    • [from primary source, list up to 5]

    Documentation Required:
    • [from primary source, list up to 5]

    Specific Actions:
    • [from primary source, list up to 5]

    ⚖️ LEGAL & PROCEDURAL ELEMENTS
    --------------------------------------------------
    [FROM PRIMARY SOURCE for accurate legal context]
    [Use full text for exact legal wording only if more specific]

    Legal Demands:
    • [from primary source, list up to 3]

    Next Steps:
    • [from primary source, list up to 3]

    Consequences of Non-Compliance: [primary source]

    🏥 MEDICAL & CLAIM INFORMATION
    --------------------------------------------------
    [FROM PRIMARY SOURCE primarily]
    [Supplement with exact identifiers from FULL TEXT headers]

    Claim Number: [full text headers first, primary if unclear]
    Case Number: [full text headers first, primary if unclear]
    Work Status: [primary source]
    Disability Information: [primary source]

    Treatment Authorizations:
    • [from primary source, list up to 3]

    📞 CONTACT & FOLLOW-UP
    --------------------------------------------------
    [Check FULL TEXT FIRST for exact contact details]
    [Use PRIMARY SOURCE for contact context if full text unclear]

    Contact Person: [full text for exact info, primary for context]
    Phone: [full text exact, primary if unclear]
    Email: [full text exact, primary if unclear]
    Submission Address: [full text exact, primary if unclear]
    Response Format: [primary source]

    🚨 CRITICAL ADMINISTRATIVE FINDINGS
    --------------------------------------------------
    • [from PRIMARY SOURCE - list up to 8 most actionable/time-sensitive items]

    REMEMBER: 
    1. PRIMARY SOURCE (accurate context) is your MAIN reference for administrative interpretations
    2. Use FULL TEXT only to supplement specific missing details (contact info, claim numbers, exact dates)
    3. NEVER override primary source administrative context with full text

    📋 OUTPUT FORMAT INSTRUCTIONS:
    You MUST output your response as a valid JSON object following this exact schema:
    {format_instructions}

    ⚠️ IMPORTANT JSON RULES:
    - Use empty string "" for text fields with no data (NOT null, NOT "N/A", NOT "unknown")
    - Use empty array [] for list fields with no data
    - Use null ONLY for optional fields like claim_number when not present
    - For all_doctors_involved: each doctor must have "name" (required), "title" (optional), "role" (optional)
    - For signature_type: use "physical" or "electronic" or null if not found
    - For content_type: always use "administrative"
    - For from_party and to_party: nested objects with name, organization, title fields
    - For legal_representation: nested object with representative and firm fields
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        logger.info(f"📄 PRIMARY SOURCE size: {len(raw_text):,} chars")
        logger.info(f"📄 SUPPLEMENTARY size: {len(text):,} chars")
        logger.info("🤖 Invoking LLM for structured administrative summary with Pydantic validation...")
        
        try:
            # Single LLM call with both sources
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "doc_type": doc_type,
                "document_actual_context": raw_text,  # PRIMARY: Accurate summarized context
                "full_document_text": text,           # SUPPLEMENTARY: Full OCR extraction
                "fallback_date": fallback_date,
                "format_instructions": format_instructions
            })
            
            raw_response = result.content.strip()
            logger.info(f"📝 Raw LLM response length: {len(raw_response)} chars")
            
            # Parse and validate with Pydantic
            try:
                parsed_summary = self.admin_summary_parser.parse(raw_response)
                logger.info(f"✅ Pydantic validation successful - content_type: {parsed_summary.content_type}")
                
                # Format the validated Pydantic model back to text format
                long_summary = format_admin_long_summary(parsed_summary)
                
                logger.info(f"⚡ Structured administrative long summary generation completed with Pydantic validation")
                logger.info(f"✅ Generated long summary: {len(long_summary):,} chars")
                logger.info("✅ Context priority maintained: PRIMARY source used for administrative findings")
                
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
                        parsed_summary = AdminLongSummary(**parsed_dict)
                        long_summary = format_admin_long_summary(parsed_summary)
                        logger.info("✅ Successfully recovered with JSON extraction fallback")
                        return long_summary
                except Exception as json_error:
                    logger.warning(f"⚠️ JSON extraction fallback failed: {json_error}")
                
                # Final fallback: use raw response if it looks like formatted text
                if "📋" in raw_response or "ADMINISTRATIVE DOCUMENT OVERVIEW" in raw_response:
                    logger.info("✅ Using raw formatted text response")
                    return raw_response
                
                # Create minimal fallback summary
                fallback = create_fallback_admin_summary(doc_type, fallback_date)
                return format_admin_long_summary(fallback)
            
        except Exception as e:
            logger.error(f"❌ Direct administrative document long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Administrative document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            # Fallback: Generate a minimal summary using Pydantic model
            fallback = create_fallback_admin_summary(doc_type, fallback_date)
            return format_admin_long_summary(fallback)

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
    
    def _create_admin_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback administrative summary directly from long summary"""
        
        # Extract key administrative information using regex patterns
        patterns = {
            'sender': r'From:\s*([^\n]+)',
            'recipient': r'To:\s*([^\n]+)',
            'deadline': r'Response Deadline:\s*([^\n]+)',
            'subject': r'Subject:\s*([^\n]+)',
            'actions': r'Required Responses:(.*?)(?:\n\n|\n[A-Z]|$)',
            'contact': r'Contact Person:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and date
        parts.append(f"{doc_type} document")
        
        # Add sender/recipient context
        if 'sender' in extracted:
            parts.append(f"from {extracted['sender']}")
        
        if 'recipient' in extracted:
            parts.append(f"to {extracted['recipient']}")
        
        # Add subject
        if 'subject' in extracted:
            subject = extracted['subject'][:60] + "..." if len(extracted['subject']) > 60 else extracted['subject']
            parts.append(f"regarding {subject}")
        
        # Add deadline
        if 'deadline' in extracted:
            parts.append(f"Deadline: {extracted['deadline']}")
        
        # Add contact
        if 'contact' in extracted:
            parts.append(f"Contact: {extracted['contact']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["with specified submission requirements", "following established procedures", "requiring timely response"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used administrative fallback summary: {len(summary.split())} words")
        return summary