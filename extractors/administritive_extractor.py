"""
AdministrativeExtractor - Enhanced Extractor for Administrative and Legal Documents
Optimized for accuracy using Gemini-style full-document processing with contextual guidance
"""
import logging
import re
import time
from typing import Dict, Optional, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from utils.extraction_verifier import ExtractionVerifier
from utils.summary_helpers import ensure_date_and_author

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
        
        logger.info("✅ AdministrativeExtractor initialized (Full Context + Context-Aware)")


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
        
        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, detected_type)
        
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
        """
        logger.info("🔍 Processing administrative document with DUAL-CONTEXT approach...")
        logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
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

    ⚠️ ANTI-HALLUCINATION RULES FOR DUAL-CONTEXT:

    1. **CONTEXT PRIORITY ENFORCEMENT**:
    - When both sources provide information about the SAME administrative element:
        ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
        ❌ NEVER override with potentially inaccurate full text version
    
    2. **LEGAL DEMANDS & REQUIREMENTS PRIORITY**:
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
    Report Date: {fallback_date}

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
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        logger.info(f"📄 PRIMARY SOURCE size: {len(raw_text):,} chars")
        logger.info(f"📄 SUPPLEMENTARY size: {len(text):,} chars")
        logger.info("🤖 Invoking LLM with DUAL-CONTEXT PRIORITY approach...")
        
        try:
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
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            
            logger.info(f"⚡ Administrative long summary generated in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary: {len(long_summary):,} chars")
            logger.info("✅ Context priority maintained: PRIMARY source used for administrative findings")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct administrative document long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Administrative document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 30–60 word administrative summary in key-value format.
        Pipe-delimited, zero hallucination, skips missing fields.
        Includes Author (report signer) but excludes patient details.
        """

        logger.info("🎯 Generating 30–60 word administrative structured summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an administrative and legal-document extraction specialist.

TASK:
Create a concise, accurate administrative summary using ONLY information explicitly present in the long summary.

STRICT REQUIREMENTS:
1. Word count MUST be **between 30 and 60 words**.
2. Output format MUST be EXACTLY:

[Document Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | Physical Exam:[value] | Vital Signs:[value] | Medication:[value] | MMI Status:[value] | Work Status:[value] | Restrictions:[value] | Action Items:[value] | Recommendations:[value] | Critical Finding:[value] | Follow-up:[value]

NEW KEY RULES (IMPORTANT):
- **ONLY include, critical, or clinically significant findings**.
- **ONLY include abnormalities or pathological findings for physical exam and vital signs (if present). Skip normal findings entirely for these (physical exam, vital signs) fields.**
- **If a value is not extracted, omit the ENTIRE key-value pair.**
- **Never output an empty key, an empty value, or placeholders.**
- **No duplicate pipes, no empty pipes (no '||').**

FORMAT & RULES:
- MUST be **30–60 words**.
- MUST be **ONE LINE**, pipe-delimited, no line breaks.
- First three fields (Document Title, Author, Date) appear without keys
- For Author never use Dr. with it
- All other fields use key-value format: Key:[value].
- DO NOT include patient details (name, DOB, ID).
- NEVER fabricate any information or infer abnormalities.

CONTENT PRIORITY (ONLY IF AND PRESENT IN THE SUMMARY):
1. Document Title
2. Author
3. Date
4. body parts or injury locations
5. diagnoses
6. Physical exam (only abnormalities)
7. Vital signs (only abnormalities)
8. Medications (only if explicitly listed)
9. MMI status (only if explicitly stated)
10. Work status & restrictions (only if given)
11. Action items (only if they indicate issues)
12. Critical findings
13. Follow-up requirements
14. Recommendations (only if given)

ABSOLUTELY FORBIDDEN (donot include in output, for LLM use only):
- Normal findings (ignore them entirely for these fields: physical exam, vital signs)
- assumptions, interpretations, invented medications, or inferred diagnoses
- placeholder text or "Not provided"
- narrative writing
- duplicate pipes or empty pipe fields (e.g., "||")
- For Author never use Dr. with it
- any patient details (patient name, DOB, ID)

Your final output MUST be between 30–60 words and follow the exact pipe-delimited style.
""")


        user_prompt = HumanMessagePromptTemplate.from_template("""
LONG ADMINISTRATIVE SUMMARY:

{long_summary}

Now produce a 30–60 word administrative structured summary following ALL rules.
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        try:
            chain = chat_prompt | self.llm
            response = chain.invoke({"long_summary": long_summary, "doc_type": doc_type})
            summary = response.content.strip()

            # Clean formatting - only whitespace, no pipe cleaning
            summary = re.sub(r"\s+", " ", summary).strip()
            
            # Programmatically add missing Date or Author if LLM missed them
            summary = ensure_date_and_author(summary, long_summary)

            # Word count validation
            wc = len(summary.split())
            if wc < 30 or wc > 60:
                logger.warning(f"⚠️ Administrative summary out of bounds ({wc} words). Auto-correcting...")

                fix_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        f"Your prior output contained {wc} words. Rewrite it to be STRICTLY between 30 and 60 words while keeping all facts accurate and key-value pipe-delimited format. DO NOT add fabricated details. Maintain format: [Document Title] | [Author] | [Date] | Body Parts:[value] | Diagnosis:[value] | etc. DO NOT include patient details."
                    ),
                    HumanMessagePromptTemplate.from_template(summary)
                ])
                chain2 = fix_prompt | self.llm
                fixed = chain2.invoke({})
                summary = re.sub(r"\s+", " ", fixed.content.strip())
                # Re-ensure date and author after correction
                summary = ensure_date_and_author(summary, long_summary)

            logger.info(f"✅ Administrative summary generated: {len(summary.split())} words")
            return summary

        except Exception as e:
            logger.error(f"❌ Administrative summary generation failed: {e}")
            return "Summary unavailable due to processing error."
    
    
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