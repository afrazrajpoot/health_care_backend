"""
DecisionDocumentExtractor - Enhanced Extractor for UR/IMR Decisions, Appeals, Authorizations
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
from utils.summary_helpers import ensure_date_and_author, clean_long_summary

logger = logging.getLogger("document_ai")


class DecisionDocumentExtractor:
    """
    Enhanced Decision Document extractor with FULL CONTEXT processing.
    
    Key Features:
    - Full document context (no chunking) = No information loss
    - Context-aware extraction for different decision types
    - Chain-of-thought reasoning = Explains decisions
    - Optimized for accuracy matching Gemini's approach
    - Supports UR/IMR, Appeals, Authorizations, RFA, DFR
    - Direct LLM generation for long summary (removes intermediate extraction)
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.verifier = ExtractionVerifier(llm)
        
        # Pre-compile regex for efficiency
        self.decision_patterns = {
            'approved': re.compile(r'\b(approved|authorized|granted|allowed|certified)\b', re.IGNORECASE),
            'denied': re.compile(r'\b(denied|rejected|disallowed|not authorized|not approved)\b', re.IGNORECASE),
            'partially_approved': re.compile(r'\b(partially|partially approved|partially authorized)\b', re.IGNORECASE),
            'pending': re.compile(r'\b(pending|under review|being reviewed|in process)\b', re.IGNORECASE),
            'appeal': re.compile(r'\b(appeal|reconsideration|reevaluation|review requested)\b', re.IGNORECASE)
        }
        
        # Document type identifiers
        self.doc_type_patterns = {
            'ur_imr': re.compile(r'\b(UR|UM|Utilization Review|IMR|Independent Medical Review)\b', re.IGNORECASE),
            'appeal': re.compile(r'\b(Appeal|Reconsideration|Level I|Level II|Final Appeal)\b', re.IGNORECASE),
            'authorization': re.compile(r'\b(Authorization|RFA|Request for Authorization|Treatment Authorization)\b', re.IGNORECASE),
            'dfr': re.compile(r'\b(DFR|Doctor First Report|First Report|Initial Report)\b', re.IGNORECASE),
            'denial': re.compile(r'\b(Denial|Notice of Denial|Determination of Denial)\b', re.IGNORECASE)
        }

        # ENHANCED: Pre-compile regex for signature extraction to assist LLM
        self.signature_patterns = {
            'electronic_signature': re.compile(r'(electronically signed|signature|e-signed|signed by|authenticated by|digital signature|verified by)[:\s]*([A-Za-z\s\.,]+?)(?=\n|\s{2,}|$)', re.IGNORECASE | re.DOTALL),
            'signed_date': re.compile(r'(signed|signature|date)[:\s]*([A-Za-z\s\.,]+?)\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}\s*(AM|PM))?', re.IGNORECASE),
            'provider_signature': re.compile(r'(provider|doctor|dc|md|do|reviewer|physician)[:\s]*([A-Za-z\s\.,]+?)(?=\s{2,}|\n\n|$)', re.IGNORECASE),
            # NEW: Patterns for physical signatures, footers, or stamps
            'physical_signature': re.compile(r'(handwritten signature|ink signature|physical sign)[:\s]*([A-Za-z\s\.,]+?)(?=\n|$)', re.IGNORECASE),
            'footer_signature': re.compile(r'^(?:\s*[-=]{3,}.*?\n){0,3}([A-Za-z\s\.,]+?)\s*(?:\d{1,2}/\d{1,2}/\d{4}.*)?$', re.IGNORECASE | re.MULTILINE | re.DOTALL),
            'auth_stamp': re.compile(r'(authentica|stamp|seal)[:\s]*([A-Za-z\s\.,]+?)(?=\s{2,}|$)', re.IGNORECASE),
            # ENHANCED: Specific pattern for last provider/signer
            'last_provider_sign': re.compile(r'(Provider:\s*)([A-Za-z\s\.,]+?)(?=\s*\(|Review|\n{2,})', re.IGNORECASE | re.DOTALL)
        }
        
        logger.info("✅ DecisionDocumentExtractor initialized (Full Context + Enhanced Signature Extraction)")

    def extract(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str
    ) -> Dict:
        """
        Extract Decision Document data with FULL CONTEXT using raw text.
        
        Args:
            text: Complete document text (layout-preserved)
            raw_text: Summarized original context from Document AI
            doc_type: Document type (UR/IMR/Appeal/Authorization/RFA/DFR)
            fallback_date: Fallback date if not found
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("⚖️ STARTING DECISION DOCUMENT EXTRACTION (FULL CONTEXT + RAW TEXT)")
        logger.info("=" * 80)
        
        logger.info(f"📋 Document Type: {doc_type}")
        
        # Check document size
        text_length = len(raw_text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # ENHANCED: Pre-extract potential signatures using regex to guide LLM
        potential_signatures = self._pre_extract_signatures(text)
        logger.info(f"🔍 Pre-extracted potential signatures: {potential_signatures}")
        
        # Stage 1: Generate long summary with dual-context approach (raw_text + text)
        long_summary = self._generate_long_summary_direct(
            text=text,
            raw_text=raw_text,
            doc_type=doc_type,
            fallback_date=fallback_date,
            potential_signatures=potential_signatures
        )
        
        # ENHANCED: Verify and inject author into long summary if needed
        verified_author = self._verify_and_extract_author(long_summary, text, potential_signatures)
        long_summary = self._inject_author_into_long_summary(long_summary, verified_author)
        
        # Stage 1.5: Clean the long summary - remove empty fields, placeholders, and instruction text
        long_summary = clean_long_summary(long_summary)

        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, doc_type)

        logger.info("=" * 80)
        logger.info("✅ DECISION DOCUMENT EXTRACTION COMPLETE (2 LLM CALLS ONLY)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

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

    def _verify_and_extract_author(self, long_summary: str, full_text: str, potential_signatures: List[str]) -> str:
        """
        ENHANCED: Use LLM to double-check author extraction, with stronger focus on last signer.
        Falls back to regex if fails. Prioritizes electronic/physical signers over reviewers.
        """
        # ENHANCED Prompt for verification: Focus on author only, emphasize last provider/signer
        verify_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
            You are verifying the AUTHOR/SIGNER of a decision document. CRITICAL: The AUTHOR is the person who ELECTRONICALLY OR PHYSICALLY SIGNED the report, often the last reviewer/provider mentioned or in the signature block (e.g., Dr. Jane Doe as electronic signer).
            Scan the FULL TEXT and LONG SUMMARY. Extract ONLY the exact name of the person who signed.
            - Prioritize end-of-document signatures or last reviewer/provider entry.
            - Differentiate from Requesting/Reviewing Provider (e.g., Dr. Smith is reviewer, but Dr. Doe signed).
            - Use candidates for confirmation, especially those with 'signed' or last 'Provider:'.
            - If unclear, return "No distinct signer identified; fallback to reviewer: [Name]".
            Output ONLY the name as JSON: {{"author": "Exact Name (e.g., Dr. Jane Doe)"}}.
            """),
            HumanMessagePromptTemplate.from_template("""
            Full Text: {full_text}
            Long Summary: {long_summary}
            Candidates: {potential_signatures}
            """)
        ])
        
        try:
            chain = verify_prompt | self.llm
            result = chain.invoke({
                "full_text": full_text,
                "long_summary": long_summary,
                "potential_signatures": "\n".join(potential_signatures)
            })
            parsed = self.parser.parse(result.content)
            verified_author = parsed.get("author", "")
            
            if not verified_author or "No distinct" in verified_author:
                # ENHANCED Fallback: Scan for last provider in sections
                last_provider_pattern = re.compile(r'(Provider:\s*)([A-Za-z\s\.,]+?)(?=\s*\(|Review|\n{2,}|$)', re.IGNORECASE | re.DOTALL)
                match = last_provider_pattern.search(full_text + "\n" + long_summary)
                if match:
                    verified_author = match.group(2).strip()
                elif potential_signatures:
                    for cand in reversed(potential_signatures):  # Prioritize last
                        if any(word in cand.lower() for word in ['signed', 'signature', 'provider']):
                            verified_author = cand.split(':')[-1].strip()
                            break
                else:
                    verified_author = "Signature not identified"
            
            logger.info(f"🔍 Verified Author: {verified_author}")
            return verified_author
        except Exception as e:
            logger.warning(f"Author verification failed: {e}. Using raw candidates.")
            # ENHANCED: Prioritize last candidate
            return potential_signatures[-1].split(':')[-1].strip() if potential_signatures else "Author not extracted"

    def _inject_author_into_long_summary(self, long_summary: str, verified_author: str) -> str:
        """
        NEW: Robust post-processing to inject the verified author prominently if not present.
        Handles cases where LLM deviates from format by inserting under PARTIES INVOLVED or at top.
        """
        if not verified_author or verified_author in ["Signature not identified", "Author not extracted"]:
            return long_summary
        
        # Check if Author already exists
        if re.search(r'Author:\s*' + re.escape(verified_author), long_summary, re.IGNORECASE):
            return long_summary
        
        # ENHANCED: Look for PARTIES INVOLVED section (exact match)
        parties_section_pattern = re.compile(r'(👥 PARTIES INVOLVED\s*[-—]+\s*)', re.IGNORECASE)
        if parties_section_pattern.search(long_summary):
            insertion = r'\1  Author: {}\n'.format(verified_author)
            long_summary = parties_section_pattern.sub(insertion, long_summary)
            logger.info(f"✅ Injected Author into PARTIES INVOLVED section: {verified_author}")
            return long_summary
        
        # Fallback 1: Inject after first "Provider:" mention
        first_provider_pattern = re.compile(r'(Provider:\s*[^\n\r]+?)(?=\n|\r|$)', re.IGNORECASE | re.MULTILINE)
        match = first_provider_pattern.search(long_summary)
        if match:
            insertion_point = match.end()
            long_summary = long_summary[:insertion_point] + f"\n  Author: {verified_author}" + long_summary[insertion_point:]
            logger.info(f"✅ Injected Author after first Provider: {verified_author}")
            return long_summary
        
        # Fallback 2: Inject at the very top under a new PARTIES header if no provider mentioned
        long_summary = f"👥 PARTIES INVOLVED\n--------------------------------------------------\nAuthor: {verified_author}\n\n{long_summary}"
        logger.info(f"✅ Injected new PARTIES section with Author: {verified_author}")
        return long_summary

    def _generate_long_summary_direct(
        self,
        text: str,
        raw_text: str,
        doc_type: str,
        fallback_date: str,
        potential_signatures: List[str]
    ) -> str:
        """
        Generate long summary with PRIORITIZED context hierarchy:
        1. PRIMARY SOURCE: raw_text (accurate Document AI summarized context)
        2. SUPPLEMENTARY: text (full OCR extraction for missing details only)
        
        This ensures accurate context preservation while capturing all necessary details.
        """
        logger.info("🔍 Processing decision document with DUAL-CONTEXT approach...")
        # logger.info(f"   📌 PRIMARY SOURCE (raw_text): {len(raw_text):,} chars (accurate context)")
        # logger.info(f"   📄 SUPPLEMENTARY (full text): {len(text):,} chars (detail reference)")
        
        # Build system prompt with CLEAR PRIORITY INSTRUCTIONS
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical-legal documentation specialist analyzing a COMPLETE {doc_type} decision document.

🎯 CRITICAL CONTEXT HIERARCHY (HIGHEST PRIORITY):

You are provided with TWO versions of the document:

1. **PRIMARY SOURCE - "ACCURATE CONTEXT" (raw_text)**:
   - This is the MOST ACCURATE, context-aware summary generated by Google's Document AI foundation model
   - It has been intelligently processed to preserve CRITICAL DECISION CONTEXT
   - **USE THIS AS YOUR PRIMARY SOURCE OF TRUTH**
   - This contains the CORRECT decision interpretations, accurate findings, and proper context
   - **ALWAYS PRIORITIZE information from this source**

2. **SUPPLEMENTARY SOURCE - "FULL TEXT EXTRACTION" (text)**:
   - This is the complete OCR text extraction (may have formatting noise, OCR artifacts)
   - Use ONLY to fill in SPECIFIC DETAILS that may be missing from the accurate context
   - Examples of acceptable supplementary use:
       * Exact claim numbers or identifiers
       * Additional doctor names mentioned
       * Precise dates or measurements
       * Specific CPT codes or authorization numbers
   - **DO NOT let this override the decision context from the primary source**

🚨 ABSOLUTE ANTI-FABRICATION RULE (HIGHEST PRIORITY):
**YOU MUST ONLY EXTRACT AND SUMMARIZE INFORMATION THAT EXISTS IN THE PROVIDED SOURCES.**
- NEVER generate, infer, assume, or fabricate ANY information
- If information is NOT explicitly stated in either source → OMIT IT ENTIRELY
- An incomplete summary is 100x better than a fabricated one
- Every single piece of information in your output MUST be traceable to the source text

⚠️ STRICT ANTI-HALLUCINATION RULES:

1. **ZERO FABRICATION TOLERANCE**:
   - If a field (e.g., DOB, Claim Number, Decision) is NOT in either source → LEAVE IT BLANK or OMIT
   - NEVER write "likely", "probably", "typically", "usually" - these indicate fabrication
   - NEVER fill in "standard" or "typical" values - only actual extracted values

2. **CONTEXT PRIORITY ENFORCEMENT**:
   - When both sources provide information about the SAME decision element:
     ✅ ALWAYS use interpretation from PRIMARY SOURCE (accurate context)
     ❌ NEVER override with potentially inaccurate full text version

3. **EXTRACT ONLY EXPLICITLY STATED INFORMATION** - Empty if not mentioned
3. **NO ASSUMPTIONS** - Do not infer or add typical values
4. **PATIENT DETAILS - EXACT EXTRACTION ONLY** - Use verbatim text from either source
5. **DECISION STATUS - EXACT WORDING ONLY** - From primary source preferred
6. **SERVICES/TREATMENTS - ZERO TOLERANCE FOR ASSUMPTIONS** - Only explicitly listed
7. **SIGNATURE/AUTHOR - CRITICAL PRIORITY** - Check both sources, use pre-extracted candidates
8. **EMPTY FIELDS BETTER THAN GUESSES** - Omit if not found
9. **CRITERIA AND REGULATIONS - EXACT REFERENCES** - From primary source preferred

EXTRACTION FOCUS - 7 CRITICAL DECISION DOCUMENT CATEGORIES:

I. DOCUMENT IDENTITY & PARTIES
- Document type, dates, identification numbers
- All parties involved: patient, provider, reviewer, insurer
- Contact information for appeals/communication

II. REQUEST DETAILS (WHAT WAS REQUESTED)
- Requested services/treatments with SPECIFICS:
  * Procedure names, CPT codes if available
  * Medication names, dosages, durations
  * Therapy types, frequencies, durations
  * Diagnostic tests with body parts
- Dates of service requested
- Requesting provider details

III. DECISION STATUS & OUTCOME (MOST CRITICAL)
- Overall decision: APPROVED, DENIED, PARTIALLY APPROVED, PENDING
- **EXACT wording used in decision**
- Decision date and effective dates
- For partial decisions: EXACT breakdown of approved vs denied services

IV. MEDICAL NECESSITY DETERMINATION
- Medical necessity determination (Medically Necessary, Not Medically Necessary)
- Specific criteria applied (ODG, MTUS, ACOEM, etc.)
- Clinical rationale for decision
- Supporting evidence referenced

V. REVIEWER ANALYSIS & FINDINGS
- Clinical summary reviewed
- Key findings from records review
- Consultant/reviewer opinions if applicable
- Gaps in documentation noted

VI. APPEAL & NEXT STEPS INFORMATION
- Appeal deadlines and procedures
- Required documentation for appeals
- Contact information for questions
- Effective dates and expiration

VII. REGULATORY COMPLIANCE
- Regulatory references (California Code, Labor Code, etc.)
- Timeliness compliance statements
- Reviewer credentials and qualifications

CHAIN-OF-THOUGHT FOR SIGNATURE EXTRACTION (CRITICAL - ABSOLUTE PRIORITY):
1. Scan the LAST 20-30% of the document FIRST for signature blocks, footers, or stamps (e.g., "Electronically Signed: Dr. Jane Doe").
2. Look for keywords: "electronically signed", "e-signed", "signed by", "authenticated", "signature date". Also check last "Provider:" or "Reviewer:" in sections.
3. Cross-verify with pre-extracted candidates - confirm if they match full text (prioritize last one).
4. If physical signature implied (e.g., "hand signed"), note it but extract name only.
5. DIFFERENTIATE: Author is the SIGNER, not necessarily the Requesting Provider.
6. If no distinct signer, use reviewing provider and note "(inferred signer)".
7. Output EXACT name only - e.g., "Dr. Jane Doe (electronic signer)".

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
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate
- PATIENT DETAILS: Extract EXACT values from demographics (e.g., "John Doe", "01/01/1980")
- DECISION STATUS: Use exact wording from document
- It is BETTER to have empty fields than incorrect information
- CRITICAL: ALWAYS INCLUDE THE AUTHOR FIELD UNDER PARTIES INVOLVED. IF NOT FOUND, STATE "Signer not identified".

Now analyze this COMPLETE {doc_type} decision document and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")

        # User prompt with DUAL-CONTEXT input
        user_prompt = HumanMessagePromptTemplate.from_template("""
DOCUMENT TYPE: {doc_type}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PRIMARY SOURCE - ACCURATE CONTEXT (USE THIS FIRST):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{primary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 SUPPLEMENTARY SOURCE - FULL TEXT (USE ONLY FOR MISSING DETAILS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{supplementary_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANDATORY SIGNATURE SCAN: Read BOTH SOURCES above. Focus on the end for electronic/physical signatures. Use these candidates to CONFIRM (prioritize last signer):
{potential_signatures}

⚠️ REPORT DATE INSTRUCTION:
- Extract the ACTUAL document/decision/report date from the sources above
- DO NOT use current/today's date - only use dates explicitly mentioned in the document
- IMPORTANT: US date format is MM/DD/YYYY. Example: 11/25/2025 means November 25, 2025 (NOT day 11 of month 25)
- If no date found, use "00/00/0000" as placeholder

**INSTRUCTIONS**: Generate a comprehensive structured summary following the DUAL-CONTEXT hierarchy. Prioritize PRIMARY SOURCE for decision context. Use SUPPLEMENTARY SOURCE only for specific details (claim numbers, exact dates, additional names, CPT codes) not found in PRIMARY SOURCE.

Generate the long summary in this EXACT STRUCTURED FORMAT:

📋 DECISION DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Document Date: [extract ACTUAL date from document; if not found use "00/00/0000"]
Decision Date: [extracted from PRIMARY SOURCE]
Document ID: [extracted from either source]
Claim/Case Number: [extracted from either source]
Jurisdiction: [extracted from PRIMARY SOURCE]
Author:
hint: check the signature block mainly last pages of the report and the closing statement the person who signed the report either physically or electronically
• Signature: [extracted name/title if physical signature present or extracted name/title if electronic signature present; otherwise omit ; should not the business name or generic title like "Medical Group" or "Health Services", "Physician", "Surgeon","Pharmacist", "Radiologist", etc.]


👥 PARTIES INVOLVED
--------------------------------------------------
Patient: [name from either source]
  DOB: [CRITICAL: Extract EXACT DOB from demographics - format as MM/DD/YYYY if possible]
  Member ID: [CRITICAL: Extract EXACT member ID from identifiers]
Requesting Provider: [name from PRIMARY SOURCE]
  Specialty: [extracted from PRIMARY SOURCE]
  NPI: [extracted from either source]
Reviewing Entity: [name from PRIMARY SOURCE]
  Reviewer: [extracted from PRIMARY SOURCE]
  Credentials: [extracted from PRIMARY SOURCE]
Claims Administrator: [name from PRIMARY SOURCE]
  Author: [CRITICAL ULTIMATE PRIORITY: Using CoT above, extract the EXACT signer name from both sources (physical or electronic). Examples: "Dr. Jane Doe (electronic signer)" from "Electronically Signed: Dr. Jane Doe" or last "Provider: Dr. Jane Doe". If none, "No distinct signature; using reviewer: [name]". REQUIRED - LLM MUST POPULATE THIS. SCAN LAST PROVIDER IN SECTIONS.]

All Doctors Involved:
• [list all extracted doctors with names and titles from BOTH sources]
━━━ ALL DOCTORS EXTRACTION ━━━
- Extract ALL physician/doctor names mentioned ANYWHERE in BOTH sources.
- Include: consulting doctor, referring doctor, ordering physician, treating physician, examining physician, PCP, specialist, etc.
- Include names with credentials (MD, DO, DPM, DC, NP, PA) or doctor titles (Dr., Doctor).
- Extract ONLY actual person names, NOT pharmacy labels, business names, or generic titles.
- Format: Include titles and credentials as they appear (e.g., "Dr. John Smith, MD", "Jane Doe, DO").
- If no doctors found, leave list empty [].
                                                               
━━━ CLAIM NUMBER EXTRACTION PATTERNS  ━━━
CRITICAL: Scan BOTH SOURCES (header, footer, cc: lines, letterhead) for claim numbers.

Common claim number patterns (case-insensitive):
- if in the structured raw_text like json formatted dat, if the fileds are first and values then handle the same way to extract the claim number of accurate filed, but most of the time the fields are first and values are second then the claim number will be in the second field
- "[Claim #XXXXXXXXX]" or "[Claim #XXXXX-XXX]"
- "Claim Number: XXXXXXXXX" or "Claim #: XXXXXXXXX"
- "Claim: XXXXXXXXX" or "Claim #XXXXXXXXX"
- "WC Claim: XXXXXXXXX" or "Workers Comp Claim: XXXXXXXXX"
- "Policy/Claim: XXXXXXXXX"
- In "cc:" lines: "Broadspire [Claim #XXXXXXXXX]"
- In subject lines or reference fields: "Claim #XXXXXXX"
                                                               
📋 REQUEST DETAILS
--------------------------------------------------
Date of Service Requested: [extracted from PRIMARY SOURCE]
Request Received: [extracted from PRIMARY SOURCE]
Requested Services:
• [list up to 10 from PRIMARY SOURCE with procedure names, CPT from either source, body parts, frequency/duration]
Clinical Reason: [extracted from PRIMARY SOURCE]

⚖️ DECISION OUTCOME
--------------------------------------------------
Overall Decision: [extracted from PRIMARY SOURCE, exact wording]
Decision Details: [extracted from PRIMARY SOURCE]
Partial Decision Breakdown:
• [list up to 5 from PRIMARY SOURCE with service: decision/quantity approved/denied]
Effective Dates: [start/end from either source]

🏥 MEDICAL NECESSITY DETERMINATION
--------------------------------------------------
Medical Necessity: [extracted]
Criteria Applied: [extracted]
Clinical Rationale: [extracted]
Guidelines Referenced:
• [list up to 5]

🔍 REVIEWER ANALYSIS
--------------------------------------------------
Clinical Summary Reviewed: [extracted]
Key Findings:
• [list up to 5]
Documentation Gaps:
• [list up to 3]

🔄 APPEAL INFORMATION
--------------------------------------------------
Appeal Deadline: [extracted]
Appeal Procedures: [extracted]
Required Documentation:
• [list up to 5]
Timeframe for Response: [extracted]

🚨 CRITICAL ACTIONS REQUIRED
--------------------------------------------------
• [list up to 8 time-sensitive items]

⚠️ CRITICAL REMINDERS (donot include in output, for LLM use only):
1. For "patient_details": Extract EXACT values from demographics/headers. Examples: Patient: "John Doe", DOB: "01/01/1980". Leave empty if not explicitly stated.
2. For "overall_decision": Extract EXACT wording from document
   - If document says "not medically necessary", use: "not medically necessary"
   - If document says "authorized", use: "authorized"
   - DO NOT simplify or categorize

3. For "requested_services": Extract ONLY services explicitly listed in the REQUEST
   - Include details ONLY if explicitly stated
   - DO NOT include services mentioned as examples or comparisons

4. For "partial_decision_breakdown": Only populate if document explicitly breaks down partial approval
   - If no breakdown provided, leave empty

5. For "critical_actions_required": Include time-sensitive actions only
   - Appeal deadlines
   - Required response dates
   - Time-limited authorizations

6. For "Author": CRITICAL PRIORITY - Extract the EXACT name from the signature block, electronic signature, or end-of-document authentication. Scan the LAST 20% of the document first. Use pre-extracted candidates to confirm. If no signature found, explicitly state "No distinct signature identified; using reviewer". This field MUST be populated accurately.

7. For "Author": ABSOLUTE MUST - LLM, re-read full text now and extract signer. Full context ensures accuracy.
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for direct full-context decision long summary generation with DUAL-CONTEXT...")
            
            # Single LLM call with DUAL-CONTEXT (primary + supplementary sources)
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "doc_type": doc_type,
                "primary_source": raw_text,  # PRIMARY: Document AI summarizer output
                "supplementary_source": text,  # SUPPLEMENTARY: Full OCR text
                "fallback_date": fallback_date,
                "potential_signatures": "\n".join(potential_signatures) if potential_signatures else "No pre-extracted candidates found."
            })
            
            long_summary = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Direct decision long summary generation completed with DUAL-CONTEXT in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary using:")
            logger.info(f"   - PRIMARY SOURCE: {len(raw_text):,} chars")
            logger.info(f"   - SUPPLEMENTARY SOURCE: {len(text):,} chars")
            
            return long_summary
            
        except Exception as e:
            logger.error(f"❌ Direct decision long summary generation failed: {e}", exc_info=True)
            
            # Check if context length exceeded
            if "context_length_exceeded" in str(e).lower() or "maximum context" in str(e).lower():
                logger.error("❌ Document exceeds GPT-4o 128K context window")
                logger.error("❌ Consider implementing chunked fallback for very large documents")
            
            # Fallback: Generate a minimal summary
            return f"Fallback long summary for {doc_type} on {fallback_date}: Document processing failed due to {str(e)}"

    def _generate_short_summary_from_long_summary(self, long_summary: str, doc_type: str) -> str:
        """
        Generate a precise 20-50 word, pipe-delimited actionable summary in key-value format.
        No hallucination, no assumptions. Missing fields are omitted.
        """
        logger.info("🎯 Generating 20-50 word actionable short summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
You are a medical-legal extraction specialist.

TASK:
Generate a concise, highly actionable summary from a VERIFIED long medical summary. 

MANDATORY FORMAT - EXACTLY THIS STRUCTURE:
[Document Type] | [Author] | [Date : value] | [Body Parts] | Decision:[value]

STRICT REQUIREMENTS:
1. Word count MUST be between **20 and 50 words** (count carefully).
2. Output format MUST be EXACTLY as shown above.
3. Use pipe separators (|) between fields.
4. ONLY include fields that have actual extracted values - omit entire field if missing.
5. DO NOT fabricate or infer missing data.
6. Output must be a SINGLE LINE (no line breaks).

FIELD EXTRACTION RULES:
- **Document Type**: Extract exact type (e.g., "UR Decision", "IMR Appeal", "Authorization")
- **Author**: Extract VERIFIED signer name from PARTIES section (e.g., "Jane Doe") but never use Dr. with it
  * If no distinct author found, OMIT this field entirely
  * Do NOT use generic titles or business names
- **Date**: Extract decision/document date with key (e.g., "Date : 01/15/2025")
- **Body Parts**: List affected body parts if mentioned (e.g., "Body Parts : Lumbar Spine, Right Knee")
  * If no body parts mentioned, OMIT this field
- **Decision**: Extract decision outcome (e.g., "Decision: APPROVED", "Decision: DENIED", "Decision: PENDING", "Decision: MODIFIED", "Decision: PARTIALLY APPROVED/DENIED (with breakdown or details if available)")

ABSOLUTELY FORBIDDEN:
- Patient personal details (name, DOB, Member ID, DOI)
- Invented or placeholder information
- Empty pipe fields (||)
- Generic text like "Not provided" or "Unknown"
- Narrative sentences
- Word count outside 20-50 range

EXAMPLES:
✅ GOOD: "UR Decision | Smith | Date : 01/15/2025 | Body Parts : Lumbar Spine | Decision: DENIED | Recommendations: File IMR appeal by 02/01"

✅ GOOD (minimal): "IMR Appeal | Date : 01/20/2025 | Decision: APPROVED | Recommendations: Schedule MRI within 7 days"

❌ BAD: "UR Decision | | | | Decision: DENIED |" (too many empty fields)
❌ BAD: "This is a UR decision letter that was denied..." (narrative format)

Your final output MUST be 20-50 words, single-line, pipe-delimited, including ONLY extracted information.
""")

        user_prompt = HumanMessagePromptTemplate.from_template("""
LONG SUMMARY:

{long_summary}

Generate the summary in EXACTLY this format:
[Document Type] | [Author] | [Date : value] | [Body Parts] | Decision:[value]

Remember:
- 20-50 words total
- Single line
- Omit fields if not found
- NO patient details
- Count words before responding
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        # Retry configuration
        max_retries = 4
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for short summary generation...")
                
                chain = chat_prompt | self.llm
                response = chain.invoke({
                    "doc_type": doc_type,
                    "long_summary": long_summary
                })
                
                summary = response.content.strip()
                end_time = time.time()
                # Programmatically add missing Date or Author if LLM missed them
                summary = ensure_date_and_author(summary, long_summary)
                # Clean whitespace and normalize pipes
                summary = re.sub(r'\s+', ' ', summary).strip()
                summary = re.sub(r'\s*\|\s*', ' | ', summary)  # Normalize pipe spacing
                
                # Remove patient details
                forbidden_patterns = [r'Patient[:\s]+[^|]+\|?', r'DOB[:\s]+[^|]+\|?', r'Member\s+ID[:\s]+[^|]+\|?', r'DOI[:\s]+[^|]+\|?']
                for pattern in forbidden_patterns:
                    summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
                
                # Clean up double pipes
                summary = re.sub(r'\|\s*\|', '|', summary)
                summary = re.sub(r'\s+', ' ', summary).strip()
                
                word_count = len(summary.split())
                
                logger.info(f"⚡ Short summary generated in {end_time - start_time:.2f}s: {word_count} words")
                logger.info(f"📝 Summary: {summary}")
                
                # Validate word count (20-50 words)
                if 20 <= word_count <= 50:
                    logger.info("✅ Perfect 20-50 word summary generated!")
                    return summary
                else:
                    logger.warning(f"⚠️ Summary has {word_count} words (expected 20-50), attempt {attempt + 1}")
                    
                    if attempt < max_retries - 1:
                        # Add specific feedback based on word count
                        if word_count > 50:
                            feedback = f"Your previous summary had {word_count} words (TOO LONG). Remove less critical details. Target: 20-50 words. Keep ONLY: Document Type, Author (if found), Date, Body Parts (if mentioned), Decision, Recommendations (if given). Use format: [Document Type] | [Author] | [Date : value] | [Body Parts] | Decision:[value] | [Recommendations: value]"
                        else:
                            feedback = f"Your previous summary had {word_count} words (TOO SHORT). Add more specific details to reach 20-50 words. Include decision rationale or specific service details. Use format: [Document Type] | [Author] | [Date : value] | [Body Parts] | Decision:[value] | [Recommendations: value]"
                        
                        feedback_prompt = SystemMessagePromptTemplate.from_template(
                            f"{feedback}\n\nCRITICAL: Count words before responding. Output must be 20-50 words, single line, pipe-delimited."
                        )
                        chat_prompt = ChatPromptTemplate.from_messages([feedback_prompt, user_prompt])
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"⚠️ Final summary has {word_count} words after {max_retries} attempts")
                        # Force adjust if close enough
                        if word_count > 50:
                            words = summary.split()
                            summary = ' '.join(words[:50])
                            logger.info(f"🔧 Force-trimmed to 50 words")
                        return summary
                        
            except Exception as e:
                logger.error(f"❌ Short summary generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay * (attempt + 1)} seconds...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"❌ All {max_retries} attempts failed for short summary generation")
                    return f"{doc_type} | Decision processing failed"
        
        return f"{doc_type} | Decision processing failed"
    def _get_word_count_feedback_prompt(self, actual_word_count: int, doc_type: str) -> SystemMessagePromptTemplate:
        """Get feedback prompt for word count adjustment for decision documents"""
        
        if actual_word_count > 60:
            feedback = f"Your previous {doc_type} summary had {actual_word_count} words (TOO LONG). Remove less critical details to reach exactly 60 words. Prioritize: decision outcome, key services, medical necessity, appeal deadline."
        else:
            feedback = f"Your previous {doc_type} summary had {actual_word_count} words (TOO SHORT). Add more specific decision details to reach exactly 60 words. Include: specific service names, decision rationale, timeframe details."
        
        return SystemMessagePromptTemplate.from_template(f"""
You are a medical-legal specialist creating PRECISE 60-word summaries of {doc_type} documents.

CRITICAL FEEDBACK: {feedback}

REQUIREMENTS:
- EXACTLY 60 words
- Include: decision outcome, key services, medical necessity determination, appeal information
- Count words carefully before responding
- Adjust length by adding/removing specific decision details

""")

    def _clean_and_validate_short_summary(self, summary: str) -> str:
        """Clean and validate the 60-word short summary (same as QME version)"""
        # Remove excessive whitespace, quotes, and markdown
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = summary.replace('"', '').replace("'", "")
        summary = re.sub(r'[\*\#\-]', '', summary)
        
        # Remove common prefixes
        summary = re.sub(r'^(60-word summary:|summary:|decision summary:)\s*', '', summary, flags=re.IGNORECASE)
        
        # Count words
        words = summary.split()
        
        # Strict word count enforcement
        if len(words) != 60:
            logger.info(f"📝 Decision word count adjustment needed: {len(words)} words")
            
            if len(words) > 60:
                summary = self._trim_to_60_words(words)
            else:
                summary = self._expand_to_60_words(words, summary)
        
        return summary

    def _trim_to_60_words(self, words: List[str]) -> str:
        """Intelligently trim words to reach exactly 60 (same as QME version)"""
        if len(words) <= 60:
            return ' '.join(words)
        
        text = ' '.join(words)
        
        # Decision-specific reductions
        reductions = [
            (r'\b(and|with|including)\s+appropriate\s+', ' '),
            (r'\bfor\s+(a|the)\s+period\s+of\s+\w+\s+\w+', ' '),
            (r'\bwith\s+follow[- ]?up\s+in\s+\w+\s+\w+', ' with follow-up'),
            (r'\bmedical\s+necessity', 'med necessity'),
            (r'\brequested\s+services?\s*', 'requested '),
            (r'\bdetermination\s+', 'determ '),
        ]
        
        for pattern, replacement in reductions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        words = text.split()
        if len(words) > 60:
            excess = len(words) - 60
            mid_point = len(words) // 2
            start_remove = mid_point - excess // 2
            words = words[:start_remove] + words[start_remove + excess:]
        
        return ' '.join(words[:60])

    def _expand_to_60_words(self, words: List[str], original_text: str) -> str:
        """Intelligently expand text to reach exactly 60 words (same as QME version)"""
        if len(words) >= 60:
            return ' '.join(words)
        
        needed_words = 60 - len(words)
        
        # Decision-specific expansions
        expansions = []
        
        if 'appeal' in original_text.lower():
            expansions.append("with specified appeal procedures")
        
        if 'medical necessity' in original_text.lower():
            expansions.append("based on established guidelines")
        
        if 'partial' in original_text.lower():
            expansions.append("with specific service limitations")
        
        # Add generic decision context
        while len(words) + len(expansions) < 60 and len(expansions) < 5:
            expansions.extend([
                "per established medical guidelines",
                "with detailed clinical rationale", 
                "based on documentation review",
                "following utilization review",
                "with specified effective dates"
            ])
        
        # Add expansions to the text
        expanded_text = original_text
        for expansion in expansions[:needed_words]:
            expanded_text += f" {expansion}"
        
        words = expanded_text.split()
        return ' '.join(words[:60])

    def _create_decision_fallback_summary(self, long_summary: str, doc_type: str) -> str:
        """Create comprehensive fallback decision summary directly from long summary"""
        
        # Extract key decision information using regex patterns
        patterns = {
            'decision': r'Overall Decision:\s*([^\n]+)',
            'provider': r'Requesting Provider:\s*([^\n]+)',
            'author': r'Author:\s*([^\n]+)',  # ENHANCED: For signer
            'services': r'Requested Services:(.*?)(?:\n\n|\n[A-Z]|$)',
            'medical_necessity': r'Medical Necessity:\s*([^\n]+)',
            'appeal_deadline': r'Appeal Deadline:\s*([^\n]+)',
            'rationale': r'Clinical Rationale:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, long_summary, re.DOTALL)
            if match:
                extracted[key] = match.group(1).strip()
        
        # Build comprehensive summary
        parts = []
        
        # Start with document type and decision
        parts.append(f"{doc_type} decision")
        
        if 'decision' in extracted:
            parts.append(f"outcome: {extracted['decision']}")
        
        # Add provider and author context
        if 'provider' in extracted:
            parts.append(f"for {extracted['provider']}")
        if 'author' in extracted:
            parts.append(f"(Signed: {extracted['author']})")
        
        # Add services
        if 'services' in extracted:
            first_service = extracted['services'].split('\n')[0].replace('•', '').strip()[:80]
            parts.append(f"regarding {first_service}")
        
        # Add medical necessity
        if 'medical_necessity' in extracted:
            parts.append(f"Medical necessity: {extracted['medical_necessity']}")
        
        # Add appeal information
        if 'appeal_deadline' in extracted:
            parts.append(f"Appeal deadline: {extracted['appeal_deadline']}")
        
        summary = ". ".join(parts)
        
        # Ensure exactly 60 words
        words = summary.split()
        if len(words) > 60:
            summary = ' '.join(words[:60])
        elif len(words) < 60:
            padding = ["based on clinical documentation review", "following established guidelines", "with specified determination date"]
            while len(words) < 60 and padding:
                words.extend(padding.pop(0).split())
            summary = ' '.join(words[:60])
        
        logger.info(f"🔄 Used decision fallback summary: {len(summary.split())} words")
        return summary