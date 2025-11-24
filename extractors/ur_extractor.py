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
        
        logger.info("✅ DecisionDocumentExtractor initialized (Full Context)")

    def extract(
        self,
        text: str,
        doc_type: str,
        fallback_date: str,
   
    ) -> Dict:
        """
        Extract Decision Document data with FULL CONTEXT.
        
        Args:
            text: Complete document text (layout-preserved)
            doc_type: Document type (UR/IMR/Appeal/Authorization/RFA/DFR)
            fallback_date: Fallback date if not found
            raw_text: Original flat text (optional)
            
        Returns:
            Dict with long_summary and short_summary
        """
        logger.info("=" * 80)
        logger.info("⚖️ STARTING DECISION DOCUMENT EXTRACTION (FULL CONTEXT)")
        logger.info("=" * 80)
        
        logger.info(f"📋 Document Type: {doc_type}")
        
        # Check document size
        text_length = len(text)
        token_estimate = text_length // 4
        logger.info(f"📄 Document size: {text_length:,} chars (~{token_estimate:,} tokens)")
        
        if token_estimate > 120000:
            logger.warning(f"⚠️ Document very large ({token_estimate:,} tokens)")
            logger.warning("⚠️ May exceed GPT-4o context window (128K tokens)")
        
        # Stage 1: Directly generate long summary with FULL CONTEXT (no intermediate extraction)
        long_summary = self._generate_long_summary_direct(
            text=text,
            doc_type=doc_type,
            fallback_date=fallback_date
        )

        # Stage 2: Generate short summary from long summary
        short_summary = self._generate_short_summary_from_long_summary(long_summary, doc_type)

        logger.info("=" * 80)
        logger.info("✅ DECISION DOCUMENT EXTRACTION COMPLETE (FULL CONTEXT)")
        logger.info("=" * 80)

        # Return dictionary with both summaries
        return {
            "long_summary": long_summary,
            "short_summary": short_summary
        }

    def _generate_long_summary_direct(
        self,
        text: str,
        doc_type: str,
        fallback_date: str
    ) -> str:
        """
        Directly generate comprehensive long summary with FULL document context using LLM.
        Adapted from original extraction prompt to output structured summary directly.
        """
        logger.info("🔍 Processing ENTIRE decision document in single context window for direct long summary...")
        
        # Adapted System Prompt for Direct Long Summary Generation
        # Reuses core anti-hallucination rules and decision focus from original extraction prompt
        system_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert medical-legal documentation specialist analyzing a COMPLETE {doc_type} decision document.

CRITICAL ADVANTAGE - FULL CONTEXT PROCESSING:
You are seeing the ENTIRE document at once, allowing you to:
- Understand the complete decision rationale from start to finish
- Connect request details with decision criteria and justification
- Identify all services/treatments being decided upon
- Provide comprehensive extraction without information loss

⚠️ CRITICAL ANTI-HALLUCINATION RULES (HIGHEST PRIORITY):

1. **EXTRACT ONLY EXPLICITLY STATED INFORMATION**
   - If a field/value is NOT explicitly mentioned in the document, return EMPTY string "" or empty list []
   - DO NOT infer, assume, or extrapolate information
   - DO NOT fill in "typical" or "common" values
   
2. **DECISION STATUS - EXACT WORDING ONLY**
   - Extract decision status using EXACT wording from document
   - DO NOT interpret or categorize (e.g., if document says "not medically necessary", use that exact phrase)
   - For partial approvals, extract EXACTLY what was approved vs denied
   
3. **SERVICES/TREATMENTS - ZERO TOLERANCE FOR ASSUMPTIONS**
   - Extract ONLY services/treatments explicitly listed in the request/decision
   - Include quantities/durations ONLY if explicitly stated
   - DO NOT extract services mentioned as examples, comparisons, or historical context
   
4. **EMPTY FIELDS ARE ACCEPTABLE - DO NOT FILL**
   - It is BETTER to return an empty field than to guess
   - If you cannot find information for a field, leave it empty
   - DO NOT use phrases like "Not mentioned", "Not stated", "Unknown" - just return ""
   
5. **CRITERIA AND REGULATIONS - EXACT REFERENCES**
   - Extract medical necessity criteria EXACTLY as stated
   - Include specific guideline references (e.g., "ODG", "MTUS", "ACOEM")
   - DO NOT add criteria not explicitly referenced

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

⚠️ FINAL REMINDER:
- If information is NOT in the document, return EMPTY ("" or [])
- NEVER assume, infer, or extrapolate
- DECISION STATUS: Use exact wording from document
- It is BETTER to have empty fields than incorrect information

Now analyze this COMPLETE {doc_type} decision document and generate a COMPREHENSIVE STRUCTURED LONG SUMMARY with the following EXACT format (use markdown headings and bullet points for clarity):
""")

        # Adapted User Prompt for Direct Long Summary - Outputs the structured summary directly
        user_prompt = HumanMessagePromptTemplate.from_template("""
COMPLETE {doc_type} DECISION DOCUMENT TEXT:

{full_document_text}

Generate the long summary in this EXACT STRUCTURED FORMAT (use the fallback date {fallback_date} if no dates found):

📋 DECISION DOCUMENT OVERVIEW
--------------------------------------------------
Document Type: {doc_type}
Document Date: [extracted or {fallback_date}]
Decision Date: [extracted]
Document ID: [extracted]
Claim/Case Number: [extracted]
Jurisdiction: [extracted]

👥 PARTIES INVOLVED
--------------------------------------------------
Patient: [name]
  DOB: [extracted]
  Member ID: [extracted]
Requesting Provider: [name]
  Specialty: [extracted]
  NPI: [extracted]
Reviewing Entity: [name]
  Reviewer: [extracted]
  Credentials: [extracted]
Claims Administrator: [name]

📋 REQUEST DETAILS
--------------------------------------------------
Date of Service Requested: [extracted]
Request Received: [extracted]
Requested Services:
• [list up to 10 with procedure names, CPT, body parts, frequency/duration]
Clinical Reason: [extracted]

⚖️ DECISION OUTCOME
--------------------------------------------------
Overall Decision: [extracted, exact wording]
Decision Details: [extracted]
Partial Decision Breakdown:
• [list up to 5 with service: decision/quantity approved/denied]
Effective Dates: [start/end]

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

⚠️ CRITICAL REMINDERS:
1. For "overall_decision": Extract EXACT wording from document
   - If document says "not medically necessary", use: "not medically necessary"
   - If document says "authorized", use: "authorized"
   - DO NOT simplify or categorize

2. For "requested_services": Extract ONLY services explicitly listed in the REQUEST
   - Include details ONLY if explicitly stated
   - DO NOT include services mentioned as examples or comparisons

3. For "partial_decision_breakdown": Only populate if document explicitly breaks down partial approval
   - If no breakdown provided, leave empty

4. For "critical_actions_required": Include time-sensitive actions only
   - Appeal deadlines
   - Required response dates
   - Time-limited authorizations
""")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        try:
            start_time = time.time()
            
            logger.info("🤖 Invoking LLM for direct full-context decision long summary generation...")
            
            # Single LLM call with FULL document context to generate long summary directly
            chain = chat_prompt | self.llm
            result = chain.invoke({
                "doc_type": doc_type,
                "full_document_text": text,
                "fallback_date": fallback_date
            })
            
            long_summary = result.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"⚡ Direct decision long summary generation completed in {processing_time:.2f}s")
            logger.info(f"✅ Generated long summary from complete {len(text):,} char document")
            
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
        Generate a precise 30–60 word, pipe-delimited actionable summary in key-value format.
        No hallucination, no assumptions. Missing fields are omitted.
        """
        logger.info("🎯 Generating 30–60 word actionable short summary (key-value format)...")

        system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a medical-legal extraction specialist.

    Your task: Generate a short, highly actionable summary from a VERIFIED long medical summary.

    STRICT REQUIREMENTS:
    1. Word count MUST be between **30 and 60 words** (min 30, max 60).
    2. Format MUST be EXACTLY:

    [Document Type] | [Decision Date] | Requesting Provider:[value] | Services:[value] | Decision:[value] | Medical Necessity:[value] | Rationale:[value] | Appeal Info:[value]

    3. DO NOT fabricate or infer missing data — simply SKIP entire key-value pairs that do not exist.
    4. Use ONLY information explicitly found in the long summary.
    5. Output must be a SINGLE LINE (no line breaks).
    6. Content priority:
    - document type
    - decision date
    - requesting provider
    - key services/treatments decided
    - final decision outcome (APPROVED/DENIED/PARTIAL)
    - medical necessity determination
    - key rationale for decision
    - appeal deadline if applicable

    7. ABSOLUTE NO:
    - assumptions
    - clinical interpretation
    - invented information
    - narrative sentences

    8. If a field is missing, SKIP THE ENTIRE KEY-VALUE PAIR—do NOT include empty key-value pairs.

    Your final output must be 30–60 words and MUST follow the exact format above.
    """)

        user_prompt = HumanMessagePromptTemplate.from_template("""
    LONG SUMMARY:

    {long_summary}

    Now produce the 30–60 word single-line summary following the strict rules.
    """)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        
        # Retry configuration
        max_retries = 3
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
                
                # Clean whitespace only
                summary = re.sub(r'\s+', ' ', summary).strip()
                
                word_count = len(summary.split())
                
                logger.info(f"⚡ Short summary generated in {end_time - start_time:.2f}s: {word_count} words")
                
                # Validate word count
                if 30 <= word_count <= 60:
                    logger.info("✅ Perfect 30-60 word summary generated!")
                    return summary
                else:
                    logger.warning(f"⚠️ Summary has {word_count} words (expected 30-60), attempt {attempt + 1}")
                    
                    if attempt < max_retries - 1:
                        # Add word count feedback to next attempt
                        feedback_prompt = SystemMessagePromptTemplate.from_template(
                            f"Your previous summary had {word_count} words. Rewrite it to be STRICTLY between 30 and 60 words while preserving accuracy and key-value format. DO NOT add invented data. Maintain the exact format: [Document Type] | [Decision Date] | Requesting Provider:[value] | Services:[value] | Decision:[value] | Medical Necessity:[value] | Rationale:[value] | Appeal Info:[value]"
                        )
                        chat_prompt = ChatPromptTemplate.from_messages([feedback_prompt, user_prompt])
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"⚠️ Final summary has {word_count} words after {max_retries} attempts")
                        return summary
                        
            except Exception as e:
                logger.error(f"❌ Short summary generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay * (attempt + 1)} seconds...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"❌ All {max_retries} attempts failed for short summary generation")
                    return "Summary unavailable due to processing error."
        
        return "Summary unavailable due to processing error."
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
        
        # Add provider context
        if 'provider' in extracted:
            parts.append(f"for {extracted['provider']}")
        
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